"""Image/depth-only reference detector. No MuJoCo import, IDs or scene pose input.

Material/shape priors describe the practice props, not an official general detector.
The output deliberately contains hypotheses and rejection reasons, never motor commands.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .geometry import Calibration, fit_table


def components(mask, minimum=15):
    contours,_ = cv2.findContours(mask.astype('uint8')*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= minimum]


def perceive(rgb, depth, cal, table_reference=(0.,0.,1.,0.)):
    h,w = depth.shape
    if rgb.shape != (h,w,3) or tuple(cal.image_size) != (w,h): raise ValueError('RGB/depth/calibration size mismatch')
    yy,xx = np.mgrid[:h,:w]; valid = np.isfinite(depth)&(depth>.05)&(depth<3.)
    sparse = valid & (yy%10==0) & (xx%10==0)
    if sparse.sum()<100:raise ValueError('Insufficient valid depth: camera stream unavailable or outside calibrated range')
    points = cal.backproject(np.c_[xx[sparse],yy[sparse]],depth[sparse])
    plane,residual = fit_table(points)
    reference=np.asarray(table_reference,float)
    angle = float(np.degrees(np.arccos(np.clip(plane[:3]@reference[:3],-1,1))))
    offset = abs(float(plane[3]-reference[3]))
    out = dict(calibration_version=cal.version, coordinate_frame=cal.base_frame,table_plane=plane.tolist(), table_residual_m=residual,
               calibration_consistent=bool(angle<.5 and offset<.003), table_angle_error_deg=angle,
               table_offset_error_m=offset, paper=[], inkstone=[], brush=[], limitations=[
                   'material/geometry prior for reference props; not trained on official objects',
                   'table plane does not detect in-plane camera translation or yaw drift',
                   'ink centre is detected; liquid depth still requires a measured datum',
                   'upright brush hypothesis does not establish tip pose or a stable grasp'])
    # All dense positions come only from the RGB-aligned depth stream and calibration.
    xyz=np.full((h,w,3),np.nan)
    xyz[valid]=cal.backproject(np.c_[xx[valid],yy[valid]],depth[valid])
    height=xyz@plane[:3]+plane[3]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # White paper, near table. Shadows allowed by saturation/value thresholds.
    near=valid&(abs(height)<.009)&(hsv[:,:,1]<65)
    values=hsv[:,:,2][near]
    threshold,_=cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) if len(values) else (255,None)
    threshold=max(90,threshold,float(np.percentile(values,98))*.85) if len(values) else 255
    mask=near&(hsv[:,:,2]>threshold)
    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    out['paper_candidate_dimensions_m']=[]
    for c in components(mask,1200):
        pixels=c[:,0,:]
        pixels=pixels[near[pixels[:,1],pixels[:,0]]]
        surface=plane.copy()
        surface[3]-=float(np.nanmedian(height[pixels[:,1],pixels[:,0]]))
        samples=cal.intersect_plane(pixels,surface)
        # Morphological closing may reconnect isolated bright table/hand specks.
        # Metric rectangle fit below is intentionally rejected if the size no longer agrees.
        samples=samples[np.isfinite(samples).all(1)]
        if len(samples)<4: continue
        rect=cv2.minAreaRect(samples[:,:2].astype('float32'))
        sides=sorted(rect[1]); sizes={'A3':(.297,.420),'A4':(.210,.297),'A5':(.148,.210)}
        out['paper_candidate_dimensions_m'].append(sides)
        name,error=min(((name,float(np.max(np.abs(np.asarray(size)-sides)))) for name,size in sizes.items()),key=lambda x:x[1])
        if error>.018: continue
        corners_xy=cv2.boxPoints(rect)
        z=float(np.median(samples[:,2])); corners=np.c_[corners_xy,np.full(4,z)]
        hullarea=cv2.contourArea(cv2.convexHull(samples[:,:2].astype('float32')))
        coverage=float(hullarea/max(np.prod(sides),1e-9))
        area_mask=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(area_mask,cv2.convexHull(c).reshape(-1,2),1)
        observed_fraction=float(np.mean(near[area_mask>0]))
        out['paper'].append(dict(center_m=corners.mean(0).tolist(),corners_m=corners.tolist(),size=name,
                                 dimension_error_m=error,visible_hull_fraction=coverage,unoccluded_depth_fraction=observed_fraction,
                                 uncertainty_m=max(.002,error/2,residual*3),pose_ready=bool(error<.008 and coverage>.96 and observed_fraction>.94)))
    dark=valid&(height>.010)&(height<.080)&(hsv[:,:,2]<100)&(hsv[:,:,1]<180)
    for c in components(dark,100):
        ps=c[:,0,:]; points3=xyz[ps[:,1],ps[:,0]]
        low=np.nanmin(points3,axis=0);high=np.nanmax(points3,axis=0);size=sorted((high-low)[:2])
        if .030<size[0]<.10 and .060<size[1]<.13 and size[1]/size[0]>1.15:
            out['inkstone'].append(dict(center_m=((low+high)/2).tolist(),outer_xy_size_m=(high-low)[:2].tolist(),
                                       uncertainty_m=max(.003,residual*3),liquid_surface_measured=False))
    # Reference brown cylindrical barrel. Narrow circular top is not enough to infer a 6D pen pose.
    brown=valid&(height>.16)&(height<.40)&(hsv[:,:,0]>=5)&(hsv[:,:,0]<=25)&(hsv[:,:,1]>100)&(hsv[:,:,2]>20)&(hsv[:,:,2]<220)
    for c in components(brown,8):
        x,y,bw,bh=cv2.boundingRect(c)
        m=np.zeros((h,w),np.uint8);cv2.drawContours(m,[c],-1,1,-1)
        p=xyz[(m>0)&valid];span=np.ptp(p[:,:2],axis=0)
        if max(span)>.025 or len(p)<10: continue
        out['brush'].append(dict(visible_center_m=np.median(p,axis=0).tolist(),visible_xy_span_m=span.tolist(),
                                hypothesis='upright_brown_barrel',full_pose_observed=False,uncertainty_m=max(.003,residual*3)))
    reasons=[]
    if not out['calibration_consistent']:reasons.append('calibration_drift')
    for key in ('paper','inkstone','brush'):
        if len(out[key])!=1:reasons.append(f'{key}_missing_or_ambiguous')
    if len(out['paper'])==1 and not out['paper'][0]['pose_ready']:reasons.append('paper_pose_uncertain')
    out['scene_geometry_ready']=not reasons
    out['rejection_reasons']=reasons
    out['hardware_motion_authorized']=False
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);a=p.parse_args()
    # Capture contract has pixels, optical-Z depth, and calibration only; no truth/scene file is opened.
    rgb=cv2.cvtColor(cv2.imdecode(np.fromfile(a.capture/'rgb.png',dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    depth=np.load(a.capture/'depth.npy',allow_pickle=False)
    cal=Calibration.from_dict(json.loads((a.capture/'calibration.json').read_text(encoding='utf-8')))
    result=perceive(rgb,depth,cal)
    (a.capture/'perception.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
