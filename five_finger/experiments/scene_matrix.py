"""MuJoCo RGB-D scene variation suite; evaluates perception, not completed writing.

Scenario truth is used by the renderer/evaluator only. Detector is a separate process
given a capture directory containing RGB, aligned optical-Z depth and calibration.
"""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.competition_scene import make_scene,TableClearance
from fivefinger.autonomy.geometry import Calibration
from fivefinger.paths import ROOT


def scenarios():
    return [dict(name='baseline'),
        *[dict(name=f'paper_shift_{s}',paper_shift=[s*.03,s*.02]) for s in (-1,1)],
        *[dict(name=f'paper_yaw_{s}',paper_yaw=s*12) for s in (-1,1)],
        *[dict(name=f'camera_recalibrated_{s}',camera_shift=[s*.05,-s*.03,s*.05],camera_tilt=s*7) for s in (-1,1)],
        *[dict(name=f'brush_shift_{s}',brush_shift=[s*.02,s*.015]) for s in (-1,1)],
        *[dict(name=f'ink_shift_{s}',ink_shift=[s*.03,-s*.02]) for s in (-1,1)],
        dict(name='dim_light',brightness=.65),dict(name='depth_noise_1mm',noise=.001),
        dict(name='depth_holes_10pct',holes=.1),dict(name='paper_missing',missing='paper'),
        dict(name='brush_missing',missing='brush'),dict(name='ink_missing',missing='inkstone'),
        dict(name='paper_occlusion',occlusion=True),dict(name='camera_blackout',brightness=0),
        dict(name='depth_disconnected',holes=1.),dict(name='calibration_z_drift_20mm',drift=.020),
        dict(name='in_plane_camera_drift_20mm',xy_drift=.020),
        *[dict(name=f'preliminary_{size}',paper_size=size) for size in ('A3','A4','A5')],
        dict(name='combined_layout',paper_shift=[.025,-.015],paper_yaw=-8,camera_shift=[.03,.02,.03],
             camera_tilt=4,brush_shift=[-.015,.010],ink_shift=[.02,-.02],brightness=.8,noise=.0005)]


def extra(root):
    ET.SubElement(root.find('worldbody'),'geom',name='matrix_occluder',type='box',size='.10 .10 .03',
                  pos='0 0 3',rgba='.2 .35 .6 1',contype='4',conaffinity='3')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='');ap.add_argument('--output',type=Path,default=ROOT/'outputs/adaptive_scenes');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    m,d,_,_,_,_=make_scene(scene_modifier=extra)
    # Compiled identity geom frames otherwise bypass later quaternion changes.
    m.geom_sameframe[:]=0
    # Rendering and initialisation, not animation: no inference takes joint/geom state.
    initial={k:getattr(m,k).copy() for k in ('geom_pos','geom_quat','geom_size','geom_rgba','cam_pos','cam_quat')}
    q0=d.qpos.copy();cam=m.camera('overhead').id
    renderer=mujoco.Renderer(m,height=960,width=1280)
    summaries=[]
    cases=[s for s in scenarios() if not a.only or s['name'] in a.only.split(',')]
    for s in cases:
        for key,value in initial.items():getattr(m,key)[:]=value
        d.qpos[:]=q0
        paper=m.geom('paper').id
        m.geom_pos[paper,:2]+=s.get('paper_shift',[0,0])
        m.geom_quat[paper]=Rotation.from_euler('z',s.get('paper_yaw',0),degrees=True).as_quat()[[3,0,1,2]]
        sizes={'A3':(.297,.420),'A4':(.210,.297),'A5':(.148,.210)}
        if 'paper_size' in s:m.geom_size[paper,:2]=np.asarray(sizes[s['paper_size']])/2
        shift=np.asarray(s.get('brush_shift',[0,0]))
        d.qpos[26:28]+=shift
        for gid in range(m.ngeom):
            name=m.geom(gid).name
            if name.startswith('support_'):m.geom_pos[gid,:2]+=shift
            if name.startswith('inkstone_') or name=='ink_liquid_surface':m.geom_pos[gid,:2]+=s.get('ink_shift',[0,0])
            if s.get('missing')=='inkstone' and (name.startswith('inkstone_') or name=='ink_liquid_surface'):m.geom_pos[gid,2]+=3
        if s.get('missing')=='paper':m.geom_pos[paper,2]+=3
        if s.get('missing')=='brush':d.qpos[28]+=3
        if s.get('occlusion'):m.geom_pos[m.geom('matrix_occluder').id]=m.geom_pos[paper]+[0,0,.03]
        m.cam_pos[cam]+=s.get('camera_shift',[0,0,0])
        rot=Rotation.from_quat(m.cam_quat[cam][[1,2,3,0]])*Rotation.from_euler('x',s.get('camera_tilt',0),degrees=True)
        m.cam_quat[cam]=rot.as_quat()[[3,0,1,2]]
        mujoco.mj_forward(m,d)
        clearance,_=TableClearance(m).measure(d)
        renderer.update_scene(d,camera='overhead');rgb=renderer.render().copy()
        renderer.enable_depth_rendering();renderer.update_scene(d,camera='overhead');depth=renderer.render().copy();renderer.disable_depth_rendering()
        rng=np.random.default_rng(2709)
        rgb=np.clip(rgb*s.get('brightness',1),0,255).astype('uint8')
        depth+=rng.normal(0,s.get('noise',0),depth.shape).astype('float32')
        depth[rng.random(depth.shape)<s.get('holes',0)]=np.nan
        T=np.eye(4);T[:3,:3]=d.cam_xmat[cam].reshape(3,3)@np.diag([1,-1,-1]);T[:3,3]=d.cam_xpos[cam]
        T[2,3]+=s.get('drift',0);T[0,3]+=s.get('xy_drift',0)
        f=960/(2*np.tan(np.deg2rad(m.cam_fovy[cam])/2));K=np.array([[f,0,639.5],[0,f,479.5],[0,0,1]])
        cal=Calibration(K,T,(1280,960),'sim-calibration-'+s['name'],base_frame='reference_workcell')
        out=a.output/s['name'];out.mkdir(exist_ok=True)
        cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))[1].tofile(out/'rgb.png');np.save(out/'depth.npy',depth)
        (out/'calibration.json').write_text(json.dumps(cal.to_dict(),indent=2),encoding='utf-8')
        # No scenario metadata or hidden ground truth is passed to detector.
        run=subprocess.run([sys.executable,'-m','fivefinger.autonomy.perception','--capture',str(out)],capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        report=dict(name=s['name'],parameters=s,level='MuJoCo rendered RGB-D perception',physical_writing_executed=False,
                    initial_arm_table_clearance_m=clearance,official_score=None)
        if run.returncode:
            report.update(scene_geometry_ready=False,exception=run.stderr.strip().splitlines()[-1],safe_rejection=True)
            # Never let a previous successful detection survive a newer failed capture.
            failed=dict(scene_geometry_ready=False,paper=[],brush=[],inkstone=[],rejection_reasons=['perception_failed'],
                        coordinate_frame=cal.base_frame,hardware_motion_authorized=False,error=report['exception'])
            (out/'perception.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            result=json.loads((out/'perception.json').read_text(encoding='utf-8'))
            report.update(scene_geometry_ready=result['scene_geometry_ready'],rejection_reasons=result['rejection_reasons'],
                          detected_counts={k:len(result[k]) for k in ('paper','brush','inkstone')})
            if len(result['paper'])==1 and s.get('missing')!='paper':
                report['paper_center_error_m']=float(np.linalg.norm(np.asarray(result['paper'][0]['center_m'])[:2]-m.geom_pos[paper,:2]))
            if len(result['brush'])==1 and s.get('missing')!='brush':
                report['brush_xy_error_m']=float(np.linalg.norm(np.asarray(result['brush'][0]['visible_center_m'])[:2]-d.qpos[26:28]))
            if len(result['inkstone'])==1 and s.get('missing')!='inkstone':
                report['ink_center_error_m']=float(np.linalg.norm(np.asarray(result['inkstone'][0]['center_m'])[:2]-m.geom_pos[m.geom('inkstone_bottom').id,:2]))
        (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        summaries.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
    renderer.close()
    (a.output/('partial_summary.json' if a.only else 'summary.json')).write_text(json.dumps(dict(cases=summaries,scope='Perception coverage, not full task or hardware qualification',
        known_unobservable_fault='table plane alone cannot detect in-plane camera drift'),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
