"""Offline acquired-grasp geometry refinement; no physical success claims.

The optimizer moves hand qpos in a scratch model only. Generated offsets are
then evaluated from the normal open-hand reset in separate physical trials.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from experiments.view_blind_competition import load
from fivefinger.paths import ROOT


def closest_axis_point(model,data,finger,center,rotation):
    geom=model.geom(f'col_r_f_link{finger}_4');pose=data.geom(geom.id)
    p=rotation.T@(pose.xpos-center)
    direction=rotation.T@pose.xmat.reshape(3,3)[:,2]*geom.size[1]
    a=p-direction;b=p+direction;v=b-a
    u=np.clip(-a[:2]@v[:2]/max(v[:2]@v[:2],1e-14),0,1)
    return a+u*v


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',choices=['paired','index-only'],default='index-only')
    parser.add_argument('--source-case',default='20260913_131527_aa0b31')
    parser.add_argument('--parent-profile',default='config/profile_palm_opposition_v35_nominal.json')
    parser.add_argument('--version-start',type=int)
    args=parser.parse_args()
    source=ROOT/'outputs/blind_competition'/args.source_case
    run,cfg,m,d,tr,physics,report,events=load(source)
    i=int(np.flatnonzero(tr['stage']=='hold')[-1])
    d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    center=d.body('brush').xpos.copy();rotation=d.body('brush').xmat.reshape(3,3).copy()
    names=[f'r_f_joint{f}_{j}' for f in (1,2,3) for j in (1,2,3,4) if (f,j)!=(3,1)]
    ids=[m.joint(n).qposadr[0] for n in names];prior=d.qpos[ids].copy()
    actuator_limits=np.array([m.actuator_ctrlrange[m.actuator('act_'+n).id] for n in names])
    nominal=np.array([cfg['grip']['grip_targets_rad'][n] for n in names])
    # Bounds also ensure applying the same delta to nominal open-to-close
    # targets remains within actuator limits.
    lower=np.maximum.reduce([prior-.1,actuator_limits[:,0],prior+actuator_limits[:,0]-nominal])
    upper=np.minimum.reduce([prior+.1,actuator_limits[:,1],prior+actuator_limits[:,1]-nominal])
    initial_points=np.array([closest_axis_point(m,d,f,center,rotation) for f in (1,2,3)])
    parent=json.loads((ROOT/args.parent_profile).read_text(encoding='utf-8'))
    results=[]
    designs=[(44,.002,-.002),(45,.004,-.004)] if args.design=='paired' else [(46,.002,0.),(47,.004,0.)]
    if args.version_start is not None:
        designs=[(args.version_start+i,shift,middle) for i,(_,shift,middle) in enumerate(designs)]
    for version,shift,middle_shift in designs:
        goal=initial_points.copy();goal[:,2]+=[0,shift,middle_shift]
        def residual(q):
            d.qpos[ids]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            points=np.array([closest_axis_point(m,d,f,center,rotation) for f in (1,2,3)])
            return np.r_[(points-goal).ravel(),.001*(q-prior)]
        result=least_squares(residual,np.clip(prior,lower+1e-10,upper-1e-10),bounds=(lower,upper),max_nfev=300,
            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        residual(result.x)
        actual=np.array([closest_axis_point(m,d,f,center,rotation) for f in (1,2,3)])
        profile=dict(parent);offsets=dict(zip(names,(result.x-prior).tolist()))
        profile['grip_target_offsets_rad']=offsets
        path=ROOT/f'config/profile_axial_grasp_v{version}.json'
        error_mm=float(np.max(np.linalg.norm(actual-goal,axis=1))*1000)
        geometry_pass=bool(result.success and error_mm<=.25)
        # Convergence alone is insufficient. Keep previous rejected artifacts,
        # but only emit new runnable profiles for geometrically feasible poses.
        if geometry_pass:
            path.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        results.append({'profile':str(path.relative_to(ROOT)),'requested_index_middle_shift_mm':[shift*1000,middle_shift*1000],
            'optimizer_success':bool(result.success),'maximum_point_error_mm':error_mm,
            'geometry_gate_pass':geometry_pass,'geometry_gate_tolerance_mm':.25,
            'disposition':'physical_trial_candidate' if geometry_pass else 'rejected_geometry_do_not_run',
            'initial_axis_points_mm':(initial_points*1000).tolist(),'optimized_axis_points_mm':(actual*1000).tolist(),
            'offsets_rad':offsets,'physical_validation_pending':geometry_pass})
    suffix='' if args.design=='paired' else '_index_only'
    if args.version_start is not None:suffix=f'_v{args.version_start}_v{args.version_start+1}'
    destination=ROOT/f'outputs/robustness/axial_grasp_design{suffix}_20260913.json'
    destination.write_text(json.dumps({'source_case':source.name,'parent_profile':args.parent_profile,'source_time_s':float(tr['time_s'][i]),
        'method':'capsule centerline closest points to fixed barrel axis; acquired pose scratch kinematics only',
        'hardware_ready':False,'candidates':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'profile':r['profile'],'error_mm':r['maximum_point_error_mm'],'geometry_gate_pass':r['geometry_gate_pass']} for r in results]))


if __name__=='__main__':main()
