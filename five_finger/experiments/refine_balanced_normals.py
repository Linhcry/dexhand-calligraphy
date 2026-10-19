"""Scratch thumb geometry to reduce nominal normal-force imbalance.

This ignores friction, gravity and contact patch moments. It is a pre-grasp
design heuristic only; generated candidates must start from the normal reset.
"""
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
from experiments.refine_axial_contacts import closest_axis_point
from experiments.view_blind_competition import load
from fivefinger.paths import ROOT


def main():
    source=ROOT/'outputs/blind_competition/20260913_141036_6f4771'
    run,cfg,m,d,tr,physics,report,events=load(source)
    i=int(np.flatnonzero(tr['stage']=='hold')[-1]);d.qpos[:]=tr['qpos'][i]
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    center=d.body('brush').xpos.copy();rotation=d.body('brush').xmat.reshape(3,3).copy()
    points=np.array([closest_axis_point(m,d,f,center,rotation) for f in (1,2,3)])
    normals=-points.copy();normals[:,2]=0;normals/=np.linalg.norm(normals,axis=1)[:,None]
    targets=np.array([2.7,1.35,1.35]);forces=normals*targets[:,None]
    thumb_normal=-forces[1:].sum(axis=0);thumb_normal/=np.linalg.norm(thumb_normal)
    goal=points[0].copy();goal[:2]=-thumb_normal[:2]*np.linalg.norm(points[0,:2])
    thumb_lever=np.cross([0,0,1],thumb_normal*targets[0])
    others_moment=np.cross(points[1:],forces[1:]).sum(axis=0)
    balanced_height=-thumb_lever@others_moment/(thumb_lever@thumb_lever)
    parent=json.loads((ROOT/'config/profile_axial_grasp_v46.json').read_text(encoding='utf-8'))
    names=[f'r_f_joint1_{j}' for j in range(1,5)];ids=[m.joint(n).qposadr[0] for n in names]
    prior=d.qpos[ids].copy();nominal=np.array([cfg['grip']['grip_targets_rad'][n] for n in names])
    existing=np.array([parent['grip_target_offsets_rad'].get(n,0.) for n in names])
    limits=np.array([m.actuator_ctrlrange[m.actuator('act_'+n).id] for n in names])
    lower=np.maximum.reduce([prior-.1-existing,limits[:,0],prior+limits[:,0]-nominal])
    upper=np.minimum.reduce([prior+.1-existing,limits[:,1],prior+limits[:,1]-nominal])
    results=[]
    for version,goal_z in [(51,points[0,2]),(52,balanced_height)]:
        desired=goal.copy();desired[2]=goal_z
        def residual(q):
            d.qpos[ids]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            actual=closest_axis_point(m,d,1,center,rotation)
            return np.r_[actual-desired,.001*(q-prior)]
        solution=least_squares(residual,np.clip(prior,lower+1e-10,upper-1e-10),bounds=(lower,upper),
            max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        residual(solution.x);actual=closest_axis_point(m,d,1,center,rotation)
        error=float(np.linalg.norm(actual-desired)*1000);passed=bool(solution.success and error<=.25)
        offsets=dict(parent['grip_target_offsets_rad']);offsets.update(dict(zip(names,(existing+solution.x-prior).tolist())))
        profile=dict(parent);profile['grip_target_offsets_rad']=offsets
        path=ROOT/f'config/profile_balanced_geometry_v{version}.json'
        if passed:path.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        results.append({'profile':str(path.relative_to(ROOT)),'geometry_gate_pass':passed,
            'maximum_point_error_mm':error,'geometry_gate_tolerance_mm':.25,
            'initial_thumb_point_mm':(points[0]*1000).tolist(),'target_thumb_point_mm':(desired*1000).tolist(),
            'optimized_thumb_point_mm':(actual*1000).tolist(),'offsets_rad':offsets,
            'physical_validation_pending':passed})
    result={'source_case':source.name,'source_time_s':float(tr['time_s'][i]),'parent_profile':'config/profile_axial_grasp_v46.json',
        'nominal_targets_N':targets.tolist(),'initial_normal_force_sum_N':forces.sum(axis=0).tolist(),
        'initial_normal_moment_Nm':np.cross(points,forces).sum(axis=0).tolist(),
        'candidates':results,'hardware_ready':False,
        'limitations':'Capsule-centerline scratch kinematics; normal directions projected radially. Nominal forces only, ignores gravity/friction/patch moments. Lower normal imbalance is not dynamic stability or a force-closure guarantee.'}
    (ROOT/'outputs/robustness/balanced_normal_geometry_20260913.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'profile':r['profile'],'geometry_gate_pass':r['geometry_gate_pass'],'error_mm':r['maximum_point_error_mm']} for r in results]))


if __name__=='__main__':main()
