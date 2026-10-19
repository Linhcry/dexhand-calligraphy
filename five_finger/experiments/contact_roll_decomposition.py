"""Separate measured contact migration, material roll and tool-axis tilt.

Kinematic reconstruction of saved state only. Rebuilds the shared scene, so
run serially after physics completes. Uses saved runtime contacts, not new forces.
"""
import argparse
import json
import mujoco
import numpy as np
from experiments.view_blind_competition import load
from fivefinger.axial_frame import axial_angles


def analyze(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    samples=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))['samples']
    hold=[s for s in samples if s['stage']=='hold']
    if not hold:raise ValueError('No acquired hold')
    start=hold[-1]['time_s'];rows=[];reference=None;maximum_rotation_error=0.;normal_angles=[]
    for s in samples:
        if s['time_s']<start:continue
        i=int(np.argmin(abs(tr['time_s']-s['time_s'])))
        if abs(tr['time_s'][i]-s['time_s'])>1e-7:raise ValueError('Contact/state timestamp mismatch')
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d)
        hand=d.body('right_hand_base').xmat.reshape(3,3)
        pen=np.asarray(s['brush_rotation_world']);actual=d.body('brush').xmat.reshape(3,3)
        error=float(np.max(abs(actual-pen)));maximum_rotation_error=max(maximum_rotation_error,error)
        if error>1e-7:raise ValueError('Saved contact and reconstructed pose disagree')
        relative=hand.T@pen
        if reference is None:reference=relative.copy()
        tilt=float(np.arccos(np.clip(reference[:,2]@relative[:,2],-1,1)))
        for group in ('thumb','index','middle'):
            active=[c for c in s['contacts'] if c['group']==group and c['normal_N']>.05]
            if not active:continue
            for contact in active:
                radial=-np.asarray(contact['position_brush_m']).copy();radial[2]=0
                if np.linalg.norm(radial)<1e-9:continue
                radial/=np.linalg.norm(radial)
                normal=pen.T@np.asarray(contact['frame_on_brush_world'])[:,0]
                normal_angles.append(float(np.rad2deg(np.arccos(np.clip(radial@normal,-1,1)))))
            point=np.average([c['position_brush_m'] for c in active],axis=0,weights=[c['normal_N'] for c in active])
            angles=axial_angles(relative,point,reference[:,2],reference[:,0])
            # Independent algebraic consistency: body = transported contact - material roll.
            residual=np.arctan2(np.sin(angles[2]-angles[0]+angles[1]),np.cos(angles[2]-angles[0]+angles[1]))
            if abs(residual)>1e-8:raise ValueError('Angle decomposition inconsistent')
            rows.append({'time_s':s['time_s'],'stage':s['stage'],'finger':group,
                'angles_rad':angles.tolist(),'axis_tilt_from_hold_deg':float(np.rad2deg(tilt))})
    summary={}
    for group in ('thumb','index','middle'):
        selected=[r for r in rows if r['finger']==group]
        if not selected:continue
        angles=np.unwrap(np.array([r['angles_rad'] for r in selected]),axis=0)
        change=np.rad2deg(angles-angles[0])
        for row,value in zip(selected,change):row['angle_changes_deg']=value.tolist()
        summary[group]={'samples':len(selected),'last_contact_time_s':selected[-1]['time_s'],
            'last_angle_changes_deg':change[-1].tolist(),
            'maximum_abs_angle_changes_deg':np.max(abs(change),axis=0).tolist(),
            'last_axis_tilt_from_hold_deg':selected[-1]['axis_tilt_from_hold_deg']}
    result={'case':run.parent.name,'source_time_s':start,'columns':['transported_hand_contact','pen_material_roll','pen_body_contact'],
        'summary':summary,'maximum_saved_rotation_error':maximum_rotation_error,'samples':rows,'hardware_ready':False,
        'normal_radial_comparison':{'samples':len(normal_angles),'max_angle_deg':max(normal_angles) if normal_angles else None},
        'limitations':'5Hz contact centroids above 0.05N. Shortest-axis transport in hand frame is a coordinate convention, not material tracking or a slip detector. Angle unwrapping across missing samples does not recover lost turns; first and last contact may differ from trial endpoints.'}
    (run/'contact_roll_decomposition.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return {'case':result['case'],'summary':summary,'maximum_saved_rotation_error':maximum_rotation_error}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cases',nargs='+')
    for case in parser.parse_args().cases:print(json.dumps(analyze(case)))
