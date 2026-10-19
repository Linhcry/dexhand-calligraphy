"""Compare issued target rates with finite-interval motion at runtime contacts.

Kinematic reconstruction only; no force solve. Model load rebuilds shared scene.
"""
import argparse
import json
import mujoco
import numpy as np
from experiments.view_blind_competition import load


def analyze(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    contacts=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))['samples']
    if not np.allclose(m.actuator_gear[:,0],1.):raise ValueError('Unit transmissions required')
    ids=m.jnt_dofadr[m.actuator_trnid[6:,0]]
    values={name:[] for name in ('thumb','index','middle')}
    records=[];v=np.zeros(m.nv);jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));fp,fr=jp.copy(),jr.copy()
    for s in contacts:
        if s['stage'] not in ('write','touchdown','pen_up'):continue
        i=int(np.searchsorted(tr['time_s'],s['time_s']))
        if i==0:continue
        dt=tr['time_s'][i]-tr['time_s'][i-1]
        mujoco.mj_differentiatePos(m,v,dt,tr['qpos'][i-1],tr['qpos'][i])
        target_rate=(tr['ctrl'][i,6:]-tr['ctrl'][i-1,6:])/dt
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        for group in values:
            active=[c for c in s['contacts'] if c['group']==group and c['normal_N']>.05]
            if not active:continue
            total=sum(c['normal_N'] for c in active);sample=np.zeros(3)
            for c in active:
                weight=c['normal_N']/total;normal=np.asarray(c['frame_on_brush_world'])[:,0]
                tangent=np.eye(3)-np.outer(normal,normal)
                mujoco.mj_jac(m,d,jp,jr,c['position_world_m'],m.body('brush').id)
                mujoco.mj_jac(m,d,fp,fr,c['position_world_m'],int(m.geom_bodyid[m.geom(c['other_geom']).id]))
                sample+=weight*np.array([
                    np.linalg.norm(tangent@(jp-fp)@v),
                    np.linalg.norm(tangent@fp[:,ids]@(v[ids]-target_rate)),
                    np.linalg.norm(tangent@np.asarray(c['relative_velocity_world_m_s']))])
            values[group].append(sample*1000)
            records.append({'time_s':s['time_s'],'finger':group,
                'ten_ms_relative_tangent_speed_mm_s':sample[0]*1000,
                'target_actual_point_velocity_error_mm_s':sample[1]*1000,
                'instantaneous_relative_tangent_speed_mm_s':sample[2]*1000})
    summary={}
    for name,rows in values.items():
        if not rows:continue
        array=np.asarray(rows)
        summary[name]={'samples':len(rows),'median_mm_s':np.median(array,axis=0).tolist(),'p95_mm_s':np.quantile(array,.95,axis=0).tolist()}
    result={'case':run.parent.name,'columns':['10ms_interval_relative_slip','target_vs_actual_point_velocity_error','instantaneous_relative_slip'],
        'summary':summary,'samples':records,'hardware_ready':False,
        'limitations':'5Hz contact snapshots. 10ms qpos differences with endpoint Jacobians approximate average velocities and do not resolve 500Hz impulses. Target velocity comparison excludes arm motion; actual pen-finger relative velocity uses all DOFs. No force recomputation.'}
    (run/'target_velocity_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return {'case':run.parent.name,'summary':summary}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+')
    for case in p.parse_args().cases:print(json.dumps(analyze(case)))
