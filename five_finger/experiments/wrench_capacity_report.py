"""Analyze runtime geometry; no replay contact solve or scene rebuild."""
import argparse
import json
from pathlib import Path

import numpy as np

from fivefinger.wrench_capacity import load_capacity


def analyze(run):
    run=Path(run)
    recorded=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))
    with np.load(run/'actuator_trace.npz') as archive:act={k:archive[k] for k in archive.files}
    with np.load(run/'replay.npz') as archive:time=archive['time_s'];stage=archive['stage']
    if not np.array_equal(act['time_s'],time):raise ValueError('Actuator timeline mismatch')
    with np.load(run/'contact_trace.npz') as archive:normals=archive['normal_N'];groups=archive['groups']
    snapshots=recorded['samples']
    if not np.array_equal([s['time_s'] for s in snapshots],time[::20]):raise ValueError('Geometry snapshot timeline mismatch')
    if not np.isfinite(act['force']).all() or len(act['force'])!=len(time):raise ValueError('Invalid actuator recording')
    for i,snapshot in enumerate(snapshots):
        total=np.array([sum(c['normal_N'] for c in snapshot['contacts'] if c['group']==group) for group in groups])
        if snapshot['stage']!=stage[i*20] or not np.allclose(total,normals[i*20],rtol=0,atol=1e-8):
            raise ValueError('Runtime geometry does not match original contact trace')
    settings=json.loads((run/'config.json').read_text(encoding='utf-8'))['scene']
    budgets=dict(zip(['thumb','index','middle'],settings['grip_force_targets_N']))
    samples=[]
    for sample in recorded['samples']:
        # One snapshot each simulated second, plus the last available geometry.
        if sample['time_s']<26 or (round((sample['time_s']-.01)*100)%100 and sample is not recorded['samples'][-1]):continue
        rotation=np.asarray(sample['brush_rotation_world'])
        gravity=np.r_[sample['gravity_force_world_N'],np.zeros(3)]
        common=(sample['contacts'],sample['brush_com_world_m'],gravity)
        supported=load_capacity(*common,np.zeros(6),budgets)['feasible']
        patch_supported=load_capacity(*common,np.zeros(6),budgets,include_contact_moments=True)['feasible']
        directions={}
        if supported:
            for name,axis in [('x',rotation[:,0]),('y',rotation[:,1])]:
                for sign in (-1,1):
                    directions[f'{name}{sign:+d}']=load_capacity(*common,np.r_[np.zeros(3),axis*sign],budgets)
        hand=[c for c in sample['contacts'] if c['group'] in budgets and c['normal_N']>.01]
        patch_torques=[np.asarray(c['wrench_world'][3:])-np.cross(np.asarray(c['position_world_m'])-sample['brush_com_world_m'],c['wrench_world'][:3]) for c in hand]
        environment=np.sum([c['wrench_world'] for c in sample['contacts'] if c['group'] not in budgets],axis=0) if any(c['group'] not in budgets for c in sample['contacts']) else np.zeros(6)
        load=load_capacity(*common,environment,budgets,search_max=2.) if supported and np.linalg.norm(environment)>1e-6 else None
        samples.append({'time_s':sample['time_s'],'stage':sample['stage'],
            'active_contact_count':len(hand),'weight_equilibrium_feasible':supported,
            'weight_equilibrium_with_patch_moments_feasible':patch_supported,
            'measured_sum_patch_torque_norms_Nm':float(sum(np.linalg.norm(v) for v in patch_torques)),
            'measured_net_patch_torque_world_Nm':np.sum(patch_torques,axis=0).tolist() if patch_torques else [0.,0.,0.],
            'bending_capacity_Nm':directions,
            'measured_environment_wrench':np.asarray(environment).tolist(),
            'environment_load_multiplier':load,
            'contact_geoms':[c['other_geom'] for c in hand]})
    execution=np.isin(stage,['write','pen_up','touchdown','grip_recovery'])
    actuators=[]
    for i,name in enumerate(act['names']):
        if not str(name).startswith('act_r_f_joint'):continue
        force=act['force'][execution,i];lo,hi=act['force_range'][i]
        limited=bool(act['force_limited'][i])
        margin=np.minimum(force-lo,hi-force)
        actuators.append({'name':str(name),'maximum_absolute_output':float(np.max(abs(force))) if len(force) else None,
            'declared_force_limited':limited,'declared_force_range':[float(lo),float(hi)],
            'at_declared_limit_fraction':float(np.mean(margin<1e-6)) if limited and len(force) else None})
    result={'case':run.parent.name,'hardware_ready':False,'timeline_and_force_crosscheck_passed':True,
        'execution_actuator_samples':int(execution.sum()),
        'method':'16-edge point-contact friction polygons; gravity equilibrium and four pure bending directions about pen COM',
        'normal_budgets_N':budgets,'samples':samples,'actuators':actuators,
        'limitations':['No joint torque/velocity or rolling compatibility constraints in capacity calculation.',
            'Bending capacities use point contacts. Additional patch equilibrium includes axial cone rays for torsion/rolling, still an inner approximation of the full ellipsoid.',
            'Only active runtime contacts above 0.01 N; 5 Hz geometry sampled at approximately 1 Hz for analysis.',
            '100 Hz actuator outputs can miss 500 Hz peaks; scalar actuator output is not a hardware sensor.',
            'Normal budgets are analysis assumptions, not measured or guaranteed hard force caps.',
            'Static feasibility cannot certify dynamic grasp stability or hardware readiness.']}
    (run/'wrench_capacity_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    active=[s for s in samples if s['stage'] in ('write','pen_up','touchdown','grip_recovery')]
    return {'case':run.parent.name,'analyzed_snapshots':len(samples),
        'execution_weight_equilibrium_feasible_fraction':sum(s['weight_equilibrium_feasible'] for s in active)/len(active) if active else None,
        'execution_weight_equilibrium_with_patch_moments_fraction':sum(s['weight_equilibrium_with_patch_moments_feasible'] for s in active)/len(active) if active else None,
        'actuators_with_recorded_saturation':[a['name'] for a in actuators if a['at_declared_limit_fraction']]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',nargs='+')
    for run in parser.parse_args().runs:print(json.dumps(analyze(run),ensure_ascii=False))
