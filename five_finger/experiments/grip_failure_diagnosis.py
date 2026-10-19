"""Read saved contact and actuator targets around grasp failure (no force solve).

Model loading rebuilds the shared scene: run only after physical trials finish.
Position tracking errors and target margins are not motor torque measurements.
"""
import argparse
import json

import numpy as np

from experiments.view_blind_competition import load


def diagnose(case):
    run, cfg, model, data, trace, physics, report, events = load(case)
    with np.load(run / 'contact_trace.npz') as archive:
        contact = {key: archive[key] for key in archive.files}
    times = trace['time_s']
    if not np.array_equal(contact['time_s'], times) or not np.allclose(
        contact['normal_N'][:, :5], physics[:, 4:9], atol=1e-8, rtol=0
    ):
        raise ValueError('Recorded forces or timelines do not agree')
    # This reference model has scalar joint transmissions. Reject another
    # transmission instead of silently interpreting actuator IDs as joint IDs.
    import mujoco
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError('Only scalar joint transmissions supported')
    joints = model.actuator_trnid[:, 0]
    if np.any(~np.isin(model.jnt_type[joints], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
        raise ValueError('Expected scalar joints')
    if not np.allclose(model.actuator_gear[:, 0], 1.):
        raise ValueError('Unit gear required for direct position comparison')
    position = trace['qpos'][:, model.jnt_qposadr[joints]]
    command = trace['ctrl']
    margin = np.minimum(command - model.actuator_ctrlrange[:, 0],
                        model.actuator_ctrlrange[:, 1] - command)
    execution = np.isin(trace['stage'], ['write', 'touchdown', 'pen_up', 'grip_recovery'])
    healthy = (physics[:, 4] > .01) & np.any(physics[:, 5:9] > .01, axis=1)
    losses = np.flatnonzero(execution & ~healthy)
    center = times[losses[0]] if len(losses) else times[-1]
    probes = [30., 45., 60., center-2, center-.5, center-.2, center, times[-1]]
    indices = sorted(set(int(np.clip(np.searchsorted(times, t), 0, len(times)-1))
                         for t in probes if 0 <= t <= times[-1]))
    fingers = [i for i in range(model.nu) if model.actuator(i).name.startswith('act_r_f_joint')]
    progress=report.get('closure_nominal_progress_at_contact')
    bases={}
    adjusted=set()
    for j in fingers:
        name=model.joint(int(joints[j])).name
        finger=int(name.split('_')[2][-1])
        initial=float(data.qpos[model.jnt_qposadr[joints[j]]])
        ratio=float(progress[finger-1]) if progress is not None and finger<=3 else 1.
        bases[j]=initial+ratio*(cfg['grip']['grip_targets_rad'][name]-initial)
        if finger in (1,2,3) and name!='r_f_joint3_1':adjusted.add(j)
    samples = []
    for i in indices:
        coordinates = contact['position_brush_m'][i, :3, 2]
        samples.append({
            'time_s': float(times[i]), 'stage': str(trace['stage'][i]),
            'finger_normal_N': physics[i, 4:9].tolist(),
            'paper_tip_normal_N': float(physics[i, 14]),
            'contact_z_brush_mm': [float(v*1000) if np.isfinite(v) else None for v in coordinates],
            'actuators': [{
                'name': model.actuator(j).name,
                'target_rad': float(command[i, j]),
                'position_rad': float(position[i, j]),
                'target_error_rad': float(command[i, j]-position[i, j]),
                'target_range_margin_rad': float(margin[i, j]),
                'maintenance_offset_rad': float(command[i,j]-bases[j]) if j in adjusted else None,
                'maintenance_offset_margin_rad': float(.25-abs(command[i,j]-bases[j])) if j in adjusted else None,
            } for j in fingers],
        })
    result = {
        'case': run.parent.name, 'stop_reason': report['stop_reason'],
        'first_execution_pinch_loss_s': float(center) if len(losses) else None,
        'source': 'runtime contact_trace and saved qpos/ctrl; model used for joint mapping only',
        'no_contact_force_reconstruction': True,
        'motor_torque_saturation_not_measured': True,
        'sample_rate_hz': 100, 'hardware_ready': False, 'samples': samples,
    }
    (run / 'grip_failure_diagnosis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {key: value for key, value in result.items() if key != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    for case in parser.parse_args().cases:
        print(json.dumps(diagnose(case), ensure_ascii=False))
