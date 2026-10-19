"""Reconstruct relative pen kinematics; never recompute saved contact forces."""
import argparse
import json

import mujoco
import numpy as np

from experiments.view_blind_competition import load


def diagnose(case):
    run, cfg, model, data, trace, physics, report, events = load(case)
    brush = model.body('brush').id
    palm = model.body('right_hand_base').id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    hp, hr = np.zeros_like(jp), np.zeros_like(jr)
    rows = []
    for qpos, qvel in zip(trace['qpos'], trace['qvel']):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        center = data.xpos[brush]
        axis = data.xmat[brush].reshape(3, 3)[:, 2]
        hand_rotation = data.xmat[palm].reshape(3, 3)
        # Both Jacobians evaluated at the SAME point, removing palm rotation.
        mujoco.mj_jac(model, data, jp, jr, center, brush)
        mujoco.mj_jac(model, data, hp, hr, center, palm)
        relative_velocity = (jp-hp)@qvel
        angular_velocity = (jr-hr)@qvel
        local_center = hand_rotation.T@(center-data.xpos[palm])
        axial_speed = float(relative_velocity@axis)
        transverse_speed = np.linalg.norm(relative_velocity-axial_speed*axis)
        spin_speed = float(angular_velocity@axis)
        tilt_speed = np.linalg.norm(angular_velocity-spin_speed*axis)
        rows.append([*local_center, axial_speed, transverse_speed, spin_speed, tilt_speed, *(hand_rotation.T@axis)])
    rows = np.asarray(rows)
    times = trace['time_s']
    # Position increments avoid treating sparse instantaneous velocity samples
    # as an accurate integral through 500 Hz contact impulses.
    delta_center = np.diff(rows[:, :3], axis=0, prepend=rows[:1, :3])
    interval_axis = rows[:, 7:10] + np.vstack([rows[:1, 7:10], rows[:-1, 7:10]])
    interval_axis /= np.linalg.norm(interval_axis, axis=1)[:, None]
    interval_axial = np.sum(delta_center*interval_axis, axis=1)
    if len(physics) != len(times) or not np.allclose(physics[:, 0], times, rtol=0, atol=1e-8):
        raise ValueError('Saved force and kinematic timelines differ')
    windows = []
    for start in np.arange(40, times[-1], 20):
        mask = (times>=start)&(times<start+20)
        values = rows[mask]
        dt = np.diff(times, prepend=times[0])[mask]
        windows.append(dict(start_s=float(start), end_s=float(times[mask][-1]),
            local_center_change_mm=((values[-1, :3]-values[0, :3])*1000).tolist(),
            sparse_velocity_integral_mm=float(np.sum(values[:, 3]*dt)*1000),
            position_interval_axial_sum_mm=float(np.sum(interval_axial[mask])*1000),
            position_interval_absolute_axial_sum_mm=float(np.sum(abs(interval_axial[mask]))*1000),
            axial_speed_p95_mm_s=float(np.quantile(abs(values[:, 3]), .95)*1000),
            tilt_speed_p95_deg_s=float(np.degrees(np.quantile(values[:, 6], .95))),
            paper_contact_fraction=float(np.mean(physics[mask, 14]>.01)),
            finger_normal_median_N=np.median(physics[mask, 4:9], axis=0).tolist()))
    samples = []
    for when in [30, 60, 100, 140, 160, 180, 220, float(times[-1])]:
        if when>times[-1]:
            continue
        i=min(int(np.searchsorted(times, when)), len(times)-1)
        samples.append(dict(time_s=float(times[i]), stage=str(trace['stage'][i]),
            center_in_palm_mm=(rows[i, :3]*1000).tolist(),
            axial_speed_mm_s=float(rows[i, 3]*1000),
            tilt_speed_deg_s=float(np.degrees(rows[i, 6])),
            world_tilt_deg=float(physics[i, 10])))
    np.savez_compressed(run/'relative_motion.npz', time_s=times,
        center_in_palm_m=rows[:, :3], axial_speed_m_s=rows[:, 3],
        transverse_speed_m_s=rows[:, 4], spin_speed_rad_s=rows[:, 5], tilt_speed_rad_s=rows[:, 6],
        axis_in_palm=rows[:, 7:10], position_interval_axial_m=interval_axial)
    result = dict(source='saved qpos/qvel with model kinematics; no contact solver rerun',
        case=run.parent.name, stop=report['stop_reason'], windows=windows, samples=samples,
        axial_travel_note='100 Hz instantaneous velocities can alias 500 Hz contact impulses; their integral is NOT reliable travel. Prefer pose changes and saved position intervals; intervals still omit between-sample motion.',
        hardware_ready=False)
    (run/'slip_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    for case in args.cases:
        print(json.dumps(diagnose(case), ensure_ascii=False))
