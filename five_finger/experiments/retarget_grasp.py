"""Retarget real upstream hand poses, then evaluate free-brush grasping.

Upstream data is generic human motion, NOT a pen grasp demonstration. Initial
brush position is known in simulation. Retargeting supplies poses, not forces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from experiments.probe_grasp import make_model
from experiments.run_suite import physical_trial
from fivefinger.paths import ROOT
from fivefinger.retargeting import create_optimizer, retarget_points, robot_points

OUT = ROOT / 'outputs/retargeting'


def generate(input_path=None):
    OUT.mkdir(exist_ok=True)
    path = ROOT / 'vendor/dex-retargeting/example/profiling/human_joint_right.pkl'
    if input_path is None:
        # Only the vendored upstream sample uses pickle; custom data never does.
        human = np.asarray(np.load(path, allow_pickle=True), dtype=float)
    else:
        path = Path(input_path).resolve()
        with np.load(path, allow_pickle=False) as package:
            human = np.asarray(package['points_m'], dtype=float)
    if human.ndim != 3 or human.shape[1:] != (21,3) or len(human)<2 or not np.isfinite(human).all():
        raise ValueError('Expected points_m with shape (T>=2,21,3), finite meter coordinates')
    tips = human[:, [0, 4, 8, 12, 16, 20]]
    tips = tips - tips[:, :1]
    model, data, center = make_model()
    robot, optimizer = create_optimizer(model, data, scaling=1.0, pinch_distance=.012)
    robot.compute_forward_kinematics(robot.q0)
    neutral = robot_points(robot)
    open_index = int(np.argmax(np.linalg.norm(tips[:, 1:] - tips[:, 1:2], axis=2).sum(axis=1)))
    rotation, _ = Rotation.align_vectors(neutral[1:], tips[open_index, 1:])
    scale = float(np.median(np.linalg.norm(neutral[1:], axis=1) / np.linalg.norm(tips[open_index, 1:], axis=1)))
    optimizer.scaling = scale
    # Select distinct thumb-index pinch frames; no simulation outcomes used here.
    distance = np.linalg.norm(tips[:, 1] - tips[:, 2], axis=1)
    selected = []
    for index in np.argsort(distance):
        if all(abs(int(index) - chosen) >= 30 for chosen in selected):
            selected.append(int(index))
        if len(selected) == 6:
            break
    candidates = []
    started = time.perf_counter()
    for index in selected:
        points = rotation.apply(tips[index])
        q = robot.q0.copy()
        for _ in range(4):
            q = retarget_points(robot, optimizer, points, q)
        actual = robot_points(robot)
        midpoint = (actual[1] + actual[2]) / 2
        base = data.body('right_hand_base')
        world_midpoint = base.xpos + base.xmat.reshape(3, 3) @ midpoint
        # Align pinch midpoint transversely to the known barrel axis. Avoid
        # shifting lengthwise into the rack supports.
        offset = center - world_midpoint
        axis = data.body('brush').xmat.reshape(3, 3)[:, 2]
        offset -= axis * np.dot(offset, axis)
        candidates.append({'id': f'human_{index:03d}', 'source_frame': index,
                           'source_thumb_index_distance_m': float(distance[index]),
                           'grip_targets_rad': dict(zip(robot.dof_joint_names, q.tolist())),
                           'approach_offset_m': offset.tolist(), 'robot_tip_points_m': actual.tolist(),
                           'human_tip_points_aligned_m': (points * scale).tolist(),
                           'joint_limit_max_violation_rad': float(max(0, np.max(q-robot.joint_limits[:, 1]), np.max(robot.joint_limits[:, 0]-q)))})
    result = {'source_repository': 'https://github.com/dexsuite/dex-retargeting' if input_path is None else None,
              'source_commit': '3f56141bc8bd2760d5e452e382937269554ebb21' if input_path is None else None,
              'source_file': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_is_pen_demonstration': False, 'source_frames': len(human),
              'alignment_open_frame': open_index, 'human_to_robot_rotation': rotation.as_matrix().tolist(),
              'scale': scale, 'generation_wall_s': time.perf_counter()-started,
              'real_motor_coupling_calibrated': False, 'candidates': candidates}
    (OUT / 'candidates.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'human_input.npz', points_m=human)
    print(json.dumps({'candidates': len(candidates), 'scale': scale, 'generation_s': result['generation_wall_s']}), flush=True)
    return result


def evaluate(source):
    rules = yaml.safe_load((ROOT / 'experiments/rules.yaml').read_text(encoding='utf-8'))
    results = []
    for candidate in source['candidates']:
        folder = OUT / candidate['id']
        report = physical_trial(folder, rules, 0., (0.,0.,0.),
                                grip_targets=candidate['grip_targets_rad'], approach_offset=candidate['approach_offset_m'])
        report['candidate_id'] = candidate['id']
        results.append(report)
        print(json.dumps({'id': candidate['id'], 'hold_s': report['longest_continuous_hold_s'],
                          'passed': report['hold_3s_passed'], 'touch': report['physical_touch']}), flush=True)
    summary = {'independent_development_trials': len(results), 'hold_successes': sum(r['hold_3s_passed'] for r in results),
               'best_hold_s': max(r['longest_continuous_hold_s'] for r in results), 'official_score': None,
               'data_is_pen_demonstration': False, 'results': results}
    (OUT / 'physical_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['generate','evaluate','all'], default='all')
    parser.add_argument('--input',type=Path,help='Custom NPZ points_m: T x 21 x 3, meters; include an open-hand frame')
    args = parser.parse_args()
    source = generate(args.input) if args.only != 'evaluate' else json.loads((OUT/'candidates.json').read_text(encoding='utf-8'))
    if args.only != 'generate':
        evaluate(source)


if __name__ == '__main__':
    main()
