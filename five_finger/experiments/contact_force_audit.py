"""Read saved measurements; do not rebuild scenes or rerun contact solvers."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_force(values, times, stages, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {'samples': 0}
    selected = values[indices]
    peak = indices[int(np.argmax(selected))]
    return {
        'samples': len(indices),
        'median_N': float(np.median(selected)),
        'p95_N': float(np.quantile(selected, .95)),
        'maximum_recorded_N': float(values[peak]),
        'maximum_sim_s': float(times[peak]),
        'maximum_stage': str(stages[peak]),
    }


def audit(run):
    """Measurements are sampled at 100 Hz; inter-sample peaks are unknown."""
    run = Path(run)
    rows = np.atleast_1d(np.genfromtxt(run/'physics.csv', delimiter=',', names=True))
    with np.load(run/'replay.npz', allow_pickle=False) as replay:
        times = replay['time_s'].copy()
        stages = replay['stage'].copy()
        path_indices = replay['path_index'].copy()
    if len(rows) != len(times) or len(stages) != len(times) or not np.allclose(rows['time'], times, rtol=0, atol=1e-8):
        raise ValueError('CSV and replay timelines do not match')
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid recorded timeline')
    fields = ('thumb', 'index', 'middle', 'ring', 'little', 'paper_N')
    if any(not np.isfinite(rows[name]).all() or np.any(rows[name] < 0) for name in fields):
        raise ValueError('Invalid normal-force measurements')
    masks = {
        'whole_run': np.ones(len(times), dtype=bool),
        'drawing_and_contact_recovery': np.isin(stages, ['write', 'touchdown']),
        'pen_up': stages == 'pen_up',
        'grip_recovery': stages == 'grip_recovery',
    }
    result = {
        'source': 'original simulation physics.csv, aligned with replay.npz',
        'force_feedback': 'simulation ground truth, not hardware sensor measurements',
        'median_record_interval_s': float(np.median(np.diff(times))),
        'force_targets_are_not_hard_force_limits': True,
        'inter_sample_force_peaks_known': False,
        'hardware_force_limits_confirmed': False,
        'hardware_ready': False,
        'groups': {group: {field: summarize_force(rows[field], times, stages, mask)
                           for field in fields} for group, mask in masks.items()},
        'last_record': {'sim_s': float(times[-1]), 'stage': str(stages[-1]),
                        'path_index': int(path_indices[-1]),
                        'forces_N': {field: float(rows[field][-1]) for field in fields}},
    }
    peak_path=run/'contact_force_peaks_500hz.json'
    if peak_path.exists():
        peak=json.loads(peak_path.read_text(encoding='utf-8'))
        report=json.loads((run/'report.json').read_text(encoding='utf-8'))
        hz=float(peak['requested_rate_hz'])
        if peak['samples']!=round(report['elapsed_sim_s']*hz) or peak['maximum_observed_interval_s']>1.001/hz:
            raise ValueError('Physics-step force coverage does not match executed time')
        comparisons={}
        for field in fields:
            group='paper' if field=='paper_N' else field
            fine=float(peak['groups'][group]['maximum_recorded_N'])
            coarse=result['groups']['whole_run'][field]['maximum_recorded_N']
            if fine+1e-8<coarse:raise ValueError('Physics-step maximum is below a recorded control-step force')
            comparisons[field]={'control_step_peak_N':coarse,'physics_step_peak_N':fine}
        result['physics_step_peak_crosscheck']={'passed':True,'samples':peak['samples'],'rate_hz':hz,
            'comparisons':comparisons,'paper_scope_note':'500 Hz paper group includes every pen geom; 100 Hz paper_N is nib-only. A larger paper peak is not necessarily sampling error.'}
    (run/'contact_force_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for run in args.runs:
        result = audit(run)
        print(json.dumps({'run': str(run), 'whole_run_maximum_N': {
            key: value['maximum_recorded_N'] for key, value in result['groups']['whole_run'].items()
        }}, ensure_ascii=False))


if __name__ == '__main__':
    main()
