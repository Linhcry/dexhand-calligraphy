"""Refresh daily indexes from saved completed measurements without simulation."""
import argparse
import json
from pathlib import Path

from experiments.contact_force_audit import audit
from experiments.contact_wrench_report import summarize
from fivefinger.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error('Use YYYYMMDD')
    rows, wrenches = [], []
    for source in sorted((ROOT/'outputs/robustness').glob(args.date+'*/summary.json')):
        batch = json.loads(source.read_text(encoding='utf-8'))
        for row in batch['runs']:
            report = row['report']
            run = ROOT/'outputs/blind_competition'/row['case']/'run'
            if report['stop_reason'] in ('in_progress', 'controller_error') or not (run/'physics.csv').exists():
                continue
            result = audit(run)
            paper = result['groups']['whole_run']['paper_N']
            rows.append(dict(case=row['case'], batch_id=source.parent.name,
                profile=Path(batch['profile']).stem, stop=report['stop_reason'],
                completed_strokes=report.get('completed_strokes', 0),
                elapsed_sim_s=report.get('elapsed_sim_s'),
                writing_ink_contact_fraction=report.get('writing_ink_contact_fraction'),
                including_recovery_ink_fraction=report.get('all_pen_down_contact_fraction'),
                paper_peak_N=paper.get('maximum_recorded_N'), paper_p95_N=paper.get('p95_N')))
            if 'physics_step_peak_crosscheck' in result:
                rows[-1]['physics_step_peak_crosscheck']=result['physics_step_peak_crosscheck']
            if (run/'contact_trace.npz').exists():
                wrenches.append(dict(case=row['case'], **summarize(run)))
    directory = ROOT/'outputs/robustness'
    for prefix, values in [('contact_force_index', rows), ('contact_wrench_index', wrenches)]:
        (directory/f'{prefix}_{args.date}.json').write_text(
            json.dumps(values, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'force_audits': len(rows), 'runtime_wrench_audits': len(wrenches)}))


if __name__ == '__main__':
    main()
