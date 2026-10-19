"""Small predeclared servo sweep followed by fixed-setting perturbation trials."""
import json
import argparse
import yaml
from experiments.retarget_grasp import OUT
from experiments.run_suite import physical_trial
from fivefinger.paths import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--balanced',action='store_true',help='Separate thumb and supporting-finger force targets')
    args=parser.parse_args()
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    source=json.loads((OUT/'refined_baseline_3mm/report.json').read_text(encoding='utf-8'))
    results=[]
    settings=[(.25,.004),(.25,.012),(.5,.004),(.5,.012)]
    if args.balanced:settings=[([.6,.15,.15],.008),([.6,.15,.15],.016),([1.,.3,.3],.012)]
    for index,(force,gain) in enumerate(settings):
        name=f'balanced_{index}' if args.balanced else f'feedback_f{force}_g{gain}'
        report=physical_trial(OUT/name,rules,0.,(0.,0.,0.),grip_targets=source['grip_targets_rad'],
                              feedback={'target_n':force,'gain_m_per_ns':gain})
        report['candidate_id']=name;results.append(report)
        print(json.dumps({'id':name,'hold_s':report['longest_continuous_hold_s'],'passed':report['hold_3s_passed']}),flush=True)
    best=max(results,key=lambda r:r['longest_continuous_hold_s'])
    validation=[]
    for i,shift in enumerate([(.001,-.001,0.),(-.001,.001,0.),(.002,0.,0.)],1):
        name=f'balanced_validation_{i}' if args.balanced else f'feedback_validation_{i}'
        report=physical_trial(OUT/name,rules,0.,shift,grip_targets=source['grip_targets_rad'],feedback=best['simulation_force_feedback'])
        report['candidate_id']=name;validation.append(report)
        print(json.dumps({'id':name,'hold_s':report['longest_continuous_hold_s'],'passed':report['hold_3s_passed']}),flush=True)
    result={'development':results,'selected_settings':best['simulation_force_feedback'],
            'validation':validation,'official_score':None,'heldout_random_test':False,
            'note':'Three declared positional perturbations, not a statistical success-rate estimate.'}
    (OUT/('balanced_summary.json' if args.balanced else 'feedback_summary.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
