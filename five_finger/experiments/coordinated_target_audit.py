"""Audit recorded hand targets, separate from physical grasp success."""
import argparse
import json
import numpy as np
from experiments.view_blind_competition import load


def audit(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    first=int(np.flatnonzero(tr['stage']=='to_ink')[0])
    names=[f'r_f_joint{i}_{j}' for i in (1,2,3) for j in (1,2,3,4) if (i,j)!=(3,1)]
    ids=[m.actuator('act_'+n).id for n in names]
    values=tr['ctrl'][first:,ids];dt=np.diff(tr['time_s'][first:])
    rates=abs(np.diff(values,axis=0))/dt[:,None]
    previous_healthy=(physics[:,4]>.01)&np.any(physics[:,5:9]>.01,axis=1)
    owned=np.r_[True,previous_healthy[first:-1]]
    lateral=[names.index(n) for n in ('r_f_joint1_1','r_f_joint2_1')]
    excursion=np.max(abs(values[owned][:,lateral]-values[0,lateral]))
    lower=m.actuator_ctrlrange[ids,0];upper=m.actuator_ctrlrange[ids,1]
    within=bool(np.all(values>=lower-1e-8) and np.all(values<=upper+1e-8))
    result={'case':run.parent.name,'maximum_target_rate_rad_s':float(rates.max()),
        'entry_maximum_hand_target_jump_rad':float(np.max(abs(tr['ctrl'][first,ids]-tr['ctrl'][first-1,ids]))),
        'maximum_owned_spread_excursion_rad':float(excursion),
        'actuator_target_bounds_passed':within,'target_rate_passed':bool(rates.max()<=.12+1e-8),
        'owned_spread_bound_passed':bool(excursion<=.05+1e-8),
        'scope':'Targets after first to_ink sample. Spread bound excludes ticks whose prior physics sample lacked pinch, when retention recovery owns commands. Does not prove actual joint or pen motion is safe/stable.',
        'hardware_ready':False}
    (run/'coordinated_target_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+')
    for case in p.parse_args().cases:print(json.dumps(audit(case)))
