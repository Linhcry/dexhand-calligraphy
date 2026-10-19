"""Measure arm target discontinuity at lift/hold to ink-transfer handoff."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(run):
    run=Path(run)
    with np.load(run/'replay.npz') as archive:t={k:archive[k] for k in archive.files}
    physics=np.loadtxt(run/'physics.csv',delimiter=',',skiprows=1)
    ids=np.flatnonzero(t['stage']=='to_ink')
    if not len(ids) or ids[0]==0:raise ValueError('No recorded handoff')
    i=int(ids[0]);start=t['time_s'][i]
    mask=(t['time_s']>=start)&(t['time_s']<=start+.3)
    result={'case':run.parent.name,'handoff_time_s':float(start),
        'arm_target_jump_rad':(t['ctrl'][i,:6]-t['ctrl'][i-1,:6]).tolist(),
        'maximum_arm_target_jump_rad':float(np.max(abs(t['ctrl'][i,:6]-t['ctrl'][i-1,:6]))),
        'first_0_3s_maximum_pen_tilt_deg':float(np.max(physics[mask,10])),
        'available_post_handoff_s':float(t['time_s'][-1]-start),
        'note':'100 Hz saved arm commands and measured pen tilt; not a hardware or global stability test.'}
    (run/'handoff_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runs',nargs='+')
    for run in parser.parse_args().runs:print(json.dumps(analyze(run)))
