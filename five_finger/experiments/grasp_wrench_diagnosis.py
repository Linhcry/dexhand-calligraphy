"""Offline contact geometry reconstruction. Forces are approximate, not logged truth."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from experiments.view_blind_competition import load
from fivefinger.contact_observation import observe_brush_contacts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--times',nargs='+',type=float,required=True);a=p.parse_args()
    run,cfg,m,d,tr,physics,report,events=load(a.case);samples=[]
    for when in a.times:
        i=min(int(np.searchsorted(tr['time_s'],when)),len(tr['time_s'])-1)
        d.qpos[:]=tr['qpos'][i];d.qvel[:]=tr['qvel'][i];d.ctrl[:]=tr['ctrl'][i];d.time=tr['time_s'][i]
        d.qacc_warmstart[:]=0.;d.qfrc_applied[:]=0.
        mujoco.mj_forward(m,d);d.qfrc_applied[:6]=np.clip(d.qfrc_bias[:6],-40,40);mujoco.mj_forward(m,d)
        observed=observe_brush_contacts(m,d)
        sample={'time_s':float(d.time),'stage':str(tr['stage'][i]),'original_finger_normal_N':physics[i,4:9].tolist(),
                'original_paper_normal_N':float(physics[i,14]),'tilt_deg':float(physics[i,10]),
                'hand_axial_span_mm':observed['hand_axial_span_m']*1000,'contacts':observed['details']}
        samples.append(sample)
        print(json.dumps({k:v for k,v in sample.items() if k!='contacts'}|{'hand_contacts':[(x['other_geom'],round(x['position_brush_m'][2]*1000,2)) for x in sample['contacts'] if x['group'] not in ('paper','other')]},ensure_ascii=False))
    result={'source':'saved state reconstructed without solver warm-start history','force_values_are_approximate':True,
            'use_original_csv_for_recorded_normal_forces':True,'case':str(Path(a.case).resolve()),'samples':samples,'hardware_ready':False}
    (run/'grasp_wrench_reconstruction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
