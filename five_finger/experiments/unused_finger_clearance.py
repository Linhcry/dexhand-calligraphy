"""Kinematic posture comparison; never counts as a successful physical trial."""
import argparse
import json
import mujoco
import numpy as np
from experiments.view_blind_competition import load


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);args=parser.parse_args()
    run,cfg,m,d,tr,phys,report,events=load(args.case);brush=m.geom('brush_barrel').id
    ids=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(('col_r_f_link4_','col_r_f_link5_'))]
    rows=[]
    for when in [30.,100.,140.,150.]:
        if when>tr['time_s'][-1]:continue
        index=np.searchsorted(tr['time_s'],when);original=tr['qpos'][index]
        for flexion in [None,0.,.4,.8,1.2]:
            d.qpos[:]=original
            if flexion is not None:
                for finger in (4,5):
                    for joint in (2,3,4):d.qpos[m.joint(f'r_f_joint{finger}_{joint}').qposadr[0]]=flexion
            mujoco.mj_forward(m,d)
            distance,name=min((float(mujoco.mj_geomDistance(m,d,g,brush,.2,None)),m.geom(g).name) for g in ids)
            rows.append({'time_s':float(tr['time_s'][index]),'counterfactual_flexion_rad':flexion,
                         'minimum_pen_clearance_mm':distance*1000,'closest_geom':name})
    result={'source':'kinematic changes to unused fingers in saved states; pen and arm unchanged',
            'physical_transition_tested':False,'force_values_not_used':True,'not_a_qualification_trial':True,
            'negative_distance_means_geometric_overlap':True,'samples':rows}
    (run/'unused_finger_clearance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
