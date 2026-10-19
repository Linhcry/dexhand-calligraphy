"""Reconstruct robot/environment contact geometry at saved 100 Hz poses.

No contact forces are solved or inferred. This cannot recover 500 Hz contacts
between saved poses. Model loading rebuilds shared XML; run after physics ends.
"""
import argparse
import json
from collections import defaultdict
import mujoco
import numpy as np
from experiments.view_blind_competition import load
from fivefinger.robot_surface_monitor import robot_group


def analyze(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    names=[m.geom(i).name or '' for i in range(m.ngeom)]
    groups=[robot_group(n) for n in names];pen=m.body('brush').id
    records=defaultdict(list)
    # Every saved state is reconstructed; mj_collision creates geometry only.
    for i,t in enumerate(tr['time_s']):
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
        per_pair={}
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if bool(groups[a])==bool(groups[b]):continue
            robot,other=(a,b) if groups[a] else (b,a)
            if int(m.geom_bodyid[other])==pen:continue
            if c.dist>=0:continue
            key=(groups[robot],names[robot],names[other])
            if key not in per_pair or c.dist<per_pair[key]['signed_distance_m']:
                per_pair[key]={'time_s':float(t),'stage':str(tr['stage'][i]),
                    'signed_distance_m':float(c.dist),'position_world_m':c.pos.tolist()}
        for key,value in per_pair.items():records[key].append(value)
    pairs=[]
    for (group,robot,environment),rows in records.items():
        worst=min(rows,key=lambda x:x['signed_distance_m'])
        pairs.append({'group':group,'robot_geom':robot,'environment_geom':environment,'overlapping_saved_states':len(rows),
            'first':rows[0],'last':rows[-1],'deepest':worst,'maximum_penetration_mm':-worst['signed_distance_m']*1000})
    result={'case':run.parent.name,'states_checked':len(tr['time_s']),'sample_rate_hz':100,
        'arm_and_palm_environment_overlap_observed':any(p['group']=='arm_and_palm' for p in pairs),
        'pairs':pairs,'force_reconstruction_performed':False,'hardware_ready':False,
        'limitations':'Signed collision-geometry overlap at saved poses only; no force reconstruction or intervening 500Hz guarantee. Moving arm/palm names collision_*, fingers col_r_f_link*. Brush contacts are excluded. Tiny overlap is a soft-contact numerical quantity, not a measured hardware penetration.'}
    (run/'robot_surface_contact_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cases',nargs='+')
    for case in parser.parse_args().cases:print(json.dumps(analyze(case)))
