"""Frozen-grasp IK probe of wrist correction and robot mesh clearance.

Planning in scratch state only. No physical pen attachment and no force evidence.
IK has no obstacle avoidance; every endpoint is inspected independently.
"""
import argparse
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.view_blind_competition import load
from experiments.blind_competition import nib_point
from experiments.competition_scene import TableClearance
from fivefinger.control import solve_arm_ik
from fivefinger.grasp_stabilization import axis_alignment_error
from fivefinger.paths import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--axial-roll',action='store_true')
    args=parser.parse_args()
    case=ROOT/'outputs/blind_competition/20260913_143958_bee5e9'
    run,cfg,m,d,tr,physics,report,events=load(case);records=[]
    for time_s in [50.,180.,235.,240.,246.,249.]:
        i=int(np.argmin(abs(tr['time_s']-time_s)));state=tr['qpos'][i].copy()
        d.qpos[:]=state;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        hand=d.site('palm_control');hand_r=hand.xmat.reshape(3,3).copy()
        tool_local=hand_r.T@(nib_point(m,d)-hand.xpos)
        axis=d.body('brush').xmat.reshape(3,3)[:,2].copy()
        upright_error=axis_alignment_error(axis,[0,0,1]);angle=np.linalg.norm(upright_error)
        goal=tr['target'][i].copy();goal[2]=.002
        for correction_deg in ([-30.,-15.,-5.,0.,5.,15.,30.] if args.axial_roll else [0.,5.,10.,20.,30.]):
            d.qpos[:]=state
            correction=axis*np.radians(correction_deg) if args.axial_roll else upright_error*min(1.,np.radians(correction_deg)/max(angle,1e-12))
            desired=Rotation.from_rotvec(correction).as_matrix()@hand_r
            desired_palm=goal-desired@tool_local
            solution=solve_arm_ik(m,d,desired_palm,desired,iterations=400)
            d.qpos[:6]=solution.joints;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            palm=d.site('palm_control');predicted_tip=palm.xpos+palm.xmat.reshape(3,3)@tool_local
            clearance,lows=TableClearance(m).measure(d)
            records.append({'source_time_s':float(tr['time_s'][i]),'requested_correction_deg':correction_deg,
                'actual_correction_magnitude_deg':float(np.rad2deg(np.linalg.norm(correction))),
                'ik_converged':bool(solution.converged),'predicted_tip_error_mm':float(np.linalg.norm(predicted_tip-goal)*1000),
                'orientation_error_deg':float(np.rad2deg(solution.orientation_error_rad)),
                'minimum_arm_table_clearance_mm':float(clearance*1000),
                'palm_clearance_above_infinite_paper_plane_mm':float((lows['right_hand_base']-.002)*1000),
                'maximum_arm_joint_change_rad':float(np.max(abs(solution.joints-state[:6]))),'joints_rad':solution.joints.tolist()})
    result={'source_case':case.name,'mode':'axial_roll' if args.axial_roll else 'upright','frozen_tool_assumption':True,'hardware_ready':False,'probes':records,
        'limitations':'Endpoint kinematics only. Paper plane extended infinitely for conservative palm check; no motion path, dynamic grasp, contact-force or hardware validation. Original 0.5mm/0.01rad IK tolerance is reported separately from predicted tip error.'}
    path=ROOT/f'outputs/robustness/wrist_clearance_probe_{"axial_roll_" if args.axial_roll else ""}20260913.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='joints_rad'} for r in records]))


if __name__=='__main__':main()
