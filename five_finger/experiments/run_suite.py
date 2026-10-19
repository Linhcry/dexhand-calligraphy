"""Run rule-driven physical grasp trials and independent final-round planning checks.

Each physical trial starts from a fresh declared setup; these are not two official rounds.
The free brush is never welded, teleported during the run or driven by a mocap body.
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from experiments.probe_grasp import make_model, GRIP, brush_contacts
from experiments.judge import HoldGate, circle_score
from fivefinger.control import set_joint_targets, solve_arm_ik, tactile_forces
from fivefinger.paths import ROOT, output_dir
from fivefinger.trajectory import smoothstep5, make_path
from fivefinger.vision import OfflineOCR

POEMS=[('静夜思','床前明月光疑是地上霜举头望明月低头思故乡',5),
       ('早发白帝城','朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山',7),
       ('春望','国破山河在城春草木深感时花溅泪恨别鸟惊心烽火连三月家书抵万金白头搔更短浑欲不胜簪',5)]


def physical_trial(folder,rules,thumb_delta,shift,grip_targets=None,approach_offset=None,feedback=None):
    cfg=rules['assumptions'];m,d,center=make_model(cfg['brush_radius_m'],shift=shift)
    site=m.site('palm_control').id;start=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3).copy()
    qa=[m.joint(f'r_a_joint{i}').qposadr[0] for i in range(1,7)]
    va=[m.joint(f'r_a_joint{i}').dofadr[0] for i in range(1,7)]
    offset=np.zeros(3) if approach_offset is None else np.asarray(approach_offset,dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.15:
        raise ValueError('Approach offset must be finite xyz within 150 mm')
    grasp_pose=solve_arm_ik(m,d,start+offset,rot)
    if not grasp_pose.converged:raise RuntimeError('Grasp alignment IK failed')
    q0=d.qpos[qa].copy() if approach_offset is None else grasp_pose.joints
    pre=solve_arm_ik(m,d,start+offset+[0,0,.03],rot);lift=solve_arm_ik(m,d,start+offset+[0,0,.05],rot)
    if not pre.converged or not lift.converged:raise RuntimeError('Approach/lift IK failed')
    d.qpos[qa]=pre.joints;set_joint_targets(m,d,{f'r_a_joint{i+1}':q for i,q in enumerate(pre.joints)})
    mujoco.mj_forward(m,d)
    grip=dict(GRIP if grip_targets is None else grip_targets)
    initial={name:float(d.qpos[m.joint(name).qposadr[0]]) for name in grip}
    grip['r_f_joint1_2']+=thumb_delta
    servo=None
    if feedback is not None:
        from fivefinger.grasp_feedback import ContactGripServo
        servo=ContactGripServo(m,**feedback)
    hold=HoldGate(rules['preliminary']['hold_seconds']);first_touch=None;lift_seen=False;drop=False
    rows=[];states=[];controls=[];native=[];previous_relative=None;stages=[];max_bias=0.
    for tick in range(1600):
        t=d.time
        close=float(smoothstep5((t-2)/4))
        commanded={k:initial[k]+close*(v-initial[k]) for k,v in grip.items()}
        if servo is not None and t>=9.5:
            sensed,_=brush_contacts(m,d)
            for name,delta in servo.update(d,sensed,.01).items():
                commanded[name]=commanded.get(name,grip.get(name,0.))+delta
        set_joint_targets(m,d,commanded)
        u=float(smoothstep5(t/2))
        target=pre.joints+u*(q0-pre.joints)
        if t>=8:
            u=float(smoothstep5((t-8)/3));target=q0+u*(lift.joints-q0)
        set_joint_targets(m,d,{f'r_a_joint{i+1}':q for i,q in enumerate(target)})
        for _ in range(5):
            d.qfrc_applied[va]=np.clip(d.qfrc_bias[va],-cfg['gravity_compensation_torque_limit_Nm'],cfg['gravity_compensation_torque_limit_Nm'])
            max_bias=max(max_bias,float(np.max(np.abs(d.qfrc_applied[va]))))
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('Invalid physics state')
        forces,support=brush_contacts(m,d);pos=d.body('brush').xpos.copy()
        palm=d.body('right_hand_base');relative=palm.xmat.reshape(3,3).T@(pos-palm.xpos)
        slip=np.linalg.norm(relative-previous_relative)/.01 if previous_relative is not None else 0.
        previous_relative=relative.copy();contact=forces>cfg['contact_threshold_N']
        if contact.any() and first_touch is None:first_touch=float(d.time)
        raised=pos[2]-center[2]>=cfg['held_lift_m'] and support<.01
        lift_seen |= raised
        drop |= lift_seen and pos[2]<center[2]+.005 and support>.03
        valid=raised and contact[0] and contact.sum()>=cfg['held_minimum_distinct_fingers'] and slip<=cfg['maximum_relative_slip_speed_m_s']
        hold.update(bool(valid),.01)
        stage='approach' if t<2 else ('close' if t<6 else ('settle' if t<8 else ('lift' if t<11 else 'hold_check')))
        rows.append([d.time,*pos,*forces,support,slip,float(valid),hold.continuous_s]);stages.append(stage)
        states.append(d.qpos.copy());controls.append(d.ctrl.copy());native.append(tactile_forces(m,d))
    folder.mkdir(parents=True,exist_ok=True);rows=np.asarray(rows);native=np.asarray(native)
    np.savetxt(folder/'physics_trace.csv',rows,delimiter=',',comments='',header='time_s,brush_x_m,brush_y_m,brush_z_m,thumb_N,index_N,middle_N,ring_N,little_N,support_N,relative_slip_m_s,valid_hold,continuous_hold_s')
    np.savez_compressed(folder/'replay.npz',time_s=rows[:,0],qpos=states,ctrl=controls,native_touch=native,stage=np.asarray(stages))
    report={'thumb_delta_rad':thumb_delta,'initial_shift_m':list(shift),'initial_brush_center_m':center.tolist(),
            'grip_targets_rad':grip,'approach_offset_m':offset.tolist(),
            'simulation_force_feedback':feedback,'feedback_uses_ground_truth_brush_pose':servo is not None,
            'elapsed_sim_s':float(d.time),'within_preliminary_deadline':d.time<=rules['preliminary']['round_seconds'],
            'physical_touch':first_touch is not None,'first_touch_s':first_touch,'lifted_20mm_clear_of_rack':bool(lift_seen),
            'longest_continuous_hold_s':hold.longest_s,'hold_3s_passed':bool(hold.passed),'brush_drop_observed':bool(drop),
            'maximum_finger_brush_contact_N':float(rows[:,4:9].max()),'maximum_native_touch_N':float(native.max()),
            'maximum_arm_bias_feedforward_Nm':max_bias,'ink_dip':'not_executed','overhead_target':'not_executed','dot':'not_executed',
            'stop_reason':'hold_gate_failed' if not hold.passed else 'later_stages_require_controller_integration',
            'official_score':None,'synthetic_process_points_before_penalties':10*int(first_touch is not None)+15*int(hold.passed),
            'sensor_note':'contact-pair force verifies physical collision; native touch geometry is not calibrated',
            'kinematic_brush_attachment':False,'brush_pose_overwritten_after_initialization':False,'physics_warnings':[]}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def final_checks(folder,rules):
    folder.mkdir(parents=True,exist_ok=True);engine=OfflineOCR();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',70);out=[]
    for title,text,columns in POEMS:
        canvas=Image.new('RGB',(1000,260),'white');ImageDraw.Draw(canvas).text((60,70),title,font=font,fill='black')
        canvas.save(folder/f'{title}_ocr_input.png');bgr=np.asarray(canvas)[:,:,::-1].copy();pred=engine.read(bgr)
        recognized=title in ''.join(p['text'] for p in pred)
        for profile in ('baseline','compact'):
            kwargs={} if profile=='baseline' else dict(character_size=.032,gap=.009,clearance=.01,speed=.06,travel_speed=.20)
            path=make_path(text,columns=columns,**kwargs)
            # Translate to the center of the paper; this is a paper-coordinate plan, not a robot controller.
            span=np.ptp(path.xyz[:,:2],axis=0);path.xyz[:,:2]+=(np.asarray(rules['final']['paper_m'])-span)/2-path.xyz[:,:2].min(0)
            fits=bool((span<=np.asarray(rules['final']['paper_m'])-.02).all())
            duration=float(path.time_s[-1]);reserved=60.
            item={'title':title,'characters':len(text),'profile':profile,'ocr_passed_on_synthetic_title':recognized,
                  'ocr_results':pred,'path_seconds':duration,'reserved_other_actions_s':reserved,
                  'planned_total_s':duration+reserved,'fits_300s_with_reserve':duration+reserved<=rules['final']['round_seconds'],
                  'fits_A3_with_10mm_margin':fits,'fits_A3_paper':bool((span<=np.asarray(rules['final']['paper_m'])).all()),
                  'path_extent_m':span.tolist(),'robot_executed':False,
                  'spoken_title_and_content':False,'official_font_similarity':None,'official_score':None}
            np.savetxt(folder/f'{title}_{profile}_path.csv',np.c_[path.time_s,path.xyz,path.pen_down.astype(int)],delimiter=',',comments='',header='time_s,x_m,y_m,z_m,pen_down')
            out.append(item)
    (folder/'report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['all','grasp','final','judge'],default='all');args=parser.parse_args()
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    folder=output_dir()/'competition';folder.mkdir(exist_ok=True)
    if args.only in ('all','grasp'):
        physical=[]
        for repeat,shift in enumerate([(0.,0.,0.),(.001,-.001,0.)],1):
            for index,delta in enumerate([-.12,0.,.12],1):
                report=physical_trial(folder/f'grasp_r{repeat}_c{index}',rules,delta,shift)
                report['trial']=f'grasp_r{repeat}_c{index}';physical.append(report)
                print(json.dumps({'trial':report['trial'],'touch':report['physical_touch'],'hold_s':report['longest_continuous_hold_s'],'passed':report['hold_3s_passed']},ensure_ascii=False),flush=True)
        (folder/'grasp_summary.json').write_text(json.dumps(physical,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.only in ('all','final'):final_checks(folder/'final_planning',rules)
    if args.only in ('all','judge'):
        cases=[]
        for paper,radius in rules['assumptions']['eye_radius_examples_m'].items():
            for fraction,offset in [(.30,0.),(.50,0.),(.5,.8),(.5,2.)]:
                result=circle_score(radius,radius*fraction,radius*offset,rules['preliminary']['base_points'][paper])
                cases.append({'paper':paper,'ink_radius_fraction':fraction,'center_offset_fraction':offset,**result})
        (folder/'geometry_judge_checks.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    from experiments.write_report import write_report
    write_report()
    print('Saved outputs/competition. These are independent development experiments, not official scored rounds.')


if __name__=='__main__':main()
