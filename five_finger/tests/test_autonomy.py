import json
from dataclasses import fields
import cv2
import numpy as np
import pytest
from fivefinger.autonomy.geometry import Calibration
from fivefinger.autonomy.planning import PaperFrame,poem_path,eye_target,transfer_waypoints
from fivefinger.autonomy.supervisor import Evidence,Phase,RoundSupervisor,TitleConsensus
from fivefinger.autonomy.devices import Commissioning,GuardedArm,UnconfiguredOfficialArm
from fivefinger.autonomy.validation import check_calibration_anchors,audit_reference_joint_path
from fivefinger.autonomy.template_eye import locate_eye
from fivefinger.autonomy.tool_estimator import ToolEstimator
from fivefinger.paths import ROOT


def calibration():
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    return Calibration(np.array([[800,0,640],[0,800,480],[0,0,1.]]),T,(1280,960),'test')


def test_depth_transform_and_bad_units():
    c=calibration();np.testing.assert_allclose(c.backproject([[720,400]],[1]),[[.1,.1,0]])
    d=c.to_dict();d['units']='mm'
    with pytest.raises(ValueError):Calibration.from_dict(d)
    with pytest.raises(ValueError):c.backproject([[640,480]],[np.nan])


def test_anchors_detect_unobservable_to_plane_xy_drift():
    c=calibration();points=np.array([[-.2,-.2,0],[.2,-.2,0],[.2,.2,0],[-.2,.2,0]])
    pixels=points[:,:2]*[800,-800]+[640,480]
    assert check_calibration_anchors(c,points,pixels)['verified']
    d=c.to_dict();d['T_base_camera'][0][3]=.02
    assert not check_calibration_anchors(Calibration.from_dict(d),points,pixels)['verified']


@pytest.mark.parametrize('length',[20,28,40])
def test_rotated_paper_layout_all_rule_lengths(length):
    bank=json.loads((ROOT/'config/adaptive_poems.json').read_text(encoding='utf-8'))
    text=next(t for t in bank.values() if len(t)==length)
    theta=.21;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    p=PaperFrame(np.array([.3,-.15,.002]),R,np.array([.297,.420]))
    path,meta=poem_path(text,p);local=p.to_local(path.xyz)
    assert (abs(local[:,:2])<=p.size/2-meta['margin_m']+1e-8).all()
    assert len(path.xyz)==len(path.pen_down) and (np.diff(path.time_s)>0).all()
    assert np.max(np.linalg.norm(np.diff(path.xyz,axis=0),axis=1)/.01)<=.10001
    assert meta['rows']==(8 if length==40 else 4)


def test_eye_small_target_rejects_uncertainty():
    with pytest.raises(ValueError):eye_target([.2,.2,.002],.003,.002,.0015)
    assert eye_target([.2,.2,.002],.008,.0005,.003).shape==(3,3)
    path=transfer_waypoints([0,0,.1],[.3,0,.01],obstacles=[([.1,-.1,0],[.2,.1,.25])])
    assert min(path[1:3,2])>=.30


def test_hold_and_dip_require_continuous_measured_evidence():
    s=RoundSupervisor();s.start(0);s.phase=Phase.HOLD
    for i in range(1,350):
        t=i*.01;e=Evidence(t,calibration_verified=True,tool_calibrated=True,trajectory_checked=True,stable_grip=i!=200)
        s.tick(t,e)
    assert s.phase==Phase.HOLD
    for i in range(350,530):
        t=i*.01;s.tick(t,Evidence(t,calibration_verified=True,tool_calibrated=True,trajectory_checked=True,stable_grip=True))
    assert s.phase==Phase.DIP
    for i in range(530,610):
        t=i*.01;s.tick(t,Evidence(t,calibration_verified=True,tool_calibrated=True,trajectory_checked=True,ink_depth_m=.004))
    assert s.phase==Phase.DIP
    s.tick(6.10,Evidence(6.10,calibration_verified=True,tool_calibrated=True,trajectory_checked=True,ink_withdrawn=True,ink_loaded=True))
    assert s.phase==Phase.EXECUTE


@pytest.mark.parametrize('fault,reason', [('camera_lost','camera_lost'),('robot_fault','robot_fault'),
    ('collision_risk','collision_risk'),('force_limit_exceeded','force_limit'),('dropped','brush_dropped'),('operator_stop','operator_stop')])
def test_faults_stop_without_action(fault,reason):
    s=RoundSupervisor();s.start(0);e=Evidence(.01);setattr(e,fault,True)
    assert s.tick(.01,e)==Phase.STOPPED and s.reason==reason


def test_paper_change_does_not_reset_round_clock():
    s=RoundSupervisor();s.start(0);s.phase=Phase.WAIT_PAPER;s.last=299.8
    s.tick(299.9,Evidence(299.9,paper_replaced=True,scene_ready=True))
    assert s.phase==Phase.OBSERVE and s.remaining(299.9)==pytest.approx(.1)
    assert s.tick(300.,Evidence(300.))==Phase.STOPPED and s.reason=='round_time_limit'
    with pytest.raises(RuntimeError):s.start(301)


def test_stale_data_paper_motion_and_reused_ocr_frame():
    s=RoundSupervisor();s.start(0)
    assert s.tick(.21,Evidence(0))==Phase.STOPPED
    s=RoundSupervisor();s.start(0);s.phase=Phase.EXECUTE
    s.tick(.01,Evidence(.01,paper_shift_m=.004,calibration_verified=True,tool_calibrated=True,trajectory_checked=True))
    assert s.reason=='paper_moved_during_contact'
    c=TitleConsensus();bank={'静夜思':'test'}
    for _ in range(4):assert c.push('静夜思',.99,.1,bank) is None
    assert c.push('静夜思',.99,.2,bank) is None
    assert c.push('静夜思',.99,.3,bank)=='静夜思'
    assert c.push('静夜想',.99,.4,bank) is None


class FakeArm:
    def __init__(self):self.sent=[];self.stops=[]
    def read_joints(self):return 1.,{'joint':0.}
    def servo_joints(self,p,dt):self.sent.append(p)
    def controlled_stop(self,reason):self.stops.append(reason)


def test_driver_rejects_uncommissioned_and_unsafe_commands():
    with pytest.raises(RuntimeError):UnconfiguredOfficialArm()
    with pytest.raises(RuntimeError):GuardedArm(FakeArm(),{'joint':(-1,1,1)},lambda a,b:True,Commissioning())
    commissioned=Commissioning(**{f.name:True for f in fields(Commissioning)})
    arm=FakeArm();driver=GuardedArm(arm,{'joint':(-1,1,1)},lambda a,b:True,commissioned)
    driver.send({'joint':.01},1.01,.02);assert len(arm.sent)==1
    for target,now in [({'joint':2},1.01),({'joint':.2},1.01),({'wrong':0},1.01),({'joint':0},2.)]:
        with pytest.raises(ValueError):driver.send(target,now,.02)
    assert len(arm.sent)==1 and len(arm.stops)==4


def test_template_registration_and_blank_rejection():
    rng=np.random.default_rng(93);src=np.full((500,350,3),255,np.uint8)
    for _ in range(100):
        a=tuple(rng.integers([20,20],[330,480]));b=tuple(rng.integers([20,20],[330,480]))
        cv2.line(src,a,b,(0,0,0),2)
    H=cv2.getPerspectiveTransform(np.float32([[0,0],[349,0],[349,499],[0,499]]),np.float32([[90,60],[440,80],[420,580],[70,550]]))
    dst=cv2.warpPerspective(src,H,(650,650),borderValue=(180,180,180))
    result=locate_eye(src,dst,(120,150),10)
    expected=cv2.perspectiveTransform(np.float32([[[120,150]]]),H)[0,0]
    assert np.linalg.norm(np.asarray(result['center_px'])-expected)<2
    with pytest.raises(ValueError):locate_eye(src,np.full_like(dst,255),(120,150),10)


def test_tool_estimator_uses_calibration_and_rejects_slip():
    T=np.eye(4);T[2,3]=.2;estimator=ToolEstimator(T)
    pose=estimator.estimate(np.eye(4),1.,1.01,visual_tip=[0,0,.2],visual_stamp=1.)
    np.testing.assert_allclose(pose['T_base_tip'][:3,3],[0,0,.2])
    with pytest.raises(ValueError):estimator.estimate(np.eye(4),1.,1.01,visual_tip=[.02,0,.2],visual_stamp=1.)
    with pytest.raises(ValueError):estimator.estimate(np.eye(4),1.,2.)


def test_nonfinite_feedback_and_interpolated_robot_table_guard():
    s=RoundSupervisor();s.start(0)
    assert s.tick(.01,Evidence(float('nan')))==Phase.STOPPED
    from experiments.competition_scene import make_scene
    m,d,*_=make_scene();q=d.qpos[:6].copy()
    assert audit_reference_joint_path(m,d,[q,q])['valid']
    bad=[-.539,3.18,2.57,-.608,2.11,0]
    verdict=audit_reference_joint_path(m,d,[q,bad])
    assert not verdict['valid'] and verdict['samples']>1


def test_disconnected_depth_rejects_before_pose_estimation():
    from fivefinger.autonomy.perception import perceive
    c=calibration()
    with pytest.raises(ValueError,match='Insufficient valid depth'):
        perceive(np.zeros((960,1280,3),np.uint8),np.full((960,1280),np.nan),c)
