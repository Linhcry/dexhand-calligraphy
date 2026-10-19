"""Contact on a proximal phalanx cannot be moved by the distal joint."""
import mujoco
import numpy as np
from fivefinger.grasp_feedback import ContactGripServo


def fixture_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <default><joint damping=".1"/><geom size=".006" mass=".01"/></default>
    <worldbody><body><joint name="r_f_joint2_1" axis="0 -1 0"/><geom/>
      <body pos="0 0 .02"><joint name="r_f_joint2_2" axis="0 1 0"/><geom/>
        <body pos="0 0 .02"><joint name="r_f_joint2_3" axis="0 1 0"/>
          <geom name="col_r_f_link2_3" type="capsule" fromto="0 0 0 0 0 .02"/>
          <body pos="0 0 .04"><joint name="r_f_joint2_4" axis="0 1 0"/>
            <geom name="col_r_f_link2_4" type="capsule" size=".006 .01"/>
          </body></body></body></body>
      <body name="brush" pos=".010 0 .047"><freejoint/>
        <geom name="brush_barrel" type="capsule" size=".006 .004"/>
      </body></worldbody><actuator>
      <position name="act_r_f_joint2_1" joint="r_f_joint2_1" ctrlrange="0 .3"/>
      <position name="act_r_f_joint2_2" joint="r_f_joint2_2" ctrlrange="0 1.3"/>
      <position name="act_r_f_joint2_3" joint="r_f_joint2_3" ctrlrange="0 1.3"/>
      <position name="act_r_f_joint2_4" joint="r_f_joint2_4" ctrlrange="0 1.3"/>
      </actuator></mujoco>''')


def test_proximal_contact_does_not_request_distal_joint_motion():
    m=fixture_model()
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    pairs=[{m.geom(c.geom1).name,m.geom(c.geom2).name} for c in d.contact]
    assert {'col_r_f_link2_3','brush_barrel'} in pairs
    servo=ContactGripServo(m)
    delta=servo.update(d,np.array([0,.1,0,0,0]),.01,fingers=(2,),measured_contact=True)
    assert delta['r_f_joint2_4']==0.
    assert abs(delta['r_f_joint2_1'])+abs(delta['r_f_joint2_2'])+abs(delta['r_f_joint2_3'])>0


def test_feedback_is_redistributed_away_from_saturated_joint():
    m=fixture_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    force=np.array([0,.1,0,0,0]);base={f'r_f_joint2_{i}':0. for i in (1,2,3,4)}
    legacy=ContactGripServo(m).update(d,force,.01,fingers=(2,),measured_contact=True)
    servo=ContactGripServo(m);servo.delta['r_f_joint2_1']=-.2
    bounded=servo.update(d,force,.01,fingers=(2,),measured_contact=True,base_targets=base)
    assert legacy['r_f_joint2_1']<0  # This request would have been clipped by the actuator.
    assert bounded['r_f_joint2_1']==0
    assert bounded['r_f_joint2_2']>legacy['r_f_joint2_2']
    assert bounded['r_f_joint2_4']==0
    assert all(0<=bounded[n]<=.25 for n in base)


def test_maintenance_preserves_acquired_spread_but_still_corrects_flexion():
    m=fixture_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    servo=ContactGripServo(m);servo.delta['r_f_joint2_1']=.04
    force=np.array([0,.1,0,0,0])
    for _ in range(10):
        delta=servo.update(d,force,.01,fingers=(2,),measured_contact=True,lock_spread=True)
    assert delta['r_f_joint2_1']==.04
    assert delta['r_f_joint2_2']>0
    assert delta['r_f_joint2_4']==0
