"""Independent finite differences for the held-tool model; pen stays a free body."""
import mujoco
import numpy as np
import pytest
from experiments.competition_scene import make_scene
from experiments.blind_competition import nib_point,command_nib
from fivefinger.grasp_feedback import ContactGripServo


@pytest.mark.parametrize('damping_s,ref_velocity',[(0.,None),(.04,None),(.04,np.array([.03,0.,-.01]))])
def test_tool_lever_arm_jacobian_matches_finite_difference(damping_s,ref_velocity):
    m,d,*_=make_scene()
    site=m.site('palm_control').id;body=int(m.site_bodyid[site])
    point=nib_point(m,d)
    local=d.xmat[body].reshape(3,3).T@(point-d.xpos[body])
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,jp,jr,point,body)
    q=d.qpos.copy();numerical=np.zeros((3,6));eps=1e-6
    for i in range(6):
        samples=[]
        for sign in (1,-1):
            d.qpos[:]=q;d.qpos[i]+=sign*eps;mujoco.mj_forward(m,d)
            samples.append(d.xpos[body]+d.xmat[body].reshape(3,3)@local)
        numerical[:,i]=(samples[0]-samples[1])/(2*eps)
    np.testing.assert_allclose(jp[:,:6],numerical,atol=1e-7)
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    control=d.ctrl.copy();external=d.qfrc_applied.copy()
    command_nib(m,d,point+[.0001,0,0],d.site_xmat[site].reshape(3,3).copy(),point_jacobian=True,normal_damping_s=damping_s,target_velocity=ref_velocity)
    np.testing.assert_array_equal(d.qpos,q)
    np.testing.assert_array_equal(d.qfrc_applied,external)
    np.testing.assert_array_equal(d.ctrl[6:],control[6:])
    assert np.max(abs(d.ctrl[:6]-q[:6]))<=.060001
    assert m.joint('brush_free').type[0]==mujoco.mjtJoint.mjJNT_FREE


@pytest.mark.parametrize('gain,deadband',[(float('nan'),.1),(.04,.1),(.001,-1),(.001,float('nan'))])
def test_maintenance_rejects_invalid_feedback_before_model_access(gain,deadband):
    servo=ContactGripServo(None)
    with pytest.raises(ValueError):servo.update(None,np.zeros(5),.01,gain=gain,deadband_n=deadband)


def test_maintenance_rejects_negative_force_and_duplicate_fingers():
    servo=ContactGripServo(None)
    with pytest.raises(ValueError):servo.update(None,-np.ones(5),.01)
    with pytest.raises(ValueError):servo.update(None,np.ones(5),.01,fingers=(1,1))


def test_palm_reference_does_not_follow_a_displaced_free_pen():
    m,d,*_=make_scene();site=m.site('palm_control').id
    target=d.site_xpos[site].copy()+[0,0,.02]
    rotation=d.site_xmat[site].reshape(3,3).copy()
    command_nib(m,d,target,rotation,point_jacobian=True,palm_reference=True)
    expected=d.ctrl.copy()
    address=int(m.joint('brush_free').qposadr[0])
    d.qpos[address:address+3]+=[.03,-.02,.05]
    mujoco.mj_forward(m,d);state=d.qpos.copy();external=d.qfrc_applied.copy()
    command_nib(m,d,target,rotation,point_jacobian=True,palm_reference=True)
    np.testing.assert_allclose(d.ctrl,expected,atol=1e-12)
    np.testing.assert_array_equal(d.qpos,state)
    np.testing.assert_array_equal(d.qfrc_applied,external)
