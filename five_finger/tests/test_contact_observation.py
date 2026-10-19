import mujoco
import numpy as np
from fivefinger.contact_observation import observe_brush_contacts,wrench_about_center,friction_utilization


def test_wrench_rotation_moment_arm_and_opposite_body_sign():
    frame=np.array([[0,0,1],[1,0,0],[0,1,0]])
    value=wrench_about_center([2,0,0,0,0,0],frame,[.1,0,0],np.zeros(3),True)
    np.testing.assert_allclose(value,[0,0,2,0,-.2,0])
    np.testing.assert_allclose(wrench_about_center([2,0,0,0,0,0],frame,[.1,0,0],np.zeros(3),False),-value)


def test_observation_supports_weight_and_does_not_change_simulation():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
    <worldbody><geom name="paper" type="plane" size="1 1 .1"/>
    <body name="brush" pos="0 0 .051"><freejoint/><geom type="sphere" size=".05" mass="1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m)
    for _ in range(1000):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    state=d.qpos.copy();vel=d.qvel.copy();external=d.qfrc_applied.copy()
    result=observe_brush_contacts(m,d)
    np.testing.assert_allclose(result['wrench_world'][5,:3],[0,0,9.81],atol=.01)
    np.testing.assert_allclose(result['wrench_world'][5,3:],0,atol=1e-8)
    np.testing.assert_allclose(np.asarray(result['details'][0]['frame_on_brush_world'])[:,0],[0,0,1],atol=1e-10)
    np.testing.assert_array_equal(d.qpos,state);np.testing.assert_array_equal(d.qvel,vel)
    np.testing.assert_array_equal(d.qfrc_applied,external)
    assert np.isnan(result['position_brush_m'][:5]).all()
    d.qvel[:]=0.;d.qvel[0]=.2;mujoco.mj_forward(m,d)
    moving=observe_brush_contacts(m,d)
    np.testing.assert_allclose(moving['tangential_relative_speed_m_s'][5],.2,atol=1e-10)
    np.testing.assert_allclose(moving['axial_relative_velocity_m_s'][5],0,atol=1e-10)


def test_friction_cone_includes_rotation_and_anisotropic_coefficients():
    # Each enabled friction component contributes half of normal capacity.
    wrench=[2.,.5,1.,.01,.02,.03]
    coefficients=[.5,1.,.01,.02,.03]
    np.testing.assert_allclose(friction_utilization(wrench,coefficients,3,True),np.sqrt(.5))
    np.testing.assert_allclose(friction_utilization(wrench,coefficients,6,True),np.sqrt(5)/2)
    np.testing.assert_allclose(friction_utilization(wrench,coefficients,6,False),2.5)
    assert friction_utilization(wrench,coefficients,1,True)==0


def test_force_on_first_contact_body_points_away_from_obstacle():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
    <worldbody><body name="brush"><freejoint/><geom name="barrel" type="sphere" size=".05" mass="1"/></body>
    <body pos=".09 0 0"><geom name="obstacle" type="sphere" size=".05"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert d.ncon==1
    assert int(m.geom_bodyid[d.contact[0].geom1])==m.body('brush').id
    result=observe_brush_contacts(m,d)
    assert result['wrench_world'][6,0]<0
    assert np.asarray(result['details'][0]['frame_on_brush_world'])[0,0]<0
    np.testing.assert_allclose(result['wrench_world'][6,1:],0,atol=1e-10)
