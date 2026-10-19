import numpy as np
import pytest
import mujoco
from fivefinger.surface_command_filter import tip_preserving_projection,SurfaceCommandFilter


def test_clearance_correction_uses_redundancy_without_changing_tip_displacement():
    j=np.column_stack([np.eye(3),np.zeros((3,3))]);h=np.array([[0,0,0,.1,0,0]])
    delta=np.array([.01,-.02,.005,-.04,0,0]);gap=np.array([.006]);bounds=np.full(6,.06)
    filtered=tip_preserving_projection(j,h,delta,gap,.005,-bounds,bounds)
    np.testing.assert_allclose(j@filtered,j@delta,atol=1e-12)
    assert float((h@filtered)[0])>=-.3*(gap[0]-.005)-1e-8
    assert np.max(abs(filtered))<=.06+1e-10


def test_infeasible_clearance_is_reported_instead_of_breaking_tip_constraint():
    j=np.column_stack([np.eye(3),np.zeros((3,3))]);h=np.array([[0,0,1.,0,0,0]])
    delta=np.array([0,0,-.01,0,0,0]);bounds=np.full(6,.06)
    with pytest.raises(RuntimeError,match='infeasible'):
        tip_preserving_projection(j,h,delta,np.array([.006]),.005,-bounds,bounds)


def test_active_filter_only_returns_commands_and_does_not_change_live_physics():
    joints=''.join(f'<joint name="j{i}" type="{kind}" axis="{axis}"/>' for i,(kind,axis) in enumerate([
        ('slide','1 0 0'),('slide','0 1 0'),('slide','0 0 1'),('hinge','1 0 0'),('hinge','0 1 0'),('hinge','0 0 1')]))
    actuators=''.join(f'<position joint="j{i}" ctrlrange="-.5 .5"/>' for i in range(6))
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="right_hand_base" pos="0 0 .016">{joints}
        <geom name="collision_palm" type="box" size=".04 .02 .005"/><site name="palm_control"/>
      </body><body name="brush" pos="0 0 .2"><freejoint/><geom size=".006"/></body>
      </worldbody><actuator>{actuators}</actuator></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    vertices=np.array([[x,y,z] for x in (-.04,.04) for y in (-.02,.02) for z in (-.005,.005)])
    f=SurfaceCommandFilter(model,[(model.geom('collision_palm').id,'right_hand_base',vertices)],0.)
    command=np.array([0,0,-.008,0,0,0.]);before={n:getattr(data,n).copy() for n in ('qpos','qvel','ctrl','qfrc_applied')}
    result=f.apply(data,command,np.array([.06,0,.002]))
    assert f.active==1 and not np.array_equal(result,command)
    assert np.max(abs(result-data.qpos[:6]))<=.06+1e-8
    for name,value in before.items():np.testing.assert_array_equal(getattr(data,name),value)
    f.require_actual_clear(data)
    data.qpos[2]=-.010;mujoco.mj_kinematics(model,data)
    with pytest.raises(RuntimeError,match='surface_actual_clearance_stop'):f.require_actual_clear(data)
    assert f.report()['actual_clearance_stops']==1
    # The stop occurs at a positive 1 mm gap in this fixture, before overlap.
    assert 0<f.report()['minimum_actual_gap_m']<.002
