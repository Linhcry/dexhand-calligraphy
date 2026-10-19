"""Physical contact negatives and independent kinematic checks for the writing demo."""
import mujoco
import numpy as np
import pytest
from experiments.writing import make_writing_model,paper_contact
from fivefinger.control import solve_arm_ik


@pytest.mark.parametrize('surface,nib_x,expect_ink',[('paper',0.,True),('table',0.,False),('paper',1.,False)])
def test_ink_requires_nib_paper_contact(surface,nib_x,expect_ink):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep="0.002"/>
    <default><geom solref=".006 1" solimp=".95 .999 .001"/></default>
    <worldbody><geom name="paper" type="box" size=".1 .1 .002" pos="{0 if surface=='paper' else 2} 0 0"/>
    <geom name="table" type="plane" size="3 3 .1" pos="0 0 -.003"/>
    <body pos="{nib_x} 0 .03"><freejoint/><geom name="brush_tip_approximation" type="sphere" size=".003" mass=".02"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert paper_contact(model,data)[0]==0.
    for _ in range(500):mujoco.mj_step(model,data)
    force,point=paper_contact(model,data)
    assert (force>.01)==expect_ink
    if expect_ink:assert abs(point[2]-.002)<.001


def test_reference_hand_reaches_writing_footprint_and_pen_is_free():
    model,data,_,rotation=make_writing_model()
    original=data.qpos.copy();offset=data.site('brush_tip').xpos-data.site('palm_control').xpos
    for x in (.25,.29,.33):
        for y in (-.09,-.05,-.01):
            result=solve_arm_ik(model,data,np.array([x,y,.002])-offset,rotation,iterations=400)
            assert result.converged,result
            assert result.position_error_m<.0005
    np.testing.assert_array_equal(data.qpos,original)
    pen_joint=model.joint('brush_free')
    assert pen_joint.type[0]==mujoco.mjtJoint.mjJNT_FREE
    assert model.neq==0 and model.nmocap==0
    assert not np.any(model.actuator_trnid[:,0]==pen_joint.id)
    assert np.all(data.qfrc_applied[pen_joint.dofadr[0]:]==0.)
