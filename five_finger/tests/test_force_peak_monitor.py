import mujoco
import numpy as np
import pytest
from fivefinger.force_peak_monitor import ForcePeakMonitor


def test_physics_step_monitor_keeps_impact_peak_and_rejects_missing_steps():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
    <worldbody><geom name="paper" type="plane" size="1 1 .1"/>
    <body name="brush" pos="0 0 .15"><freejoint/><geom type="sphere" size=".05" mass="1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);monitor=ForcePeakMonitor(m)
    for i in range(1000):
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        state=d.qpos.copy();velocity=d.qvel.copy()
        monitor.update(d,'impact' if i<200 else 'settled')
        np.testing.assert_array_equal(d.qpos,state);np.testing.assert_array_equal(d.qvel,velocity)
    report=monitor.report()
    assert report['samples']==1000
    assert report['groups']['paper']['maximum_recorded_N']>20
    assert report['groups']['paper']['stage']=='impact'
    np.testing.assert_allclose(report['stage_maxima_N']['settled']['paper'],9.81,atol=.05)
    for _ in range(2):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    with pytest.raises(ValueError,match='skipped'):monitor.update(d,'settled')
