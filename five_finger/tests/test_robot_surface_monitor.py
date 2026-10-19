import mujoco
import numpy as np
from fivefinger.robot_surface_monitor import RobotSurfaceMonitor,surface_trace_passes


def test_brush_contact_allowed_but_palm_paper_overlap_retained_without_state_changes():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><geom name="paper" type="plane" size="1 1 .01"/>
        <body pos="0 0 .006"><freejoint/><geom name="collision_palm" type="box" size=".005 .005 .005"/></body>
        <body name="brush" pos=".008 0 .006"><freejoint/><geom name="brush_barrel" type="sphere" size=".005"/></body>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);monitor=RobotSurfaceMonitor(model)
    assert any({model.geom(c.geom1).name,model.geom(c.geom2).name}=={'collision_palm','brush_barrel'} for c in data.contact)
    monitor.update(data,'hold');assert monitor.report()['robot_non_tool_surface_contact_free']
    data.qpos[2]=.004;mujoco.mj_forward(model,data)
    state=data.qpos.copy();force=data.qfrc_applied.copy();control=data.ctrl.copy()
    monitor.update(data,'write');report=monitor.report()
    assert not report['robot_non_tool_surface_contact_free'] and report['rate_hz']==500
    assert report['pairs'][0]['environment_geom']=='paper'
    assert abs(report['pairs'][0]['maximum_penetration_mm']-1)<1e-9
    np.testing.assert_array_equal(data.qpos,state);np.testing.assert_array_equal(data.qfrc_applied,force)
    np.testing.assert_array_equal(data.ctrl,control)
    data.qpos[2]=.006;mujoco.mj_forward(model,data);monitor.update(data,'done')
    assert monitor.report()['pairs'][0]['overlapping_steps']==1


def test_missing_incomplete_or_contradictory_surface_evidence_cannot_pass():
    good={'physics_steps_checked':50,'rate_hz':500.,'robot_non_tool_surface_contact_free':True,'pairs':[]}
    assert surface_trace_passes(good,10,True)
    assert not surface_trace_passes(None,10,True)
    assert not surface_trace_passes(good,11,True)
    assert not surface_trace_passes(good,10,None)
    assert not surface_trace_passes(dict(good,pairs=[{'robot_geom':'collision_palm'}]),10,True)
