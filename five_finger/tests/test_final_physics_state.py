import mujoco
import numpy as np
from fivefinger.physics_recording import save_final_physics_state


def test_partial_control_interval_records_exact_integrator_state_without_mutation(tmp_path):
    m=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody><body pos="0 0 1"><freejoint/><geom size=".01"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m)
    for _ in range(9):mujoco.mj_step(m,d)
    q=d.qpos.copy();path=tmp_path/'final_physics_state.npz'
    save_final_physics_state(path,m,d,9,.01)
    with np.load(path,allow_pickle=False) as saved:
        assert int(saved['physics_steps'])==9
        assert abs(float(saved['time_s'])-.018)<1e-12
        assert float(saved['last_complete_control_time_s'])==.01
        restored=mujoco.MjData(m);spec=int(saved['state_spec'])
        mujoco.mj_setState(m,restored,saved['integration_state'],spec)
        actual=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,restored,actual,spec)
        np.testing.assert_array_equal(actual,saved['integration_state'])
    np.testing.assert_array_equal(d.qpos,q)
