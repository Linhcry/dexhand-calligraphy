"""Save the last physical state, including stops between 100 Hz trace rows."""
from pathlib import Path
import mujoco
import numpy as np


def save_final_physics_state(path,model,data,physics_steps,last_control_time):
    spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,state,spec)
    destination=Path(path);temporary=destination.with_name(destination.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,integration_state=state,state_spec=spec,mujoco_version=mujoco.__version__,
            nq=model.nq,nv=model.nv,time_s=data.time,physics_steps=physics_steps,
            last_complete_control_time_s=last_control_time,qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl)
    temporary.replace(destination)
