from dataclasses import dataclass
import mujoco
import numpy as np


def set_joint_targets(model, data, targets):
    """Targets are joint positions in radians, not physical motor CAN commands."""
    for joint_name, angle in targets.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + joint_name)
        if actuator_id < 0:
            raise KeyError(joint_name)
        if not np.isfinite(angle):
            raise ValueError("Joint targets must be finite")
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = np.clip(angle, low, high)


def tactile_forces(model, data):
    forces = []
    for finger in range(1, 6):
        sensor = model.sensor(f"touch_r_f_link{finger}_pad")
        forces.append(float(data.sensordata[sensor.adr[0]]))
    return np.array(forces)


class ForceLimitedClosure:
    """Simulation-only proportional curl controller, pending real coupling calibration."""
    def __init__(self, target_n=0.5, maximum_n=2.0):
        if not 0 < target_n < maximum_n:
            raise ValueError("Invalid development force settings")
        self.target_n, self.maximum_n = target_n, maximum_n
        self.curl = np.zeros(5)

    def update(self, forces_n, dt):
        forces = np.asarray(forces_n, float)
        if forces.shape != (5,) or not np.isfinite(forces).all() or dt <= 0:
            raise ValueError("Expected five finite contact force readings and positive dt")
        velocity = np.clip(0.35 * (self.target_n-forces), -0.5, 0.12)
        velocity[forces > self.maximum_n] = -0.5
        self.curl = np.clip(self.curl + dt*velocity, 0, .9)
        return {f"r_f_joint{i+1}_{joint}": float(self.curl[i])
                for i in range(5) for joint in (2, 3, 4)}


@dataclass
class IKResult:
    joints: np.ndarray
    converged: bool
    position_error_m: float
    orientation_error_rad: float


def solve_arm_ik(model, data, target_position, target_rotation=None, iterations=150):
    """Damped least-squares pose IK, joint limits enforced; no obstacle avoidance."""
    target_position = np.asarray(target_position, float)
    if target_position.shape != (3,) or not np.isfinite(target_position).all():
        raise ValueError("Expected finite xyz target")
    if target_rotation is not None:
        target_rotation = np.asarray(target_rotation, float)
        if target_rotation.shape != (3, 3) or not np.allclose(target_rotation.T@target_rotation, np.eye(3), atol=1e-6) or np.linalg.det(target_rotation) < 0:
            raise ValueError("Expected a proper rotation matrix")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    site_id = model.site("palm_control").id
    joints = [model.joint(f"r_a_joint{i}").id for i in range(1, 7)]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    orientation_error = np.zeros(3)
    for _ in range(iterations):
        mujoco.mj_forward(model, scratch)
        position_error = target_position-scratch.site_xpos[site_id]
        current_r = scratch.site_xmat[site_id].reshape(3, 3)
        if target_rotation is not None:
            from scipy.spatial.transform import Rotation
            orientation_error = Rotation.from_matrix(target_rotation@current_r.T).as_rotvec()
        if np.linalg.norm(position_error) < 0.0005 and np.linalg.norm(orientation_error) < 0.01:
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, vadr], .3*jacr[:, vadr]]) if target_rotation is not None else jacp[:, vadr]
        error = np.r_[position_error, .3*orientation_error] if target_rotation is not None else position_error
        delta = jac.T@np.linalg.solve(jac@jac.T + .002**2*np.eye(len(error)), error)
        delta *= min(1., .08/max(np.max(np.abs(delta)), 1e-9))
        scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + delta, model.jnt_range[joints, 0], model.jnt_range[joints, 1])
    mujoco.mj_forward(model, scratch)
    pos_error = float(np.linalg.norm(target_position-scratch.site_xpos[site_id]))
    if target_rotation is not None:
        from scipy.spatial.transform import Rotation
        orientation_error = Rotation.from_matrix(target_rotation@scratch.site_xmat[site_id].reshape(3, 3).T).as_rotvec()
    rot_error = float(np.linalg.norm(orientation_error))
    return IKResult(scratch.qpos[qadr].copy(), pos_error < .0005 and rot_error < .01, pos_error, rot_error)
