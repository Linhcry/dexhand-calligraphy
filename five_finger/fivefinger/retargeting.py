"""Offline DexPilot hand-pose retargeting using the MuJoCo reference model."""
import numpy as np
from .mujoco_kinematics import MujocoHandKinematics


def create_optimizer(model, data, scaling=1.0, pinch_distance=.012):
    from .third_party.dex_optimizer import DexPilotOptimizer
    robot = MujocoHandKinematics(model, data)
    # The middle-finger spread coordinate has a near-zero range in this model.
    names = [name for name in robot.dof_joint_names if name != 'r_f_joint3_1']
    opt = DexPilotOptimizer(robot, names, robot.link_names[1:], robot.link_names[0],
                           scaling=scaling, eta1=pinch_distance, norm_delta=0.)
    ids = [robot.get_joint_index(name) for name in names]
    opt.set_joint_limit(robot.joint_limits[ids], epsilon=0.)
    opt.opt.set_maxeval(100)
    return robot, opt


def tip_vectors(points):
    """points: wrist, thumb, index, middle, ring, little, in meters."""
    from .third_party.dex_optimizer import DexPilotOptimizer
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 3) or not np.isfinite(points).all():
        raise ValueError('Expected six finite xyz points in meters')
    origin, task = DexPilotOptimizer.generate_link_indices(5)
    return points[task] - points[origin]


def retarget_points(robot, optimizer, points, initial=None):
    initial = robot.q0.copy() if initial is None else np.asarray(initial).copy()
    ids = optimizer.idx_pin2target
    result = optimizer.retarget(tip_vectors(points), initial[optimizer.idx_pin2fixed], initial[ids])
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite retargeting result')
    angles = initial.copy()
    angles[ids] = np.clip(result, robot.joint_limits[ids, 0], robot.joint_limits[ids, 1])
    robot.compute_forward_kinematics(angles)
    return angles


def robot_points(robot):
    return np.asarray([robot.get_link_pose(i)[:3, 3] for i in range(6)])
