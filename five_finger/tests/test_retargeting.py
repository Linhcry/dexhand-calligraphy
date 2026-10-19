"""Meaningful frame/Jacobian and source-port regression checks."""
import numpy as np
import pytest

pytest.importorskip('nlopt')
pytest.importorskip('torch')
from fivefinger.cli import load_workcell
from fivefinger.mujoco_kinematics import MujocoHandKinematics
from fivefinger.retargeting import create_optimizer, robot_points, retarget_points


def test_local_jacobian_matches_finite_difference_without_mutating_live_state():
    model, live = load_workcell()
    original = live.qpos.copy()
    robot = MujocoHandKinematics(model, live)
    q = np.linspace(.05, .4, 20)
    q = np.clip(q, robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    for link in range(1, 6):
        robot.compute_forward_kinematics(q)
        rotation = robot.get_link_pose(link)[:3, :3]
        analytic = rotation @ robot.compute_single_link_local_jacobian(q, link)[:3]
        numeric = np.zeros((3, 20))
        for joint in range(20):
            delta = np.zeros(20); delta[joint] = 1e-6
            robot.compute_forward_kinematics(q + delta)
            plus = robot.get_link_pose(link)[:3, 3]
            robot.compute_forward_kinematics(q - delta)
            minus = robot.get_link_pose(link)[:3, 3]
            numeric[:, joint] = (plus - minus) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    np.testing.assert_array_equal(live.qpos, original)


def test_known_robot_pose_reconstruction_reduces_error_and_respects_limits():
    model, live = load_workcell()
    robot, optimizer = create_optimizer(model, live)
    # Disable pinch projection for a pure known-pose regression test.
    optimizer.project_dist = 0.
    q = robot.q0.copy()
    q[0] = .35
    for i in range(5):
        q[i*4+1:i*4+4] = [.55, .45, .35]
    robot.compute_forward_kinematics(q)
    reference = robot_points(robot)
    robot.compute_forward_kinematics(robot.q0)
    before = np.linalg.norm(robot_points(robot) - reference)
    out = retarget_points(robot, optimizer, reference)
    after = np.linalg.norm(robot_points(robot) - reference)
    assert after < .003
    assert after < before / 10
    assert (out >= robot.joint_limits[:, 0]).all()
    assert (out <= robot.joint_limits[:, 1]).all()
