"""Dex-retargeting kinematics interface backed by the actual MuJoCo hand.

Only finger hinge coordinates are exposed. Arm and brush states stay fixed in
private scratch data. Returned frames are relative to right_hand_base.
"""
import mujoco
import numpy as np


class MujocoHandKinematics:
    def __init__(self, model, initial_data):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_data.qpos
        self.dof_joint_names = [f'r_f_joint{i}_{j}' for i in range(1, 6) for j in range(1, 5)]
        joints = [model.joint(name).id for name in self.dof_joint_names]
        self.qadr = model.jnt_qposadr[joints]
        self.vadr = model.jnt_dofadr[joints]
        self.joint_limits = model.jnt_range[joints].copy()
        self.dof = len(joints)
        self.q0 = self.data.qpos[self.qadr].copy()
        self.link_names = ['right_hand_base'] + [f'r_f_link{i}_tip' for i in range(1, 6)]
        self.base_id = model.body('right_hand_base').id
        self.bodies = [model.body(name).id for name in self.link_names[1:]]
        self.compute_forward_kinematics(self.q0)

    def get_joint_index(self, name):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name):
        return self.link_names.index(name)

    def compute_forward_kinematics(self, qpos):
        values = np.asarray(qpos)
        if values.shape != (self.dof,) or not np.isfinite(values).all():
            raise ValueError('Expected 20 finite finger joint angles')
        self.data.qpos[self.qadr] = values
        mujoco.mj_forward(self.model, self.data)

    def get_link_pose(self, link_id):
        pose = np.eye(4)
        if link_id == 0:
            return pose
        body = self.bodies[link_id - 1]
        base_r = self.data.xmat[self.base_id].reshape(3, 3)
        pose[:3, :3] = base_r.T @ self.data.xmat[body].reshape(3, 3)
        pose[:3, 3] = base_r.T @ (self.data.xpos[body] - self.data.xpos[self.base_id])
        return pose

    def compute_single_link_local_jacobian(self, qpos, link_id):
        self.compute_forward_kinematics(qpos)
        if link_id == 0:
            return np.zeros((6, self.dof))
        body = self.bodies[link_id - 1]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
        local_r = self.data.xmat[body].reshape(3, 3).T
        return np.vstack([local_r @ jp[:, self.vadr], local_r @ jr[:, self.vadr]])
