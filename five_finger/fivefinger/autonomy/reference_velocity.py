"""Bounded reference-motion feedforward for a position-command inner loop.

This is a simulation development model, not a vendor SDK implementation.
Never infer hardware gains from a MuJoCo reference model.
"""
import numpy as np


class ReferenceVelocity:
    def __init__(self):
        self.previous = None
        self.phase = None

    def update(self, target, phase, dt):
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(dt) or not 0 < dt <= .05:
            raise ValueError('Invalid reference sample')
        velocity = np.zeros(3)
        if self.previous is not None and phase == self.phase:
            velocity = (target-self.previous)/dt
            speed = np.linalg.norm(velocity[:2])
            velocity[:2] *= min(1., .12/max(speed, 1e-12))
            velocity[2] = np.clip(velocity[2], -.06, .06)
        self.previous = target.copy()
        self.phase = phase
        return velocity


def position_servo_feedforward(joint_velocity, kp, kv):
    """u_ff=(kv/kp)*qdot_ref, with existing development command bounds.

For tau=kp*(u-q)-kv*qdot, this adds desired-velocity damping compensation.
It does not cancel measured-velocity damping or modify actuator gains.
"""
    joint_velocity, kp, kv = (np.asarray(x, float) for x in (joint_velocity, kp, kv))
    if joint_velocity.shape != (6,) or kp.shape != (6,) or kv.shape != (6,):
        raise ValueError('Expected six arm joints')
    if not all(np.isfinite(x).all() for x in (joint_velocity, kp, kv)) or np.any(kp <= 0) or np.any(kv < 0) or np.any(kv/kp > .2):
        raise ValueError('Invalid reference position-servo gains')
    return np.clip((kv/kp)*np.clip(joint_velocity, -.4, .4), -.04, .04)
