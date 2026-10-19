import numpy as np
import pytest
from fivefinger.autonomy.reference_velocity import ReferenceVelocity, position_servo_feedforward


def test_phase_change_does_not_generate_a_velocity_spike():
    estimator = ReferenceVelocity()
    np.testing.assert_array_equal(estimator.update([0,0,0], 'travel', .01), np.zeros(3))
    np.testing.assert_allclose(estimator.update([.0003,0,-.0002], 'travel', .01), [.03,0,-.02])
    np.testing.assert_array_equal(estimator.update([1,1,1], 'contact', .01), np.zeros(3))
    np.testing.assert_array_equal(estimator.update([1,1,1], 'contact', .01), np.zeros(3))


def test_reference_limits_and_invalid_samples():
    estimator = ReferenceVelocity(); estimator.update([0,0,0], 'travel', .01)
    velocity = estimator.update([1,1,-1], 'travel', .01)
    assert np.linalg.norm(velocity[:2]) == pytest.approx(.12)
    assert velocity[2] == -.06
    with pytest.raises(ValueError): estimator.update([0,0,np.nan], 'travel', .01)
    with pytest.raises(ValueError): estimator.update([0,0,0], 'travel', 0)


def test_feedforward_realizes_desired_velocity_damping_in_inner_servo():
    kp=np.full(6,1000.);kv=np.full(6,100.)
    q=np.linspace(0,.5,6);qref=q+.001;qdot=np.full(6,.01);qdotref=np.full(6,.02)
    offset=position_servo_feedforward(qdotref,kp,kv)
    np.testing.assert_allclose(kp*(qref+offset-q)-kv*qdot, kp*(qref-q)+kv*(qdotref-qdot))
    assert np.max(abs(position_servo_feedforward(np.full(6,100.),kp,kv))) <= .04
    with pytest.raises(ValueError): position_servo_feedforward(qdotref,np.zeros(6),kv)
