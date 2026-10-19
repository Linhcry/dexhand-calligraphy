import numpy as np
import pytest
from fivefinger.autonomy.grip_guard import LiftGripGate


def test_lift_waits_for_continuous_force_reserve_and_releases():
    gate=LiftGripGate();strong=[2.,1.,0,0,0];weak=[1.,1.,0,0,0]
    assert gate.allow(False,weak,.01)
    assert gate.allow(True,strong,.01)
    assert not gate.allow(True,weak,.01)
    for _ in range(9):assert not gate.allow(True,strong,.01)
    assert not gate.allow(True,weak,.01)  # One bad sample resets continuous confirmation.
    for _ in range(9):assert not gate.allow(True,strong,.01)
    assert gate.allow(True,strong,.01)
    assert not gate.active and gate.activations==1


def test_grip_recovery_is_bounded_and_rejects_bad_feedback():
    gate=LiftGripGate()
    for _ in range(79):assert not gate.allow(True,[1,1,0,0,0],.01)
    with pytest.raises(RuntimeError,match='grip_reserve_timeout'):gate.allow(True,[1,1,0,0,0],.01)
    for forces in [[np.nan]*5,[-1]*5,[1,1]]:
        with pytest.raises(ValueError):LiftGripGate().allow(True,forces,.01)
