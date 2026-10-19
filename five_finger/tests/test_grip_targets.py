import numpy as np
import pytest
from fivefinger.grasp_feedback import validated_grip_targets


@pytest.mark.parametrize('targets', [0, -1, 3.01, np.nan, np.inf, [2.7, 3.1, 1.35], [2.7, 2.7]])
def test_invalid_maintenance_forces_cannot_bypass_acquisition_limits(targets):
    with pytest.raises(ValueError):
        validated_grip_targets(targets)


def test_target_validation_copies_input_and_preserves_force_units():
    targets = np.array([2.7, 2.7, 1.35])
    result = validated_grip_targets(targets)
    targets[1] = 100
    np.testing.assert_allclose(result, [2.7, 2.7, 1.35])
