import numpy as np
import pytest
from experiments.blind_competition import normal_damping_correction


def test_damping_opposes_motion_and_disappears_away_from_paper():
    assert normal_damping_correction(.002, -.02, .04) == pytest.approx(.0008)
    assert normal_damping_correction(.002, .02, .04) == pytest.approx(-.0008)
    assert normal_damping_correction(.03, -.1, .04) == 0.
    assert normal_damping_correction(.011, -.02, .04) == pytest.approx(.0004)
    assert normal_damping_correction(.002, 0., .04) == 0.


def test_damping_cannot_command_more_than_two_mm_or_bypass_parameter_limits():
    assert normal_damping_correction(.002, -10., .1) == .002
    assert normal_damping_correction(.002, 10., .1) == -.002
    for values in [(np.nan,0.,.04),(.002,np.inf,.04),(.002,0.,-.01),(.002,0.,.101)]:
        with pytest.raises(ValueError):normal_damping_correction(*values)
