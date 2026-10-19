import numpy as np
import pytest
from fivefinger.autonomy.command_clock import CommandClock


def test_55_hz_schedule_on_500_hz_physics_grid_and_no_catchup_burst():
    clock=CommandClock(55,[0.,0.]);times=[];old=0
    for now in np.arange(0,1,.002):
        clock.update(now,[now,-now])
        if clock.emissions!=old:times.append(now);old=clock.emissions
    assert len(times)==55
    assert np.max(abs(np.diff(times)-1/55))<=.002
    clock.update(10,[2,3]);assert clock.emissions==56
    clock.update(10.001,[7,8]);assert clock.emissions==56
    np.testing.assert_array_equal(clock.value,[2,3])
    with pytest.raises(ValueError):clock.update(9,[2,3])


def test_held_output_is_not_aliased_and_rejects_invalid_values():
    clock=CommandClock(55,[0.,0.]);command=np.array([1.,2.])
    out=clock.update(0,command);command[:]=9;out[:]=8
    np.testing.assert_array_equal(clock.update(.001,[3,4]),[1,2])
    with pytest.raises(ValueError):clock.update(.002,[float('nan'),0])
    with pytest.raises(ValueError):clock.update(.002,[1])
