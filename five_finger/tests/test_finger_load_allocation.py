import numpy as np
import pytest
from fivefinger.finger_load_allocation import FingerLoadAllocator


def test_allocation_reduces_load_error_with_force_bounds_and_slew():
    allocator=FingerLoadAllocator([2.7,1.35,1.35],[3.,1.5,1.5])
    normal=np.array([2.7,1.35,1.35])
    directions=np.array([[1.,0,0,0,0,0],[-1.,0,0,0,0,.01],[-1.,0,0,0,0,-.01]])
    external=np.array([0.,0,0,0,0,.02])
    original=np.linalg.norm((directions.T@normal+external)*[1,1,1,20,20,20])
    old=allocator.target.copy()
    for i in range(1000):
        value=allocator.update(normal,directions*normal[:,None],external,.01)
        assert np.max(abs(value-old))<=.001+1e-12
        assert np.all(value>=allocator.lower) and np.all(value<=allocator.upper)
        old=value.copy()
    final=np.linalg.norm((directions.T@value+external)*[1,1,1,20,20,20])
    assert final<original
    assert value[2]>value[1]  # redistribute toward the contact opposing +Z torque
    assert allocator.updates==100


def test_allocator_freezes_on_lost_contact_and_rejects_nonfinite_input():
    allocator=FingerLoadAllocator([2.7,1.35,1.35],[3.,1.5,1.5])
    np.testing.assert_array_equal(allocator.update([2.,0.,1.],np.zeros((3,6)),np.zeros(6),.01),allocator.reference)
    assert allocator.updates==0 and allocator.unobserved_updates==1
    with pytest.raises(ValueError):allocator.update([2.,1.,1.],np.full((3,6),np.nan),np.zeros(6),.01)
