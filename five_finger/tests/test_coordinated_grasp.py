import numpy as np
from scipy.linalg import block_diag
from fivefinger.coordinated_grasp import solve_contact_velocity


def test_shared_translation_and_normal_compression_can_coexist():
    jac=block_diag(*[.02*np.eye(3)]*3)
    obj=np.vstack([np.column_stack([np.eye(3),np.zeros((3,3))])]*3)
    compression=np.array([.0001,0,0,-.0001,0,0,0,.0001,0])
    desired=np.array([0,0,.0005,0,0,0])
    velocity,twist,residual=solve_contact_velocity(jac,obj,compression,desired,np.zeros(9),np.full(9,-.12),np.full(9,.12))
    assert residual<1e-6
    assert twist[2]>.0004
    assert max(abs(velocity))<=.12
    np.testing.assert_allclose(jac@velocity-obj@twist,compression,atol=1e-6)


def test_fixed_joint_limits_are_respected_and_incompatible_motion_has_residual():
    jac=np.eye(3);obj=np.zeros((3,6))
    velocity,twist,residual=solve_contact_velocity(jac,obj,[.001,0,0],np.zeros(6),np.zeros(3),np.zeros(3),np.zeros(3))
    np.testing.assert_array_equal(velocity,np.zeros(3))
    assert residual>=.001-1e-12


def test_posture_weight_reduces_joint_motion_while_preserving_contact_consistency():
    jac=block_diag(*[.02*np.eye(3)]*3)
    obj=np.vstack([np.column_stack([np.eye(3),np.zeros((3,3))])]*3)
    args=(jac,obj,np.zeros(9),[0,0,.0005,0,0,0],np.zeros(9),np.full(9,-.12),np.full(9,.12))
    weak,_,_=solve_contact_velocity(*args,posture_weight=.003)
    strong,_,residual=solve_contact_velocity(*args,posture_weight=.02)
    assert np.linalg.norm(strong)<np.linalg.norm(weak)
    assert residual<1e-6
