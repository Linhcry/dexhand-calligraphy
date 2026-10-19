import numpy as np
from fivefinger.grasp_stabilization import tangential_joint_velocity,desired_contact_velocity,axis_alignment_error,bound_combined_delta
from scipy.spatial.transform import Rotation


def test_tangent_correction_preserves_normal_with_coupled_joints():
    jac=np.array([[.02,.01,0,.01],[.01,0,.02,-.01],[0,.02,.01,.02]])
    normal=np.array([1.,2.,3.]);normal/=np.linalg.norm(normal)
    desired=np.array([.0002,-.0003,.0001]);desired-=normal*(normal@desired)
    for free in (None,[0,1,2]):
        velocity=tangential_joint_velocity(jac,normal,desired,free)
        actual=jac@velocity
        assert abs(normal@actual)<1e-12
        assert np.linalg.norm(actual-desired)<np.linalg.norm(desired)
        assert np.max(abs(velocity))<=.04
    assert velocity[3]==0


def test_contact_lever_rotation_and_bounded_velocity():
    # Positive Z rotation moves a contact on +X toward +Y.
    v=desired_contact_velocity(np.zeros(3),[0,0,.1],[.01,0,0],.25)
    np.testing.assert_allclose(v,[0,.00025,0])
    v=desired_contact_velocity([1,2,3],[1,2,3],[.1,.2,.4],1.)
    np.testing.assert_allclose(np.linalg.norm(v),.001)


def test_axis_error_ignores_barrel_spin_and_corrects_tilt_including_opposite_axis():
    target=np.array([0.,0.,1.])
    spin=Rotation.from_euler('z',120,degrees=True).as_matrix()
    np.testing.assert_allclose(axis_alignment_error(spin[:,2],target),0.)
    for current in (Rotation.from_euler('x',20,degrees=True).apply(target),-target):
        error=axis_alignment_error(current,target)
        np.testing.assert_allclose(Rotation.from_rotvec(error).apply(current),target,atol=1e-12)


def test_combined_normal_and_pose_terms_keep_shared_rate_budget():
    before={'a':.1,'b':-.1};candidate={'a':.1016,'b':-.1008}
    actual,limited=bound_combined_delta(before,candidate,.01)
    assert limited
    np.testing.assert_allclose([actual['a']-.1,actual['b']+.1],[.0012,-.0006],atol=1e-12)
    assert max(abs(actual[n]-before[n]) for n in before)<=.0012+1e-12


def test_redundant_posture_motion_keeps_contact_point_and_respects_blocked_joints():
    jac=np.array([[.02,.01,0,.01],[.01,0,.02,-.01],[0,.02,.01,.02]])
    normal=np.array([1.,0.,0.]);posture=np.array([.002,-.003,.001,.004])
    v=tangential_joint_velocity(jac,normal,np.zeros(3),posture_velocity=posture)
    assert np.linalg.norm(v)>1e-5
    np.testing.assert_allclose(jac@v,0,atol=1e-12)
    assert v@posture>0  # moves toward the acquired posture
    constrained=tangential_joint_velocity(jac,normal,np.zeros(3),[0,1,2],posture)
    np.testing.assert_allclose(constrained,0,atol=1e-12)
    fast=tangential_joint_velocity(jac,normal,np.zeros(3),posture_velocity=posture*1000)
    assert max(abs(fast))<=.04+1e-12
    np.testing.assert_allclose(jac@fast,0,atol=1e-12)


def test_common_pose_reserve_slows_before_exhausting_offset_budget():
    from fivefinger.grasp_stabilization import coordinated_pose_scale
    assert coordinated_pose_scale([.1,-.1],.05)==1.
    assert 0<coordinated_pose_scale([.21,-.23],.05)<1.
    assert coordinated_pose_scale([.1,-.25],.05)==0.
    assert coordinated_pose_scale([.25],0.)==1.
