import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from fivefinger.axial_frame import axial_angles,transported_basis


def test_material_roll_does_not_become_hand_contact_migration_even_with_tilt():
    x=np.array([1.,0,0]);z=np.array([0.,0,1])
    tilt=Rotation.from_rotvec(np.array([.3,-.4,0])).as_matrix()
    roll=Rotation.from_rotvec(z*1.2).as_matrix()
    # The hand-space contact stays on the transported +X side while the
    # round tool rotates about its axis. Its body coordinates must change.
    point_pen=roll.T@np.array([.006,0,.02])
    angles=axial_angles(tilt@roll,point_pen,z,x)
    np.testing.assert_allclose(angles,[0,1.2,-1.2],atol=1e-12)
    u,v,a=transported_basis(z,x,tilt@z)
    np.testing.assert_allclose(np.column_stack((u,v,a)),tilt,atol=1e-12)


def test_contact_migration_is_preserved_and_ambiguous_geometry_is_rejected():
    x=np.array([1.,0,0]);z=np.array([0.,0,1]);angle=.6
    point=np.array([.006*np.cos(angle),.006*np.sin(angle),.02])
    np.testing.assert_allclose(axial_angles(np.eye(3),point,z,x),[angle,0,angle],atol=1e-12)
    with pytest.raises(ValueError):transported_basis(z,x,-z)
    with pytest.raises(ValueError):axial_angles(np.eye(3),np.array([0,0,.02]),z,x)
