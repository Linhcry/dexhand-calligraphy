import numpy as np
from experiments.contact_motion_compatibility import compatibility


def test_full_point_control_allows_rigid_motion_but_three_fixed_contacts_do_not():
    levers=[np.array([.01,0,0]),np.array([0,.01,0]),np.array([0,0,.01])]
    normals=[np.eye(3)[i] for i in range(3)]
    assert compatibility([np.eye(3)]*3,normals,levers,False)['compatible_object_twist_dimension']==6
    assert compatibility([np.zeros((3,3))]*3,normals,levers,False)['compatible_object_twist_dimension']==0
    assert compatibility([np.eye(3)]*3,normals,levers,True)['compatible_object_twist_dimension']<6
