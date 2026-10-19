import numpy as np
from fivefinger.wrench_capacity import load_capacity


def contacts():
    # Opposing fingers at x=+-0.01 m, 1 N normal each, friction 0.5.
    return [dict(group=name,normal_N=1.,position_world_m=[x,0,0],
        frame_on_brush_world=np.diag([sign,1,1]).tolist(),friction=[.5,.5,0,0,0],contact_dim=3)
        for name,x,sign in [('thumb',-.01,1),('index',.01,-1)]]


def test_point_pinch_resists_bending_but_not_moment_along_contact_line():
    budget={'thumb':1.,'index':1.}
    result=load_capacity(contacts(),[0,0,0],[0]*6,[0,0,0,0,1,0],budget)
    np.testing.assert_allclose(result['capacity'],.01,atol=1e-8)
    axis=load_capacity(contacts(),[0,0,0],[0]*6,[0,0,0,1,0,0],budget)
    np.testing.assert_allclose(axis['capacity'],0.,atol=1e-8)
    loaded=load_capacity(contacts(),[0,0,0],[0,0,-.5,0,0,0],[0,0,0,0,1,0],budget)
    np.testing.assert_allclose(loaded['capacity'],.005,atol=1e-8)


def test_overweight_and_missing_contacts_are_infeasible():
    budget={'thumb':1.,'index':1.}
    assert not load_capacity(contacts(),[0,0,0],[0,0,-2,0,0,0],[0,0,0,1,0,0],budget)['feasible']
    assert not load_capacity([], [0,0,0],[0]*6,[0,0,0,1,0,0],budget)['feasible']


def test_patch_torsion_adds_capacity_missing_from_two_point_contacts():
    patches=contacts()
    for c in patches:c.update(contact_dim=4,friction=[.5,.5,.005,0,0])
    result=load_capacity(patches,[0,0,0],[0]*6,[0,0,0,1,0,0],{'thumb':1.,'index':1.},include_contact_moments=True)
    np.testing.assert_allclose(result['capacity'],.01,atol=1e-8)
