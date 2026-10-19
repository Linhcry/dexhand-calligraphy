import numpy as np
from fivefinger.trajectory import make_path

def test_cruise_reduces_time_with_same_stroke_count_and_bounded_speed():
    kwargs=dict(character_size=.032,speed=.045,travel_speed=.1,continuous_strokes=True)
    slow=make_path('永',**kwargs)
    fast=make_path('永',stroke_timing='cosine_cruise',**kwargs)
    def boundaries(p):return np.flatnonzero(np.diff(np.r_[False,p.pen_down,False].astype(int)))
    assert len(boundaries(slow))==len(boundaries(fast))
    assert fast.time_s[-1]<slow.time_s[-1]
    velocity=np.linalg.norm(np.diff(fast.xyz,axis=0),axis=1)/.01
    assert velocity.max()<=.10001
    assert velocity[fast.pen_down[1:] & fast.pen_down[:-1]].max()<=.04501
    np.testing.assert_allclose(fast.xyz[boundaries(fast)[::2]],slow.xyz[boundaries(slow)[::2]],atol=.00002)
