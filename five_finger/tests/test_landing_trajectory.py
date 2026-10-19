import numpy as np
import pytest
from fivefinger.trajectory import make_path


def test_slow_final_descent_preserves_strokes_and_respects_approach_speed():
    kwargs=dict(clearance=.009,paper_z=.0019,travel_speed=.1,continuous_strokes=True,stroke_timing='cosine_cruise')
    baseline=make_path('永',**kwargs);slow=make_path('永',landing_speed=.02,landing_band=.002,**kwargs)
    np.testing.assert_array_equal(slow.xyz[slow.pen_down],baseline.xyz[baseline.pen_down])
    assert slow.time_s[-1]>baseline.time_s[-1]
    delta=np.diff(slow.xyz,axis=0);start=slow.xyz[:-1]
    descending=(delta[:,2]<-1e-10)&(start[:,2]<=.0039+1e-10)&(~slow.pen_down[1:])
    assert descending.any()
    assert np.max(np.linalg.norm(delta[descending],axis=1)/.01)<=.02+1e-9
    np.testing.assert_allclose(delta[descending,:2],0,atol=1e-12)
    with pytest.raises(ValueError):make_path('永',landing_speed=.2,travel_speed=.1)
    with pytest.raises(ValueError):make_path('永',landing_speed=.02,landing_band=.01,clearance=.009)
