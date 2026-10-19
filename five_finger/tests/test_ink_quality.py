import numpy as np
from fivefinger.ink_quality import measure_ink_precision


def test_unwanted_bridge_is_detected_even_when_its_endpoints_are_valid():
    planned=np.c_[np.r_[np.linspace(0,.01,30),np.linspace(.03,.04,30)],np.zeros(60)]
    ink=np.array([[1,0,0],[1,.01,0],[1,.03,0],[1,.04,0]])
    r=measure_ink_precision(ink,planned)
    assert r['ink_sample_precision_within_2_5mm']==1.
    assert .35<1-r['ink_length_precision_within_2_5mm']<.4


def test_pen_up_break_does_not_create_a_false_bridge():
    planned=np.c_[np.r_[np.linspace(0,.01,30),np.linspace(.03,.04,30)],np.zeros(60)]
    ink=np.array([[1,0,0],[1,.01,0],[0,.02,0],[1,.03,0],[1,.04,0]])
    assert measure_ink_precision(ink,planned)['ink_length_precision_within_2_5mm']==1.
