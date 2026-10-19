import numpy as np
from fivefinger.contact_closure import ContactAwareClosure


def test_only_contacted_finger_stops_nominal_approach_and_does_not_jump_after_loss():
    gate=ContactAwareClosure()
    gate.update(.4,np.zeros(5),1.)
    np.testing.assert_allclose(gate.update(.5,[.1,0,0,0,0],1.01),[.4,.5,.5,.5,.5])
    np.testing.assert_allclose(gate.update(.8,np.zeros(5),1.02),[.4,.8,.8,.8,.8])
    np.testing.assert_allclose(gate.update(1.,[0,.1,.2,0,0],1.03),[.4,.8,.8,1,1])
    assert gate.latched.all()
    np.testing.assert_allclose(gate.first_contact_s,[1.01,1.03,1.03])
