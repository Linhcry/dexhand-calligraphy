import numpy as np
from fivefinger.retention_envelope import RetentionEnvelope


def test_recovery_needs_a_healthy_reference_and_rejects_fallen_or_rotated_pen():
    guard=RetentionEnvelope();p=np.array([.07,.02,.20]);axis=np.array([0.,0.,1.])
    assert not guard.observe(p,axis,False)
    assert guard.observe(p,axis,True)
    assert guard.observe(p+[.003,0,0],axis,False)
    assert not guard.observe(p+[.011,0,0],axis,False)
    assert not guard.observe(p,[np.sin(.3),0,np.cos(.3)],False)
    assert not guard.observe(p,-axis,False)
    assert guard.observe(p+[.02,0,0],axis,True)
    assert guard.observe(p+[.025,0,0],axis,False)
