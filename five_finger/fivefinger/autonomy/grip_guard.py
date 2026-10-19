"""Sensor-only motion interlock: restore grip reserve before lifting the pen."""
import numpy as np


class LiftGripGate:
    def __init__(self,thumb_n=1.5,index_n=.6,settle_s=.1,timeout_s=.8):
        values=np.array([thumb_n,index_n,settle_s,timeout_s])
        if not np.isfinite(values).all() or not (values>0).all() or settle_s>=timeout_s:
            raise ValueError('Invalid lift grip gate settings')
        self.minimum=np.array([thumb_n,index_n]);self.settle=settle_s;self.timeout=timeout_s
        self.active=False;self.elapsed=0.;self.stable=0.;self.total_wait=0.;self.activations=0

    def allow(self,request_lift,forces,dt):
        force=np.asarray(forces,float)
        if force.shape!=(5,) or not np.isfinite(force).all() or (force<0).any() or not 0<dt<=.05:
            raise ValueError('Invalid grip observation or time step')
        sufficient=bool((force[:2]>=self.minimum).all())
        if request_lift and not sufficient and not self.active:
            self.active=True;self.elapsed=0.;self.stable=0.;self.activations+=1
        if not self.active:return True
        self.elapsed+=dt;self.total_wait+=dt
        self.stable=self.stable+dt if sufficient else 0.
        if self.stable+1e-9>=self.settle:
            self.active=False
            return True
        if self.elapsed+1e-9>=self.timeout:
            raise RuntimeError('grip_reserve_timeout_before_lift')
        return False
