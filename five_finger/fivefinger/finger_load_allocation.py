"""Experimental normal-target allocation from local measured wrench ratios.

Scaling a measured contact wrench is a local approximation, not a friction or
dynamic contact model. This only requests bounded normal-force targets; it does
not apply an object wrench. Hardware use requires calibrated contact feedback.
"""
import numpy as np
from scipy.optimize import lsq_linear


class FingerLoadAllocator:
    def __init__(self, reference, acquisition_caps):
        self.reference=np.asarray(reference,float).copy()
        self.upper=np.asarray(acquisition_caps,float).copy()
        if self.reference.shape!=(3,) or self.upper.shape!=(3,) or not np.isfinite(np.r_[self.reference,self.upper]).all():
            raise ValueError('Expected three finite normal targets')
        if (self.reference<.5).any() or (self.reference>self.upper).any() or (self.upper>3.).any():
            raise ValueError('Invalid normal-force allocation bounds')
        self.lower=np.maximum(.5,self.reference-.3)
        self.upper=np.minimum(self.upper,self.reference+.3)
        if (self.upper<=self.lower).any():raise ValueError('Allocation requires nonzero force adjustment intervals')
        self.target=self.reference.copy();self.desired=self.reference.copy();self.elapsed=0.;self.unit_wrench=None
        self.updates=0;self.unobserved_updates=0;self.last_residual=None
        self.minimum_target=self.target.copy();self.maximum_target=self.target.copy()

    def update(self, normal_N, wrench_world, external_wrench, dt):
        normal=np.asarray(normal_N,float);wrench=np.asarray(wrench_world,float)
        external=np.asarray(external_wrench,float)
        if normal.shape!=(3,) or wrench.shape!=(3,6) or external.shape!=(6,) or not np.isfinite(np.r_[normal,wrench.ravel(),external]).all() or (normal<0).any() or not 0<dt<=.05:
            raise ValueError('Invalid allocator observation')
        # Do not allocate a contact direction that is no longer observed.
        if np.any(normal<.1):
            self.unobserved_updates+=1;self.elapsed=0.;self.unit_wrench=None;self.desired=self.target.copy()
            return self.target.copy()
        estimate=(wrench/normal[:,None]).T
        self.unit_wrench=estimate.copy() if self.unit_wrench is None else self.unit_wrench+dt/(.1+dt)*(estimate-self.unit_wrench)
        self.elapsed+=dt
        # 5 cm characteristic length balances force and moment residual units.
        scale=np.array([1.,1.,1.,20.,20.,20.])
        matrix=self.unit_wrench*scale[:,None]
        if self.elapsed>=.1-1e-10:
            self.elapsed=0.
            reg=np.sqrt(.1)
            problem=np.vstack([matrix,reg*np.eye(3)])
            rhs=np.r_[-external*scale,reg*self.reference]
            result=lsq_linear(problem,rhs,bounds=(self.lower,self.upper),method='bvls',tol=1e-9,max_iter=50)
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError('Normal target allocation failed')
            self.desired=result.x.copy();self.updates+=1
        # Apply slew at every control tick, not as a 10 Hz staircase.
        self.target+=np.clip(self.desired-self.target,-.1*dt,.1*dt)
        self.target=np.clip(self.target,self.lower,self.upper)
        self.last_residual=float(np.linalg.norm(matrix@self.target+external*scale))
        self.minimum_target=np.minimum(self.minimum_target,self.target)
        self.maximum_target=np.maximum(self.maximum_target,self.target)
        return self.target.copy()

    def report(self):
        return {'updates':self.updates,'unobserved_contact_ticks':self.unobserved_updates,
            'minimum_target_N':self.minimum_target.tolist(),'maximum_target_N':self.maximum_target.tolist(),
            'normal_target_slew_N_s':.1,'normal_target_minimum_N':self.lower.tolist(),
            'target_slew_at_every_control_tick':True,
            'normal_target_maximum_N':self.upper.tolist(),
            'last_scaled_wrench_residual':self.last_residual,
            'measured_wrench_scaling_is_approximate':True,'hardware_ready':False}
