"""Permit bounded finger recovery only near the last observed healthy pinch.

The caller supplies pen position/axis in the palm frame. These development
thresholds are not a substitute for calibrated pose uncertainty on hardware.
"""
import numpy as np


class RetentionEnvelope:
    def __init__(self):
        self.reference=None;self.maximum_missing_distance_m=0.;self.maximum_missing_axis_deg=0.

    def observe(self,position,axis,healthy_pinch):
        p=np.asarray(position,float);a=np.asarray(axis,float)
        if p.shape!=(3,) or a.shape!=(3,) or not np.isfinite([p,a]).all() or np.linalg.norm(a)<1e-9:
            raise ValueError('Invalid relative pen pose')
        a=a/np.linalg.norm(a)
        if healthy_pinch:
            self.reference=(p.copy(),a.copy());return True
        if self.reference is None:return False
        distance=float(np.linalg.norm(p-self.reference[0]))
        angle=float(np.degrees(np.arccos(np.clip(a@self.reference[1],-1.,1.))))
        self.maximum_missing_distance_m=max(self.maximum_missing_distance_m,distance)
        self.maximum_missing_axis_deg=max(self.maximum_missing_axis_deg,angle)
        return distance<=.01 and angle<=15.
