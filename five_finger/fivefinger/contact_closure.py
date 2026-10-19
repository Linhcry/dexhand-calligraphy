"""Stop the nominal closing trajectory on first contact; force control continues.

This is an approach-trajectory limiter, not a certified force limit.
"""
import numpy as np


class ContactAwareClosure:
    def __init__(self):
        self.progress=np.zeros(5)
        self.latched=np.zeros(3,dtype=bool)
        self.first_contact_s=np.full(3,np.nan)

    def update(self,planned_progress,forces,now):
        forces=np.asarray(forces,float)
        if forces.shape!=(5,) or not np.isfinite(forces).all() or np.any(forces<0):
            raise ValueError('Expected five nonnegative contact forces')
        if not np.isfinite([planned_progress,now]).all() or not 0<=planned_progress<=1 or now<0:
            raise ValueError('Invalid closure trajectory input')
        for i in range(3):
            if not self.latched[i] and forces[i]>.05:
                self.latched[i]=True;self.first_contact_s[i]=now
            if not self.latched[i]:self.progress[i]=planned_progress
        self.progress[3:]=planned_progress
        return self.progress.copy()
