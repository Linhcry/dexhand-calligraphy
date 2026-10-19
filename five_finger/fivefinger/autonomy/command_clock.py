"""Sample-and-hold target cadence for simulation bandwidth experiments.

No SDK transport or safety certification. The actual device must implement its
own watchdog. This clock never emits catch-up bursts after a missed deadline.
"""
import numpy as np


class CommandClock:
    def __init__(self,hz,initial):
        if not np.isfinite(hz) or not 1<=hz<=500:
            raise ValueError('Command frequency must be 1..500 Hz')
        self.period=1./hz
        self.value=np.asarray(initial,float).copy()
        if self.value.ndim!=1 or not np.isfinite(self.value).all():
            raise ValueError('Invalid initial command')
        self.next_time=0.;self.last_time=-1.;self.emissions=0

    def update(self,now,command):
        command=np.asarray(command,float)
        if not np.isfinite(now) or now<0 or now<self.last_time or command.shape!=self.value.shape or not np.isfinite(command).all():
            raise ValueError('Invalid command or non-monotonic clock')
        self.last_time=now
        if now+1e-9>=self.next_time:
            self.value=command.copy();self.emissions+=1
            self.next_time=(np.floor((now+1e-9)/self.period)+1)*self.period
        return self.value.copy()
