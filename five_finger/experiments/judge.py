"""Rules-based evidence gates. Unobserved later stages cannot earn points."""
from dataclasses import dataclass
import numpy as np


@dataclass
class HoldGate:
    required_s: float = 3.
    continuous_s: float = 0.
    longest_s: float = 0.
    passed: bool = False

    def update(self, valid, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt required')
        self.continuous_s = self.continuous_s+dt if valid else 0.
        self.longest_s = max(self.longest_s,self.continuous_s)
        self.passed |= self.continuous_s+1e-9 >= self.required_s
        return self.passed


def circle_score(eye_radius, ink_radius, offset, base):
    """Geometric proxy only: no official dragon/font images or judge calibration."""
    R,r,d=map(float,(eye_radius,ink_radius,offset))
    if not np.isfinite([R,r,d,base]).all() or R<=0 or r<0 or d<0 or base<0:
        raise ValueError('invalid circle or score')
    if r<=R/3 or d>=R+r:
        return {'points':0.,'category':'invalid_or_missed','overlap_fraction':0.,'official_score':False}
    if d+r<=R:
        fraction,factor,category=1.,1.5,'perfect'
    else:
        if d+R<=r:area=np.pi*R*R
        else:
            area=r*r*np.arccos(np.clip((d*d+r*r-R*R)/(2*d*r),-1,1))+R*R*np.arccos(np.clip((d*d+R*R-r*r)/(2*d*R),-1,1))
            area-=.5*np.sqrt(max(0.,(-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))
        fraction=float(area/(np.pi*r*r))
        factor,category=(1.2,'accurate') if fraction>=.8 else ((1.,'effective') if fraction>=.2 else (.5,'edge'))
    return {'points':float(base*factor),'category':category,'overlap_fraction':fraction,'official_score':False}


def round_score(attempts,elapsed_s,rules):
    """Score only finalized supplied events by deadline; this is a proxy, not a judge."""
    cfg=rules['preliminary'];points=0.
    for attempt in attempts[:cfg['maximum_dots_per_round']]:
        for name,value in cfg['process_points'].items():
            event=attempt.get(name)
            if event and event['passed'] and event['time_s']<=cfg['round_seconds']:
                points+=value
        dot=attempt.get('dot')
        if dot and dot['time_s']<=cfg['round_seconds'] and dot.get('physical_ink_observed',False):
            points+=circle_score(dot['eye_radius'],dot['ink_radius'],dot['offset'],cfg['base_points'][attempt['paper']])['points']
    points-=sum(a.get('drop_count',0) for a in attempts)*rules['penalties']['brush_drop']
    if elapsed_s>cfg['round_seconds']:points-=rules['penalties']['overtime_points']
    return {'proxy_points':float(min(cfg['per_round_cap'],max(0.,points))), 'official_score':False}
