"""Observe normal-force maxima after EACH physics step, without applying forces."""
import mujoco
import numpy as np
from fivefinger.contact_observation import GROUPS


class ForcePeakMonitor:
    def __init__(self,model):
        self.model=model;self.brush=model.body('brush').id
        self.groups={}
        for gid in range(model.ngeom):
            name=model.geom(gid).name or ''
            group=5 if name=='paper' else 6
            for finger in range(1,6):
                if name.startswith(f'col_r_f_link{finger}_'):group=finger-1;break
            self.groups[gid]=group
        self.samples=0;self.last_time=None;self.maximum_interval_s=0.
        self.maxima=np.zeros(7);self.peak_times=np.zeros(7);self.peak_stages=['none']*7
        self.by_stage={};self.scratch=np.zeros(6)

    def update(self,data,stage):
        now=float(data.time)
        if self.last_time is not None:
            dt=now-self.last_time
            if dt<=0:raise ValueError('Non-increasing force audit time')
            self.maximum_interval_s=max(self.maximum_interval_s,dt)
            if dt>self.model.opt.timestep*1.001:raise ValueError('Force audit skipped physics steps')
        self.last_time=now;self.samples+=1;forces=np.zeros(7)
        for index,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            pa=int(self.model.geom_bodyid[a])==self.brush;pb=int(self.model.geom_bodyid[b])==self.brush
            if pa==pb:continue
            mujoco.mj_contactForce(self.model,data,index,self.scratch)
            forces[self.groups[b if pa else a]]+=max(0.,float(self.scratch[0]))
        if not np.isfinite(forces).all():raise RuntimeError('Nonfinite contact force')
        peak=forces>self.maxima
        self.maxima[peak]=forces[peak];self.peak_times[peak]=now
        for i in np.flatnonzero(peak):self.peak_stages[i]=stage
        if stage not in self.by_stage:self.by_stage[stage]=np.zeros(7)
        self.by_stage[stage]=np.maximum(self.by_stage[stage],forces)

    def report(self):
        return dict(source='simulation normal forces after every mj_step and mj_forward',
            requested_rate_hz=1./self.model.opt.timestep,samples=self.samples,
            maximum_observed_interval_s=self.maximum_interval_s,
            forces_are_simulation_truth=True,hardware_ready=False,
            note='Physics-step maxima, not continuous-time force bounds or hardware measurements.',
            groups={name:dict(maximum_recorded_N=float(self.maxima[i]),sim_s=float(self.peak_times[i]),stage=self.peak_stages[i]) for i,name in enumerate(GROUPS)},
            stage_maxima_N={stage:dict(zip(GROUPS,values.tolist())) for stage,values in self.by_stage.items()})
