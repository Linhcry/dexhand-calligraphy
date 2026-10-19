"""Deterministic offline decision layer. Evidence comes from skills, not elapsed time alone."""
from dataclasses import dataclass
from enum import Enum
import math


class Phase(str,Enum):
    PREPARE='prepare'
    OBSERVE='observe'
    RECOGNIZE='recognize'
    GRASP='grasp'
    HOLD='hold'
    DIP='dip'
    EXECUTE='execute'
    VERIFY='verify'
    WAIT_PAPER='wait_paper'
    STOPPED='stopped'


@dataclass
class Evidence:
    timestamp_s: float
    scene_ready: bool=False
    calibration_verified: bool=False
    tool_calibrated: bool=False
    trajectory_checked: bool=False
    title_consensus: bool=False
    offline_assets_ready: bool=False
    speech_completed: bool=False
    eye_semantics_verified: bool=False
    touching: bool=False
    stable_grip: bool=False
    ink_depth_m: float=0.
    ink_withdrawn: bool=False
    ink_loaded: bool=False
    skill_finished: bool=False
    result_inspected: bool=False
    paper_replaced: bool=False
    paper_shift_m: float=0.
    camera_lost: bool=False
    robot_fault: bool=False
    collision_risk: bool=False
    dropped: bool=False
    force_limit_exceeded: bool=False
    operator_stop: bool=False


class RoundSupervisor:
    def __init__(self,stage='final'):
        if stage not in ('preliminary','final'):raise ValueError(stage)
        self.stage=stage;self.limit=300. if stage=='final' else 180.
        self.phase=Phase.PREPARE;self.started=None;self.last=None;self.reason=None
        self.hold_since=None;self.dip_since=None;self.dipped=False;self.items=0;self.events=[]

    def transition(self,phase,now,reason):
        self.phase=phase;self.reason=reason;self.events.append(dict(t=now,phase=phase.value,reason=reason))

    def start(self,now):
        if not math.isfinite(now):raise ValueError('Invalid clock')
        if self.phase!=Phase.PREPARE or self.started is not None:raise RuntimeError('Round already started; changing paper must not reset time')
        self.started=now;self.last=now;self.transition(Phase.OBSERVE,now,'referee_start')

    def remaining(self,now):return max(0.,self.limit-(now-self.started)) if self.started is not None else self.limit

    def tick(self,now,e):
        if self.phase in (Phase.PREPARE,Phase.STOPPED):return self.phase
        if not all(math.isfinite(v) for v in (now,e.timestamp_s,e.ink_depth_m,e.paper_shift_m)):
            self.transition(Phase.STOPPED,self.last,'nonfinite_observation');return self.phase
        dt=now-self.last;self.last=now
        reason=next((name for condition,name in (
            (dt<0 or dt>.25,'control_clock_gap'),(self.remaining(now)<=0,'round_time_limit'),
            (e.timestamp_s>now+.001 or now-e.timestamp_s>.2,'stale_observation'),
            (e.operator_stop,'operator_stop'),(e.camera_lost,'camera_lost'),(e.robot_fault,'robot_fault'),
            (e.collision_risk,'collision_risk'),(e.force_limit_exceeded,'force_limit'),
            (e.dropped,'brush_dropped')) if condition),None)
        if reason:self.transition(Phase.STOPPED,now,reason);return self.phase
        moving=self.phase in (Phase.GRASP,Phase.HOLD,Phase.DIP,Phase.EXECUTE)
        if moving and (not e.calibration_verified or not e.tool_calibrated or not e.trajectory_checked):
            self.transition(Phase.STOPPED,now,'motion_evidence_invalidated');return self.phase
        if self.phase==Phase.EXECUTE and e.paper_shift_m>.003:
            # Our conservative stop threshold is below the official >10 mm penalty threshold.
            self.transition(Phase.STOPPED,now,'paper_moved_during_contact');return self.phase
        if self.phase==Phase.OBSERVE and e.scene_ready and e.calibration_verified and e.tool_calibrated:
            self.transition(Phase.RECOGNIZE,now,'scene_verified')
        elif self.phase==Phase.RECOGNIZE:
            ready=(e.title_consensus and e.offline_assets_ready and e.speech_completed) if self.stage=='final' else e.eye_semantics_verified
            if ready and e.trajectory_checked:self.transition(Phase.GRASP,now,'task_and_plan_verified')
        elif self.phase==Phase.GRASP and e.touching:self.transition(Phase.HOLD,now,'contact_measured')
        elif self.phase==Phase.HOLD:
            if not e.stable_grip:self.hold_since=None
            elif self.hold_since is None:self.hold_since=now
            elif now-self.hold_since>=3:self.transition(Phase.DIP,now,'continuous_hold_3s')
        elif self.phase==Phase.DIP:
            if e.ink_depth_m>=.003:
                if self.dip_since is None:self.dip_since=now
                self.dipped|=now-self.dip_since>=.5
            else:self.dip_since=None
            if self.dipped and e.ink_withdrawn and e.ink_loaded:self.transition(Phase.EXECUTE,now,'dip_and_withdraw_verified')
        elif self.phase==Phase.EXECUTE and e.skill_finished:self.transition(Phase.VERIFY,now,'motion_finished_not_official_score')
        elif self.phase==Phase.VERIFY and e.result_inspected:
            self.items+=1
            self.transition(Phase.STOPPED if self.stage=='preliminary' and self.items>=3 else Phase.WAIT_PAPER,now,'item_finished')
        elif self.phase==Phase.WAIT_PAPER and e.paper_replaced and e.scene_ready:
            self.hold_since=None;self.dip_since=None;self.dipped=False
            self.transition(Phase.OBSERVE,now,'new_paper_same_round_clock')
        return self.phase


class TitleConsensus:
    """No fuzzy correction to a likely poem; distinct fresh frames must agree."""
    def __init__(self,frames=3,min_confidence=.9):self.frames=frames;self.threshold=min_confidence;self.history=[];self.last=-1
    def push(self,text,confidence,timestamp,bank):
        if timestamp<=self.last:return None
        self.last=timestamp
        if confidence<self.threshold or text not in bank:self.history=[];return None
        self.history.append((text,timestamp));self.history=self.history[-self.frames:]
        return text if len(self.history)==self.frames and len({x[0] for x in self.history})==1 and self.history[-1][1]-self.history[0][1]<=1. else None
