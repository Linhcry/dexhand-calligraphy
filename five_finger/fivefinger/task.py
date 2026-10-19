"""Observation-gated task skeleton; reaching a time alone never implies task success."""
from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    ACQUIRE = "acquire_target"
    APPROACH = "approach_brush"
    GRIP = "close_fingers"
    HOLD = "hold_brush"
    DIP = "dip_ink"
    TRANSFER = "move_above_paper"
    EXECUTE = "dot_or_write"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Observation:
    target_valid: bool = False
    approach_reached: bool = False
    brush_contact: bool = False
    brush_held: bool = False
    ink_depth_m: float = 0.
    ink_visible: bool = False
    paper_reached: bool = False
    action_finished: bool = False
    result_verified: bool = False
    unsafe: bool = False


class TaskMachine:
    def __init__(self, limit_s=180.):
        self.stage = Stage.ACQUIRE
        self.limit_s = limit_s
        self.elapsed = self.hold_s = self.dip_s = 0.

    def update(self, o: Observation, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.stage in (Stage.DONE, Stage.FAILED):
            return self.stage
        self.elapsed += dt
        if o.unsafe or self.elapsed >= self.limit_s:
            self.stage = Stage.FAILED
            return self.stage
        if self.stage == Stage.ACQUIRE and o.target_valid:
            self.stage = Stage.APPROACH
        elif self.stage == Stage.APPROACH and o.approach_reached:
            self.stage = Stage.GRIP
        elif self.stage == Stage.GRIP and o.brush_contact and o.brush_held:
            self.stage = Stage.HOLD
        elif self.stage == Stage.HOLD:
            self.hold_s = self.hold_s + dt if o.brush_held else 0.
            if not o.brush_held:
                self.stage = Stage.GRIP
            elif self.hold_s >= 3.:
                self.stage = Stage.DIP
        elif self.stage == Stage.DIP:
            if not o.brush_held:
                self.stage = Stage.FAILED
            else:
                self.dip_s = self.dip_s + dt if o.ink_depth_m >= 0.003 else 0.
                if self.dip_s >= .5 and o.ink_visible:
                    self.stage = Stage.TRANSFER
        elif self.stage == Stage.TRANSFER:
            if not o.brush_held:
                self.stage = Stage.FAILED
            elif o.paper_reached:
                self.stage = Stage.EXECUTE
        elif self.stage == Stage.EXECUTE:
            if not o.brush_held:
                self.stage = Stage.FAILED
            elif o.action_finished and o.result_verified:
                self.stage = Stage.DONE
        return self.stage
