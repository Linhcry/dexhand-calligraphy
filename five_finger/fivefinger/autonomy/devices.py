"""Vendor-independent contracts. No guessed SDK calls, CAN IDs or motor mappings."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


class ArmDriver(Protocol):
    def read_joints(self) -> tuple[float,dict[str,float]]: ...
    def servo_joints(self,positions_rad:dict[str,float],duration_s:float) -> None: ...
    def controlled_stop(self,reason:str) -> None: ...

class HandDriver(Protocol):
    def read_feedback(self) -> dict: ...
    def set_motor_targets(self,targets:dict[str,float]) -> None: ...
    def hold(self) -> None: ...

class RGBDCamera(Protocol):
    def capture(self) -> tuple[float,np.ndarray,np.ndarray]: ...


@dataclass
class Commissioning:
    official_model_confirmed:bool=False
    joint_mapping_verified:bool=False
    calibration_verified:bool=False
    tool_transform_verified:bool=False
    force_units_verified:bool=False
    watchdog_verified:bool=False
    emergency_stop_verified:bool=False
    collision_model_verified:bool=False
    low_speed_hardware_trial_passed:bool=False

    def require(self):
        missing=[name for name,value in vars(self).items() if value is not True]
        if missing:raise RuntimeError('Hardware not commissioned: '+', '.join(missing))


class GuardedArm:
    """Validate by joint NAME, freshness, position/velocity and full-robot collision callback.

    The callback must check the entire interpolated move and current world model. This wrapper
    is not a safety-rated controller; vendor-side timeout/stop remains mandatory.
    """
    def __init__(self,driver,limits,collision_checker,commissioning):
        commissioning.require()
        if not limits or collision_checker is None:raise ValueError('Limits and collision checker are required')
        for low,high,velocity in limits.values():
            if not np.isfinite([low,high,velocity]).all() or low>=high or velocity<=0:raise ValueError('Invalid joint limits')
        self.driver=driver;self.limits=limits;self.check=collision_checker

    def send(self,target,now,dt):
        try:
            stamp,current=self.driver.read_joints()
            if now-stamp>.10 or stamp>now+.001 or not 0<dt<=.10:raise ValueError('Feedback or control period invalid')
            if set(target)!=set(self.limits) or set(current)!=set(self.limits):raise ValueError('Joint names mismatch')
            for name,(low,high,velocity) in self.limits.items():
                value=target[name]
                if not np.isfinite([value,current[name]]).all() or not low<=value<=high:raise ValueError('Joint range violation')
                if abs(value-current[name])/dt>velocity:raise ValueError('Joint speed violation')
            if not self.check(current,target):raise ValueError('Swept collision check rejected command')
            self.driver.servo_joints(target,dt)
        except Exception as exc:
            self.driver.controlled_stop(str(exc))
            raise


class UnconfiguredOfficialArm:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Official arm SDK/model not supplied. Implement ArmDriver and commission it before hardware use.')
