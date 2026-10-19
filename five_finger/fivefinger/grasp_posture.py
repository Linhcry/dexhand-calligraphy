"""Bounded development changes to acquired and unused finger target postures."""
import math


def apply_grasp_offsets(targets,offsets):
    result=dict(targets)
    for name,value in offsets.items():
        if name not in result or not name.startswith(('r_f_joint1_','r_f_joint2_','r_f_joint3_')):
            raise ValueError('Unknown grasp joint offset')
        if not math.isfinite(value) or abs(value)>.1:
            raise ValueError('Grasp offset must be within 0.1 rad')
        result[name]+=float(value)
    return result


def unused_finger_posture(targets, flexion_rad):
    result=dict(targets)
    if flexion_rad is None:return result
    if not math.isfinite(flexion_rad) or not 0<=flexion_rad<=.3:
        raise ValueError('Unused-finger flexion must be 0..0.3 rad in development')
    for finger in (4,5):
        for joint in (2,3,4):
            name=f'r_f_joint{finger}_{joint}'
            if name not in result:raise ValueError('Incomplete hand posture mapping')
            result[name]=float(flexion_rad)
    return result
