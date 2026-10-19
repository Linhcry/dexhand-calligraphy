"""Measured flange pose + calibrated grasp transform; no free-object simulator state."""
import numpy as np


class ToolEstimator:
    def __init__(self,T_flange_tip,maximum_visual_age_s=.15,maximum_disagreement_m=.005):
        T=np.asarray(T_flange_tip,float)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('Invalid calibrated tool transform')
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<.999:raise ValueError('Improper tool rotation')
        self.T=T.copy();self.max_age=maximum_visual_age_s;self.max_error=maximum_disagreement_m

    def estimate(self,T_base_flange,joint_stamp,now,visual_tip=None,visual_stamp=None):
        if not 0<=now-joint_stamp<=.1:raise ValueError('Stale flange feedback')
        flange=np.asarray(T_base_flange,float)
        if flange.shape!=(4,4) or not np.isfinite(flange).all():raise ValueError('Invalid flange pose')
        tip=flange@self.T
        if visual_tip is not None:
            if visual_stamp is None or not 0<=now-visual_stamp<=self.max_age:raise ValueError('Stale visual tip estimate')
            point=np.asarray(visual_tip,float)
            if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid visual tip')
            if np.linalg.norm(point-tip[:3,3])>self.max_error:raise ValueError('Grasp transform invalid: slip or calibration disagreement')
        return dict(T_base_tip=tip,source='encoder_forward_kinematics_and_calibrated_grasp',
                    visually_checked=visual_tip is not None,force_estimate_available=False)
