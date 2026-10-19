"""Independent calibration anchors and arm mesh trajectory auditing."""
import cv2
import numpy as np


def check_calibration_anchors(cal, known_base_points, observed_pixels, max_error_px=2.):
    """Use fixed workcell features measured during preparation, never newly moved paper.

    Caller supplies feature correspondences verified by a template detector. At least four
    spread points are needed. Table normal alone cannot detect XY or yaw calibration drift.
    """
    points=np.asarray(known_base_points,float);pixels=np.asarray(observed_pixels,float)
    if points.shape!=(len(pixels),3) or pixels.shape[1:]!=(2,) or len(points)<4:raise ValueError('Need four verified anchors')
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():raise ValueError('Invalid anchors')
    if np.linalg.svd(points-points.mean(0),compute_uv=False)[1]<.05:raise ValueError('Anchors insufficiently spread')
    camera=(points-cal.T_base_camera[:3,3])@cal.T_base_camera[:3,:3]
    if (camera[:,2]<=0).any():raise ValueError('Anchor behind camera')
    predicted,_=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),cal.K,np.asarray(cal.distortion,float))
    error=np.linalg.norm(predicted.reshape(-1,2)-pixels,axis=1)
    return dict(verified=bool(error.max()<=max_error_px),maximum_error_px=float(error.max()),anchor_count=len(points))


def audit_reference_joint_path(model,data,waypoints,max_joint_step_rad=.01):
    """Interpolate EVERY joint segment, check table meshes and robot/environment contacts.

    This is sampled kinematic verification, not continuous collision detection or dynamics.
    Robot self-collision is not represented by the current reference collision masks.
    """
    import mujoco
    from experiments.competition_scene import TableClearance
    path=np.asarray(waypoints,float)
    if path.ndim!=2 or path.shape[1]!=6 or not np.isfinite(path).all():raise ValueError('Need finite six-joint path')
    scratch=mujoco.MjData(model);scratch.qpos[:]=data.qpos
    guard=TableClearance(model);count=0
    for a,b in zip(path[:-1],path[1:]):
        n=max(2,int(np.ceil(np.max(np.abs(b-a))/max_joint_step_rad))+1)
        for u in np.linspace(0,1,n):
            scratch.qpos[:6]=a+(b-a)*u;mujoco.mj_forward(model,scratch);count+=1
            try:guard.require_clear(scratch)
            except RuntimeError as e:return dict(valid=False,reason=str(e),samples=count)
            for c in scratch.contact:
                names=[model.geom(int(g)).name for g in (c.geom1,c.geom2)]
                if c.dist<-.0001 and any(name.startswith('collision_') for name in names):
                    return dict(valid=False,reason='robot_environment_intersection',geoms=names,samples=count)
    return dict(valid=True,samples=count,minimum_moving_arm_table_m=guard.minimum_arm,
                self_collision_verified=False,dynamic_execution_verified=False)
