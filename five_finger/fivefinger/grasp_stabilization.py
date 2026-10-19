"""Bounded in-hand pose maintenance using simulated contact observations.

Only position-actuator targets are changed. No attachment or pen wrench is used.
Actual deployment needs calibrated object pose and finger contact observations.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def bound_combined_delta(previous,candidate,dt):
    """Preserve the 0.12 rad/s maintenance-offset budget after adding terms."""
    if previous.keys()!=candidate.keys() or not 0<dt<=.05:
        raise ValueError('Invalid combined grip update')
    names=list(previous)
    before=np.array([previous[n] for n in names]);after=np.array([candidate[n] for n in names])
    if not np.isfinite([before,after]).all():raise ValueError('Nonfinite grip update')
    step=after-before;scale=min(1.,.12*dt/max(float(np.max(abs(step))),1e-12))
    return dict(zip(names,(before+scale*step).tolist())),scale<1.


def axis_alignment_error(current,target):
    a=np.asarray(current,float);b=np.asarray(target,float)
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all() or min(np.linalg.norm(a),np.linalg.norm(b))<1e-9:
        raise ValueError('Invalid pen axes')
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    cross=np.cross(a,b);sine=np.linalg.norm(cross);cosine=float(np.clip(a@b,-1.,1.))
    if sine>1e-9:return cross/sine*np.arctan2(sine,cosine)
    if cosine>=0:return np.zeros(3)
    basis=np.eye(3)[int(np.argmin(abs(a)))];cross=np.cross(a,basis)
    return cross/np.linalg.norm(cross)*np.pi


def desired_contact_velocity(position_error, rotation_error, lever, gain):
    value=gain*(np.asarray(position_error)+np.cross(rotation_error,lever))
    return value*min(1.,.001/max(float(np.linalg.norm(value)),1e-12))


def coordinated_pose_scale(offsets, reserve_rad):
    """Slow all fingers together before any software offset budget is spent.

This protects correction authority, not a contact-force or stability guarantee.
Normal force control remains separate; the 0.25 rad offset cap is unchanged.
"""
    values=np.asarray(offsets,float)
    if not np.isfinite(values).all() or not np.isfinite(reserve_rad) or not 0<=reserve_rad<=.1:
        raise ValueError('Invalid pose reserve')
    if not reserve_rad:return 1.
    margin=.25-float(np.max(abs(values)))
    return float(np.clip((margin-.005)/reserve_rad,0.,1.))


def tangential_joint_velocity(jacobian, normal, desired, free=None, posture_velocity=None):
    """Solve in the nullspace of first-order motion into the contact normal."""
    jac=np.asarray(jacobian,float);n=np.asarray(normal,float)
    n=n/max(np.linalg.norm(n),1e-12)
    selected=np.arange(jac.shape[1]) if free is None else np.asarray(free,int)
    result=np.zeros(jac.shape[1])
    if not len(selected):return result
    j=jac[:,selected];jn=n@j;norm=float(jn@jn)
    null=np.eye(len(selected))-np.outer(jn,jn)/norm if norm>1e-14 else np.eye(len(selected))
    tangent=np.eye(3)-np.outer(n,n)
    effective=tangent@j@null
    velocity=null@effective.T@np.linalg.solve(effective@effective.T+1e-6*np.eye(3),tangent@desired)
    if posture_velocity is not None:
        # Secondary posture motion must leave the complete contact point
        # unchanged to first order, including its normal component.
        posture=np.asarray(posture_velocity,float)
        if posture.shape!=(jac.shape[1],) or not np.isfinite(posture).all():
            raise ValueError('Invalid posture velocity')
        kernel=np.eye(len(selected))-np.linalg.pinv(j,rcond=1e-10)@j
        velocity+=kernel@posture[selected]
    velocity*=min(1.,.04/max(float(np.max(abs(velocity))),1e-12))
    result[selected]=velocity
    return result


class InHandPoseServo:
    def __init__(self,model,gain_s,orientation_mode='full',posture_gain_s=0.,reserve_rad=0.,lock_spread=False):
        if not np.isfinite(gain_s) or not 0<gain_s<=1.:
            raise ValueError('In-hand pose gain must be in (0, 1] per second')
        if orientation_mode not in ('full','axis'):raise ValueError('Invalid pose orientation mode')
        if not np.isfinite(posture_gain_s) or not 0<=posture_gain_s<=.1:
            raise ValueError('Posture gain must be 0..0.1 per second')
        self.model=model;self.gain=gain_s;self.anchor=None;self.orientation_mode=orientation_mode
        self.posture_gain=posture_gain_s;self.posture_anchor=None
        coordinated_pose_scale([0.],reserve_rad)
        self.reserve=reserve_rad;self.reserve_limited_updates=0;self.minimum_reserve_scale=1.
        self.lock_spread=bool(lock_spread)
        self.maximum_position_error_m=0.;self.maximum_rotation_error_rad=0.
        self.active_updates=0

    def update(self,data,grip_servo,base_targets,dt):
        if not 0<dt<=.05:raise ValueError('Invalid in-hand control interval')
        model=self.model;hand=data.body('right_hand_base');pen=data.body('brush')
        rh=hand.xmat.reshape(3,3);rp=pen.xmat.reshape(3,3)
        relative=rh.T@(pen.xpos-hand.xpos)
        if self.anchor is None:
            self.anchor=(relative.copy(),rh.T@rp)
            self.posture_anchor={n:float(base_targets[n]+v) for n,v in grip_servo.delta.items()}
            return
        position_error=rh@(self.anchor[0]-relative)
        rotation_error=(axis_alignment_error(rp[:,2],(rh@self.anchor[1])[:,2]) if self.orientation_mode=='axis'
                        else Rotation.from_matrix(rh@self.anchor[1]@rp.T).as_rotvec())
        self.maximum_position_error_m=max(self.maximum_position_error_m,float(np.linalg.norm(position_error)))
        self.maximum_rotation_error_rad=max(self.maximum_rotation_error_rad,float(np.linalg.norm(rotation_error)))
        position_error*=min(1.,.01/max(np.linalg.norm(position_error),1e-12))
        rotation_error*=min(1.,np.radians(10)/max(np.linalg.norm(rotation_error),1e-12))
        common_scale=coordinated_pose_scale(list(grip_servo.delta.values()),self.reserve)
        self.reserve_limited_updates+=int(common_scale<1.)
        self.minimum_reserve_scale=min(self.minimum_reserve_scale,common_scale)
        scratch=np.zeros(6);jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        groups={i:[] for i in (1,2,3)}
        for index,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            if int(model.geom_bodyid[a])==pen.id:other=b;normal=-np.asarray(c.frame[:3])
            elif int(model.geom_bodyid[b])==pen.id:other=a;normal=np.asarray(c.frame[:3])
            else:continue
            name=model.geom(other).name or ''
            finger=next((i for i in groups if name.startswith(f'col_r_f_link{i}_')),None)
            if finger is None:continue
            mujoco.mj_contactForce(model,data,index,scratch)
            if scratch[0]<=.05:continue
            mujoco.mj_jac(model,data,jp,jr,c.pos,int(model.geom_bodyid[other]))
            groups[finger].append((float(scratch[0]),c.pos.copy(),normal.copy(),jp.copy()))
        # Do not chase a pen after either side of the pinch has lost contact.
        if not groups[1] or not groups[2]:return
        for finger,contacts in groups.items():
            if not contacts:continue
            total=sum(c[0] for c in contacts)
            point=sum(c[0]*c[1] for c in contacts)/total
            normal=sum(c[0]*c[2] for c in contacts)/total
            if np.linalg.norm(normal)<1e-6:continue
            names=[n for n in grip_servo.delta if n.startswith(f'r_f_joint{finger}_') and not (self.lock_spread and n.endswith('_1'))]
            ids=[model.joint(n).dofadr[0] for n in names]
            jac=sum(c[0]*c[3][:,ids] for c in contacts)/total
            desired=desired_contact_velocity(position_error,rotation_error,point-pen.xpos,self.gain)
            posture=(np.array([self.posture_anchor[n]-base_targets[n]-grip_servo.delta[n] for n in names])
                     *self.posture_gain) if self.posture_gain else None
            free=list(range(len(names)))
            # Remove directions blocked by target limits, then solve again so
            # component-wise clipping cannot undo the normal nullspace.
            for _ in range(len(names)+1):
                velocity=tangential_joint_velocity(jac,normal,desired,free,posture)
                blocked=[]
                for k in free:
                    name=names[k];base=float(base_targets[name]);delta=grip_servo.delta[name]
                    lo,hi=model.actuator_ctrlrange[model.actuator('act_'+name).id]
                    low,high=max(-.25,lo-base),min(.25,hi-base)
                    if (delta<=low+1e-9 and velocity[k]<0) or (delta>=high-1e-9 and velocity[k]>0):blocked.append(k)
                if not blocked:break
                free=[k for k in free if k not in blocked]
            scale=1.
            for k in free:
                name=names[k];base=float(base_targets[name]);delta=grip_servo.delta[name]
                lo,hi=model.actuator_ctrlrange[model.actuator('act_'+name).id]
                low,high=max(-.25,lo-base),min(.25,hi-base)
                step=dt*velocity[k]
                if step>0 and delta<=high:scale=min(scale,(high-delta)/step)
                elif step<0 and delta>=low:scale=min(scale,(low-delta)/step)
            for k in free:
                name=names[k];grip_servo.delta[name]+=common_scale*max(0.,scale)*dt*velocity[k]
            self.active_updates+=1
