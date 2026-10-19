"""Bounded local joint/contact motion solve for a freely held pen.

Finger target velocities and ONE object twist are solved together. Normal
compression goals come from the existing force servo. All geometry/forces are
simulator observations; no pen state or external wrench is written.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from fivefinger.grasp_stabilization import axis_alignment_error


def solve_contact_velocity(jac, object_map, compression, desired, preferred, lower, upper,posture_weight=.003):
    jac=np.asarray(jac,float);obj=np.asarray(object_map,float)
    compression=np.asarray(compression,float);desired=np.asarray(desired,float)
    preferred=np.asarray(preferred,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    n=jac.shape[1]
    if not np.isfinite(posture_weight) or not 0<posture_weight<=.05:raise ValueError('Invalid posture weight')
    if obj.shape!=(len(jac),6) or compression.shape!=(len(jac),) or desired.shape!=(6,) or any(v.shape!=(n,) for v in (preferred,lower,upper)):
        raise ValueError('Invalid coupled motion dimensions')
    if not np.isfinite(np.r_[jac.ravel(),obj.ravel(),compression,desired,preferred,lower,upper]).all() or np.any(lower>upper):
        raise ValueError('Invalid coupled motion data')
    # Translation and rotation*0.05 both use m/s. Contact consistency has
    # higher weight than pose tracking; this is a soft constraint, not equality.
    matrix=np.vstack([np.column_stack([20*jac,-20*obj]),
        np.column_stack([np.zeros((6,n)),np.eye(6)]),
        np.column_stack([posture_weight*np.eye(n),np.zeros((n,6))])])
    rhs=np.r_[20*compression,desired,posture_weight*preferred]
    lo=np.r_[lower,np.full(6,-.001)];hi=np.r_[upper,np.full(6,.001)]
    fixed=hi-lo<1e-12;value=(hi+lo)/2
    free=~fixed
    solved=lsq_linear(matrix[:,free],rhs-matrix[:,fixed]@value[fixed],
        bounds=(lo[free],hi[free]),method='bvls',tol=1e-9,max_iter=80)
    if not solved.success:raise RuntimeError('Coupled contact velocity solve failed')
    value[free]=solved.x
    residual=jac@value[:n]-obj@value[n:]-compression
    return value[:n],value[n:],float(np.max(abs(residual)))


class CoordinatedGraspServo:
    def __init__(self,model,gain,posture_weight=.003):
        if not np.isfinite(gain) or not 0<=gain<=1:raise ValueError('Invalid coordinated pose gain')
        self.model=model;self.gain=gain;self.anchor=None;self.target_anchor=None
        if not np.isfinite(posture_weight) or not 0<posture_weight<=.05:raise ValueError('Invalid coordinated posture weight')
        self.posture_weight=posture_weight
        self.updates=0;self.missing_contact_updates=0;self.maximum_residual=0.
        self.maximum_position_error=0.;self.maximum_axis_error=0.;self.maximum_target_rate=0.

    def update(self,data,force_servo,base,previous,dt):
        if not 0<dt<=.05:raise ValueError('Invalid coordinated control interval')
        m=self.model;hand=data.body('right_hand_base');pen=data.body('brush')
        rh=hand.xmat.reshape(3,3);axis=pen.xmat.reshape(3,3)[:,2]
        position=rh.T@(pen.xpos-hand.xpos)
        if self.anchor is None:
            self.anchor=(position.copy(),rh.T@axis)
            # Start from actual issued (already clipped) targets, not hidden
            # acquisition offsets that may have integrated beyond a ctrl bound.
            self.target_anchor={n:float(data.ctrl[m.actuator('act_'+n).id]) for n in previous}
            for n in previous:force_servo.delta[n]=self.target_anchor[n]-base[n]
            return
        names=list(previous);dofs=[m.joint(n).dofadr[0] for n in names]
        groups={i:[] for i in (1,2,3)};scratch=np.zeros(6)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        for index,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            if int(m.geom_bodyid[a])==pen.id:other=b;normal=-np.asarray(c.frame[:3])
            elif int(m.geom_bodyid[b])==pen.id:other=a;normal=np.asarray(c.frame[:3])
            else:continue
            geom_name=m.geom(other).name or ''
            finger=next((f for f in groups if geom_name.startswith(f'col_r_f_link{f}_')),None)
            if finger is None:continue
            mujoco.mj_contactForce(m,data,index,scratch)
            if scratch[0]<=.05:continue
            mujoco.mj_jac(m,data,jp,jr,c.pos,int(m.geom_bodyid[other]))
            groups[finger].append((float(scratch[0]),c.pos.copy(),normal.copy(),jp[:,dofs].copy()))
        if any(not c for c in groups.values()):
            self.missing_contact_updates+=1
            return  # Keep the already bounded normal-force maintenance update.
        before=np.array([previous[n] for n in names]);normal_velocity=(np.array([force_servo.delta[n] for n in names])-before)/dt
        jacobians=[];object_maps=[];compression=[]
        for contacts in groups.values():
            total=sum(c[0] for c in contacts)
            point=sum(c[0]*c[1] for c in contacts)/total
            normal=sum(c[0]*c[2] for c in contacts)/total
            if np.linalg.norm(normal)<1e-8:
                self.missing_contact_updates+=1
                return
            normal/=np.linalg.norm(normal)
            jac=sum(c[0]*c[3] for c in contacts)/total
            r=point-pen.xpos;skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
            jacobians.append(jac);object_maps.append(np.column_stack([np.eye(3),-skew/.05]))
            compression.extend(normal*(normal@jac@normal_velocity))
        pos_error=rh@(self.anchor[0]-position);axis_error=axis_alignment_error(axis,rh@self.anchor[1])
        self.maximum_position_error=max(self.maximum_position_error,float(np.linalg.norm(pos_error)))
        self.maximum_axis_error=max(self.maximum_axis_error,float(np.linalg.norm(axis_error)))
        translation=self.gain*pos_error;translation*=min(1.,.001/max(np.linalg.norm(translation),1e-12))
        angular=self.gain*axis_error;angular*=min(1.,.02/max(np.linalg.norm(angular),1e-12))
        lower=[];upper=[];preferred=[]
        for i,n in enumerate(names):
            lo,hi=m.actuator_ctrlrange[m.actuator('act_'+n).id]
            lo=max(lo-base[n],-.25);hi=min(hi-base[n],.25)
            if n.endswith('_1'):
                lo=max(lo,self.target_anchor[n]-base[n]-.05)
                hi=min(hi,self.target_anchor[n]-base[n]+.05)
            # Interval includes force-maintenance corrections at the rate limit.
            lower.append(max(-.12,(lo-before[i])/dt));upper.append(min(.12,(hi-before[i])/dt))
            preferred.append(np.clip(.05*(self.target_anchor[n]-base[n]-before[i]),-.04,.04))
        velocity,twist,residual=solve_contact_velocity(np.vstack(jacobians),np.vstack(object_maps),
            compression,np.r_[translation,.05*angular],preferred,lower,upper,self.posture_weight)
        for n,v in zip(names,velocity):force_servo.delta[n]=previous[n]+dt*float(v)
        self.maximum_residual=max(self.maximum_residual,residual)
        self.maximum_target_rate=max(self.maximum_target_rate,float(np.max(abs(velocity))))
        self.updates+=1

    def report(self):
        return {'updates':self.updates,'missing_contact_normal_only_updates':self.missing_contact_updates,
            'pose_gain_s':self.gain,
            'posture_weight_m':self.posture_weight,
            'maximum_contact_velocity_residual_m_s':self.maximum_residual,
            'maximum_relative_position_error_mm':self.maximum_position_error*1000,
            'maximum_relative_axis_error_deg':float(np.degrees(self.maximum_axis_error)),
            'maximum_joint_target_rate_rad_s':self.maximum_target_rate,
            'spread_excursion_bound_rad':.05,'soft_contact_consistency_constraint':True,'hardware_ready':False}
