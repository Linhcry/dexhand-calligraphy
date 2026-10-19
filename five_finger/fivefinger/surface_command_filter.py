"""Optional frozen-tool command projection away from a horizontal surface.

Preserves the desired tool-point displacement to first order when feasible.
It is a kinematic position-target filter, not a dynamics or hardware guarantee.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize,LinearConstraint


def tip_preserving_projection(tip_jacobian,height_jacobian,delta,clearance,floor,lower,upper):
    j=np.asarray(tip_jacobian,float);h=np.asarray(height_jacobian,float);delta=np.asarray(delta,float)
    clearance=np.asarray(clearance,float);lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    if j.shape!=(3,6) or h.shape!=(len(clearance),6) or delta.shape!=(6,) or lower.shape!=(6,) or upper.shape!=(6,):
        raise ValueError('Expected a six-joint arm and point/height Jacobians')
    if not all(np.isfinite(x).all() for x in (j,h,delta,clearance,lower,upper)) or not .001<=floor<=.01:
        raise ValueError('Invalid clearance projection inputs')
    _,singular,vh=np.linalg.svd(j,full_matrices=True);rank=int(np.sum(singular>1e-8));null=vh[rank:].T
    # Position-target barrier: do not spend more than 30% of current clearance
    # reserve in a command. This does not bound the actual dynamic trajectory.
    required=-.3*(clearance-floor)-h@delta
    matrix=np.vstack([h@null,null,-null]);bound=np.r_[required,lower-delta,delta-upper]
    result=minimize(lambda a:.5*float(a@a),np.zeros(null.shape[1]),jac=lambda a:a,
        constraints=[LinearConstraint(matrix,bound,np.inf)],method='SLSQP',
        options={'ftol':1e-12,'maxiter':80})
    if not result.success or np.min(matrix@result.x-bound)<-1e-7:
        raise RuntimeError('surface_command_projection_infeasible')
    candidate=delta+null@result.x
    if np.max(abs(j@(candidate-delta)))>1e-8:raise RuntimeError('surface_tip_nullspace_error')
    return candidate


class SurfaceCommandFilter:
    def __init__(self,model,mesh_entries,plane_z,floor=.005,actual_guard=.002):
        if not np.isfinite(plane_z) or not .001<=floor<=.01:raise ValueError('Invalid surface plane or floor')
        if not np.isfinite(actual_guard) or not 0<actual_guard<=floor:raise ValueError('Actual guard must be positive and at most the reference clearance')
        self.model=model;self.entries=[e for e in mesh_entries if e[1]!='r_a_link0']
        if not self.entries:raise ValueError('Robot mesh evidence required')
        self.plane_z=float(plane_z);self.floor=float(floor);self.scratch=mujoco.MjData(model)
        self.calls=0;self.active=0;self.backtracked=0;self.minimum_predicted_gap=None;self.maximum_target_delta=0.
        self.maximum_tip_linear_change=0.;self.failures=0
        self.actual_guard=float(actual_guard);self.actual_steps=0;self.minimum_actual_gap=None;self.actual_stops=0

    def critical(self,data):
        points=[];heights=[];bodies=[]
        for gid,name,vertices in self.entries:
            geom=data.geom(gid);world=vertices@geom.xmat.reshape(3,3).T+geom.xpos
            point=world[np.argmin(world[:,2])]
            points.append(point);heights.append(point[2]-self.plane_z);bodies.append(int(self.model.geom_bodyid[gid]))
        return points,np.asarray(heights),bodies

    def apply(self,data,command,tool_point):
        model=self.model;self.calls+=1;command=np.asarray(command,float).copy()
        delta=command-data.qpos[:6];points,gap,bodies=self.critical(data)
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));rows=[]
        for point,body in zip(points,bodies):
            mujoco.mj_jac(model,data,jp,jr,point,body);rows.append(jp[2,:6].copy())
        height_jac=np.asarray(rows)
        if np.min(gap+height_jac@delta)>=.020:return command
        self.active+=1
        mujoco.mj_jac(model,data,jp,jr,tool_point,int(model.site_bodyid[model.site('palm_control').id]))
        tip_jac=jp[:,:6].copy()
        lower=np.maximum(-.06,model.actuator_ctrlrange[:6,0]-data.qpos[:6])
        upper=np.minimum(.06,model.actuator_ctrlrange[:6,1]-data.qpos[:6])
        try:candidate=tip_preserving_projection(tip_jac,height_jac,delta,gap,self.floor,lower,upper)
        except RuntimeError:
            self.failures+=1;raise
        # Nonlinear frozen-hand endpoint check, without changing the live state.
        threshold=min(float(np.min(gap)),self.floor)
        for scale in (1.,.5,.25,.125):
            step=scale*candidate;self.scratch.qpos[:]=data.qpos;self.scratch.qpos[:6]+=step
            mujoco.mj_kinematics(model,self.scratch)
            _,predicted,_=self.critical(self.scratch)
            minimum=float(np.min(predicted))
            if minimum>=threshold-1e-7:
                self.backtracked+=int(scale<1)
                self.minimum_predicted_gap=minimum if self.minimum_predicted_gap is None else min(self.minimum_predicted_gap,minimum)
                self.maximum_target_delta=max(self.maximum_target_delta,float(np.max(abs(step))))
                self.maximum_tip_linear_change=max(self.maximum_tip_linear_change,float(np.linalg.norm(tip_jac@(step-delta))))
                return data.qpos[:6]+step
        self.failures+=1
        raise RuntimeError('surface_nonlinear_endpoint_infeasible')

    def require_actual_clear(self,data):
        _,gap,_=self.critical(data);minimum=float(np.min(gap));self.actual_steps+=1
        self.minimum_actual_gap=minimum if self.minimum_actual_gap is None else min(self.minimum_actual_gap,minimum)
        if minimum<self.actual_guard:
            self.actual_stops+=1
            raise RuntimeError(f'surface_actual_clearance_stop: gap={minimum:.6f} m')

    def report(self):
        return {'calls':self.calls,'active_updates':self.active,'backtracked_updates':self.backtracked,
            'infeasible_updates':self.failures,'floor_m':self.floor,'plane_z_m':self.plane_z,
            'minimum_checked_endpoint_gap_m':self.minimum_predicted_gap,'maximum_joint_target_delta_rad':self.maximum_target_delta,
            'maximum_linear_tool_displacement_change_m':self.maximum_tip_linear_change,
            'actual_guard_m':self.actual_guard,'actual_steps_checked':self.actual_steps,
            'minimum_actual_gap_m':self.minimum_actual_gap,'actual_clearance_stops':self.actual_stops,
            'limitations':'Infinite horizontal paper plane; frozen held-tool Jacobian and endpoint mesh checks. Finger meshes excluded from proactive filter but included in 500Hz non-tool contact audit. No dynamic clearance guarantee.'}
