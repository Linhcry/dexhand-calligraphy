"""Simulation-only normal-force servo using known barrel pose and pair forces."""
import mujoco
import numpy as np


def validated_grip_targets(target_n):
    """Development force limits apply to acquisition and maintenance alike."""
    targets = np.broadcast_to(np.asarray(target_n, dtype=float), (3,)).copy()
    if not np.isfinite(targets).all() or not ((targets > 0) & (targets <= 3.)).all():
        raise ValueError('Force target outside development limits (0, 3] N')
    return targets


class ContactGripServo:
    def __init__(self, model, target_n=.35, gain_m_per_ns=.008):
        targets=validated_grip_targets(target_n)
        if not 0 < gain_m_per_ns <= .03:
            raise ValueError('Force target or feedback gain outside development limits')
        self.model = model
        self.target_n = targets
        self.gain = gain_m_per_ns
        self.delta = {f'r_f_joint{i}_{j}': 0. for i in (1,2,3) for j in (1,2,3,4)
                      if (i,j) != (3,1)}

    def update(self, data, forces, dt, *, gain=None, deadband_n=0., fingers=(1,2,3), close_only=False, measured_contact=False, base_targets=None, lock_spread=False):
        forces = np.asarray(forces, float)
        if forces.shape != (5,) or not np.isfinite(forces).all() or (forces<0).any() or not 0 < dt <= .05:
            raise ValueError('Expected five forces and dt in (0, 0.05]')
        gain = self.gain if gain is None else gain
        if not np.isfinite(gain) or not 0 < gain <= .03 or not 0 <= deadband_n <= .5:
            raise ValueError('Invalid maintenance feedback parameters')
        if not fingers or any(f not in (1,2,3) for f in fingers) or len(set(fingers))!=len(fingers):
            raise ValueError('Invalid force-controlled fingers')
        brush = data.body('brush')
        axis = brush.xmat.reshape(3,3)[:,2]
        projection = np.eye(3)-np.outer(axis,axis)
        for finger in fingers:
            geom = self.model.geom(f'col_r_f_link{finger}_4')
            pose = data.geom(geom.id)
            half = geom.size[1]*pose.xmat.reshape(3,3)[:,2]
            a,b = pose.xpos-half,pose.xpos+half
            ab=projection@(b-a)
            u=np.clip(np.dot(projection@(brush.xpos-a),ab)/max(np.dot(ab,ab),1e-12),0,1)
            point=a+u*(b-a)
            toward=projection@(brush.xpos-point)
            norm=np.linalg.norm(toward)
            if norm < 1e-6:
                continue
            jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
            mujoco.mj_jac(self.model,data,jp,jr,point,int(geom.bodyid[0]))
            # Keep the acquired opposition/spread target when requested. Only
            # flexion joints then receive maintenance corrections; no joint is
            # physically locked and acquisition still uses the full hand.
            names=[name for name in self.delta if name.startswith(f'r_f_joint{finger}_') and not (lock_spread and name.endswith('_1'))]
            ids=[self.model.joint(name).dofadr[0] for name in names]
            jac=(toward/norm)@jp[:,ids]
            if measured_contact:
                weighted=np.zeros(len(ids));weight=0.;wrench=np.zeros(6)
                for contact_id,contact in enumerate(data.contact):
                    ga,gb=int(contact.geom1),int(contact.geom2)
                    na,nb=self.model.geom(ga).name or '',self.model.geom(gb).name or ''
                    if na.startswith(f'col_r_f_link{finger}_') and nb.startswith('brush_'):gid=ga
                    elif nb.startswith(f'col_r_f_link{finger}_') and na.startswith('brush_'):gid=gb
                    else:continue
                    mujoco.mj_contactForce(self.model,data,contact_id,wrench)
                    normal=max(0.,float(wrench[0]))
                    direction=projection@(brush.xpos-contact.pos)
                    distance=np.linalg.norm(direction)
                    if normal<1e-6 or distance<1e-6:continue
                    mujoco.mj_jac(self.model,data,jp,jr,contact.pos,int(self.model.geom_bodyid[gid]))
                    weighted += normal*((direction/distance)@jp[:,ids]);weight+=normal
                # A contact on link 3 has zero derivative with respect to joint 4.
                # Retain the geometric approach fallback only while out of contact.
                if weight>1e-6:jac=weighted/weight
            error=self.target_n[finger-1]-forces[finger-1]
            error=np.sign(error)*max(0.,abs(error)-deadband_n)
            if close_only:error=max(0.,error)
            speed=np.clip(gain*error,-.004,.004)
            bounds={name:(-.25,.25) for name in names}
            if base_targets is not None:
                for k,name in enumerate(names):
                    base=float(base_targets[name])
                    if not np.isfinite(base):raise ValueError('Nonfinite base joint target')
                    actuator=self.model.actuator('act_'+name)
                    low,high=self.model.actuator_ctrlrange[actuator.id]
                    if not low<=base<=high:raise ValueError('Base joint target outside actuator range')
                    lo,hi=max(-.25,low-base),min(.25,high-base)
                    bounds[name]=(lo,hi)
                    self.delta[name]=float(np.clip(self.delta[name],lo,hi))
                    intended=jac[k]*speed
                    if (self.delta[name]<=lo+1e-9 and intended<0) or (self.delta[name]>=hi-1e-9 and intended>0):
                        jac[k]=0.
                # Redistribute normal motion over feasible joints. A downstream
                # actuator clip must not silently consume the feedback correction.
            velocity=np.clip(jac*speed/(np.dot(jac,jac)+1e-6),-.12,.12)
            for name,v in zip(names,velocity):
                self.delta[name]=float(np.clip(self.delta[name]+dt*v,*bounds[name]))
        return self.delta.copy()
