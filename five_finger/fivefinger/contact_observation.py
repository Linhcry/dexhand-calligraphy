"""Read-only simulator contact geometry and wrenches acting ON the free brush.

Groups: thumb, index, middle, ring, little, paper, other environment contacts.
Wrench order is force XYZ then torque XYZ, in world axes, about brush COM.
"""
import mujoco
import numpy as np

GROUPS=('thumb','index','middle','ring','little','paper','other')


def friction_utilization(local_wrench, friction, dim, elliptic):
    """Contact cone ratio, including enabled torsional/rolling dimensions."""
    force=np.asarray(local_wrench,float)
    if force[0]<=1e-6 or dim==1:return 0.
    scaled=force[1:dim]/np.maximum(np.asarray(friction,float)[:dim-1],1e-12)
    return float((np.linalg.norm(scaled) if elliptic else np.sum(abs(scaled)))/force[0])


def wrench_about_center(local_wrench, frame, point, center, brush_is_geom2):
    rotation=np.asarray(frame,float).reshape(3,3).T
    sign=1. if brush_is_geom2 else -1.
    force=sign*(rotation@np.asarray(local_wrench[:3]))
    torque=sign*(rotation@np.asarray(local_wrench[3:]))+np.cross(np.asarray(point)-center,force)
    return np.r_[force,torque]


def observe_brush_contacts(model,data):
    brush=model.body('brush').id
    center=data.xipos[brush];origin=data.xpos[brush];rotation=data.xmat[brush].reshape(3,3)
    wrench=np.zeros((7,6));normal=np.zeros(7);positions=np.zeros((7,3));counts=np.zeros(7,dtype=int)
    details=[];hand_z=[];scratch=np.zeros(6)
    axial_velocity=np.zeros(7);tangent_speed=np.zeros(7);cone_ratio=np.zeros(7)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    hp,hr=np.zeros_like(jp),np.zeros_like(jr)
    for index,contact in enumerate(data.contact):
        ga,gb=int(contact.geom1),int(contact.geom2)
        ba,bb=int(model.geom_bodyid[ga]),int(model.geom_bodyid[gb])
        if (ba==brush)==(bb==brush):continue
        is_second=bb==brush;other=ga if is_second else gb;name=model.geom(other).name or ''
        group=5 if name=='paper' else 6
        for finger in range(1,6):
            if name.startswith(f'col_r_f_link{finger}_'):group=finger-1;break
        mujoco.mj_contactForce(model,data,index,scratch)
        f=max(0.,float(scratch[0]));local=rotation.T@(contact.pos-origin)
        value=wrench_about_center(scratch,contact.frame,contact.pos,center,is_second)
        mujoco.mj_jac(model,data,jp,jr,contact.pos,brush)
        mujoco.mj_jac(model,data,hp,hr,contact.pos,int(model.geom_bodyid[other]))
        relative_velocity=(jp-hp)@data.qvel
        contact_normal=np.asarray(contact.frame[:3])
        tangential=relative_velocity-contact_normal*(relative_velocity@contact_normal)
        axial_velocity[group]+=f*float(relative_velocity@rotation[:,2])
        tangent_speed[group]+=f*float(np.linalg.norm(tangential))
        utilization=friction_utilization(scratch,contact.friction,int(contact.dim),model.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC)
        if f>.01:cone_ratio[group]=max(cone_ratio[group],utilization)
        wrench[group]+=value;normal[group]+=f;positions[group]+=f*local;counts[group]+=1
        if group<5 and f>1e-4:hand_z.append(float(local[2]))
        details.append({'other_geom':name,'group':GROUPS[group],'normal_N':f,
                        'position_world_m':contact.pos.tolist(),
                        'frame_on_brush_world':((1. if is_second else -1.)*np.asarray(contact.frame).reshape(3,3).T).tolist(),
                        'friction':np.asarray(contact.friction).tolist(),'contact_dim':int(contact.dim),
                        'position_brush_m':local.tolist(),'wrench_world':value.tolist(),
                        'relative_velocity_world_m_s':relative_velocity.tolist(),'friction_utilization':utilization})
    positions=np.divide(positions,normal[:,None],out=np.full((7,3),np.nan),where=normal[:,None]>1e-8)
    return {'wrench_world':wrench,'normal_N':normal,'position_brush_m':positions,'contact_count':counts,
            'hand_axial_span_m':max(hand_z)-min(hand_z) if len(hand_z)>1 else 0.,
            'brush_axis_world':rotation[:,2].copy(),'details':details,
            'axial_relative_velocity_m_s':np.divide(axial_velocity,normal,out=np.full(7,np.nan),where=normal>1e-8),
            'tangential_relative_speed_m_s':np.divide(tangent_speed,normal,out=np.full(7,np.nan),where=normal>1e-8),
            'friction_utilization_max':cone_ratio}
