"""Separate axial material rotation from contact azimuth for a round tool.

All vectors are expressed in the hand frame. The reference transverse axis is
transported by the shortest rotation aligning the initial and current tool axes.
This coordinate convention is a diagnostic, not a slip or stability detector.
"""
import numpy as np


def transported_basis(reference_axis,reference_x,current_axis):
    a=np.asarray(reference_axis,float);x=np.asarray(reference_x,float);b=np.asarray(current_axis,float)
    if any(v.shape!=(3,) for v in (a,x,b)) or not np.isfinite([a,x,b]).all():
        raise ValueError('Expected three finite 3D frame vectors')
    if min(np.linalg.norm(a),np.linalg.norm(b))<1e-9:raise ValueError('Undefined tool axis')
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    x=x-a*(a@x)
    if np.linalg.norm(x)<1e-9:raise ValueError('Undefined transverse reference')
    x/=np.linalg.norm(x)
    cosine=float(np.clip(a@b,-1,1))
    if cosine<=-1+1e-7:raise ValueError('Antiparallel tool axes have ambiguous shortest transport')
    v=np.cross(a,b)
    # Rodrigues formula, applied directly to the reference transverse vector.
    u=x+np.cross(v,x)+np.cross(v,np.cross(v,x))/(1+cosine)
    u-=b*(b@u);u/=np.linalg.norm(u)
    return u,np.cross(b,u),b


def axial_angles(hand_from_pen,point_pen,reference_axis,reference_x):
    rotation=np.asarray(hand_from_pen,float);point=np.asarray(point_pen,float)
    if rotation.shape!=(3,3) or point.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(point).all():
        raise ValueError('Expected finite tool rotation and contact point')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or np.linalg.det(rotation)<0:
        raise ValueError('Expected a proper rotation')
    if np.linalg.norm(point[:2])<1e-9:raise ValueError('Contact on tool axis has undefined azimuth')
    u,v,_=transported_basis(reference_axis,reference_x,rotation[:,2])
    radial=rotation@point
    material_angle=np.arctan2(v@rotation[:,0],u@rotation[:,0])
    hand_contact_angle=np.arctan2(v@radial,u@radial)
    body_contact_angle=np.arctan2(point[1],point[0])
    return np.array([hand_contact_angle,material_angle,body_contact_angle])
