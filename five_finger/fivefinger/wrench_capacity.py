"""Static point-contact load estimate, NOT dynamic grasp certification.

Friction is approximated by an inscribed polygon. Patch moments are omitted by
default; optional axial torque rays provide a second, still conservative model.
Finger normal budgets are declared assumptions; actuation/rolling constraints
and model-to-hardware error are not represented. No simulated state is modified.
"""
import numpy as np
from scipy.optimize import linprog


def load_capacity(contacts, center, external_wrench, disturbance, budgets, edges=16, search_max=1.,include_contact_moments=False):
    center=np.asarray(center,float);external=np.asarray(external_wrench,float)
    disturbance=np.asarray(disturbance,float)
    if center.shape!=(3,) or external.shape!=(6,) or disturbance.shape!=(6,) or not np.isfinite(np.r_[center,external,disturbance]).all():
        raise ValueError('Invalid wrench geometry')
    if edges<4 or edges%4 or not budgets or any(not np.isfinite(v) or v<=0 for v in budgets.values()) or not np.isfinite(search_max) or search_max<=0:
        raise ValueError('Invalid friction discretization or normal budgets')
    names=list(budgets);columns=[];groups=[]
    for contact in contacts:
        if contact['group'] not in budgets or contact['normal_N']<=.01:continue
        frame=np.asarray(contact['frame_on_brush_world'],float)
        point=np.asarray(contact['position_world_m'],float)
        if frame.shape!=(3,3) or point.shape!=(3,) or not np.isfinite(np.r_[frame.ravel(),point]).all() or not np.allclose(frame.T@frame,np.eye(3),atol=1e-7):
            raise ValueError('Invalid contact frame')
        mu=np.asarray(contact['friction'],float)[:2] if contact['contact_dim']>=3 else np.zeros(2)
        if mu.shape!=(2,) or not np.isfinite(mu).all() or (mu<0).any():raise ValueError('Invalid friction')
        rays=[np.r_[1.,mu[0]*np.cos(angle),mu[1]*np.sin(angle),0.,0.,0.]
              for angle in np.arange(edges)*2*np.pi/edges]
        if include_contact_moments:
            coefficients=np.asarray(contact['friction'],float)
            if coefficients.shape!=(5,) or not np.isfinite(coefficients).all() or (coefficients<0).any():
                raise ValueError('Invalid contact moment coefficients')
            for component in range(3,contact['contact_dim']):
                for sign in (-1,1):
                    ray=np.zeros(6);ray[0]=1.;ray[component]=sign*coefficients[component-1];rays.append(ray)
        for ray in rays:
            force=frame@ray[:3]
            columns.append(np.r_[force,np.cross(point-center,force)+frame@ray[3:]])
            groups.append(names.index(contact['group']))
    if not columns:return {'feasible':False,'capacity':None,'reason':'no_active_hand_contacts'}
    matrix=np.column_stack(columns)
    # Torque rows divided by 0.1 m only to improve numerical conditioning.
    scale=np.array([1.,1.,1.,10.,10.,10.])
    eq=np.column_stack([matrix,disturbance])*scale[:,None]
    limits=np.zeros((len(names),len(columns)+1))
    limits[groups,np.arange(len(columns))]=1.
    objective=np.zeros(len(columns)+1);objective[-1]=-1.
    result=linprog(objective,A_ub=limits,b_ub=list(budgets.values()),
        A_eq=eq,b_eq=-external*scale,bounds=[(0,None)]*len(columns)+[(0,search_max)],method='highs')
    if not result.success:return {'feasible':False,'capacity':None,'solver_status':int(result.status)}
    residual=matrix@result.x[:-1]+disturbance*result.x[-1]+external
    if np.max(abs(residual*scale))>1e-6:raise RuntimeError('Wrench equilibrium residual too large')
    return {'feasible':True,'capacity':float(result.x[-1]),
        'search_cap_reached':bool(result.x[-1]>search_max-1e-6),
        'normal_budget_usage_N':dict(zip(names,(limits@result.x).tolist())),
        'equilibrium_residual':residual.tolist()}
