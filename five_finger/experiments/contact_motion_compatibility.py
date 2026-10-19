"""Local no-slip contact compatibility, from saved geometry and joint kinematics.

No simulation stepping or contact force solve. Shared scene rebuild requires
serial use after physical trials. Rank depends on model/linearization/tolerance.
"""
import argparse
import json

import mujoco
import numpy as np
from scipy.linalg import block_diag

from experiments.view_blind_competition import load


def compatibility(jacobians,normals,levers,preserve_normals):
    blocks=[];object_maps=[]
    for jac,normal,r in zip(jacobians,normals,levers):
        if preserve_normals:
            row=normal@jac
            jac=jac@(np.eye(jac.shape[1])-np.outer(row,row)/max(row@row,1e-20))
        blocks.append(jac)
        skew=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
        object_maps.append(np.column_stack([np.eye(3),-skew/.05]))
    fingers=block_diag(*blocks);obj=np.vstack(object_maps)
    residual=(np.eye(len(obj))-fingers@np.linalg.pinv(fingers,rcond=1e-8))@obj
    singular=np.linalg.svd(residual,compute_uv=False)
    rank=int(np.sum(singular>1e-5))
    return {'compatible_object_twist_dimension':6-rank,
        'constraint_singular_values':singular.tolist(),
        'unit_motion_residual_xyz_then_rotation':np.linalg.norm(residual,axis=0).tolist(),
        'characteristic_length_m':.05,'rank_absolute_tolerance':1e-5}


def analyze(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    source=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))['samples']
    output=[]
    for when in [30.,45.,60.,90.,105.]:
        if when>tr['time_s'][-1]:continue
        sample=min(source,key=lambda s:abs(s['time_s']-when));i=int(np.searchsorted(tr['time_s'],sample['time_s']))
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        results={}
        for spread in (False,True):
            jacobians=[];normals=[];levers=[]
            for finger,name in enumerate(('thumb','index','middle'),1):
                contacts=[c for c in sample['contacts'] if c['group']==name and c['normal_N']>.01]
                if not contacts:break
                names=[f'r_f_joint{finger}_{j}' for j in (1,2,3,4) if (spread or j!=1) and (finger,j)!=(3,1)]
                ids=[m.joint(n).dofadr[0] for n in names]
                jac=np.zeros((3,len(ids)));point=np.zeros(3);normal=np.zeros(3);total=sum(c['normal_N'] for c in contacts)
                for c in contacts:
                    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));weight=c['normal_N']/total
                    mujoco.mj_jac(m,d,jp,jr,c['position_world_m'],int(m.geom_bodyid[m.geom(c['other_geom']).id]))
                    jac+=weight*jp[:,ids];point+=weight*np.asarray(c['position_world_m'])
                    normal+=weight*np.asarray(c['frame_on_brush_world'])[:,0]
                if np.linalg.norm(normal)<1e-8:break
                jacobians.append(jac);normals.append(normal/np.linalg.norm(normal));levers.append(point-d.body('brush').xpos)
            if len(jacobians)!=3:continue
            for preserve in (False,True):
                key=f'spread_{spread}_preserve_normals_{preserve}'
                results[key]=compatibility(jacobians,normals,levers,preserve)
        output.append({'time_s':sample['time_s'],'stage':sample['stage'],'modes':results})
    result={'case':run.parent.name,'samples':output,'hardware_ready':False,
        'interpretation':'Local velocity equality at force-weighted representative contacts. No actuator bounds, finger dynamics, compliance or changing contact modes. Dimension is not a task success probability.'}
    (run/'contact_motion_compatibility.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return {'case':run.parent.name,'samples':[{'time_s':s['time_s'],'dimensions':{k:v['compatible_object_twist_dimension'] for k,v in s['modes'].items()}} for s in output]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+')
    for case in p.parse_args().cases:print(json.dumps(analyze(case)))
