"""Find tabletop-safe elbow branches without stepping penetrating initial states."""
import json
import mujoco
import numpy as np
from experiments.writing import make_writing_model,OUT
from fivefinger.control import solve_arm_ik


def mesh_lowest(model,data):
    lows=[]
    for i in range(1,7):
        mesh=model.mesh(f'r_a_link{i}').id
        gid=next(g for g in range(model.ngeom) if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g]==mesh)
        a,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        lows.append(float(np.min(model.mesh_vert[a:a+n]@data.geom_xmat[gid].reshape(3,3)[2,:])+data.geom_xpos[gid,2]))
    return lows


def main():
    m,d,center,R=make_writing_model();target=d.site('palm_control').xpos.copy();rng=np.random.default_rng(42)
    candidates=[]
    for i in range(100):
        q=np.array([rng.uniform(-2,2),rng.uniform(-1,2),rng.uniform(-3,0),rng.uniform(-1,4),rng.uniform(-3,3),rng.uniform(-3,3)])
        d.qpos[:6]=q
        solved=solve_arm_ik(m,d,target,R,iterations=450)
        if not solved.converged:continue
        d.qpos[:6]=solved.joints;mujoco.mj_forward(m,d);lows=mesh_lowest(m,d)
        if min(lows)>.005:
            candidates.append({'q':solved.joints.tolist(),'minimum_z_by_link':lows,'residual':solved.position_error_m})
            print(json.dumps(candidates[-1]),flush=True)
            break
    (OUT/'safe_arm_candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    if not candidates:raise RuntimeError('No tabletop-safe IK solution')


if __name__=='__main__':main()
