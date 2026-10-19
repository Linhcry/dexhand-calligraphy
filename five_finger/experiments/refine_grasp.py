"""Contact-aware pose refinement; development optimization, not human demonstration."""
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
import yaml

from experiments.probe_grasp import make_model, GRIP
from experiments.run_suite import physical_trial
from experiments.retarget_grasp import OUT
from fivefinger.mujoco_kinematics import MujocoHandKinematics
from fivefinger.paths import ROOT


def refine(targets, penetration):
    model, data, center = make_model()
    robot = MujocoHandKinematics(model, data)
    prior = np.asarray([targets.get(name, robot.q0[i]) for i, name in enumerate(robot.dof_joint_names)])
    ids = np.array([i for i, name in enumerate(robot.dof_joint_names) if name != 'r_f_joint3_1' and i < 12])
    base = data.body('right_hand_base')
    rotation = base.xmat.reshape(3,3).copy()
    origin = base.xpos.copy()
    line = rotation.T @ (center-origin)
    # Opposing surfaces: thumb on one side, index/middle on the other.
    normals = {1: np.array([0., -1.]), 2: np.array([-.15, 1.]), 3: np.array([.15, 1.])}

    def closest(finger, segment):
        name = f'col_r_f_link{finger}_{segment}'
        geom = model.geom(name)
        pos = rotation.T @ (robot.data.geom(name).xpos-origin)
        axis = rotation.T @ robot.data.geom(name).xmat.reshape(3,3)[:,2]
        half = geom.size[1]
        a, b = (pos-half*axis)[[0,2]], (pos+half*axis)[[0,2]]
        ab=b-a; u=np.clip(np.dot(line[[0,2]]-a,ab)/max(np.dot(ab,ab),1e-12),0,1)
        return a+u*ab, float(geom.size[0])

    def residual(x):
        q=prior.copy(); q[ids]=x
        robot.compute_forward_kinematics(q)
        errors=[]
        for finger in (1,2,3):
            point,radius=closest(finger,4)
            normal=normals[finger]/np.linalg.norm(normals[finger])
            desired=line[[0,2]]+normal*(radius+.006-penetration)
            errors.extend(point-desired)
            for segment in (2,3):
                other,r=closest(finger,segment)
                errors.append(max(0.,r+.006-np.linalg.norm(other-line[[0,2]])))
        errors.extend(.0005*(x-prior[ids]))
        return np.asarray(errors)

    solved=least_squares(residual,np.clip(prior[ids],robot.joint_limits[ids,0],robot.joint_limits[ids,1]),
                         bounds=(robot.joint_limits[ids,0],robot.joint_limits[ids,1]),max_nfev=200)
    q=prior.copy();q[ids]=solved.x
    return dict(zip(robot.dof_joint_names,q.tolist())),float(np.linalg.norm(solved.fun)),bool(solved.success)


def main():
    source=json.loads((OUT/'candidates.json').read_text(encoding='utf-8'))
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    # Fixed small ablation: prior vs geometric correction, two closure depths.
    priors=[('baseline',GRIP)]+[(c['id'],c['grip_targets_rad']) for c in source['candidates'][:2]]
    reports=[]
    for name,prior in priors:
        for depth in (.001,.003):
            grip,error,converged=refine(prior,depth)
            trial=f'refined_{name}_{int(depth*1000)}mm'
            folder=OUT/trial
            report=physical_trial(folder,rules,0.,(0.,0.,0.),grip_targets=grip)
            report.update(candidate_id=trial,optimizer_residual_m=error,optimizer_converged=converged,
                          target_overlap_m=depth,contact_refinement=True)
            (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            reports.append(report)
            print(json.dumps({'id':trial,'hold_s':report['longest_continuous_hold_s'],'passed':report['hold_3s_passed'],'residual_m':error}),flush=True)
    summary={'independent_development_trials':len(reports),'hold_successes':sum(r['hold_3s_passed'] for r in reports),
             'best_hold_s':max(r['longest_continuous_hold_s'] for r in reports),'results':reports,'official_score':None}
    (OUT/'refinement_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
