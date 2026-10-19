"""Free-body pen pickup and contact-only writing in a declared practice fixture."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from fivefinger.cli import load_workcell
from fivefinger.scene import build_scene
from fivefinger.control import set_joint_targets, solve_arm_ik
from fivefinger.grasp_feedback import ContactGripServo
from fivefinger.trajectory import smoothstep5, make_path
from fivefinger.paths import ROOT
from experiments.probe_grasp import brush_contacts

OUT = ROOT / 'outputs' / 'writing'
DEFAULT_CONFIG = ROOT / 'experiments/writing_config.json'


def read_config(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('writing_speed_m_s','travel_speed_m_s','character_size_m','clearance_m','brush_radius_m'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(f'Invalid {key}')
    return cfg


def make_writing_model(yaw=0., shift=(0., 0., 0.),cfg=None,scene_hook=None):
    cfg=read_config() if cfg is None else cfg
    m0, d0 = load_workcell()
    rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    center = np.array([.17, .09, .15])
    palm = center - rotation @ np.array([.073136,.022,.209114])
    result = solve_arm_ik(m0, d0, palm + rotation @ [0.,0.,.1], rotation, iterations=400)
    if not result.converged:
        raise RuntimeError(f'Upright grasp IK: {result}')
    root = ET.parse(build_scene()).getroot()
    default_geom=root.find('./default/geom')
    default_geom.set('solref',' '.join(map(str,cfg['contact_solref'])))
    default_geom.set('solimp',' '.join(map(str,cfg['contact_solimp'])))
    default_geom.set('friction',' '.join(map(str,cfg['finger_contact_friction'])))
    default_geom.set('condim',str(cfg['finger_contact_dimensions']))
    root.find('option').set('cone','elliptic')
    root.find('option').set('noslip_iterations','5')
    world = root.find('worldbody')
    paper=world.find("geom[@name='paper']")
    paper.set('priority','1');paper.set('condim','3');paper.set('friction',' '.join(map(str,cfg['paper_friction'])))
    for name in ('rack','dot_target','inkstone','ink_surface_visual'):
        world.remove(world.find(f"geom[@name='{name}']"))
    brush = world.find("body[@name='brush']")
    brush.set('pos', ' '.join(map(str,center + shift)))
    brush.set('quat', '1 0 0 0')
    brush.find("geom[@name='brush_barrel']").set('size',f"{cfg['brush_radius_m']} .085")
    brush.find("geom[@name='brush_tip_approximation']").set('solref','.01 1')
    # Open-top cup. All contact remains physical; the pen has its original freejoint.
    bottom = center[2] - .107
    ET.SubElement(world,'geom',name='support_base',type='box',pos=f'{center[0]} {center[1]} {bottom/2}',size=f'.025 .025 {bottom/2}',rgba='.25 .28 .32 1')
    for axis in (0,1):
        for sign in (-1,1):
            pos = center.copy(); pos[axis] += sign*.0095; pos[2] = bottom+.038
            size = [.018,.018,.038]; size[axis] = .003
            ET.SubElement(world,'geom',name=f'support_wall_{axis}_{sign}',type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba='.25 .28 .32 1')
    for geom in world.findall('geom'):
        if geom.get('name','').startswith('support_'):
            geom.set('priority','1');geom.set('friction',' '.join(map(str,cfg['rack_friction'])))
    if scene_hook is not None:scene_hook(root)
    assets = {}
    for i,mesh in enumerate(root.findall('./asset/mesh')):
        if not mesh.get('file'):continue
        src=Path(mesh.get('file')); key=f'mesh_{i}{src.suffix}';assets[key]=src.read_bytes();mesh.set('file',key)
    for i,texture in enumerate(root.findall('./asset/texture')):
        if texture.get('file'):
            src=Path(texture.get('file'));key=f'texture_{i}{src.suffix}';assets[key]=src.read_bytes();texture.set('file',key)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    data=mujoco.MjData(model)
    data.qpos[:26]=d0.qpos[:26]; data.qpos[:6]=result.joints
    data.ctrl[:]=d0.ctrl; data.ctrl[:6]=result.joints
    mujoco.mj_forward(model,data)
    return model,data,center,rotation


def paper_contact(model,data):
    """Return force and weighted contact point ONLY for actual nib/paper pairs.

    The 2026 final scene may split the drawing area into several A3 sheets named
    ``paper`` or ``paper_N``. All such geoms count as the same scoring surface.
    """
    nib=model.geom('brush_tip_approximation').id
    force=0.;point=np.zeros(3);scratch=np.zeros(6)
    for index,contact in enumerate(data.contact):
        names={model.geom(contact.geom1).name or '',model.geom(contact.geom2).name or ''}
        paper_names={name for name in names if name=='paper' or name.startswith('paper_')}
        if nib not in {contact.geom1,contact.geom2} or not paper_names:continue
        mujoco.mj_contactForce(model,data,index,scratch)
        normal=max(0.,float(scratch[0]));force+=normal;point+=normal*contact.pos
    return force,point/max(force,1e-12)


def command_tip(model,data,target,rotation):
    """Resolved pose servo; simulation truth estimates the moving hand-to-pen offset."""
    site=model.site('palm_control').id
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,jp,jr,site)
    error=np.asarray(target)-data.site('brush_tip').xpos
    orientation=Rotation.from_matrix(rotation@data.site_xmat[site].reshape(3,3).T).as_rotvec()
    jac=np.vstack([jp[:,:6],.3*jr[:,:6]])
    delta=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),np.r_[error,.3*orientation])
    delta*=min(1.,.015/max(np.max(np.abs(delta)),1e-12))
    data.ctrl[:6]=np.clip(data.qpos[:6]+delta,model.actuator_ctrlrange[:6,0],model.actuator_ctrlrange[:6,1])


def run_experiment(args):
    cfg=read_config(args.config)
    cfg['yaw_deg']=args.yaw if args.yaw is not None else cfg['yaw_deg']
    cfg['initial_shift_m']=args.shift if args.shift is not None else cfg['initial_shift_m']
    cfg['grip_force_targets_N']=args.forces if args.forces is not None else cfg['grip_force_targets_N']
    cfg['force_gain_m_per_Ns']=args.gain if args.gain is not None else cfg['force_gain_m_per_Ns']
    cfg['feedback_start_s']=args.feedback_start if args.feedback_start is not None else cfg['feedback_start_s']
    m,d,center,rotation=make_writing_model(cfg['yaw_deg'],cfg['initial_shift_m'],cfg)
    grip=cfg['grip_targets_rad']
    initial={k:float(d.qpos[m.joint(k).qposadr[0]]) for k in grip}
    start=d.site('palm_control').xpos.copy()
    home=d.qpos[:6].copy()
    lifted=solve_arm_ik(m,d,start+[0,0,.12],rotation)
    assert lifted.converged, lifted
    servo=ContactGripServo(m,target_n=cfg['grip_force_targets_N'],gain_m_per_ns=cfg['force_gain_m_per_Ns'])
    path=make_path(cfg['text'],origin_xy=cfg['origin_xy_m'],character_size=cfg['character_size_m'],paper_z=cfg['pen_down_z_m'],
                   clearance=cfg['clearance_m'],speed=cfg['writing_speed_m_s'],travel_speed=cfg['travel_speed_m_s']) if args.write else None
    seconds=22+path.time_s[-1]+15 if path is not None else args.seconds
    rows=[]; states=[];controls=[];ink=[];stages=[];desired=[];transfer_tip=None
    relative_previous=None;hold_time=0.;longest_hold=0.;hold_passed=False;max_finger_torque=0.;lost_grip_s=0.;stop_reason='completed'
    path_index=0;touchdown_wait=0.;done_started=None
    for tick in range(int(seconds*100)):
        t=d.time; u=float(smoothstep5((t-1)/3))
        targets={k:initial[k]+u*(v-initial[k]) for k,v in grip.items()}
        if t>=cfg['feedback_start_s']:
            forces,_=brush_contacts(m,d)
            for k,v in servo.update(d,forces,.01).items():targets[k]+=v
        set_joint_targets(m,d,targets)
        lift=float(smoothstep5((t-7)/3))
        d.ctrl[:6]=home+lift*(lifted.joints-home)
        stage='close' if t<7 else ('lift' if t<10 else 'hold')
        target=d.site('brush_tip').xpos.copy()
        if path is not None and t>=14:
            if not hold_passed:
                stop_reason='hold_gate_failed';break
            if transfer_tip is None:transfer_tip=d.site('brush_tip').xpos.copy()
            if t<18:
                stage='transfer';target=transfer_tip.copy()
                target[:2]+=float(smoothstep5((t-14)/4))*(path.xyz[0,:2]-transfer_tip[:2])
            elif t<22:
                stage='lower';target=path.xyz[0].copy()
                target[2]=transfer_tip[2]+float(smoothstep5((t-18)/4))*(path.xyz[0,2]-transfer_tip[2])
            else:
                idx=path_index
                target=path.xyz[idx].copy();stage='write' if path.pen_down[idx] else 'pen_up'
                starts_stroke=bool(path.pen_down[idx] and idx>0 and not path.pen_down[idx-1])
                observed_force,_=paper_contact(m,d)
                if starts_stroke and observed_force<.03:
                    stage='touchdown';touchdown_wait+=.01
                    if touchdown_wait>2.:stop_reason='touchdown_timeout';break
                else:
                    touchdown_wait=0.;path_index=min(idx+1,len(path.xyz)-1)
                if idx==len(path.xyz)-1:
                    if done_started is None:done_started=t
                    stage='done';target[2]+=.02*float(smoothstep5((t-done_started)/2))
            command_tip(m,d,target,rotation)
        for _ in range(5):
            d.qfrc_applied[:6]=np.clip(d.qfrc_bias[:6],-40,40)
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
            raise RuntimeError('Non-finite physics state or MuJoCo warning')
        forces,support=brush_contacts(m,d)
        p=d.body('brush').xpos.copy();tip=d.site('brush_tip').xpos.copy()
        tilt=np.degrees(np.arccos(np.clip(d.body('brush').xmat.reshape(3,3)[2,2],-1,1)))
        paper_n,point=paper_contact(m,d)
        palm=d.body('right_hand_base');relative=palm.xmat.reshape(3,3).T@(p-palm.xpos)
        slip=0. if relative_previous is None else float(np.linalg.norm(relative-relative_previous)/.01)
        relative_previous=relative.copy()
        valid=bool(p[2]-center[2]>=.02 and support<.01 and forces[0]>.01 and (forces[1:]>.01).any() and slip<=.02)
        hold_time=hold_time+.01 if valid else 0.;longest_hold=max(longest_hold,hold_time);hold_passed|=hold_time>=3.
        gripping=bool(forces[0]>.01 and (forces[1:]>.01).any())
        lost_grip_s=lost_grip_s+.01 if t>=14 and not gripping else 0.
        max_finger_torque=max(max_finger_torque,float(np.max(np.abs(d.actuator_force[6:]))))
        rows.append([d.time,*p,*forces,support,tilt,*tip,paper_n,*point,slip,float(valid)])
        # No desired trajectory or pen-down flag is allowed to create ink.
        ink.append([paper_n>.01,*point,paper_n]);stages.append(stage);desired.append(target)
        states.append(d.qpos.copy());controls.append(d.ctrl.copy())
        if tick%200==0:print(round(d.time,2),stage,np.round(p,3),np.round(forces,2),'support',round(support,3),'tilt',round(tilt,1),'paper',round(paper_n,2),flush=True)
        if lost_grip_s>.3:stop_reason='grip_lost';break
        if done_started is not None and t-done_started>=3.:break
    folder=OUT/args.name;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'replay.npz',qpos=states,ctrl=controls,time_s=np.array(rows)[:,0],ink=ink,stage=stages,target=desired)
    np.savetxt(folder/'physics.csv',rows,delimiter=',',header='time,x,y,z,thumb,index,middle,ring,little,support,tilt,tip_x,tip_y,tip_z,paper_N,contact_x,contact_y,contact_z,relative_slip_m_s,valid_hold',comments='')
    (folder/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=np.asarray(rows);ink=np.asarray(ink);stages=np.asarray(stages);desired=np.asarray(desired)
    down=stages=='write';xy_error=np.linalg.norm(desired[down,:2]-rows[down,11:13],axis=1)
    strokes=[]
    transitions=np.diff(np.r_[False,down,False].astype(int));starts=np.where(transitions==1)[0];ends=np.where(transitions==-1)[0]
    for a,b in zip(starts,ends):
        strokes.append({'start_s':float(rows[a,0]),'end_s':float(rows[b-1,0]),'contact_fraction':float(np.mean(ink[a:b,0])),
                        'contact_path_length_m':float(sum(np.linalg.norm(ink[i,1:3]-ink[i-1,1:3]) for i in range(a+1,b) if ink[i,0] and ink[i-1,0]))})
    report={'stop_reason':stop_reason,'elapsed_sim_s':float(d.time),'hold_3s_passed':bool(hold_passed),'longest_hold_s':longest_hold,
            'writing_contact_fraction':float(np.mean(ink[down,0])) if down.any() else 0.,'strokes':strokes,
            'xy_error_median_mm':float(np.median(xy_error)*1000) if down.any() else None,
            'xy_error_p95_mm':float(np.quantile(xy_error,.95)*1000) if down.any() else None,
            'maximum_finger_contact_N':float(rows[:,4:9].max()),'maximum_finger_actuator_torque_Nm':max_finger_torque,
            'paper_force_median_N':float(np.median(rows[down,14])) if down.any() else None,
            'maximum_paper_force_N':float(rows[:,14].max()),'pen_tilt_final_deg':float(rows[-1,10]),
            'pen_held_at_end':bool(rows[-1,4]>.01 and (rows[-1,5:9]>.01).any() and rows[-1,9]<.01),
            'physical_step_s':float(m.opt.timestep),'free_pen':True,'pen_pose_overwritten_after_reset':False,
            'pen_external_force_applied':False,'control_uses_simulation_truth':True,'ink_from_contact_only':True,
            'fixture':'upright practice cup, not official contest fixture','model':'DexHand021 right + JAKA Zu7 reference',
            'brush_model':'rigid barrel and compliant contact at capsule nib, no bristle or ink-fluid simulation',
            'arm_collision_check':'official reference arm mesh collisions disabled','official_score':None,'physics_warnings':[]}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('warnings',[w.number for w in d.warning])
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--yaw',type=float)
    parser.add_argument('--shift',type=float,nargs=3)
    parser.add_argument('--forces',type=float,nargs=3)
    parser.add_argument('--gain',type=float)
    parser.add_argument('--feedback-start',type=float)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--name',default='demo')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if not args.name or any(s in args.name for s in ('/','\\','..')):parser.error('Expected a simple output folder name')
    run_experiment(args)


if __name__=='__main__':main()
