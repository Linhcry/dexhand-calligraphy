"""Camera-only title acquisition, physical pickup/dip and collision-audited poem writing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from experiments.competition_scene import make_scene,TableClearance
from experiments.probe_grasp import brush_contacts
from experiments.writing import paper_contact
from fivefinger.control import solve_arm_ik,set_joint_targets
from fivefinger.grasp_feedback import ContactGripServo, validated_grip_targets
from fivefinger.trajectory import make_path,smoothstep5
from fivefinger.paths import ROOT
from fivefinger.autonomy.command_clock import CommandClock
from fivefinger.autonomy.grip_guard import LiftGripGate
from fivefinger.autonomy.reference_velocity import ReferenceVelocity, position_servo_feedforward
from fivefinger.contact_observation import observe_brush_contacts, GROUPS
from fivefinger.grasp_posture import unused_finger_posture
from fivefinger.grasp_stabilization import InHandPoseServo, bound_combined_delta
from fivefinger.contact_closure import ContactAwareClosure
from fivefinger.force_peak_monitor import ForcePeakMonitor
from fivefinger.retention_envelope import RetentionEnvelope
from fivefinger.finger_load_allocation import FingerLoadAllocator
from fivefinger.coordinated_grasp import CoordinatedGraspServo
from fivefinger.robot_surface_monitor import RobotSurfaceMonitor
from fivefinger.surface_command_filter import SurfaceCommandFilter
from fivefinger.physics_recording import save_final_physics_state


def nib_point(m,d):
    geom=m.geom('brush_tip_approximation');pose=d.geom(geom.id);axis=pose.xmat.reshape(3,3)[:,2]
    return pose.xpos-np.sign(axis[2])*geom.size[1]*axis-np.array([0,0,geom.size[0]])


def upright_rotation(d):
    axis=d.body('brush').xmat.reshape(3,3)[:,2];cross=np.cross(axis,[0,0,1]);sine=np.linalg.norm(cross)
    correction=np.eye(3) if sine<1e-8 else Rotation.from_rotvec(cross/sine*np.arctan2(sine,axis[2])).as_matrix()
    return correction@d.site('palm_control').xmat.reshape(3,3)


def normal_damping_correction(height, velocity, damping_s):
    """Bounded velocity damping near the reference paper (top z = 2 mm)."""
    if not np.isfinite([height,velocity,damping_s]).all() or not 0<=damping_s<=.1:
        raise ValueError('Invalid normal motion damping')
    proximity=float(np.clip((.014-height)/.006,0.,1.))
    return float(np.clip(-proximity*damping_s*velocity,-.002,.002))


def command_nib(m,d,target,rotation,point_jacobian=False,normal_damping_s=0.,target_velocity=None,palm_reference=False):
    site=m.site('palm_control').id;jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,jr,site)
    if point_jacobian and not palm_reference:
        # Held-tool approximation: include the palm-to-tip lever arm in the
        # arm Jacobian. No actuator or external force is applied to the pen.
        mujoco.mj_jac(m,d,jp,jr,nib_point(m,d),int(m.site_bodyid[site]))
    error=np.asarray(target)-(d.site_xpos[site] if palm_reference else nib_point(m,d))
    error[:2]*=2.0
    if normal_damping_s:
        # Observe the FREE pen's point velocity; use only the six arm actuators
        # to brake descent/rebound. No change to contact physics or pen state.
        vp,vr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,vp,vr,nib_point(m,d),m.body('brush').id)
        error[2]+=normal_damping_correction(nib_point(m,d)[2],float(vp[2]@d.qvel),normal_damping_s)
    rot_error=Rotation.from_matrix(rotation@d.site_xmat[site].reshape(3,3).T).as_rotvec()
    jac=np.vstack([jp[:,:6],.3*jr[:,:6]])
    delta=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),np.r_[error,.3*rot_error])
    if target_velocity is not None:
        velocity=np.asarray(target_velocity,float)
        if velocity.shape!=(3,) or not np.isfinite(velocity).all():
            raise ValueError('Invalid Cartesian reference velocity')
        qdot_ref=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),np.r_[velocity,np.zeros(3)])
        delta+=position_servo_feedforward(qdot_ref,m.actuator_gainprm[:6,0],-m.actuator_biasprm[:6,2])
    delta*=min(1.,.06/max(np.max(np.abs(delta)),1e-12))
    d.ctrl[:6]=np.clip(d.qpos[:6]+delta,m.actuator_ctrlrange[:6,0],m.actuator_ctrlrange[:6,1])


class InkState:
    def __init__(self,s):
        self.s=s;self.dwell=0.;self.longest=0.;self.loaded=False;self.withdrawn=False;self.remaining=0.;self.previous=None;self.maximum_depth=0.
    def update(self,point,dt):
        inside=np.all(np.abs(point[:2]-self.s['ink_center_xy_m'])<np.asarray(self.s['ink_inner_half_size_m'])-.004)
        depth=self.s['ink_surface_z_m']-point[2] if inside else 0.
        self.maximum_depth=max(self.maximum_depth,depth)
        valid=inside and depth>=self.s['dip_required_depth_m'] and point[2]>self.s['ink_bottom_z_m']+.002
        self.dwell=self.dwell+dt if valid else 0.;self.longest=max(self.longest,self.dwell)
        if self.dwell>=self.s['dip_required_dwell_s']:
            self.loaded=True;self.remaining=self.s['ink_capacity_m']
        if self.loaded and point[2]>self.s['ink_surface_z_m']+.01:self.withdrawn=True
        return depth
    def mark(self,force,point):
        active=self.loaded and self.withdrawn and self.remaining>0 and force>.01
        if active and self.previous is not None:self.remaining=max(0.,self.remaining-float(np.linalg.norm(point[:2]-self.previous[:2])))
        self.previous=point.copy() if active else None
        return bool(active)


def recognize(public,run,m,d):
    # No title/poem argument exists on this boundary. Only the camera's pixels enter OCR.
    image_path=run/'camera_rgb.png'
    with mujoco.Renderer(m,width=1600,height=1200) as renderer:
        renderer.update_scene(d,camera='overhead');Image.fromarray(renderer.render()).save(image_path)
        renderer.enable_depth_rendering();renderer.update_scene(d,camera='overhead');np.save(run/'camera_depth_m.npy',renderer.render())
    ocr_python=os.environ.get('FIVEFINGER_OCR_PYTHON',str(ROOT/'.venv/Scripts/python.exe'))
    subprocess.run([ocr_python,'-m','experiments.ocr_camera_worker',str(image_path),str(run/'ocr_result.json')],cwd=ROOT,check=True)
    observed=json.loads((run/'ocr_result.json').read_text(encoding='utf-8'))
    candidates=[x for x in observed['detections'] if re.fullmatch(r'[\u4e00-\u9fff]{2,8}',x['text']) and x['confidence']>=.85]
    if len(candidates)!=1:raise RuntimeError(f'OCR needs one confident Chinese title, got {candidates}')
    title=candidates[0]['text']
    # A known poem library is required by the published title-to-whole-poem task; lookup follows OCR.
    bank=json.loads((ROOT/'config/example_poems.json').read_text(encoding='utf-8'))
    if title not in bank:raise RuntimeError(f'Unsupported visually recognized title: {title}')
    poem=bank[title]
    (run/'recognized.json').write_text(json.dumps({'title':title,'poem':poem,'confidence':candidates[0]['confidence'],
        'source':'camera_rgb -> image-only OCR -> exact title lookup','camera_sha256':observed['input_image_sha256']},ensure_ascii=False,indent=2),encoding='utf-8')
    speech=f'识别到古诗，{title}。'+ '。'.join(poem[i:i+5] for i in range(0,len(poem),5))+'。'
    (run/'spoken_text.txt').write_text(speech,encoding='utf-8')
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/speak_recognized.ps1'),
                    '-InputTextPath',str(run/'spoken_text.txt'),'-OutputWavePath',str(run/'announcement.wav')],check=True)
    with wave.open(str(run/'announcement.wav'),'rb') as audio:spoken_s=audio.getnframes()/audio.getframerate()
    print(json.dumps({'vision_recognized_title':title,'confidence':candidates[0]['confidence'],'characters_to_write':len(poem)},ensure_ascii=False),flush=True)
    return title,poem,observed['elapsed_wall_s'],spoken_s


def run_case(public,profile=None,settings=None,viewer=False):
    """Run the legacy blind-poem controller.

    ``settings`` defaults to the frozen single-A3 baseline. The 2026 final
    wrapper passes the four-A3 2x2 scene explicitly; old callers are unchanged.
    """
    public=Path(public).resolve();run=public.parent/'run';run.mkdir(exist_ok=True)
    manifest_path=public/'controller_manifest.json'
    manifest_verified=False
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for name,digest in manifest.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT.resolve()):raise ValueError('Invalid manifest path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Controller changed after random draw: {name}')
        manifest_verified=True
    if settings is None:
        settings=json.loads((ROOT/'config/competition_scene.json').read_text(encoding='utf-8')) if profile is None else None
    if profile is not None:
        profile_path=(ROOT/profile).resolve()
        if not profile_path.is_relative_to((ROOT/'config').resolve()):
            raise ValueError('Profile must be inside project config directory')
        key=profile_path.relative_to(ROOT).as_posix()
        if not manifest_verified or key not in manifest:
            raise ValueError('Profile must be frozen in the pre-draw manifest')
        settings=json.loads(profile_path.read_text(encoding='utf-8'))
    events=[];access=[]
    def deny_answer(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(args[0])
            if 'examiner_private' in name:raise PermissionError('Controller cannot read examiner answer')
            if any(v in name for v in ('scene.json','screen_pixels.png','camera_rgb.png','ocr_result.json','example_poems.json')):access.append(name)
    sys.addaudithook(deny_answer)
    m,d,center,R,s,cfg=make_scene(public,settings=settings);guard=TableClearance(m);guard.require_clear(d)
    viewer_window=None;viewer_start=0.
    if viewer:
        from mujoco import viewer as mj_viewer
        viewer_window=mj_viewer.launch_passive(m,d)
        viewer_window.cam.lookat[:]=[.06,0,.30]
        viewer_window.cam.distance,viewer_window.cam.azimuth,viewer_window.cam.elevation=1.7,130,-28
        viewer_start=time.perf_counter()

    title,poem,ocr_s,speech_s=recognize(public,run,m,d)
    # Development-only smoke runs may physically execute a prefix of the poem.
    # Official final rounds never set max_characters and always write full text.
    max_chars=int(s.get('max_characters',0) or 0)
    if max_chars and not 1<=max_chars<=len(poem):raise ValueError('max_characters must be between 1 and recognized poem length')
    write_text=poem[:max_chars] if max_chars else poem
    observe_s=max(1.,ocr_s);grasp_start=float(np.ceil((observe_s+speech_s)*100)/100)
    events.append({'event':'vision_recognized','sim_s':observe_s,'title':title,'path_exists_before_ocr':False})
    events.append({'event':'announcement','start_s':observe_s,'duration_s':speech_s})
    path=make_path(write_text,origin_xy=s['origin_xy_m'],character_size=s['character_size_m'],gap=s['character_gap_m'],
        landing_speed=s.get('landing_speed_m_s'),landing_band=s.get('landing_band_m',.002),
                   paper_z=.0019,clearance=s['clearance_m'],speed=s['writing_speed_m_s'],travel_speed=s['travel_speed_m_s'],columns=s['columns'],continuous_strokes=s.get('continuous_strokes',False),stroke_timing=s.get('stroke_timing','quintic'))
    low=np.asarray(s['paper_center_xy_m'])-np.asarray(s['paper_size_m'])/2+.005
    high=np.asarray(s['paper_center_xy_m'])+np.asarray(s['paper_size_m'])/2-.005
    if (path.xyz[:,:2]<low).any() or (path.xyz[:,:2]>high).any():raise RuntimeError('Writing plan outside A3 margins')
    (run/'planned_text.txt').write_text(write_text,encoding='utf-8')
    grip=unused_finger_posture(cfg['grip_targets_rad'],s.get('unused_finger_flexion_rad'))
    cfg['grip_targets_rad']=grip.copy()
    initial={k:float(d.qpos[m.joint(k).qposadr[0]]) for k in grip}
    home=d.qpos[:6].copy();lift=solve_arm_ik(m,d,d.site('palm_control').xpos+[0,0,.12],R,iterations=400)
    if not lift.converged:raise RuntimeError('Safe lift IK failed')
    servo=ContactGripServo(m,target_n=s['grip_force_targets_N'],gain_m_per_ns=.008);ink=InkState(s)
    maintenance=s.get('grip_maintenance',False)
    normal_maintenance_mode=s.get('normal_maintenance_mode','legacy')
    if normal_maintenance_mode not in ('legacy','three_finger_bidirectional'):
        raise ValueError('Invalid normal maintenance mode')
    if normal_maintenance_mode!='legacy' and not (maintenance and s.get('measured_grip_contact_jacobian',False) and s.get('grip_antiwindup',False)):
        raise ValueError('Three-finger maintenance requires measured contact feedback and target-bound antiwindup')
    maintenance_gain=float(s.get('grip_maintenance_gain',.001))
    if not np.isfinite(maintenance_gain) or not 0 < maintenance_gain <= .008:
        raise ValueError('Maintenance gain outside bounded development range')
    filtered_grip=None
    closure=ContactAwareClosure() if s.get('contact_limited_closure',False) else None
    force_peak_monitor=ForcePeakMonitor(m)
    cartesian_pickup=bool(s.get('cartesian_pickup',False));pickup_anchor=None;pickup_rotation=None
    pickup_control_point=s.get('pickup_control_point','nib')
    if pickup_control_point not in ('nib','palm'):raise ValueError('Invalid pickup control point')
    tracking_pause_seconds=0.
    retention=RetentionEnvelope();retention_recovery_seconds=0.;retention_rejected_steps=0;retention_rate_limited_steps=0
    allow_retained_recovery=bool(s.get('retained_pen_recovery',True))
    pose_gain=float(s.get('grasp_pose_stabilization_gain_s',0.))
    posture_gain=float(s.get('grasp_posture_gain_s',0.))
    if not np.isfinite(posture_gain) or not 0<=posture_gain<=.1 or (posture_gain and not pose_gain):
        raise ValueError('Posture maintenance requires pose control and gain 0..0.1')
    pose_reserve=float(s.get('grasp_pose_reserve_rad',0.))
    if not np.isfinite(pose_reserve) or not 0<=pose_reserve<=.1 or (pose_reserve and not pose_gain):
        raise ValueError('Pose reserve requires pose control and range 0..0.1 rad')
    coordinate_grasp=bool(s.get('coordinated_grasp',False))
    if coordinate_grasp and (normal_maintenance_mode!='three_finger_bidirectional' or not s.get('maintenance_lock_spread',False) or posture_gain or pose_reserve):
        raise ValueError('Coordinated grasp requires three-finger feedback, retained normal spread and no independent pose options')
    coordinated_servo=CoordinatedGraspServo(m,pose_gain,float(s.get('coordinated_posture_weight_m',.003))) if coordinate_grasp else None
    pose_servo=InHandPoseServo(m,pose_gain,s.get('grasp_pose_orientation_mode','full'),posture_gain,pose_reserve,s.get('grasp_pose_lock_spread',False)) if pose_gain and not coordinate_grasp else None
    pose_rate_limited_steps=0;pose_maximum_delta_rate=0.
    orientation=R.copy()
    reserve_fraction=float(s.get('pinch_reserve_fraction',.65))
    if not .3<=reserve_fraction<=1.:
        raise ValueError('Pinch reserve fraction outside bounded range')
    maintenance_targets=validated_grip_targets(s.get('pinch_maintenance_targets_N',np.asarray(s['grip_force_targets_N'])*reserve_fraction))
    middle_support=s.get('middle_support_target_N')
    if middle_support is not None:
        middle_support=float(middle_support)
        if not np.isfinite(middle_support) or not .1<=middle_support<=.5:
            raise ValueError('Middle support target must be 0.1..0.5 N in development')
        if not maintenance or not s.get('pinch_reserve_only',False):
            raise ValueError('Middle support requires pinch maintenance')
        maintenance_targets[2]=middle_support
    allocate_load=bool(s.get('contact_load_allocation',False))
    if allocate_load and normal_maintenance_mode!='three_finger_bidirectional':
        raise ValueError('Load allocation requires bidirectional three-finger maintenance')
    load_allocator=FingerLoadAllocator(maintenance_targets,s['grip_force_targets_N']) if allocate_load else None
    observed_contact=None
    normal_damping_s=float(s.get('normal_motion_damping_s',0.))
    normal_damping_correction(.002,0.,normal_damping_s)  # Validate before stepping.
    velocity_ff_scale=float(s.get('velocity_feedforward_scale',0.))
    if not np.isfinite(velocity_ff_scale) or not 0<=velocity_ff_scale<=1.:
        raise ValueError('Invalid reference velocity feedforward scale')
    velocity_ff_axes=s.get('velocity_feedforward_axes','xyz')
    if velocity_ff_axes not in ('xyz','normal','tangent'):
        raise ValueError('Unknown reference velocity feedforward axes')
    reference_velocity=ReferenceVelocity()
    command_clock=CommandClock(s['command_update_hz'],d.ctrl) if 'command_update_hz' in s else None
    lift_gate=LiftGripGate(thumb_n=s.get('lift_gate_thumb_n',1.5),index_n=s.get('lift_gate_index_n',.6))
    frozen_arm=None
    state='wait_vision';entered=0.;anchor=nib_point(m,d);path_index=0;touchdown_wait=0.;lost=0.
    hold=0.;longest_hold=0.;hold_passed=False;previous_relative=None;done_at=None;reason='time_limit';touched_brush_evidenced=False
    rows=[];states=[];velocities=[];controls=[];inks=[];stages=[];targets=[];indices=[];contact_trace=[];minimum_clearance=10.;normal_offset=0.;filtered_normal=0.
    contact_snapshots=[];actuator_forces=[];grip_target_trace=[]
    robot_surface_monitor=RobotSurfaceMonitor(m)
    surface_filter=None
    if s.get('surface_command_filter',False):
        if not s.get('tool_point_jacobian',False):raise ValueError('Surface filter requires held-tool point Jacobian')
        paper=d.geom('paper')
        if not np.allclose(paper.xmat.reshape(3,3)[:,2],[0,0,1],atol=1e-8):
            raise ValueError('Surface filter currently requires a horizontal paper plane')
        plane_z=paper.xpos[2]+m.geom('paper').size[2]
        surface_filter=SurfaceCommandFilter(m,guard.entries,plane_z,s.get('surface_clearance_floor_m',.005),s.get('surface_actual_guard_m',.002))
    def transition(next_state):
        nonlocal state,entered,anchor
        events.append({'event':next_state,'sim_s':float(d.time)})
        state=next_state;entered=float(d.time);anchor=nib_point(m,d).copy()
    try:
        for tick in range(int(s['round_limit_s']*100)):
            t=d.time;g=t-grasp_start;elapsed=t-entered
            previous_applied_command=d.ctrl.copy()
            force,_=brush_contacts(m,d)
            touched_brush_evidenced |= bool(np.any(force>.01))
            healthy_pinch=bool(force[0]>.01 and np.any(force[1:]>.01))
            hand=d.body('right_hand_base');pen_state=d.body('brush');hand_r=hand.xmat.reshape(3,3)
            retained=retention.observe(hand_r.T@(pen_state.xpos-hand.xpos),hand_r.T@pen_state.xmat.reshape(3,3)[:,2],healthy_pinch)
            retained=retained and allow_retained_recovery
            tracking_paused=g>=7 and not healthy_pinch
            if tracking_paused:tracking_pause_seconds+=.01
            execute_stage=None;nominal_reference=None
            close=float(smoothstep5((g-1)/3))
            progress=closure.update(close,force,d.time) if closure is not None else np.full(5,close)
            cmd={k:initial[k]+progress[int(k.split('_')[2][-1])-1]*(v-initial[k]) for k,v in grip.items()}
            if g>=3.5:
                previous_grip_delta=servo.delta.copy()
                if tracking_paused and retained:
                    # Arm motion remains paused; only a nearby pen may receive
                    # a slow, bounded closing correction. Never chase a fall.
                    adjustment=servo.update(d,force,.01,gain=.001,deadband_n=.1,close_only=True,measured_contact=True,base_targets=cmd,lock_spread=coordinate_grasp)
                    servo.delta,limited=bound_combined_delta(previous_grip_delta,servo.delta,.01)
                    adjustment=servo.delta.copy();retention_rate_limited_steps+=int(limited)
                    retention_recovery_seconds+=.01
                elif tracking_paused:
                    adjustment=servo.delta
                    retention_rejected_steps+=1
                elif g<14:
                    adjustment=servo.update(d,force,.01)
                    filtered_grip=force.copy()
                elif maintenance:
                    filtered_grip += .01/(.1+.01)*(force-filtered_grip)
                    if normal_maintenance_mode=='three_finger_bidirectional':
                        if load_allocator is not None and observed_contact is not None:
                            gravity=np.r_[m.body_mass[m.body('brush').id]*m.opt.gravity,np.zeros(3)]
                            environment=observed_contact['wrench_world'][3:].sum(axis=0)
                            servo.target_n=load_allocator.update(observed_contact['normal_N'][:3],
                                observed_contact['wrench_world'][:3],gravity+environment,.01)
                        else:servo.target_n=maintenance_targets
                        adjustment=servo.update(d,filtered_grip,.01,gain=maintenance_gain,deadband_n=.1,
                            fingers=(1,2,3),close_only=False,measured_contact=True,base_targets=cmd,
                            lock_spread=s.get('maintenance_lock_spread',False))
                    elif s.get('pinch_reserve_only',False):
                        servo.target_n=maintenance_targets
                        adjustment=servo.update(d,filtered_grip,.01,gain=maintenance_gain,deadband_n=.1,fingers=(1,2,3) if middle_support is not None else (1,2),close_only=True,measured_contact=s.get('measured_grip_contact_jacobian',False),base_targets=cmd if s.get('grip_antiwindup',False) else None,lock_spread=s.get('maintenance_lock_spread',False))
                    else:
                        adjustment=servo.update(d,filtered_grip,.01,gain=maintenance_gain,deadband_n=.1)
                else:
                    adjustment=servo.delta
                if coordinated_servo is not None and g>=14 and not tracking_paused:
                    coordinated_servo.update(d,servo,cmd,previous_grip_delta,.01)
                    adjustment=servo.delta.copy()
                elif pose_servo is not None and g>=14 and not tracking_paused:
                    pose_servo.update(d,servo,cmd,.01)
                    servo.delta,limited=bound_combined_delta(previous_grip_delta,servo.delta,.01)
                    pose_rate_limited_steps+=int(limited)
                    pose_maximum_delta_rate=max(pose_maximum_delta_rate,max(abs(servo.delta[n]-previous_grip_delta[n])/.01 for n in servo.delta))
                    adjustment=servo.delta.copy()
                for k,v in adjustment.items():cmd[k]+=v
            set_joint_targets(m,d,cmd)
            d.ctrl[:6]=home+float(smoothstep5((g-7)/3))*(lift.joints-home)
            target=nib_point(m,d).copy()
            if state=='wait_vision' and t>=observe_s:transition('announce')
            if state=='announce' and t>=grasp_start:transition('close')
            if state=='close' and g>=7:transition('lift')
            if state=='lift' and g>=10:transition('hold')
            if state=='hold' and g>=14:
                if not hold_passed:reason='hold_gate_failed';break
                transition('to_ink')
            elapsed=t-entered
            if cartesian_pickup and state in ('lift','hold'):
                if pickup_anchor is None:
                    pickup_anchor=d.site('palm_control').xpos.copy() if pickup_control_point=='palm' else anchor.copy()
                    pickup_rotation=d.site('palm_control').xmat.reshape(3,3).copy()
                target=pickup_anchor+np.array([0.,0.,.12*(float(smoothstep5(elapsed/3)) if state=='lift' else 1.)])
            elif state=='to_ink':
                target=anchor.copy();target[:2]+=float(smoothstep5(elapsed/4))*(np.asarray(s['ink_center_xy_m'])-anchor[:2])
                if elapsed>=4:transition('dip_lower')
            elif state=='dip_lower':
                target=anchor.copy();target[:2]=s['ink_center_xy_m'];target[2]+=float(smoothstep5(elapsed/4))*(s['ink_surface_z_m']-s['dip_target_depth_m']-anchor[2])
                if elapsed>=4:transition('dip_dwell')
            elif state=='dip_dwell':
                target=np.r_[s['ink_center_xy_m'],s['ink_surface_z_m']-s['dip_target_depth_m']]
                if ink.loaded:transition('dip_lift')
                elif elapsed>4:reason='dip_gate_failed';break
            elif state=='dip_lift':
                target=anchor.copy();target[2]+=float(smoothstep5(elapsed/3))*(.16-anchor[2])
                if elapsed>=3:
                    if not ink.withdrawn:reason='ink_not_withdrawn';break
                    transition('to_paper')
            elif state=='to_paper':
                target=anchor.copy();target[:2]+=float(smoothstep5(elapsed/4))*(path.xyz[0,:2]-anchor[:2])
                if elapsed>=4:transition('paper_lower')
            elif state=='paper_lower':
                target=path.xyz[0].copy();target[2]=anchor[2]+float(smoothstep5(elapsed/4))*(path.xyz[0,2]-anchor[2])
                if elapsed>=4:transition('execute')
            elif state=='execute':
                idx=path_index;target=path.xyz[idx].copy();down=bool(path.pen_down[idx]);force,_=paper_contact(m,d)
                starts=down and idx>0 and not path.pen_down[idx-1]
                starts_lift=not down and idx>0 and bool(path.pen_down[idx-1])
                inhibit=tracking_paused or (s.get('lift_grip_interlock',False) and not lift_gate.allow(starts_lift,brush_contacts(m,d)[0],.01))
                if inhibit:
                    if frozen_arm is None:
                        frozen_arm=d.qpos[:6].copy()
                        events.append({'event':'grip_reserve_wait','sim_s':float(d.time),'path_index':idx})
                    execute_stage='grip_recovery'
                else:
                    if frozen_arm is not None:events.append({'event':'grip_reserve_restored','sim_s':float(d.time),'path_index':idx})
                    frozen_arm=None
                if inhibit:pass
                elif (starts or (down and s.get('recover_contact_midstroke',False))) and force<.03:
                    execute_stage='touchdown'
                    touchdown_wait+=.01
                    if touchdown_wait>2.:reason='touchdown_timeout';break
                else:
                    execute_stage='write' if down else 'pen_up'
                    touchdown_wait=0.;path_index=min(idx+1,len(path.xyz)-1)
                # Feedforward follows the planned motion, not the derivative
                # of the pressure feedback correction (which can amplify noise).
                nominal_reference=target.copy()
                # Small normal-force correction, only while drawing.
                filtered_normal+=.01/(s['normal_force_filter_s']+.01)*(force-filtered_normal)
                if down:normal_offset=float(np.clip(normal_offset+.01*s['normal_admittance_m_Ns']*(filtered_normal-s['normal_force_target_N']),-.0008,.001))
                else:normal_offset*=.95
                target[2]+=normal_offset
                if down and s.get('normal_velocity_feedback',False):
                    # Increment about the measured tip, avoiding accumulated
                    # position error when the compliant grasp changes tool length.
                    velocity=float(np.clip(.04*(filtered_normal-s['normal_force_target_N']),-.006,.006))
                    target[2]=float(np.clip(nib_point(m,d)[2]+.01*velocity,.0011,.004))
                if idx==len(path.xyz)-1 and not inhibit:transition('done');done_at=t
            elif state=='done':
                target=anchor.copy();target[2]+=.025*float(smoothstep5(elapsed/2))
            if state in ('to_ink','dip_lower','dip_dwell','dip_lift','to_paper','paper_lower','execute','done') or (cartesian_pickup and state in ('lift','hold')):
                pitch=s['writing_wrist_pitch_deg'] if state in ('paper_lower','execute','done') else (s['writing_wrist_pitch_deg']*float(smoothstep5(elapsed/4)) if state=='to_paper' else 0.)
                writing_R=Rotation.from_euler('y',pitch,degrees=True).as_matrix()@R
                palm_pickup=cartesian_pickup and pickup_control_point=='palm' and state in ('lift','hold')
                if palm_pickup:
                    # The palm pickup owns orientation. Do not accumulate an
                    # inactive upright target and apply it at mode handoff.
                    orientation=pickup_rotation.copy()
                elif s.get('upright_tool_feedback',False):
                    desired=upright_rotation(d)
                    correction=Rotation.from_matrix(desired@orientation.T).as_rotvec()
                    correction *= min(1.,.001/max(np.linalg.norm(correction),1e-12))
                    candidate=Rotation.from_rotvec(correction).as_matrix()@orientation
                    excursion=Rotation.from_matrix(candidate@R.T).as_rotvec()
                    excursion *= min(1.,np.radians(30)/max(np.linalg.norm(excursion),1e-12))
                    orientation=Rotation.from_rotvec(excursion).as_matrix()@R
                    writing_R=orientation
                paper_phase=state in ('paper_lower','execute','done')
                ff_target=nominal_reference if s.get('feedforward_nominal_path_only',False) and nominal_reference is not None else target
                ref_velocity=velocity_ff_scale*reference_velocity.update(ff_target,state,.01) if velocity_ff_scale and paper_phase else None
                if ref_velocity is not None:
                    if velocity_ff_axes=='normal':ref_velocity[:2]=0.
                    elif velocity_ff_axes=='tangent':ref_velocity[2]=0.
                if palm_pickup:writing_R=pickup_rotation
                command_nib(m,d,target,writing_R,point_jacobian=s.get('tool_point_jacobian',False),normal_damping_s=normal_damping_s if paper_phase else 0.,target_velocity=ref_velocity,palm_reference=palm_pickup)
            if frozen_arm is not None:d.ctrl[:6]=frozen_arm
            if tracking_paused:
                # A free pen is no longer a valid arm tracking reference. Hold
                # the arm while the unchanged 0.3 s loss timer runs.
                d.ctrl[:6]=d.qpos[:6]
                if not retained:d.ctrl[6:]=previous_applied_command[6:]
            if surface_filter is not None and state in ('paper_lower','execute','done') and not tracking_paused and frozen_arm is None:
                d.ctrl[:6]=surface_filter.apply(d,d.ctrl[:6],nib_point(m,d))
            pending_command=d.ctrl.copy()
            for _ in range(5):
                if command_clock is not None:d.ctrl[:]=command_clock.update(d.time,pending_command)
                d.qfrc_applied[:6]=np.clip(d.qfrc_bias[:6],-40,40)
                if np.any(d.qfrc_applied[26:]!=0.):raise RuntimeError('Unexpected external pen force')
                mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                force_peak_monitor.update(d,execute_stage if state=='execute' and execute_stage else state)
                robot_surface_monitor.update(d,execute_stage if state=='execute' and execute_stage else state)
                minimum_clearance=min(minimum_clearance,guard.require_clear(d))
                if surface_filter is not None and state in ('paper_lower','execute','done'):
                    surface_filter.require_actual_clear(d)
            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('MuJoCo numerical warning')
            if viewer_window is not None:
                if not viewer_window.is_running():
                    reason='viewer_closed';break
                viewer_window.sync()
                # Keep the development window near simulated real time; the
                # official headless run does not enter this branch.
                wall=time.perf_counter()-viewer_start
                if d.time>wall:time.sleep(d.time-wall)

            forces,support=brush_contacts(m,d);pen=d.body('brush');point=nib_point(m,d);paper_n,contact=paper_contact(m,d)
            observed_contact=observe_brush_contacts(m,d)
            contact_trace.append([observed_contact['wrench_world'],observed_contact['normal_N'],observed_contact['position_brush_m'],observed_contact['hand_axial_span_m'],observed_contact['brush_axis_world'],observed_contact['axial_relative_velocity_m_s'],observed_contact['tangential_relative_speed_m_s'],observed_contact['friction_utilization_max']])
            depth=ink.update(point,.01);mark=ink.mark(paper_n,contact)
            palm=d.body('right_hand_base');relative=palm.xmat.reshape(3,3).T@(pen.xpos-palm.xpos)
            slip=0. if previous_relative is None else float(np.linalg.norm(relative-previous_relative)/.01);previous_relative=relative.copy()
            gripping=forces[0]>.01 and np.any(forces[1:]>.01)
            valid=gripping and support<.01 and pen.xpos[2]-center[2]>.02 and slip<.02
            hold=hold+.01 if valid else 0.;longest_hold=max(longest_hold,hold);hold_passed|=hold>=3
            lost=lost+.01 if g>=7 and not gripping else 0.
            tilt=float(np.degrees(np.arccos(np.clip(pen.xmat.reshape(3,3)[2,2],-1,1))))
            stage=state
            if state=='execute':stage=execute_stage or 'pen_up'
            actuator_forces.append(d.actuator_force.copy())
            grip_target_trace.append(servo.target_n.copy())
            if tick%20==0:
                contact_snapshots.append({'time_s':float(d.time),'stage':stage,
                    'brush_com_world_m':d.xipos[pen.id].tolist(),
                    'brush_rotation_world':pen.xmat.reshape(3,3).tolist(),
                    'gravity_force_world_N':(m.body_mass[pen.id]*m.opt.gravity).tolist(),
                    'contacts':observed_contact['details']})
            rows.append([d.time,*pen.xpos,*forces,support,tilt,*point,paper_n,*contact,slip,float(valid),minimum_clearance,depth,ink.dwell,ink.remaining])
            states.append(d.qpos.copy());velocities.append(d.qvel.copy());controls.append(d.ctrl.copy());stages.append(stage);targets.append(target);indices.append(path_index)
            inks.append([mark,*contact,paper_n])
            if tick%500==0:print(json.dumps({'t':round(d.time,2),'stage':stage,'tilt':round(tilt,1),'fingers':forces[:3].round(2).tolist(),'paper_N':round(paper_n,3),'depth_mm':round(depth*1000,2),'arm_clear_mm':round(minimum_clearance*1000,2),'path':path_index}),flush=True)
            if lost>.3:reason='grip_lost';break
            if tilt>60 and g>14:reason='pen_tilt_exceeded';break
            if done_at is not None and t-done_at>=3.:reason='completed';break
    except RuntimeError as error:reason=str(error)
    if viewer_window is not None:
        try:viewer_window.close()
        except Exception:pass

    save_final_physics_state(run/'final_physics_state.npz',m,d,robot_surface_monitor.steps,rows[-1][0] if rows else 0.)
    final_forces,_=brush_contacts(m,d)
    final_gripping=bool(final_forces[0]>.01 and np.any(final_forces[1:]>.01))
    surface_report=robot_surface_monitor.report()
    (run/'robot_surface_contacts_500hz.json').write_text(json.dumps(surface_report,indent=2)+'\n',encoding='utf-8')
    rows=np.asarray(rows);stages=np.asarray(stages);targets=np.asarray(targets);inks=np.asarray(inks)
    np.savez_compressed(run/'replay.npz',qpos=states,qvel=velocities,ctrl=controls,time_s=rows[:,0],ink=inks,stage=stages,target=targets,path_index=indices)
    np.savez_compressed(run/'actuator_trace.npz',time_s=rows[:,0],force=actuator_forces,grip_normal_targets_N=grip_target_trace,
        names=np.asarray([m.actuator(i).name for i in range(m.nu)]),
        force_range=m.actuator_forcerange,force_limited=m.actuator_forcelimited,
        joint_ids=m.actuator_trnid[:,0],gear=m.actuator_gear)
    (run/'contact_geometry_samples.json').write_text(json.dumps({'sample_rate_hz':5,
        'source':'runtime contacts, not replay force reconstruction','samples':contact_snapshots},
        ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(run/'contact_trace.npz',schema_version=2,groups=np.asarray(GROUPS),time_s=rows[:,0],wrench_world=np.asarray([x[0] for x in contact_trace]),normal_N=np.asarray([x[1] for x in contact_trace]),position_brush_m=np.asarray([x[2] for x in contact_trace]),hand_axial_span_m=np.asarray([x[3] for x in contact_trace]),brush_axis_world=np.asarray([x[4] for x in contact_trace]),axial_relative_velocity_m_s=np.asarray([x[5] for x in contact_trace]),tangential_relative_speed_m_s=np.asarray([x[6] for x in contact_trace]),friction_utilization_max=np.asarray([x[7] for x in contact_trace]))
    np.savetxt(run/'physics.csv',rows,delimiter=',',comments='',header='time,x,y,z,thumb,index,middle,ring,little,support,tilt,nib_x,nib_y,nib_z,paper_N,contact_x,contact_y,contact_z,relative_slip,valid_hold,minimum_arm_clearance,ink_depth,ink_dwell,ink_remaining')
    down=stages=='write';error=np.linalg.norm(targets[down,:2]-rows[down,11:13],axis=1)
    boundaries=np.flatnonzero(np.diff(np.r_[False,path.pen_down].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,path.pen_down,False].astype(int))==-1)
    written_strokes=int(np.sum(ends<=path_index))
    report={'recognized_title':title,'recognized_poem':poem,'stop_reason':reason,'elapsed_sim_s':float(d.time),
            'final_physics_state_saved':True,'partial_final_control_steps':robot_surface_monitor.steps-5*len(rows),
            'surface_command_filter':surface_filter.report() if surface_filter is not None else None,
            'robot_non_tool_surface_contact_free_500hz':surface_report['robot_non_tool_surface_contact_free'],
            'hold_3s_passed':bool(hold_passed),'longest_hold_s':longest_hold,'touched_brush_evidenced':touched_brush_evidenced,
            'ink_loaded':ink.loaded,'ink_withdrawn':ink.withdrawn,
            'dip_maximum_depth_mm':ink.maximum_depth*1000,'dip_longest_dwell_s':ink.longest,
            'planned_characters':len(write_text),'planned_strokes':len(boundaries),'completed_strokes':written_strokes,
            'path_completed_fraction':path_index/max(1,len(path.xyz)-1),'writing_ink_contact_fraction':float(np.mean(inks[down,0])) if down.any() else 0.,
            'xy_error_median_mm':float(np.median(error)*1000) if down.any() else None,
            'all_pen_down_contact_fraction':float(np.mean(inks[np.isin(stages,['write','touchdown']),0])) if np.isin(stages,['write','touchdown']).any() else 0.,
            'contact_recovery_seconds':float(np.sum(stages=='touchdown')*.01),
            'continuous_strokes':s.get('continuous_strokes',False),
            'minimum_moving_arm_table_clearance_mm':min(minimum_clearance,guard.minimum_arm)*1000,'minimum_including_fixed_mount_base_mm':guard.minimum*1000,
            'arm_table_collision_enabled':True,'arm_table_mesh_audit_rate_hz':500,'pen_held_at_end':final_gripping,
            'initial_answer_known_to_controller':False,'recognition_input':'1600x1200 MuJoCo overhead RGB image',
            'controller_frozen_before_draw':manifest_verified,
            'private_answer_reads':0,'pen_pose_overwrites_after_initialization':False,'pen_external_force':False,
            'ink_model':'depth/dwell loading and contact-distance consumption, not fluid or bristle dynamics',
            'robot_state_feedback':'simulation ground truth; OCR receives camera pixels only',
            'grip_control':('bounded filtered force maintenance; simulation contact feedback; no weld' if maintenance else 'force acquisition then fixed finger targets after 3 s hold; no weld'),
            'grip_maintenance':maintenance,'grip_maintenance_gain':maintenance_gain,
            'contact_limited_closure':closure is not None,
            'cartesian_pickup':cartesian_pickup,
            'pickup_control_point':pickup_control_point,
            'pickup_target_semantics':'palm during lift/hold; nib otherwise' if cartesian_pickup and pickup_control_point=='palm' else 'nib',
            'early_tracking_loss_gate_enabled':True,'tracking_pause_seconds':tracking_pause_seconds,
            'retention_recovery_seconds':retention_recovery_seconds,'retention_rejected_steps':retention_rejected_steps,
            'retained_pen_recovery':allow_retained_recovery,'retention_rate_limited_steps':retention_rate_limited_steps,
            'retention_maximum_missing_distance_mm':retention.maximum_missing_distance_m*1000,
            'retention_maximum_missing_axis_deg':retention.maximum_missing_axis_deg,
            'closure_nominal_progress_at_contact':closure.progress[:3].tolist() if closure is not None else None,
            'closure_first_contact_s':[float(v) if np.isfinite(v) else None for v in closure.first_contact_s] if closure is not None else None,
            'force_peak_audit_rate_hz':1./m.opt.timestep,
            'normal_maintenance_mode':normal_maintenance_mode,
            'contact_load_allocation':load_allocator.report() if load_allocator else None,
            'coordinated_grasp':coordinated_servo.report() if coordinated_servo else None,
            'contact_geometry_sample_rate_hz':5,
            'actuator_force_sample_rate_hz':100,
            'tool_point_jacobian':s.get('tool_point_jacobian',False),'upright_tool_feedback':s.get('upright_tool_feedback',False),
            'normal_velocity_feedback':s.get('normal_velocity_feedback',False),
            'normal_motion_damping_s':normal_damping_s,
            'velocity_feedforward_scale':velocity_ff_scale,
            'velocity_feedforward_axes':velocity_ff_axes,
            'feedforward_nominal_path_only':s.get('feedforward_nominal_path_only',False),
            'pinch_reserve_only':s.get('pinch_reserve_only',False),
            'pinch_reserve_fraction':reserve_fraction,
            'pinch_maintenance_targets_N':maintenance_targets.tolist(),
            'middle_support_target_N':middle_support,
            'maintenance_lock_spread':s.get('maintenance_lock_spread',False),
            'grasp_pose_stabilization_gain_s':pose_gain,
            'grasp_pose_orientation_mode':s.get('grasp_pose_orientation_mode','full'),
            'grasp_pose_maximum_position_error_mm':pose_servo.maximum_position_error_m*1000 if pose_servo else None,
            'grasp_pose_maximum_rotation_error_deg':float(np.degrees(pose_servo.maximum_rotation_error_rad)) if pose_servo else None,
            'grasp_pose_active_updates':pose_servo.active_updates if pose_servo else 0,
            'grasp_posture_gain_s':posture_gain,
            'grasp_pose_reserve_rad':pose_reserve,
            'grasp_pose_lock_spread':s.get('grasp_pose_lock_spread',False),
            'grasp_pose_reserve_limited_updates':pose_servo.reserve_limited_updates if pose_servo else 0,
            'grasp_pose_minimum_reserve_scale':pose_servo.minimum_reserve_scale if pose_servo else None,
            'grasp_pose_combined_offset_rate_limit_rad_s':.12 if pose_servo else None,
            'grasp_pose_maximum_combined_offset_rate_rad_s':pose_maximum_delta_rate if pose_servo else None,
            'grasp_pose_rate_limited_control_steps':pose_rate_limited_steps,
            'unused_finger_flexion_rad':s.get('unused_finger_flexion_rad'),
            'contact_wrench_recording':'100 Hz runtime contact_trace.npz; world force then torque about brush COM; simulation truth',
            'measured_grip_contact_jacobian':s.get('measured_grip_contact_jacobian',False),
            'grip_antiwindup':s.get('grip_antiwindup',False),
            'lift_grip_interlock':s.get('lift_grip_interlock',False),'grip_recovery_seconds':lift_gate.total_wait,'grip_recovery_count':lift_gate.activations,
            'command_update_hz':s.get('command_update_hz',100),'command_emissions':command_clock.emissions if command_clock is not None else len(rows),
            'contact_parameters_unvalidated_on_hardware':True,'official_hardware_and_bank_confirmed':False,'official_score':None}
    (run/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (run/'contact_force_peaks_500hz.json').write_text(json.dumps(force_peak_monitor.report(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (run/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    (run/'access_audit.json').write_text(json.dumps(access,ensure_ascii=False,indent=2),encoding='utf-8')
    (run/'config.json').write_text(json.dumps({'scene':s,'grip':cfg,'public_directory':str(public)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',required=True);p.add_argument('--profile');a=p.parse_args();result=run_case(a.public,a.profile);raise SystemExit(0 if result['stop_reason']=='completed' else 2)


if __name__=='__main__':main()
