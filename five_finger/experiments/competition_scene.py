"""Published five-finger layout with explicit uncertain dimensions and solid arm collisions."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from experiments.writing import make_writing_model,read_config
from fivefinger.control import solve_arm_ik
from fivefinger.paths import ROOT
from fivefinger.grasp_posture import apply_grasp_offsets


def load_settings():
    return json.loads((ROOT/'config/competition_scene.json').read_text(encoding='utf-8'))


def make_scene(public=None,settings=None,cfg=None,scene_modifier=None):
    settings=load_settings() if settings is None else settings
    # An explicitly supplied cfg is already resolved (e.g. saved replay config).
    if cfg is None:
        cfg=read_config()
        cfg['grip_targets_rad']=apply_grasp_offsets(cfg['grip_targets_rad'],settings.get('grip_target_offsets_rad',{}))
    else:cfg=dict(cfg)
    cfg.pop('text',None)
    cfg['finger_contact_friction']=settings['finger_contact_friction']
    metadata=json.loads((Path(public)/'scene.json').read_text(encoding='utf-8')) if public else {
        'screen_center_m':[.60,.28,.12],'screen_yaw_deg':0,'screen_tilt_deg':35}
    def change(root):
        root.set('model','five_finger_competition_reference')
        root.find('option').set('impratio',str(settings.get('friction_impedance_ratio',1.)))
        root.find('option').set('noslip_iterations',str(settings.get('noslip_iterations',5)))
        world=root.find('worldbody');asset=root.find('asset')
        root.find('./visual/global').set('offheight','1200')
        world.find("body[@name='r_a_base']").set('pos',f"-.35 0 {settings['arm_mount_height_m']}")
        # Paper layout: keep the legacy single-A3 baseline by default. The
        # 2026 final uses four explicit A3 sheets arranged as a 2x2 A1 plane.
        paper=world.find("geom[@name='paper']")
        if settings.get('paper_sheets'):
            world.remove(paper);paper=None
            for index,sheet in enumerate(settings['paper_sheets']):
                center=np.asarray(sheet.get('center_xy_m',settings['paper_center_xy_m']),float)
                size=np.asarray(sheet.get('size_m',settings['paper_size_m']),float)
                if center.shape!=(2,) or size.shape!=(2,) or not np.isfinite(np.r_[center,size]).all():
                    raise ValueError(f'Invalid paper sheet definition at index {index}')
                if min(size)<=0:
                    raise ValueError(f'Paper sheet must have positive size at index {index}')
                ET.SubElement(world,'geom',name=sheet.get('name',f'paper_{index}'),type='box',
                              pos=f'{center[0]} {center[1]} .001',size=f'{size[0]/2} {size[1]/2} .001',
                              rgba='.96 .93 .86 1',friction=' '.join(map(str,cfg['paper_friction'])),priority='1',condim='3',contype='4',conaffinity='3')
        else:
            xy=settings['paper_center_xy_m'];paper.set('pos',f'{xy[0]} {xy[1]} .001')
        if settings.get('dot_target_visible',True) is False:
            dot=world.find("geom[@name='dot_target']")
            if dot is not None:world.remove(dot)
        # Robot/environment masks: arm and palm meshes physically collide with every work surface.
        for geom in world.findall('geom'):
            geom.set('contype','4');geom.set('conaffinity','3')
        for body in root.findall('.//body'):
            seen=set()
            for geom in list(body.findall('geom')):
                key=tuple(sorted(geom.attrib.items()))
                if key in seen:body.remove(geom);continue
                seen.add(key)
                mesh=geom.get('mesh','')
                if mesh.startswith('r_a_link'):
                    geom.set('name',f'collision_{mesh}');geom.set('contype','2');geom.set('conaffinity','4')
                elif mesh=='right_hand_base':
                    geom.set('name','collision_palm');geom.set('contype','2');geom.set('conaffinity','4')
        # Thin mounting plate keeps even the fixed base mesh above the tabletop.
        ET.SubElement(world,'geom',name='arm_mount',type='box',pos='-.36 0 .0003',size='.12 .10 .0003',rgba='.22 .24 .27 1',contype='4',conaffinity='1')
        x,y=settings['ink_center_xy_m'];hx,hy=settings['ink_inner_half_size_m'];bottom=settings['ink_bottom_z_m'];surface=settings['ink_surface_z_m']
        ET.SubElement(world,'geom',name='inkstone_bottom',type='box',pos=f'{x} {y} {bottom/2}',size=f'{hx+.004} {hy+.004} {bottom/2}',rgba='.10 .12 .14 1',contype='4',conaffinity='3')
        for axis in (0,1):
            for sign in (-1,1):
                p=np.array([x,y,(surface+.008)/2]);size=[hx+.004,hy+.004,(surface+.008)/2]
                p[axis]+=sign*([hx,hy][axis]+.002);size[axis]=.002
                ET.SubElement(world,'geom',name=f'inkstone_wall_{axis}_{sign}',type='box',pos=' '.join(map(str,p)),size=' '.join(map(str,size)),rgba='.10 .12 .14 1',contype='4',conaffinity='3')
        ET.SubElement(world,'geom',name='ink_liquid_surface',type='box',pos=f'{x} {y} {surface}',size=f'{hx} {hy} .0001',rgba='.01 .02 .035 .85',contype='0',conaffinity='0')
        if settings.get('task_display_enabled',True):
            screenR=Rotation.from_euler('zy',[metadata['screen_yaw_deg'],0],degrees=True).as_matrix()@Rotation.from_euler('x',metadata['screen_tilt_deg'],degrees=True).as_matrix()
            quat=Rotation.from_matrix(screenR).as_quat()[[3,0,1,2]]
            screen=ET.SubElement(world,'body',name='task_display',pos=' '.join(map(str,metadata['screen_center_m'])),quat=' '.join(map(str,quat)))
            ET.SubElement(screen,'geom',name='screen_bezel',type='box',size='.113 .052 .005',rgba='.08 .12 .15 1',contype='4',conaffinity='3')
            attrs=dict(name='task_screen',type='box',pos='0 0 .0055',size='.105 .044 .0005',contype='0',conaffinity='0')
            if public:
                ET.SubElement(asset,'texture',name='screen_texture',type='2d',file=str((Path(public)/'screen_pixels.png').resolve()))
                ET.SubElement(asset,'mesh',name='screen_quad',inertia='shell',vertex='-.105 -.044 0 .105 -.044 0 .105 .044 0 -.105 .044 0',face='0 1 2 0 2 3',texcoord='0 1 1 1 1 0 0 0')
                ET.SubElement(asset,'material',name='screen_material',texture='screen_texture',texrepeat='1 1',texuniform='false',emission='.6',specular='0')
                attrs.pop('size');attrs.update(type='mesh',mesh='screen_quad',material='screen_material')
            else:attrs['rgba']='.95 .95 .95 1'
            ET.SubElement(screen,'geom',**attrs)
            sx,sy,sz=metadata['screen_center_m'];ET.SubElement(world,'geom',name='screen_stand',type='box',pos=f'{sx} {sy} {sz/2}',size=f'.014 .012 {sz/2}',rgba='.1 .12 .15 1',contype='4',conaffinity='3')
        cam=world.find("camera[@name='overhead']");cam.set('pos',' '.join(map(str,settings['camera_position_m'])));cam.set('fovy',str(settings['camera_fovy_deg']))
        if scene_modifier is not None:scene_modifier(root)
    shift=np.asarray(settings.get('brush_initial_shift_m',[0.,0.,0.]),float)
    if shift.shape!=(3,) or not np.isfinite(shift).all() or np.max(abs(shift))>.005:
        raise ValueError('Development pen perturbation must be within 5 mm per axis')
    m,d,center,R=make_writing_model(cfg=cfg,scene_hook=change,shift=shift)
    for name,value in cfg['grip_targets_rad'].items():
        low,high=m.actuator_ctrlrange[m.actuator('act_'+name).id]
        if not low<=value<=high:raise ValueError(f'Grasp target outside actuator range: {name}')
    # Re-solve after mounting-height change, seeded on the elbow-up branch.
    target=center-R@np.array([.073136,.022,.209114])+R@np.array([0,0,.1])+np.array([0,0,settings.get('grasp_height_offset_m',0.)])
    d.qpos[:6]=settings['elbow_up_seed_rad']
    solved=solve_arm_ik(m,d,target,R,iterations=400)
    if not solved.converged:raise RuntimeError('Safe grasp inverse kinematics failed')
    d.qpos[:6]=solved.joints;d.ctrl[:6]=solved.joints;mujoco.mj_forward(m,d)
    return m,d,center,R,settings,cfg


class TableClearance:
    """Check full visual mesh vertices, independently of the collision hulls."""
    def __init__(self,model):
        self.entries=[];self.minimum=float('inf');self.minimum_arm=float('inf')
        for gid in range(model.ngeom):
            if model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=model.geom_dataid[gid];name=model.mesh(mid).name
            if not (name.startswith('r_a_link') or name=='right_hand_base'):continue
            a,n=model.mesh_vertadr[mid],model.mesh_vertnum[mid]
            vertices=model.mesh_vert[a:a+n].astype(float)
            # A linear plane distance attains its minimum on a hull vertex: exact, not a fitted primitive.
            vertices=vertices[ConvexHull(vertices).vertices]
            self.entries.append((gid,name,vertices))
    def measure(self,data):
        lows={name:float(np.min(v@data.geom_xmat[gid].reshape(3,3)[2,:])+data.geom_xpos[gid,2]) for gid,name,v in self.entries}
        moving=min(v for k,v in lows.items() if k!='r_a_link0')
        self.minimum=min(self.minimum,min(lows.values()));self.minimum_arm=min(self.minimum_arm,moving)
        return moving,lows
    def require_clear(self,data):
        value,lows=self.measure(data)
        if min(lows.values()) < -1e-6 or value<.001:
            raise RuntimeError(f'Robot/table clearance violation: {lows}')
        return value


if __name__=='__main__':
    m,d,c,R,s,cfg=make_scene();audit=TableClearance(m);print('start clearance',audit.require_clear(d),d.qpos[:6])
    tip_offset=d.site('brush_tip').xpos-d.site('palm_control').xpos
    for target in ([.17,.09,.163],[.30,.14,.030],[.18,-.02,.002],[.35,-.15,.002]):
        solved=solve_arm_ik(m,d,np.asarray(target)-tip_offset,R,iterations=400)
        if not solved.converged:raise RuntimeError(f'Unreachable {target}')
        d.qpos[:6]=solved.joints;mujoco.mj_forward(m,d)
        print('target',target,'clearance',audit.require_clear(d),'contacts',d.ncon)
