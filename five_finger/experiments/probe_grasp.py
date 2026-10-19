"""Physical grasp development probe. No welds or kinematic brush attachment."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from fivefinger.cli import load_workcell
from fivefinger.scene import build_scene
from fivefinger.control import set_joint_targets, solve_arm_ik
from fivefinger.trajectory import smoothstep5

GRIP = {'r_f_joint1_1': .477051, 'r_f_joint1_2': .878808, 'r_f_joint1_3': .571996, 'r_f_joint1_4': .542113,
        'r_f_joint2_2': .636853, 'r_f_joint2_3': .610628, 'r_f_joint2_4': .586317,
        'r_f_joint3_2': .814250, 'r_f_joint3_3': .595934, 'r_f_joint3_4': .423636}


def make_model(radius=.006, shift=(0., 0., 0.)):
    m0,d0=load_workcell(); b=m0.body('right_hand_base').id
    R=d0.xmat[b].reshape(3,3).copy(); center=d0.xpos[b]+R@np.array([.073136,.022,.209114])+shift
    orient=np.column_stack([R[:,0],-R[:,2],R[:,1]])
    xyzw=Rotation.from_matrix(orient).as_quat(); quat=xyzw[[3,0,1,2]]
    root=ET.parse(build_scene()).getroot(); w=root.find('worldbody')
    w.remove(w.find("geom[@name='rack']"))
    brush=w.find("body[@name='brush']"); brush.set('pos',' '.join(map(str,center)));brush.set('quat',' '.join(map(str,quat)))
    brush.find("geom[@name='brush_barrel']").set('size',f'{radius} .085')
    for sign in [-1,1]:
        top=center+sign*.065*orient[:,2]; height=top[2]-radius
        ET.SubElement(w,'geom',name=f'support_{sign}',type='box',pos=f'{top[0]} {top[1]} {height/2}',size=f'.009 .012 {height/2}',rgba='.3 .18 .1 1')
        for side in [-1,1]:
            wallheight=top[2]+.015
            ET.SubElement(w,'geom',name=f'support_{sign}_wall_{side}',type='box',pos=f'{top[0]} {top[1]+side*.012} {wallheight/2}',size=f'.009 .003 {wallheight/2}',rgba='.3 .18 .1 1')
    assets={}
    for i,mesh in enumerate(root.findall('./asset/mesh')):
        path=Path(mesh.get('file'));key=f'asset_{i}{path.suffix}';assets[key]=path.read_bytes();mesh.set('file',key)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    data=mujoco.MjData(model);data.qpos[:26]=d0.qpos[:26];data.ctrl[:]=d0.ctrl
    mujoco.mj_forward(model,data)
    return model,data,center


def brush_contacts(m,d):
    forces=np.zeros(5); supports=0.; f=np.zeros(6)
    for c in range(d.ncon):
        contact=d.contact[c];g1=m.geom(contact.geom1).name or '';g2=m.geom(contact.geom2).name or ''
        if g1.startswith('brush_'): other=g2
        elif g2.startswith('brush_'):other=g1
        else:continue
        mujoco.mj_contactForce(m,d,c,f)
        for i in range(1,6):
            if other.startswith(f'col_r_f_link{i}_'):forces[i-1]+=max(0.,f[0])
        if other.startswith('support_') or other in ('table','paper'):supports+=max(0.,f[0])
    return forces,supports


def main():
    m,d,center=make_model();site=m.site('palm_control').id;start=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3).copy()
    armids=[m.joint(f'r_a_joint{i}').qposadr[0] for i in range(1,7)];q0=d.qpos[armids].copy()
    lifted=solve_arm_ik(m,d,start+[0,0,.05],rot);assert lifted.converged
    initial={name:d.qpos[m.joint(name).qposadr[0]] for name in GRIP};rows=[]
    for tick in range(1600):
        t=d.time
        grip=float(smoothstep5((t-1)/4))
        set_joint_targets(m,d,{k:initial[k]+grip*(v-initial[k]) for k,v in GRIP.items()})
        u=float(smoothstep5((t-8)/3));qa=q0+u*(lifted.joints-q0)
        set_joint_targets(m,d,{f'r_a_joint{i+1}':q for i,q in enumerate(qa)})
        for _ in range(5):
            d.qfrc_applied[armids]=d.qfrc_bias[armids]
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);forces,support=brush_contacts(m,d);p=d.body('brush').xpos.copy()
        rows.append([d.time,*p,*forces,support])
        if tick%100==0:print(round(d.time,2),np.round(p-center,3),np.round(forces,2),round(support,2))
    out=Path('outputs/competition');out.mkdir(exist_ok=True)
    np.savetxt(out/'probe.csv',rows,delimiter=',')
    print('warnings',[w.number for w in d.warning])


if __name__=='__main__':main()
