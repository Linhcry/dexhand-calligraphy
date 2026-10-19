"""Native MuJoCo replay/video with ink derived exclusively from measured contacts."""
import argparse
import json
import subprocess
import time
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.writing import OUT,make_writing_model

STAGES={'close':'手指闭合抓笔','lift':'提笔离架','hold':'空中稳定保持','transfer':'移向纸面',
        'lower':'下降到纸面上方','touchdown':'等待真实落笔接触','write':'接触书写','pen_up':'抬笔换笔画','done':'写完提笔'}


def ink_segments(trace):
    result=[];previous=None
    for index,entry in enumerate(trace['ink']):
        if not entry[0]:previous=None;continue
        point=np.array([entry[1],entry[2],.0022])
        if previous is None:
            result.append((index,point.copy(),point.copy()));previous=point
        elif np.linalg.norm(point-previous)>=.00025:
            result.append((index,previous.copy(),point.copy()));previous=point
    return result


def add_ink(scene,segments,index):
    for stamp,a,b in segments:
        if stamp>index:break
        if scene.ngeom>=scene.maxgeom:raise RuntimeError('Ink exceeds render scene capacity')
        geom=scene.geoms[scene.ngeom]
        if np.linalg.norm(a-b)<1e-9:
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0008,0,0]),a,np.eye(3).ravel(),np.array([.065,.08,.09,1.]))
        else:
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.065,.08,.09,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0008,a,b)
        scene.ngeom+=1


def ink_image(segments,index,size=560):
    canvas=Image.new('RGB',(size,size),'#f5f0e5');draw=ImageDraw.Draw(canvas)
    def pixel(xy):return ((xy[0]-.24)/.1*size,(.005-xy[1])/.1*size)
    radius=size*.0008/.1
    for stamp,a,b in segments:
        if stamp>index:break
        start,end=pixel(a),pixel(b)
        draw.line([start,end],fill='#172029',width=max(1,int(2*radius)))
        for p in (start,end):
            draw.ellipse([p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius],fill='#172029')
    return canvas


def camera(cam):
    cam.lookat[:]=[.21,.025,.16];cam.distance=.65;cam.azimuth=-45;cam.elevation=-32


def load_trial(name):
    if not name or any(s in name for s in ('/','\\','..')):raise ValueError('Expected trial directory name')
    folder=OUT/name;config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    trace=np.load(folder/'replay.npz',allow_pickle=False)
    m,d,_,_=make_writing_model(config['yaw_deg'],config['initial_shift_m'],config)
    if trace['qpos'].shape[1]!=m.nq:raise RuntimeError('Replay model dimensions disagree')
    physics=np.loadtxt(folder/'physics.csv',delimiter=',',skiprows=1)
    return folder,trace,m,d,physics,ink_segments(trace)


def apply_frame(model,data,trace,index):
    data.qpos[:]=trace['qpos'][index];data.ctrl[:]=trace['ctrl'][index];data.time=float(trace['time_s'][index])
    mujoco.mj_forward(model,data)


def view(trace,m,d,segments):
    from mujoco import viewer
    state={'paused':False,'index':0,'restart':False,'seek':0}
    def key(code):
        if code==32:state['paused']=not state['paused']
        elif code in (82,114):state['restart']=True
        elif code==262:state['seek']=500
        elif code==263:state['seek']=-500
    print('RECORDED PHYSICS. Space pause, R restart, Left/Right seek 5 s. Close to exit.',flush=True)
    with viewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False) as window:
        camera(window.cam)
        while window.is_running():
            started=time.perf_counter()
            if state['restart']:state.update(index=0,restart=False,paused=False)
            state['index']=int(np.clip(state['index']+state['seek'],0,len(trace['time_s'])-1));state['seek']=0
            with window.lock():
                apply_frame(m,d,trace,state['index']);window.user_scn.ngeom=0
                add_ink(window.user_scn,segments,state['index'])
            window.sync()
            if not state['paused']:
                state['index']=min(state['index']+3,len(trace['time_s'])-1)
                if state['index']==len(trace['time_s'])-1:state['paused']=True
            time.sleep(max(0.,.03-(time.perf_counter()-started)))


def render(folder,trace,m,d,physics,segments,video=False,index=-1):
    cam=mujoco.MjvCamera();camera(cam)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',34)
    if index<0:index=len(trace['time_s'])-1
    def compose(renderer,i):
        apply_frame(m,d,trace,i);renderer.update_scene(d,camera=cam);add_ink(renderer.scene,segments,i)
        sheet=Image.new('RGB',(1600,900),'#142431')
        sheet.paste(Image.fromarray(renderer.render()),(0,95));sheet.paste(ink_image(segments,i),(1020,170))
        draw=ImageDraw.Draw(sheet)
        draw.text((28,18),'五指灵巧手 · 拿笔写「永」',font=title,fill='#f5f0e5')
        draw.text((30,62),'MuJoCo 原生物理记录   /   立式笔架练习场景   /   参考模型',font=small,fill='#93b9bf')
        draw.text((1020,120),'实际接触笔迹 · 100 × 100 mm',font=font,fill='#f5f0e5')
        draw.text((1020,750),'指尖夹持自由笔杆，笔尖接触时留痕',font=small,fill='#93b9bf')
        draw.rectangle((0,800,1600,900),fill='#142431')
        draw.text((30,814),f'{trace["time_s"][i]:05.2f} s     {STAGES[str(trace["stage"][i])]}',font=font,fill='#f5f0e5')
        f=physics[i,4:7]
        draw.text((510,814),f'拇 / 食 / 中指接触力   {f[0]:.2f} / {f[1]:.2f} / {f[2]:.2f} N',font=small,fill='#c8dfe0')
        draw.text((1180,814),f'纸面接触力  {physics[i,14]:.2f} N',font=small,fill='#c8dfe0')
        draw.text((30,856),'笔尖与墨迹均为简化模型；本视频不代表赛场成绩或真机验证。',font=small,fill='#d4b588')
        draw.rectangle((0,895,int(1600*(i+1)/len(trace['time_s'])),900),fill='#68bba6')
        return sheet
    with mujoco.Renderer(m,height=700,width=1000,max_geom=10000) as renderer:
        apply_frame(m,d,trace,0);renderer.update_scene(d,camera=cam);renderer.render()
        compose(renderer,index).save(folder/'writing_result.png')
        ink_image(segments,len(trace['time_s'])-1,size=1000).save(folder/'contact_ink.png')
        if video:
            fps=30;output=folder/'拿笔写永_物理仿真.mp4'
            command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r',str(fps),'-i','-',
                     '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
            count=int(np.ceil(trace['time_s'][-1]*fps));process=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for frame in range(count):
                    i=min(int(frame/fps*100),len(trace['time_s'])-1)
                    process.stdin.write(compose(renderer,i).tobytes())
                    if frame%150==0:print(f'Rendered {frame}/{count}',flush=True)
            finally:process.stdin.close()
            if process.wait()!=0:raise RuntimeError('FFmpeg encoding failed')
            print(output,flush=True)
    print(folder/'writing_result.png',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='demo');parser.add_argument('--frame',type=int,default=-1)
    parser.add_argument('--viewer',action='store_true');parser.add_argument('--video',action='store_true')
    args=parser.parse_args();folder,trace,m,d,physics,segments=load_trial(args.trial)
    if args.viewer:view(trace,m,d,segments)
    else:render(folder,trace,m,d,physics,segments,args.video,args.frame)


if __name__=='__main__':main()
