"""Render saved physics, measured ink and the actual camera input; never redraw font outlines."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from experiments.competition_scene import make_scene
from experiments.view_writing import add_ink
from fivefinger.paths import ROOT

STAGES={'grip_recovery':'保持机械臂，恢复握力','wait_vision':'相机取图 / OCR','announce':'播报识别结果与诗文','close':'闭指抓笔',
        'lift':'提笔离架','hold':'连续持笔检查','to_ink':'移向砚台','dip_lower':'笔尖蘸墨',
        'dip_dwell':'浸墨停留','dip_lift':'提笔离墨','to_paper':'移向 A3 纸',
        'paper_lower':'下降准备书写','touchdown':'等待纸面接触','write':'实际接触书写',
        'pen_up':'抬笔换笔画','done':'完成后提笔'}


def load(case):
    folder=Path(case).resolve();run=folder/'run'
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    m,d,*_=make_scene(folder/'public',settings=cfg['scene'],cfg=cfg['grip'])
    with np.load(run/'replay.npz',allow_pickle=False) as archive:
        trace={key:archive[key] for key in archive.files}
    physics=np.loadtxt(run/'physics.csv',delimiter=',',skiprows=1)
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    events=json.loads((run/'events.json').read_text(encoding='utf-8'))
    return run,cfg,m,d,trace,physics,report,events


def segments(trace):
    result=[];previous=None
    for i,row in enumerate(trace['ink']):
        if not row[0]:previous=None;continue
        point=np.array([row[1],row[2],.0022])
        if previous is None or np.linalg.norm(point-previous)>=.0005:
            result.append((i,point.copy() if previous is None else previous.copy(),point.copy()))
            previous=point
    return result


class Paper:
    def __init__(self,settings,width=1000):
        self.s=settings;self.width=width;self.scale=width/settings['paper_size_m'][0]
        self.image=Image.new('RGB',(width,round(settings['paper_size_m'][1]*self.scale)),'#f5f0e5')
        self.draw=ImageDraw.Draw(self.image);self.previous=None;self.index=-1
    def pixel(self,point):
        cx,cy=self.s['paper_center_xy_m'];w,h=self.s['paper_size_m']
        return ((point[0]-cx+w/2)*self.scale,(cy+h/2-point[1])*self.scale)
    def update(self,trace,index):
        if index<self.index:raise ValueError('Paper renderer only advances')
        radius=.00065*self.scale
        for active,x,y,z,force in trace['ink'][self.index+1:index+1]:
            if not active:self.previous=None;continue
            p=self.pixel((x,y))
            if self.previous is not None:self.draw.line([self.previous,p],fill='#20282a',width=max(1,round(2*radius)))
            self.draw.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill='#20282a')
            self.previous=p
        self.index=index
        return self.image


def apply(m,d,tr,i):
    d.qpos[:]=tr['qpos'][i];d.qvel[:]=tr['qvel'][i];d.ctrl[:]=tr['ctrl'][i];d.time=float(tr['time_s'][i])
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)


def camera(cam):
    cam.lookat[:]=[.10,.02,.25];cam.distance=1.55;cam.azimuth=120;cam.elevation=-28


def render(case,video=False,qualification=False):
    run,cfg,m,d,tr,physics,report,events=load(case);segs=segments(tr)
    cam=mujoco.MjvCamera();camera(cam)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    heading=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
    observed=next(x['sim_s'] for x in events if x['event']=='vision_recognized')
    camera_input=Image.open(run/'camera_rgb.png').convert('RGB');camera_input.thumbnail((282,150))
    paper=Paper(cfg['scene']);paper.update(tr,len(tr['time_s'])-1).save(run/'contact_ink_A3.png')
    paper=Paper(cfg['scene'])
    def compose(renderer,i):
        apply(m,d,tr,i);renderer.update_scene(d,camera=cam);add_ink(renderer.scene,segs,i)
        sheet=Image.new('RGB',(1280,720),'#142431');sheet.paste(Image.fromarray(renderer.render()),(0,76))
        ink=paper.update(tr,i).resize((272,385),Image.Resampling.LANCZOS);sheet.paste(ink,(990,100))
        sheet.paste(camera_input,(987,516));draw=ImageDraw.Draw(sheet)
        title=report['recognized_title'] if tr['time_s'][i]>=observed else '等待相机识别'
        draw.text((20,12),f'五指决赛情景 · 随机诗题 → 蘸墨 → 全文书写   |   {title}',font=heading,fill='#f5f0e5')
        draw.text((20,49),'原生 MuJoCo 物理记录 / 参考机器人与示例题库 / 1× 实际仿真时间',font=small,fill='#a8c6cc')
        draw.text((990,78),'A3 实测笔迹',font=small,fill='#dce9e6')
        draw.text((990,492),'OCR 实际输入：俯视相机',font=small,fill='#dce9e6')
        draw.rectangle((0,638,975,720),fill='#142431')
        draw.text((20,647),f'{tr["time_s"][i]:06.2f} / 300 s    {STAGES.get(str(tr["stage"][i]),str(tr["stage"][i]))}',font=font,fill='#f5f0e5')
        draw.text((20,681),f'机械臂最小离桌 {physics[i,20]*1000:.2f} mm    纸面力 {physics[i,14]:.2f} N    浸墨深度 {max(0,physics[i,21])*1000:.2f} mm',font=small,fill='#a8c6cc')
        draw.text((990,682),'简化笔尖和墨迹；非官方评分',font=small,fill='#d4b588')
        return sheet
    with mujoco.Renderer(m,width=975,height=560,max_geom=max(10000,len(segs)+1000)) as renderer:
        # Independent snapshots use fresh accumulators; replay must not run physics again.
        for label,seconds in [('initial',0),('dip',next((e['sim_s'] for e in events if e['event']=='dip_lift'),35)),('writing',90),('result',tr['time_s'][-1])]:
            paper=Paper(cfg['scene']);i=min(np.searchsorted(tr['time_s'],seconds),len(tr['time_s'])-1)
            compose(renderer,i).save(run/f'{label}.png')
        if video or qualification:
            paper=Paper(cfg['scene']);fps=15
            raw=run/('qualification_silent.mp4' if qualification else 'physics_silent.mp4')
            output=run/('资格演示_参考仿真.mp4' if qualification else '随机诗题_五指物理仿真.mp4')
            end=float(tr['time_s'][-1])
            intervals=[(0.,50.),(65.,90.),(end-30.,end)] if qualification else [(0.,end)]
            if qualification and end<120:intervals=[(0.,end)]
            source_times=np.concatenate([np.arange(start,stop,1/fps) for start,stop in intervals])
            manifest=dict(source_case=str(case),source_seconds=end,intervals_s=intervals,playback_rate=1.,
                          edited_excerpt=qualification,simulation_only=True,official_acceptance_confirmed=False)
            (run/('qualification_edit.json' if qualification else 'full_video_edit.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-',
                '-an','-c:v','libx264','-preset','fast','-crf','24' if qualification else '21','-maxrate','1400k','-bufsize','2800k','-pix_fmt','yuv420p',str(raw)],stdin=subprocess.PIPE)
            count=len(source_times)
            try:
                for frame in range(count):
                    i=min(np.searchsorted(tr['time_s'],source_times[frame]),len(tr['time_s'])-1)
                    picture=compose(renderer,i)
                    if qualification:
                        dd=ImageDraw.Draw(picture);dd.rectangle((0,43,1279,75),fill='#142431')
                        dd.text((20,47),'参考仿真 · 1×片段剪辑（非连续全程） · 非官方设备实测 / 无官方评分',font=small,fill='#ffd48c')
                    process.stdin.write(picture.tobytes())
                    if frame%300==0:print(f'Rendered {frame}/{count}',flush=True)
            finally:process.stdin.close()
            if process.wait():raise RuntimeError('Video encode failed')
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-i',str(run/'announcement.wav'),
                '-filter_complex',f'[1:a]adelay={round(observed*1000)}:all=1,apad[a]',
                '-map','0:v','-map','[a]','-c:v','copy','-c:a','aac','-b:a','64k','-shortest','-movflags','+faststart',str(output)],check=True)
            print(output,flush=True)
    print(run/'result.png',flush=True)


def view(case):
    from mujoco import viewer
    run,cfg,m,d,tr,physics,report,events=load(case);segs=segments(tr)
    # Native user scene has a 10,000 geometry budget; reject rather than omit ink silently.
    state={'i':0,'paused':False,'restart':False,'seek':0}
    def key(code):
        if code==32:state['paused']=not state['paused']
        elif code in (82,114):state['restart']=True
        elif code==262:state['seek']=500
        elif code==263:state['seek']=-500
    print('Recorded physics. Space: pause; R: restart; arrows: seek 5 s. Close window to exit.',flush=True)
    with viewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False) as window:
        camera(window.cam)
        while window.is_running():
            start=time.perf_counter()
            if state['restart']:state.update(i=0,paused=False,restart=False)
            state['i']=int(np.clip(state['i']+state['seek'],0,len(tr['time_s'])-1));state['seek']=0
            with window.lock():
                apply(m,d,tr,state['i']);window.user_scn.ngeom=0;add_ink(window.user_scn,segs,state['i'])
            window.sync()
            if not state['paused']:state['i']=min(state['i']+3,len(tr['time_s'])-1)
            if state['i']==len(tr['time_s'])-1:state['paused']=True
            time.sleep(max(0,.03-(time.perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--video',action='store_true');p.add_argument('--qualification',action='store_true');p.add_argument('--viewer',action='store_true');a=p.parse_args()
    case=Path(a.case) if a.case else ROOT/'outputs/blind_competition'/(ROOT/'outputs/blind_competition/latest_case.txt').read_text().strip()
    if a.viewer:view(case)
    else:render(case,a.video,a.qualification)
