"""Replay recorded physical states, or encode their MuJoCo renderings as MP4."""
import argparse
import json
import shutil
import subprocess
import time

import mujoco
import numpy as np
from mujoco import viewer
from PIL import Image,ImageDraw,ImageFont
from experiments.probe_grasp import make_model
from fivefinger.paths import output_dir


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',default='grasp_r1_c2');parser.add_argument('--record',action='store_true');args=parser.parse_args()
    if not args.trial.startswith('grasp_r') or '/' in args.trial or '\\' in args.trial or '..' in args.trial:parser.error('Invalid trial name')
    folder=output_dir()/'competition'/args.trial
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'));trace=np.load(folder/'replay.npz')
    physics=np.loadtxt(folder/'physics_trace.csv',delimiter=',',skiprows=1)
    m,d,_=make_model(shift=report['initial_shift_m'])
    if trace['qpos'].shape[1]!=m.nq:raise RuntimeError('Replay model changed')
    def apply(index):
        # Replay only: restore recorded states. This is not a new physics success test.
        d.qpos[:]=trace['qpos'][index];d.ctrl[:]=trace['ctrl'][index];d.time=float(trace['time_s'][index]);mujoco.mj_forward(m,d)
    if not args.record:
        with viewer.launch_passive(m,d) as window:
            with window.lock():
                window.cam.lookat[:]=[.14,.15,.20];window.cam.distance=.75;window.cam.azimuth=135;window.cam.elevation=-25
            started=time.perf_counter()
            for index in range(0,len(trace['time_s']),3):
                if not window.is_running():break
                with window.lock():apply(index)
                window.sync();time.sleep(max(0.,float(trace['time_s'][index])-(time.perf_counter()-started)))
        return
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg not found')
    path=output_dir()/'competition'/'grasp_rule_experiment.mp4';cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-25
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    labels={'approach':'靠近笔架','close':'闭合手指','settle':'抓握接触','lift':'提笔','hold_check':'持笔稳定性检查'}
    frames=[]
    with (folder/'encode.log').open('wb') as err:
        enc=subprocess.Popen([ffmpeg,'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
        try:
            with mujoco.Renderer(m,height=720,width=1280) as renderer:
                for frame in range(480):
                    i=min(round(frame/30*100),len(trace['time_s'])-1);apply(i)
                    cam.lookat[:]=[.14,.15,.20];cam.distance=1.15 if frame<120 else .53
                    renderer.update_scene(d,camera=cam);picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
                    draw.rectangle((0,0,1280,55),fill='#162632');draw.text((20,12),'五指赛项抓笔实验｜实际物理状态回放',font=font,fill='white')
                    draw.text((900,15),f'{d.time:05.2f}s / 16s   {labels[str(trace["stage"][i])]}',font=small,fill='#b9dbea')
                    draw.rectangle((0,656,1280,720),fill='#162632')
                    draw.text((20,662),f'连续稳定持笔：{physics[i,-1]:.2f}s / 规则要求 3.00s；本次最长 {report["longest_continuous_hold_s"]:.2f}s',font=small,fill='white')
                    draw.text((20,690),'参考模型；本次持笔未通过，未继续蘸墨、点睛。',font=small,fill='#ffd190')
                    enc.stdin.write(picture.tobytes())
                    if frame in (0,150,270,360,479):frames.append(picture.copy())
            enc.stdin.close()
            if enc.wait(timeout=30)!=0:raise RuntimeError('FFmpeg failed')
        except BaseException:
            enc.kill();enc.wait();raise
    sheet=Image.new('RGB',(1280,720*len(frames)))
    for i,picture in enumerate(frames):sheet.paste(picture,(0,i*720))
    sheet.save(folder/'contact_sheet.jpg',quality=90)
    print(path)


if __name__=='__main__':main()
