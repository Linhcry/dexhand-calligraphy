"""Inspect recorded free-body physics, with pause/restart and static snapshots."""
import argparse
import json
import time
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from experiments.probe_grasp import make_model
from experiments.retarget_grasp import OUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trial',default='balanced_0')
    parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args()
    if not args.trial or any(s in args.trial for s in ('/','\\','..')):
        parser.error('Expected a trial directory name')
    folder=OUT/args.trial
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    trace=np.load(folder/'replay.npz',allow_pickle=False)
    physics=np.loadtxt(folder/'physics_trace.csv',delimiter=',',skiprows=1)
    model,data,_=make_model(shift=report['initial_shift_m'])
    if trace['qpos'].shape[1]!=model.nq or trace['ctrl'].shape[1]!=model.nu:
        raise RuntimeError('Replay dimensions do not match the model')
    def apply(i):
        data.qpos[:]=trace['qpos'][i];data.ctrl[:]=trace['ctrl'][i];data.time=float(trace['time_s'][i])
        mujoco.mj_forward(model,data)
    def camera(cam):
        cam.lookat[:]=[.14,.15,.21];cam.distance=.58;cam.azimuth=135;cam.elevation=-20
    if args.snapshot:
        cam=mujoco.MjvCamera();camera(cam)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
        # The selected frame is the measured end of the longest continuous hold.
        indices=[599,int(np.argmax(physics[:,-1]))]
        sheet=Image.new('RGB',(1600,650),'#12212c')
        with mujoco.Renderer(model,height=580,width=800) as renderer:
            for panel,index in enumerate(indices):
                apply(index);renderer.update_scene(data,camera=cam)
                sheet.paste(Image.fromarray(renderer.render()),(panel*800,0))
                draw=ImageDraw.Draw(sheet)
                draw.text((panel*800+15,588),f'{args.trial}   t={data.time:.2f}s   连续稳定 {physics[index,-1]:.2f}s',font=font,fill='white')
                draw.text((panel*800+15,617),'MuJoCo 记录状态；参考模型；未执行蘸墨与点睛',font=font,fill='#ffce7a')
        sheet.save(OUT/'grasp_progress.png')
        print(OUT/'grasp_progress.png')
        return
    from mujoco import viewer
    state={'paused':False,'index':0,'restart':False}
    def key(code):
        if code==32:state['paused']=not state['paused']
        if code in (82,114):state['restart']=True
    print(f'Recorded physics: {args.trial}. Space: pause/resume. R: restart. Close window: exit.')
    print(f'Longest hold: {report["longest_continuous_hold_s"]:.2f}s. This viewer does not run a new physics trial.')
    with viewer.launch_passive(model,data,key_callback=key) as window:
        camera(window.cam)
        while window.is_running():
            start=time.perf_counter()
            if state['restart']:
                state.update(index=0,restart=False,paused=False)
            with window.lock():apply(state['index'])
            window.sync()
            if not state['paused']:
                state['index']=min(state['index']+3,len(trace['time_s'])-1)
                if state['index']==len(trace['time_s'])-1:state['paused']=True
            time.sleep(max(0.,.03-(time.perf_counter()-start)))


if __name__=='__main__':
    main()
