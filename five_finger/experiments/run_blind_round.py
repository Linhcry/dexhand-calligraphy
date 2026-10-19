"""One command: independent draw, camera/control, audit, judge, and optional native video."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from fivefinger.paths import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');p.add_argument('--open-video',action='store_true');p.add_argument('--profile',help='Pre-draw frozen JSON under config/');a=p.parse_args()
    if a.profile:
        profile=(ROOT/a.profile).resolve()
        if not profile.is_relative_to((ROOT/'config').resolve()) or not profile.is_file():
            p.error('Profile must be an existing JSON inside config/')
        json.loads(profile.read_text(encoding='utf-8'))
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    draw=subprocess.run([sys.executable,'-m','experiments.blind_challenge','create'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
    print(draw.stdout,flush=True);case=json.loads(draw.stdout)['case_id'];folder=ROOT/'outputs/blind_competition'/case
    command=[sys.executable,'-m','experiments.blind_competition','--public',str(folder/'public')]
    if a.profile:command.extend(['--profile',str(profile)])
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    with (folder/'controller.log').open('w',encoding='utf-8') as log:
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
    code=process.wait()
    if code:
        (folder/'run/failure.json').write_text(json.dumps({'returncode':code,'log':'../controller.log','status':'controller stopped; no task success claim'},indent=2),encoding='utf-8')
    # The examiner reveals its private answer only after execution has ended.
    subprocess.run([sys.executable,'-m','experiments.blind_challenge','judge','--case',case],cwd=ROOT,env=env,check=True)
    if code:raise SystemExit(code)
    subprocess.run([sys.executable,'-m','experiments.audit_blind_result','--case',str(folder)],cwd=ROOT,env=env,check=True)
    command=[sys.executable,'-m','experiments.view_blind_competition','--case',str(folder)]
    if a.video or a.open_video:command.append('--video')
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    (folder.parent/'latest_verified.txt').write_text(case,encoding='utf-8')
    if a.video or a.open_video:(folder.parent/'latest_video.txt').write_text(case,encoding='utf-8')
    if a.open_video:os.startfile(folder/'run/随机诗题_五指物理仿真.mp4')


if __name__=='__main__':main()
