"""Serial, retained development/evaluation trials; never selects only successes.

This is bounded controller tuning, not reinforcement learning. The simulator
currently writes a shared workcell XML, so concurrent runs are unsupported.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import shutil
from fivefinger.paths import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',required=True)
    p.add_argument('--titles',nargs='*',help='Known development titles; omit for fresh random draws')
    p.add_argument('--rounds',type=int,default=3)
    a=p.parse_args()
    profile=(ROOT/a.profile).resolve()
    if not profile.is_relative_to(ROOT/'config') or not profile.is_file():
        p.error('Use an existing profile under config')
    if not 1 <= a.rounds <= 100:
        p.error('Rounds must be 1..100')
    bank=json.loads((ROOT/'config/example_poems.json').read_text(encoding='utf-8'))
    if a.titles and any(t not in bank for t in a.titles):
        p.error('Unknown development title')
    directory=ROOT/'outputs/robustness'/time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    lock=ROOT/'outputs/robustness/training.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    results=[]
    trials=a.titles or [None]*a.rounds
    def persist(status):
        summary={'profile':str(profile),'runs':results,'expected_runs':len(trials),'batch_status':status,
                 'batch_complete':status=='complete','process_passes':sum(x['process_pass'] for x in results),
                 'quality_passes':sum(x['quality_pass'] for x in results),'hardware_ready':False,
                 'note':'Known five-poem example bank; not official scoring, not unseen-poem generalization. All attempts retained.'}
        temporary=directory/'summary.tmp'
        temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(directory/'summary.json')
    persist('running')
    try:
        for title in trials:
            command=[sys.executable,'-m','experiments.blind_challenge','create']
            if title:command += ['--development-title',title]
            draw=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
            case=json.loads(draw.stdout)['case_id'];folder=ROOT/'outputs/blind_competition'/case
            results.append({'case':case,'mode':'development' if title else 'random_blind','report':{'stop_reason':'in_progress'},'process_pass':False,'quality_pass':False})
            persist('running')
            manifest=json.loads((folder/'public/controller_manifest.json').read_text(encoding='utf-8'))
            for name,digest in manifest.items():
                source=(ROOT/name).resolve()
                if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Pre-run snapshot mismatch')
                if source.suffix in ('.py','.json','.ps1') and not name.startswith('assets/'):
                    destination=folder/'controller_snapshot'/name
                    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            print(json.dumps({'case':case,'mode':'development' if title else 'random_blind','profile':str(profile)},ensure_ascii=False),flush=True)
            with (folder/'controller.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-m','experiments.blind_competition','--public',str(folder/'public'),'--profile',str(profile)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            with (folder/'evaluation.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-m','experiments.blind_challenge','judge','--case',case],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                if (folder/'run/report.json').exists():
                    subprocess.run([sys.executable,'-m','experiments.audit_blind_result','--case',str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            report=json.loads((folder/'run/report.json').read_text(encoding='utf-8')) if (folder/'run/report.json').exists() else {'stop_reason':'controller_error'}
            audit=json.loads((folder/'run/audit.json').read_text(encoding='utf-8')) if (folder/'run/audit.json').exists() else {}
            judge=json.loads((folder/'judge.json').read_text(encoding='utf-8'))
            # Predeclared diagnostic quality gates, not official contest scoring.
            process=bool(audit.get('all_process_checks_passed')) and judge['recognized_title_matches'] and judge['planned_body_matches']
            quality=process and audit.get('mean_stroke_coverage_2_5mm',0)>=.95 and report.get('writing_ink_contact_fraction',0)>=.95 and audit.get('ink_length_precision_within_2_5mm',0)>=.95
            row={'case':case,'mode':'development' if title else 'random_blind','returncode':result.returncode,'report':report,'process_pass':process,'quality_pass':quality,'mean_stroke_coverage_2_5mm':audit.get('mean_stroke_coverage_2_5mm'),'ink_length_precision_within_2_5mm':audit.get('ink_length_precision_within_2_5mm'),'quality_gate_version':'v2-add-unplanned-ink'}
            results[-1]=row
            row['process_gate_version']=audit.get('process_gate_version','legacy-table-only')
            persist('running')
            print(json.dumps({k:v for k,v in row.items() if k!='report'}|{'title':report.get('recognized_title'),'stop':report['stop_reason']},ensure_ascii=False),flush=True)
        persist('complete')
    except BaseException:
        persist('interrupted')
        raise
    finally:
        lock.unlink(missing_ok=True)
    print(str(directory/'summary.json'),flush=True)


if __name__=='__main__':main()
