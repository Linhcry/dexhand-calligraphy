"""Local simulation package entry point. Hardware submission entry is not yet published."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','run','view'])
    p.add_argument('--profile');p.add_argument('--video',action='store_true');p.add_argument('--case')
    a=p.parse_args();os.chdir(ROOT)
    if a.mode=='check':
        manifest=ROOT/'PACKAGE_MANIFEST.json'
        if manifest.exists():
            entries=json.loads(manifest.read_text(encoding='utf-8'))
            for x in entries:
                f=(ROOT/x['file']).resolve()
                if not f.is_relative_to(ROOT) or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=x['sha256']:
                    raise RuntimeError(f'Package integrity check failed: {x["file"]}')
        from experiments.competition_scene import make_scene,TableClearance
        m,d,*_=make_scene();clearance=TableClearance(m).require_clear(d)
        ocr=os.environ.get('FIVEFINGER_OCR_PYTHON',str(ROOT/'.venv/Scripts/python.exe'))
        subprocess.run([ocr,'-c','from fivefinger.vision import OfflineOCR; OfflineOCR(); print("Local OCR weights loaded")'],cwd=ROOT,check=True)
        result=dict(model_loaded=True,nq=m.nq,nu=m.nu,initial_moving_arm_clearance_mm=clearance*1000,hardware_ready=False,package_manifest_checked=manifest.exists())
        print(json.dumps(result,ensure_ascii=False));return
    if a.mode=='run':
        cmd=[sys.executable,'-m','experiments.run_blind_round']
        if a.profile:cmd+=['--profile',a.profile]
        if a.video:cmd+=['--video']
    else:
        if not a.case:p.error('view requires --case outputs/blind_competition/CASE_ID')
        cmd=[sys.executable,'-m','experiments.view_blind_competition','--case',a.case,'--viewer']
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))
if __name__=='__main__':main()
