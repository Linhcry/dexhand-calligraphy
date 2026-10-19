"""Separate examiner: sample a poem and expose pixels, never plaintext to control."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time
import uuid
from PIL import Image,ImageDraw,ImageFont
from fivefinger.paths import ROOT

CASES=ROOT/'outputs/blind_competition'


def create(development_title=None):
    files=list((ROOT/'fivefinger').rglob('*.py'))+list((ROOT/'assets/hanzi').glob('*.json'))
    files += [f for f in (ROOT/'assets/ocr').glob('*') if f.is_file()]
    files += [ROOT/name for name in ('experiments/blind_competition.py','experiments/competition_scene.py',
        'experiments/writing.py','experiments/probe_grasp.py','experiments/ocr_camera_worker.py',
        'experiments/writing_config.json','config/competition_scene.json','config/example_poems.json',
        'scripts/speak_recognized.ps1')]
    files += list((ROOT/'config').glob('*.json'))
    manifest={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    bank=json.loads((ROOT/'config/example_poems.json').read_text(encoding='utf-8'))
    if development_title is not None and development_title not in bank:
        raise ValueError('Development title must be in the example bank')
    title=development_title or secrets.choice(list(bank));nonce=secrets.token_hex(32)
    case=CASES/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    public=case/'public';private=case/'examiner_private';public.mkdir(parents=True);private.mkdir()
    image=Image.new('RGB',(1000,400),(237+secrets.randbelow(15),)*3)
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',160)
    draw.text((500,200),title,font=font,anchor='mm',fill=(18,20,25))
    image.save(public/'screen_pixels.png')
    metadata={'case_id':case.name,'evaluation_mode':'development_known_to_evaluator' if development_title else 'random_blind_draw','answer_commitment_sha256':hashlib.sha256((nonce+title+bank[title]).encode()).hexdigest(),
              'screen_pixels_sha256':hashlib.sha256((public/'screen_pixels.png').read_bytes()).hexdigest(),
              'screen_center_m':[.60+(secrets.randbelow(21)-10)*.001,.28+(secrets.randbelow(21)-10)*.001,.12],
              'screen_yaw_deg':secrets.randbelow(13)-6,'screen_tilt_deg':35,
              'examiner_note':'Non-official example bank; target sampled independently. Controller receives only this metadata and image bytes.'}
    (public/'scene.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (public/'controller_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (private/'answer.json').write_text(json.dumps({'title':title,'poem':bank[title],'nonce':nonce},ensure_ascii=False,indent=2),encoding='utf-8')
    CASES.mkdir(exist_ok=True);(CASES/'latest_case.txt').write_text(case.name,encoding='utf-8')
    # Deliberately never print the selected title, poem or seed here.
    print(json.dumps({'case_id':case.name,'answer_commitment_sha256':metadata['answer_commitment_sha256']}),flush=True)


def judge(case):
    folder=CASES/case
    result=json.loads((folder/'run/report.json').read_text(encoding='utf-8')) if (folder/'run/report.json').exists() else {
        'recognized_title':None,'recognized_poem':None,'stop_reason':'controller_failed_before_physical_execution'}
    secret=json.loads((folder/'examiner_private/answer.json').read_text(encoding='utf-8'))
    public=json.loads((folder/'public/scene.json').read_text(encoding='utf-8'))
    committed=hashlib.sha256((secret['nonce']+secret['title']+secret['poem']).encode()).hexdigest()==public['answer_commitment_sha256']
    verdict={'commitment_valid':committed,'recognized_title_matches':result['recognized_title']==secret['title'],
             'planned_body_matches':result['recognized_poem']==secret['poem'],'target_title':secret['title'],
             'target_poem':secret['poem'],'physical_result':result['stop_reason'],'official_score':None}
    (folder/'judge.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(verdict,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','judge']);p.add_argument('--case');p.add_argument('--development-title');a=p.parse_args()
    if a.action=='create':create(a.development_title)
    else:judge(a.case or (CASES/'latest_case.txt').read_text(encoding='utf-8'))


if __name__=='__main__':main()
