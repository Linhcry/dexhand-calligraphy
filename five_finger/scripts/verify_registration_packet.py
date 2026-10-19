"""Verify an actual encoded native simulation clip, and copy evidence into review packet."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'outputs/blind_competition/20260910_130143_ac00cb'
OUT=ROOT/'deliverables/报名准备_20260910'
def main():
    source=CASE/'run/资格演示_参考仿真.mp4';dest=OUT/source.name
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(source)],encoding='utf-8'))
    duration=float(probe['format']['duration']);size=source.stat().st_size
    if duration>120 or size>30_000_000:raise RuntimeError('Video exceeds published bounds')
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-f','null','-'],check=True)
    qa=ROOT/'outputs/registration_video_qa';qa.mkdir(parents=True,exist_ok=True)
    for name,seconds in [('title',3),('grasp',25),('dip',35),('writing',60),('result',duration-1)]:
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(seconds),'-i',str(source),'-frames:v','1',str(qa/f'{name}.png')],check=True)
    audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(source),'-vn','-f','s16le','-ac','1','-ar','16000','-'])
    import numpy as np
    peak=int(np.abs(np.frombuffer(audio,dtype=np.int16).astype(np.int32)).max())
    if peak==0:raise RuntimeError('Speech track is silent')
    shutil.copy2(source,dest)
    for name in ['report.json','audit.json','recognized.json','events.json','qualification_edit.json','config.json','access_audit.json']:
        folder=OUT/'实验依据';folder.mkdir(exist_ok=True);shutil.copy2(CASE/'run'/name,folder/name)
    shutil.copy2(CASE/'judge.json',OUT/'实验依据/judge.json')
    shutil.copy2(CASE/'run/result.png',OUT/'实验依据/实际墨迹.png')
    result=dict(source_case=CASE.name,seconds=duration,bytes=size,decimal_MB=size/1e6,under_120_seconds=True,
        under_30_decimal_MB=True,full_decode_passed=True,audio_peak_s16=peak,
        video_codec=next(s['codec_name'] for s in probe['streams'] if s['codec_type']=='video'),
        audio_codec=next(s['codec_name'] for s in probe['streams'] if s['codec_type']=='audio'),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),simulation_only=True,edited_excerpt=True,playback_rate=1,
        official_acceptance_confirmed=False,official_score=None)
    (OUT/'视频与实验验证.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
