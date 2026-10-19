import json,shutil,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'outputs/blind_competition/20260910_130143_ac00cb'
OUT=ROOT/'deliverables/报名准备_20260910'
def main():
    video=CASE/'run/随机诗题_五指物理仿真.mp4'
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)],encoding='utf-8'))
    duration=float(data['format']['duration'])
    assert abs(duration-232.02)<.1
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    dest=OUT/'完整复核视频_非资格短片.mp4';shutil.copy2(video,dest)
    subprocess.run(['ffmpeg','-y','-v','error','-ss','230','-i',str(video),'-frames:v','1',str(ROOT/'outputs/registration_video_qa/full_end.png')],check=True)
    record=dict(source_case=CASE.name,seconds=duration,bytes=dest.stat().st_size,full_decode_passed=True,edited_excerpt=False,playback_rate=1,simulation_only=True,qualification_attachment=False,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (OUT/'完整视频验证.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    (CASE.parent/'latest_video.txt').write_text(CASE.name,encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
