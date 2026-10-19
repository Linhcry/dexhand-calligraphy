"""Build a local review bundle; this is not an official upload format or submission."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/报名准备_20260910'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=OUT/'代码审阅快照_非官方提交格式.zip'
    files=[]
    for folder in ('fivefinger','experiments','tests','config','lessons'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.xml','.md') and '__pycache__' not in p.parts)
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.md','.toml','.txt','.ps1','.cmd'))
    files.extend(p for p in (ROOT/'scripts').iterdir() if p.is_file() and p.suffix in ('.py','.ps1'))
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('快照说明.txt','这是项目自有代码及配置的审阅快照，不是官方规定的提交格式，也不是离线即用环境。\n不含虚拟环境、第三方模型、OCR权重、字形数据、物理回放大文件。\n复现请使用本机完整five_finger项目，或按setup脚本和model_sources.json准备所锁定依赖及模型，并保留第三方许可证。\n尚无官方SDK，不能用于直接驱动比赛设备。\n')
    summary={
        'date':'2026-09-10','physical_control_tests':47,'ocr_tests':1,
        'test_commands':['.venv-retarget/Scripts/python.exe -m pytest -q -k "not offline_ocr_recognizes_chinese_title"','.venv/Scripts/python.exe -m pytest tests/test_baseline.py::test_offline_ocr_recognizes_chinese_title -q'],
        'technical_doc_pages':2,'team_form_pages':1,'doc_visual_review_passed':True,
        'qualification_clip_seconds':105,'qualification_clip_bytes':2437531,
        'qualified_by_organizer':False,'submitted':False,'hardware_ready':False,
        'physical_scope':'example-bank 20-character poem at fixed reference workcell; not all event tasks',
        'new_continuous_validation_case':'20260910_132551_99983a',
        'known_failures':['cosine_cruise 春晓 pen_tilt_exceeded','increased grip forces 春晓 grip_lost'],
        'missing_inputs':['学校和队伍信息','报名系统生成的正式表格','签字及学院盖章','仿真资格材料是否接受的确认','官方SDK和设备训练安排']}
    (OUT/'验收范围.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    listing=[dict(file=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='文件清单.json' and not p.name.startswith('~$')]
    (OUT/'文件清单.json').write_text(json.dumps(listing,ensure_ascii=False,indent=2),encoding='utf-8')
    packet=ROOT/'deliverables/五指报名准备包_20260910.zip'
    with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.startswith('~$'):z.write(p,p.relative_to(OUT.parent).as_posix())
    with zipfile.ZipFile(packet) as z:
        assert z.testzip() is None
        for item in listing:assert hashlib.sha256(z.read(OUT.name+'/'+item['file'])).hexdigest()==item['sha256']
    print(json.dumps(dict(packet=str(packet),bytes=packet.stat().st_size,files=len(listing)+1,archive_crc_and_hashes_passed=True),ensure_ascii=False))
if __name__=='__main__':main()
