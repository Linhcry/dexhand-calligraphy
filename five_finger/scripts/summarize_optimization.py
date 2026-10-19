"""Preserve all optimization outcomes and restore the conservative development default."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/报名准备_20260910'
CASES=ROOT/'outputs/blind_competition'
def main():
    ids=['20260910_125724_819afb','20260910_125856_b3a4d4','20260910_130143_ac00cb','20260910_130447_f30ab3','20260910_130732_f852a1','20260910_130834_47d54c','20260910_132551_99983a']
    def scene(case):return json.loads((CASES/case/'run/config.json').read_text(encoding='utf-8'))['scene']
    baseline=scene(ids[0]);baseline.update(continuous_strokes=False,recover_contact_midstroke=False,stroke_timing='quintic')
    for name,data in [('competition_scene',baseline),('profile_continuous_recovery',scene(ids[1])),('profile_cosine_recovery',scene(ids[2]))]:
        (ROOT/'config'/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    labels=['原基线','连续轨迹＋接触恢复','匀速轨迹＋接触恢复','匀速随机复测','3.5N越界参数预检查','3/2/2N夹持试验','连续轨迹随机复测（同题）']
    reports=[];lines=['# 2026年9月10日优化记录','', '范围：参考模型、示例诗库、固定纸笔布局。不是官方验收；单次成功不能代表稳健性。','', '| 方案 | OCR诗题 | 结果 | 耗时/秒 | 完成笔画 |','|---|---|---|---:|---|']
    for case,label in zip(ids,labels):
        src=CASES/case/'run';rpath=src/'report.json'
        report=json.loads(rpath.read_text(encoding='utf-8')) if rpath.exists() else {'stop_reason':'rejected_before_physics','recognized_title':'—'}
        reports.append(dict(case=case,variant=label,report=report))
        evidence=OUT/'优化试验记录'/case;evidence.mkdir(parents=True,exist_ok=True)
        for name in ('report.json','failure.json','config.json','audit.json'):
            if (src/name).exists():shutil.copy2(src/name,evidence/name)
        lines.append(f"| {label} | {report.get('recognized_title','—')} | {report['stop_reason']} | {report.get('elapsed_sim_s',0):.2f} | {report.get('completed_strokes','—')}/{report.get('planned_strokes','—')} |")
    lines+=['','同题《登鹳雀楼》由265.18秒缩短至232.02秒（约12.5%），但XY误差中位数由0.79增至1.33毫米。两次匀速完整物理试验仅一次完成；不能估计可靠成功率。3.5N试验被开发范围检查拒绝，不计入物理运行。','', '默认配置已恢复原基线。连续与匀速方案独立保存，均为实验参数，没有改变手指夹持限幅、桌面碰撞或自由笔刚体。','', '## 怎么运行','', '在five_finger目录中：','', '```powershell', '.\\.venv-retarget\\Scripts\\python.exe -m experiments.run_blind_round', '# 连续轨迹实验', '.\\.venv-retarget\\Scripts\\python.exe -m experiments.run_blind_round --profile config/profile_continuous_recovery.json', '# 提速实验：已有随机诗题失败记录', '.\\.venv-retarget\\Scripts\\python.exe -m experiments.run_blind_round --profile config/profile_cosine_recovery.json', '```','', '可增加 --video 导出完整视频。抽题前锁定所有配置文件；控制器只在OCR后取得诗题。不要修改历史案例清单以伪装新代码在旧案例上通过。','', '## 尚需解决','', '1. 长时间握持的笔杆转动和轴向滑移；应研究有界微调与接触估计，不能用刚性焊接替代抓取。','2. 运笔断墨：提速增加跟踪误差；需联合改进笔尖柔顺、法向力控制与曲率限速。','3. 变化布局的RGB-D感知和在线重规划尚未完整接入物理执行，当前26场景是感知验证。','4. 无官方SDK、题库、字帖与真机参数，暂不能承诺接机即运行或达到官方书法评分。']
    text='\n'.join(lines)+'\n'
    (OUT/'优化结果与复现.md').write_text(text,encoding='utf-8')
    (ROOT/'优化记录_20260910.md').write_text(text,encoding='utf-8')
    (OUT/'优化试验记录/汇总.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
