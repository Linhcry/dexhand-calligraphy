"""Summarize every retained trial, including rejected candidates and failures."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from fivefinger.paths import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--date',default=time.strftime('%Y%m%d'));a=p.parse_args()
    groups=defaultdict(list);sources=[]
    for source in sorted((ROOT/'outputs/robustness').glob(a.date+'*/summary.json')):
        summary=json.loads(source.read_text(encoding='utf-8'));profile=Path(summary['profile']).stem
        for row in summary['runs']:groups[profile].append(row)
        sources.append(str(source.relative_to(ROOT)))
    candidate=groups.get('profile_pinch_reserve_v7',[])
    fixed=[x for x in candidate if x['mode']=='development']
    blind=[x for x in candidate if x['mode']=='random_blind']
    observations=[]
    if fixed:observations.append(f"固定布局示例诗 {sum(x['process_pass'] for x in fixed)}/{len(fixed)} 次流程通过")
    if blind:observations.append(f"重新随机抽题 {sum(x['process_pass'] for x in blind)}/{len(blind)} 次流程通过")
    headline='本日候选 V7：'+'；'.join(observations)+'。这些是当前已记录样本的数量，不是对比赛成功率的估计。'
    if not candidate:
        headline='本日报告只统计下列新增候选试验。此前 V7 的固定布局记录见 [20260912 训练报告](训练进展与接机差距_20260912.md)，不与新候选合并计算。'
    lines=['# 五指稳定性训练与验收记录', '',
        '本轮属于有边界的控制器参数整定与完整物理回归，不是强化学习训练。目标保持五指艺术技能赛项。', '',
        '**当前不得直接接机：正式设备型号、SDK、标定与真机验收仍缺失。默认仿真配置未自动替换。**', '',headline, '',
        '历史流程通过按各批次原口径保留，不代表最新工作表面无碰撞验收通过。v3-robot-surfaces 增加 500 Hz 非笔具机器人环境接触检查；缺失完整记录不能证明通过。旧 V7 完成案例的新几何复核发现手掌碰纸，详见本轮报告。', '',
        '## 全部尝试', '',
        '| 配置 | 尝试数（含中断） | 流程通过 | 质量通过 |', '|---|---:|---:|---:|']
    for name,rows in groups.items():
        lines.append(f"| {name} | {len(rows)} | {sum(x['process_pass'] for x in rows)} | {sum(x['quality_pass'] for x in rows)} |")
    lines += ['', '不同候选版本的结果不能合并为一个模型的成功率。中断和仍在运行的记录不算已完成物理试验；详见每行状态。', '', '流程通过要求：相机 OCR 与考官答案匹配，连续持笔和蘸墨通过，300 秒内完成全部笔画，每笔有实际纸面接触墨迹，机械臂桌面检查通过，结束仍持笔。',
        '', '质量通过额外要求：实际着墨时间占书写时间至少 95%，各笔路径在 2.5 mm 容差内的平均空间覆盖至少 95%。从 V21 批次起使用 v2 门槛，另要求已记录墨线长度中至少 95% 位于计划笔画的 2.5 mm 范围内，用于检查多余连笔；旧批次保留原判定，不追改为 v2。各批次 quality_gate_version 字段记录新口径，旧记录无该字段。这些是工程筛查门槛，不是官方书法评分；不能用空间覆盖掩盖断墨。',
        '', '开发集的题目由评估程序指定，但控制器仍先读取相机图像，OCR 后才查整首诗。random_blind 表示重新随机抽题，仍来自已知五首示例诗库，不等于未见诗题泛化。', '',
        '## 每次物理结果（包含失败）','', '| 配置 / 模式 | 诗题 | 结果 | 秒 | 笔画 | 着墨比例 | 最小臂桌间距 mm | 原始记录 |', '|---|---|---|---:|---:|---:|---:|---|']
    for name,rows in groups.items():
        for row in rows:
            r=row['report'];stop=r['stop_reason'].split(':')[0]
            folder=ROOT/'outputs/blind_competition'/row['case']
            lines.append(f"| {name} / {row['mode']} | {r.get('recognized_title','—')} | {stop} | {r.get('elapsed_sim_s',0):.2f} | {r.get('completed_strokes',0)}/{r.get('planned_strokes',0)} | {r.get('writing_ink_contact_fraction',0):.1%} | {r.get('minimum_moving_arm_table_clearance_mm',0):.3f} | [{row['case']}](<{folder.as_posix()}/run/report.json>) |")
    lines += ['', '## 控制器与实验边界', '',
        '候选 V7 把笔尖相对手掌的距离纳入机械臂运动计算；握稳之后继续监视拇指和食指的接触力，先滤除快速抖动，仅在力低于保留阈值时有限收紧。中指不持续追赶目标力，避免反复推动笔杆转动。指令变化、手指偏移和目标力仍受原有边界约束，笔始终是自由刚体，没有焊接到手上，也没有施加辅助外力。', '',
        '字形轨迹使用连续笔画插值与缓慢起停的速度安排。程序同时检查纸面接触、握持、蘸墨和时间；机械臂及掌部网格每个物理步检查与桌面的间距，触发保护的候选仍被记为失败。', '',
        '每轮由本地程序独立执行：图像 OCR 后查本地诗库，再执行状态机和闭环控制，不需要联网或大模型参与实时决策。当前笔姿态和接触力来自模拟器，尚须接入真实视觉、编码器及触觉反馈。', '',
        '新增了指令频率保持器，能够在 500 Hz 物理步进中模拟 55 Hz 指令发布。训练批次保存抽题前哈希及控制器快照，并在开始每轮前持久化记录；中断不能留下一个看似完整成功的批次。', '',
        '## 接机前尚未通过的项目', '',
        '- 官方机械臂型号、手部电机映射/关节耦合、SDK、通信协议与设备侧停止接口确认。',
        '- 使用真实 RGB-D 与触觉反馈替换当前模拟器笔姿态、接触力真值；当前完整物理动作不能宣称已经仅靠视觉自主适应任意纸笔位置。',
        '- 笔、纸、相机、砚台变化的多次完整动作验证。物理笔偏移实验若通过，只证明对应毫米范围的扰动容忍。相机变化只直接考察当前 OCR 输入。',
        '- 产品资料中的 55 Hz 作为指令带宽实验条件，未确认是比赛接口频率。该实验仍使用 100 Hz 控制计算和 500 Hz 物理步进，不能充当真实设备网络/传感器延迟测试。',
        '- 初赛画龙点睛完整流程、官方龙图和决赛 28/40 字诗题仍需要额外全流程验证；五首 20 字示例诗不能代表全部赛题。',
        '- 真机低速单步、抓笔、持续持笔、接触写字、断连停止、连续多轮复验。仿真不能证明真实毛笔柔顺性、墨液行为或比赛得分。',
        '', '## 文件与复现', '',
        '每个案例保留 camera_rgb.png、OCR 结果、controller.log、physics.csv、replay.npz、report.json、audit.json 和 judge.json。较新的案例另存 controller_snapshot，用于保留该轮控制器和配置源码。失败案例不删除。', '',
        '在本项目 five_finger 目录中运行：','', '```powershell',
        r'.\.venv-retarget\Scripts\python.exe -m experiments.train_robustness --profile config/profile_pinch_reserve_v7.json --titles 春晓 静夜思 登鹳雀楼 江雪 鹿柴',
        r'.\.venv-retarget\Scripts\python.exe -m experiments.train_robustness --profile config/profile_pinch_reserve_v7.json --rounds 3',
        r'.\.venv-retarget\Scripts\python.exe -m experiments.training_report', '```', '',
        '不同物理批次请串行执行。train_robustness 会拒绝第二个同时运行的训练批次；不要另开其他场景构建/物理命令。', '',
        '新资料核对见 [新增资料核对](新增资料核对_20260912.md)。']
    diagnosis=ROOT/f'接触与限位诊断_{a.date}.md'
    if diagnosis.exists():
        lines += ['',f'本日新增控制改动和失败诊断见 [{diagnosis.stem}]({diagnosis.name})；V7 是对照配置，不应把其他候选的失败或通过归入 V7。']
    contact_note=ROOT/f'落笔阻尼与轨迹前馈_{a.date}.md'
    if contact_note.exists():
        lines += ['',f'落笔速度阻尼、前馈及统计口径说明见 [{contact_note.stem}]({contact_note.name})。']
    support_note=ROOT/f'握姿支撑与抬笔高度诊断_{a.date}.md'
    if support_note.exists():
        lines += ['',f'接触位置、力矩、手指支撑与降低抬笔高度的验证见 [{support_note.stem}]({support_note.name})。用户已确认暂时不能预约实际平台调试；真机验收保持待完成。']
    slip_note=ROOT/f'滑移识别与手内姿态保持_{a.date}.md'
    if slip_note.exists():
        lines += ['',f'相对滑移、手内姿态保持与同题复验见 [{slip_note.stem}]({slip_note.name})。全部笔画完成但提笔掉笔仍属于流程失败。']
    pickup_note=ROOT/f'闭指与提笔受力诊断_{a.date}.md'
    if pickup_note.exists():
        lines += ['',f'闭指进度、逐物理步力峰值、掉笔暂停与提笔路径见 [{pickup_note.stem}]({pickup_note.name})。保护复验按实际任务失败计数。']
    validation_path=ROOT/'outputs/robustness'/f'validation_{a.date}.json'
    clearance_note=ROOT/f'间距约束与低速落笔复验_{a.date}.md'
    if clearance_note.exists():
        lines += ['',f'机械臂指令间距约束、低速落笔与实际距离保护复验见 [{clearance_note.stem}]({clearance_note.name})。保护停止仍为任务失败，最终物理状态快照的完整试验产物尚待验证。']
    surface_note=ROOT/f'自转分解与纸面碰撞复核_{a.date}.md'
    if surface_note.exists():
        lines += ['',f'圆笔自转、接触位置分解及新增工作表面碰撞验收见 [{surface_note.stem}]({surface_note.name})。检测与拒绝错误通过已实现，提前避碰的动作修正尚待完成。']
    geometry_note=ROOT/f'接触几何筛选与握姿复验_{a.date}.md'
    if geometry_note.exists():
        lines += ['',f'握姿几何筛选与从张手重新抓笔的物理复验见 [{geometry_note.stem}]({geometry_note.name})。静态可行姿态仍需要动力学验证，新候选未替换默认方案。']
    coupled_note=ROOT/f'共同运动求解与实际接触误差_{a.date}.md'
    if coupled_note.exists():
        lines += ['',f'共同运动求解器的实现与失败验证见 [{coupled_note.stem}]({coupled_note.name})。小模型残差和指令边界通过均不代表物理接触稳定。']
    allocation_note=ROOT/f'握力分配与接触运动协调_{a.date}.md'
    if allocation_note.exists():
        lines += ['',f'三指目标分配、逐控制步平滑及接触运动兼容性见 [{allocation_note.stem}]({allocation_note.name})。该阶段的三次物理试验均失败；后续共同运动求解器的结果见对应报告。']
    wrench_note=ROOT/f'接触力矩与扶正切换诊断_{a.date}.md'
    if wrench_note.exists():
        lines += ['',f'接触面力矩敏感性、执行器输出与扶正模式切换见 [{wrench_note.stem}]({wrench_note.name})。静态可行性与短时切换修复都不等于完整任务通过。']
    posture_note=ROOT/f'握笔关节余量与夹持角度诊断_{a.date}.md'
    if posture_note.exists():
        lines += ['',f'关节调整余量、姿态回拉与保留夹持角度的试验见 [{posture_note.stem}]({posture_note.name})。增加完成笔画但仍失败的候选未被提升为默认方案。']
    if validation_path.exists():
        validation=json.loads(validation_path.read_text(encoding='utf-8'))
        lines += ['',f"软件回归验证：{validation['total_tests_passed']} 项通过。单元/集成测试通过不代表物理任务或真机验收通过。"]
    for row in blind:
        video=ROOT/'outputs/blind_competition'/row['case']/'run/随机诗题_五指物理仿真.mp4'
        if row['process_pass'] and video.exists():
            lines += ['',f'[完整随机测试视频（参考仿真，非参赛合格证明）](<{video.as_posix()}>)']
    destination=ROOT/f'训练进展与接机差距_{a.date}.md'
    destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'outputs/robustness'/f'index_{a.date}.json').write_text(json.dumps({'sources':sources,'profiles':groups,'hardware_ready':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(destination)


if __name__=='__main__':main()
