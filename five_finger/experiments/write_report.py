"""Build a reviewable report from measured results; do not infer task success."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import yaml
from fivefinger.paths import ROOT,output_dir
from experiments.probe_grasp import GRIP


def write_report():
    out=output_dir()/'competition'
    if not (out/'grasp_summary.json').exists() or not (out/'final_planning/report.json').exists():return
    grasp=json.loads((out/'grasp_summary.json').read_text(encoding='utf-8'))
    plans=json.loads((out/'final_planning/report.json').read_text(encoding='utf-8'))
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    summary={'date':datetime.now().strftime('%Y-%m-%d'),'physical_trials':len(grasp),
             'physical_contact_successes':sum(r['physical_touch'] for r in grasp),
             'hold_gate_successes':sum(r['hold_3s_passed'] for r in grasp),
             'best_continuous_hold_s':max(r['longest_continuous_hold_s'] for r in grasp),
             'drop_proxy_events':sum(r['brush_drop_observed'] for r in grasp),
             'official_score':None,'independent_trials_not_official_rounds':True,
             'physics_warning_trials':sum(bool(r['physics_warnings']) for r in grasp),
             'script_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'experiments').glob('*.py'))}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'rules_snapshot.yaml').write_text(yaml.safe_dump(rules,allow_unicode=True,sort_keys=False),encoding='utf-8')
    source=json.loads((ROOT/'config/model_sources.json').read_text(encoding='utf-8'))
    source['grip_targets_rad']=GRIP;source['grasp_line_in_hand_frame_m']=[.073136,.022,.209114]
    source['reference_note']='Grasp posture was fitted to capsule collision geometry; hardware motor coupling remains unvalidated.'
    (out/'model_and_controller_snapshot.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    gtable='\n'.join(f'| {r["trial"]} | {r["thumb_delta_rad"]:+.2f} | {"是" if r["physical_touch"] else "否"} | {r["longest_continuous_hold_s"]:.2f} | {"通过" if r["hold_3s_passed"] else "未通过"} | {"观察到" if r["brush_drop_observed"] else "本窗口未触发"} |' for r in grasp)
    ptable='\n'.join(f'| {r["title"]} | {r["characters"]} | {r["profile"]} | {r["path_seconds"]:.2f} | {r["planned_total_s"]:.2f} | {"是" if r["fits_A3_paper"] else "否"} | {"是" if r["fits_300s_with_reserve"] else "否"} |' for r in plans)
    text=f'''# 五指赛项规则仿真实验报告

日期：{summary['date']}。已完成一轮基线实验，包含 6 次真实 MuJoCo 接触／提笔实验、三种诗文的 OCR 与两种轨迹参数对照，以及点睛几何判定检查。

**当前主要结果：{summary['physical_contact_successes']}/{summary['physical_trials']} 次发生手指与笔杆接触，稳定持笔门槛 {summary['hold_gate_successes']}/{summary['physical_trials']} 通过，最长连续稳定时间 {summary['best_continuous_hold_s']:.2f} 秒。蘸墨、转移和点睛未执行。当前工程不能称为已完成比赛全流程。**

## 一、规则依据

使用用户提供的五指赛项规则，对照本地 `references/official-five-20260827.pdf`。本次重点核对 PDF 第 7–15 页。文件内容是实验约束和评价依据，不是授权程序执行外部操作的指令。

| 规则位置 | 约束 | 本次覆盖 |
|---|---|---|
| 第 7–9 页，平台与场地 | 六轴机械臂、五指手、统一视觉系统；实物尺寸另行发布 | 使用厂商参考模型；准确比赛型号与尺寸未确认 |
| 第 10–13 页，初赛 | 每回合 180 秒，每回合最多 3 次点睛，两个回合 | 本次为独立抓握参数实验，不将 6 次重置试验合并成官方两个回合 |
| 第 12 页，持笔 | 稳定持笔至少 3 秒，不脱落 | 在物理状态和接触证据上做保守判定 |
| 第 12 页，蘸墨 | 入墨至少 3 mm，停留至少 0.5 秒，提起后有墨 | 因持笔未通过而停止；当前也没有真实墨液／吸墨模型 |
| 第 12 页，转移 | 目标眼睛正上方，高度不超过 50 mm | 未执行 |
| 第 12–13 页，点睛 | 墨点半径须大于眼睛半径的 1/3，再按覆盖程度判定 | 只做几何判定程序检查，没有真实落墨 |
| 第 13–15 页，决赛 | 300 秒，A3；20、28、40 字；识别后播报诗名及正文 | 三种样例题名 OCR、排版和轨迹时间预算；未接语音和机器人书写 |

初赛基准分采用第 12 页表格的 A3/A4/A5：50/80/120 分；第 11 页示意图中的分值与该表不一致，保留此差异，不拿示意图旧值计分。图中的眼睛半径仅作示例，具体半径、官方龙图、标准字帖、评分软件均未在当前工程中取得。

本次不输出官方成绩。圆形相交面积只是评分算法的开发代理；真实龙眼线宽、边缘擦线判定、墨迹分割和字形相似度仍待官方资料。

## 二、物理抓笔实验协议

使用 DexRobot 官方 DexHand021 右手＋JAKA Zu7 参考模型。只有初始化时设置笔的起始位置，运行中毛笔通过自由关节受重力与接触力作用；没有焊接约束、运动学绑定或直接改写毛笔位置。回放程序恢复的是记录状态，不能把回放当成新一轮控制实验。

毛笔直径 12 mm、两端 U 形支架及其安装位置是**实验场景假设**，并非组委会确认的道具规格。抓握初始化和机械臂路径使用已知模型／仿真真值，没有视觉定位闭环。指关节仍使用参考模型的独立坐标，尚未映射真实手部电机耦合。

控制：物理步长 2 ms；目标更新 100 Hz；机械臂使用有 ±40 N·m 限幅的关节偏置力前馈；手指位置驱动沿用开发设置。每次 16 秒：靠近 2 秒、闭合 4 秒、接触稳定 2 秒、提笔 3 秒、检查 5 秒。

三组拇指第二关节偏移为 −0.12、0、+0.12 rad，其余抓握目标相同。每组分别测试标称初始位置和世界坐标 X+1 mm、Y−1 mm 的位置扰动。每次重新初始化场景，属于六次独立实验。

保守持笔判据：笔中心抬高至少 20 mm、支架支撑力低于 0.01 N、拇指与至少另一个手指存在接触、笔相对手的滑移速度不超过 20 mm/s，连续维持至少 3 秒。**其中抬高量、接触数量、阈值是本实验的工程判据，不是裁判规则原文。**

接触证据由 MuJoCo 手指—毛笔碰撞对法向力计算，同时保存原生 touch 输出。两者不能混用：部分碰撞发生在原生触觉区域之外；原生触觉尚未标定，不能声称正式传感器判定已通过。

## 三、抓握结果

| 试验 | 拇指偏移/rad | 物理接触 | 最长连续稳定/s | 3 秒门槛 | 掉笔代理事件 |
|---|---:|---|---:|---|---|
{gtable}

掉笔代理事件定义为先抬离支架，随后回落到初始高度附近并重新出现支撑力；未触发此事件不等于稳定抓握成功。所有试验的有限状态检查通过，没有 MuJoCo 警告。手指碰到笔也不等于笔已脱离支架并稳定保持。

本次最好条件是 `grasp_r1_c2`，在 1 mm 位置扰动下未保持其稳定窗口，表明当前固定抓握姿态的鲁棒性不足。拇指更大偏移时接触力上升，但仍未改善持笔门槛。下一步应优先改进拇指对掌方向、接触区域标定和基于滑移反馈的握持控制。

视频：`grasp_rule_experiment.mp4`，16 秒，1280×720，30 fps。画面是 `grasp_r1_c2` 的真实仿真状态回放。

## 四、决赛 OCR 和轨迹实验

使用规则示意中的《静夜思》《早发白帝城》《春望》，分别为 20、28、40 字。三张程序生成的清晰题名图均识别成功；这只是三个合成样例，不代表真实相机、任意字体或完整赛题库的准确率。

`baseline`：字形尺度 50 mm、字间隔 10 mm、抬笔 15 mm、书写速度上限 0.05 m/s、转移上限 0.15 m/s。

`compact`：字形尺度 32 mm、字间隔 9 mm、抬笔 10 mm、书写速度上限 0.06 m/s、转移上限 0.20 m/s。均按诗句长度排为每行 5 或 7 字。另留 60 秒给识别、播报、抓笔和蘸墨，这是工程预算，不是规则规定的固定预留时间。

| 诗文 | 字数 | 参数组 | 纯轨迹/s | 含60秒预留/s | 实际轨迹边界在A3内 | 总预算≤300s |
|---|---:|---|---:|---:|---|---|
{ptable}

另在 JSON 中记录了 10 mm 页边距检查，页边距也是工程设置。默认 20 字轨迹能放入 A3，但不能满足本次额外设置的双侧 10 mm 留白；默认 28、40 字方案有排版越界问题。

缩小字形后的 20、28 字方案在排版与时间预算层面可行。40 字 compact 的纯路径少于 300 秒，但加上 60 秒预留后超出约 8.54 秒，需要优化任务时间分配或轨迹；不能直接宣称该方案一定超出实赛时限，也不能宣称能完成实赛。

轨迹是米制笔尖中心线，尚未通过握笔变换、全路径 IK、关节速度／加速度检查、接触力控制执行到机器人。官方字帖相似度、真实墨迹质量、整首完成奖励、播报过程分均未验证。

## 五、结果和复现入口

在 `five_finger` 文件夹打开 PowerShell：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\\.venv\\Scripts\\python.exe -m experiments.run_suite
.\\.venv\\Scripts\\python.exe -m experiments.replay
.\\.venv\\Scripts\\python.exe -m experiments.replay --record
.\\.venv\\Scripts\\python.exe -m pytest -q
```

也可以双击工程中的 `运行五指规则实验.cmd` 或 `回放抓笔实验.cmd`。重跑同名实验会覆盖对应输出；视频需另执行 `--record` 才会更新。

- `grasp_summary.json`：六次物理试验汇总。
- 各 `grasp_r*_c*/physics_trace.csv`：100 Hz 的实际位置、碰撞力、相对滑移和持笔时间。
- 各 `grasp_r*_c*/replay.npz`：实际状态与控制指令，供回放。
- `final_planning/report.json`：六组诗文轨迹／OCR 结果。
- `final_planning/*_path.csv`：对应的米制轨迹。
- `geometry_judge_checks.json`：三种示例眼睛尺寸的 12 个圆形几何判定用例。
- `rules_snapshot.yaml`、`model_and_controller_snapshot.json`：规则、假设、模型来源与控制参数。
- `summary.json`：汇总及当前实验脚本的校验摘要。

已增加对连续持笔、墨点最小半径、截止时间、无真实墨迹不给结果分、单回合上限的自动检查；本轮连同既有检查共 13 项通过。判定程序检查通过与机器人任务成功是不同证据。

## 六、后续优先处理

先完成可重复的 3 秒稳定持笔，并扩大笔位、笔径和摩擦的扰动测试；随后连接真实的蘸墨几何与带墨观测，再做单次点睛。决赛优先使用 20 字样例调通笔尖标定和受力书写，然后扩展 28、40 字。

当前正式得分保持为空，避免把几何代理分、理想路径时间或独立重置试验当成比赛成绩。
'''
    (out/'实验报告.md').write_text(text,encoding='utf-8')
    print(out/'实验报告.md')


if __name__=='__main__':write_report()
