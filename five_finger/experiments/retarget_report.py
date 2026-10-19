"""Generate a report from measured trial outputs, not hardcoded success claims."""
import hashlib
import json
import platform
import importlib.metadata
from pathlib import Path
import numpy as np
from fivefinger.paths import ROOT
from experiments.retarget_grasp import OUT


def main():
    rows=[]
    for label,filename,key in [('通用人手姿态','physical_summary.json','results'),
                                ('接触几何修正','refinement_summary.json','results'),
                                ('同目标力调参','feedback_summary.json','development'),
                                ('同目标力位置扰动','feedback_summary.json','validation'),
                                ('分指目标力调参','balanced_summary.json','development'),
                                ('分指目标力位置扰动','balanced_summary.json','validation')]:
        path=OUT/filename
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'))
            trials=data[key]
            rows.append(f'| {label} | {len(trials)} | {sum(r["hold_3s_passed"] for r in trials)} | {max(r["longest_continuous_hold_s"] for r in trials):.2f} |')
    balanced=json.loads((OUT/'balanced_summary.json').read_text(encoding='utf-8'))
    best=max(balanced['development'],key=lambda r:r['longest_continuous_hold_s'])
    validation=balanced['validation']
    detail=[]
    for trial in [best]+validation:
        trace=np.loadtxt(OUT/trial['candidate_id']/'physics_trace.csv',delimiter=',',skiprows=1)
        detail.append({'id':trial['candidate_id'],'longest_hold_s':trial['longest_continuous_hold_s'],
                       'valid_at_end':bool(trace[-1,11]),'support_force_at_end_N':float(trace[-1,9]),
                       'drop_proxy':trial['brush_drop_observed']})
    runtime={p:importlib.metadata.version(p) for p in ['mujoco','torch','nlopt','numpy','opencv-python','scipy']}
    source_paths=['fivefinger/mujoco_kinematics.py','fivefinger/retargeting.py','fivefinger/grasp_feedback.py',
                  'experiments/run_suite.py','experiments/retarget_grasp.py','experiments/refine_grasp.py','experiments/feedback_grasp.py']
    snapshot={'python':platform.python_version(),'packages':runtime,'best_settings':balanced['selected_settings'],
              'best_development_trial':best['candidate_id'],'validation_hold_passes':sum(r['hold_3s_passed'] for r in validation),
              'validation_trials':len(validation),'detail':detail,'official_score':None,
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}}
    (OUT/'implementation_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# 拟人手势接入与接触反馈抓笔实验

日期：2026-09-08。DexPilot 优化器已接入并实际运行；正运动学和雅可比改由 MuJoCo 提供。保留上游 MIT 声明，优化器计算代码未修改，仅替换模块导入；来源见 `fivefinger/third_party/provenance.json`。

**标称场景最好连续稳定持笔 {best['longest_continuous_hold_s']:.2f} 秒；使用该组固定参数的三个位置扰动实验，{sum(r['hold_3s_passed'] for r in validation)}/{len(validation)} 通过 3 秒工程门槛。尚未达到稳定覆盖所有测试位置的程度，未执行蘸墨、点睛或书法。**

## 已实现什么

- MuJoCo 模型的五个真实指尖 body、20 个手指坐标与 DexPilot 对接；中指侧摆坐标固定，实际优化 19 个坐标。它们不是已经标定好的真实电机指令。
- 上游 621 帧通用人手动作中选出六个不同时间段的捏持候选；数据不是毛笔示范。添加自定义 `points_m` NPZ 输入接口。
- 接触修正：让拇指、食指和中指围绕笔杆两侧形成接触；目标重叠量属于位置伺服的预压设定，动态仿真仍由接触求解器处理。
- 力反馈：依据手指与笔杆接触对的力，以及已知仿真笔杆姿态，沿接触法向调整关节目标。它使用仿真真值，尚未替换成实机可用的感知估计。

## 实测对照

| 组别 | 独立实验数 | 通过 3 秒数 | 最长连续稳定/s |
|---|---:|---:|---:|
| 原工程历史基线 | 6 | 0 | 1.09 |
{chr(10).join(rows)}

不同组使用不同初始化或控制设置；这些是开发对照，不是随机独立留出集，也不是正式比赛的两个回合。**本轮握持改善来自接触几何修正与分指力反馈；通用人手姿态组没有通过，因此不能宣称拟人重定向已提高抓笔成功率。**

## 选定的分指力参数

`target_n = {balanced['selected_settings']['target_n']}` N，依次为拇指、食指、中指；`gain_m_per_ns = {balanced['selected_settings']['gain_m_per_ns']}` m/(N·s)。法向速度限幅 ±4 mm/s，关节修正速度限幅 ±0.12 rad/s，累计修正限幅 ±0.25 rad。实验在 t=9.5 s 开始反馈，是该固定阶段时序的开发设置；后续须改为真实状态触发。

固定参数位置扰动：

| 试验 | 世界坐标偏移/m | 连续稳定/s | 门槛 |
|---|---|---:|---|
'''
    for trial in validation:
        report+=f'| {trial["candidate_id"]} | {trial["initial_shift_m"]} | {trial["longest_continuous_hold_s"]:.2f} | {"通过" if trial["hold_3s_passed"] else "未通过"} |\n'
    report+='''
通过门槛指曾连续满足保守工程判据至少 3 秒，不代表直到实验结束都稳定，更不代表提笔后的转腕、蘸墨可完成。最终帧状态另存 `implementation_snapshot.json`。所有毛笔都为自由刚体，控制运行中未焊接、未直接改写笔的位置。回放恢复记录状态，不是新物理实验。

判据沿用既有规则实验：笔中心抬高 >=20 mm、支架支撑力 <0.01 N、拇指和至少另一根手指有接触，笔相对手的滑移速度 <=20 mm/s，连续 >=3 s。除规则中的 3 秒外，其余阈值是工程代理。

## 如何查看与复现

在 `five_finger` 文件夹中运行：

```powershell
.\\.venv-retarget\\Scripts\\python.exe -m experiments.view_retargeting
.\\.venv-retarget\\Scripts\\python.exe -m experiments.view_retargeting --trial balanced_validation_3
```

空格暂停／继续，R 从头回放；鼠标旋转与滚轮缩放。双击 `查看拟人握笔实验.cmd` 可打开标称场景的回放。

双击 `运行拟人握笔实验.cmd` 可重新生成六个候选、运行几何修正、分指反馈和三组扰动，再生成此报告。完整代码说明和参数入口见 `拟人握笔接入与调参.md`。

## 仍需完成

修复 X+1 mm、Y−1 mm 的失败条件；采集真正的毛笔示范；校准真实电机耦合；将仿真真值替换为视觉／触觉估计；在保持握持的同时转为可书写姿态，再完成蘸墨和点睛。手臂碰撞、真实笔毛及吸墨、正式比赛设备和评分均尚未验证。
'''
    (OUT/'接入与实验报告.md').write_text(report,encoding='utf-8')
    print(OUT/'接入与实验报告.md')


if __name__=='__main__':
    main()
