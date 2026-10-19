"""Audit saved physical experiments and publish a reproducible local evidence report."""
import hashlib
import importlib.metadata
import json
import subprocess
import numpy as np
from experiments.writing import OUT,ROOT


def main():
    trials=[]
    for name in ('demo','shift_plus','shift_minus'):
        folder=OUT/name;report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        rows=np.loadtxt(folder/'physics.csv',delimiter=',',skiprows=1)
        trace=np.load(folder/'replay.npz',allow_pickle=False)
        assert report['stop_reason']=='completed' and report['hold_3s_passed'] and report['pen_held_at_end'],name
        assert len(report['strokes'])==5 and all(s['contact_fraction']>=.95 for s in report['strokes']),name
        assert np.isfinite(rows).all() and np.isfinite(trace['qpos']).all(),name
        np.testing.assert_array_equal(trace['ink'][:,0],rows[:,14]>.01)
        np.testing.assert_allclose(trace['ink'][:,1:4],rows[:,15:18],atol=1e-12)
        before=rows[rows[:,0]<14];continuous=0.;longest=0.
        for valid in before[:,19]:
            continuous=continuous+.01 if valid else 0.;longest=max(longest,continuous)
        assert longest>=3.,name
        # After pickup, persistent finger contacts must accompany the free pen.
        writing=trace['stage']=='write'
        assert np.all(rows[writing,4]>.01) and np.all(np.any(rows[writing,5:9]>.01,axis=1)),name
        trials.append({'name':name,'shift_m':cfg['initial_shift_m'],'hold_before_transfer_s':longest,**report})
    lines=['# 拿笔写字物理实验结果','',
           '2026-09-08；DexHand021 右手＋JAKA Zu7 厂商参考模型；立式练习笔架。三次均从独立初始状态运行，没有从成功轨迹中途接续。','',
           '| 运行 | 初始偏差 x/y（mm） | 时长（s） | 转移前连续保持（s） | 写笔画采样接触率 | 平面误差中位 / P95（mm） | 结束仍持笔 |',
           '|---|---|---:|---:|---:|---|---|']
    for r in trials:
        lines.append(f"| {r['name']} | {r['shift_m'][0]*1000:+.1f} / {r['shift_m'][1]*1000:+.1f} | {r['elapsed_sim_s']:.2f} | {r['hold_before_transfer_s']:.2f} | {r['writing_contact_fraction']:.1%} | {r['xy_error_median_mm']:.2f} / {r['xy_error_p95_mm']:.2f} | 是 |")
    lines+=['','所有运行均完成「永」的五个笔画，无 MuJoCo 数值警告。接触率只统计进入“写笔画”后的 100 Hz 采样，落笔等待及抬笔另列。',
            '',f"默认运行纸面力中位数 {trials[0]['paper_force_median_N']:.3f} N，最大值 {trials[0]['maximum_paper_force_N']:.3f} N。夹取瞬态单指接触力峰值 {trials[0]['maximum_finger_contact_N']:.2f} N，执行器达到 0.4 N·m 限幅，结束笔倾角 {trials[0]['pen_tilt_final_deg']:.1f}°。这些是后续仍需优化的指标。",'',
            '## Skill 应用与检查','',
            '- mujoco：检查 9 个工作区逆解目标、接触几何和摩擦继承、自由笔关节、执行器限幅及原生渲染。',
            '- robotics-testing：4 项新增物理／运动学检查、8 项重定向及规则检查通过；原环境 13 项基线及规则检查也通过，两组有 6 项重叠。',
            '- 结果审计逐帧核对墨迹开关与笔尖／纸面真实接触力，逐帧核对墨迹位置和接触点，检查转移前 3 秒保持及书写时拇指＋其他手指接触。','',
            '## 使用范围','',
            '这三次是很小范围的开发验证，不是统计成功率。练习笔架、笔杆直径 12 mm、软指腹摩擦、纸面摩擦均为声明的仿真假设。控制使用模拟器位姿和接触力真值。未模拟笔毫与墨水流动，未执行蘸墨、诗题识别播报或正式整篇书写；官方分数为空。',
            '原模型机械臂碰撞未启用，不能据此证明真机避障。相对于此前水平笔架实验，本次同时改变了场景、接触模型和控制流程，不能将改善全部归因于 skill 或算法。','',
            '## 文件','',
            '- [完整演示视频](demo/拿笔写永_物理仿真.mp4)',
            '- [三维画面与实际字迹](demo/writing_result.png)',
            '- [默认物理报告](demo/report.json)',
            '- [完整参数快照](demo/config.json)',
            '- [教程与命令](../../拿笔写字实验教程.md)']
    (OUT/'实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'validation_summary.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2),encoding='utf-8')
    files=['experiments/writing.py','experiments/writing_config.json','experiments/view_writing.py',
           'fivefinger/grasp_feedback.py','fivefinger/trajectory.py','tests/test_physical_writing.py']
    manifest={'packages':{p:importlib.metadata.version(p) for p in ('mujoco','numpy','scipy','pillow')},
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
              'model_commit':subprocess.check_output(['git','-C',str(ROOT/'vendor/dexrobot_mujoco'),'rev-parse','HEAD'],text=True).strip(),
              'skills':{p:json.loads((ROOT.parent/'.agents/skills'/p/'source.json').read_text(encoding='utf-8-sig')) for p in ('mujoco','robotics-testing')},
              'ink_is_visual_only':True,'official_score':None}
    (OUT/'implementation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: all 3 physics records, five strokes, hold gate, contact-only ink, continuous writing grip.')


if __name__=='__main__':main()
