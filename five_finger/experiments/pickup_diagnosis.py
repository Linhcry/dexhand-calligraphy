"""Audit pickup kinematics and recorded forces; do not solve contact forces again."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from experiments.view_blind_competition import load


def audit(case):
    run,cfg,m,d,tr,physics,report,events=load(case)
    indices=np.flatnonzero(np.isin(tr['stage'],['lift','hold']))
    if not len(indices):raise ValueError('No pickup samples')
    palms=[]
    for i in indices:
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d)
        palms.append(d.site('palm_control').xpos.copy())
    palms=np.asarray(palms);times=tr['time_s'][indices]
    healthy=(physics[indices,4]>.01)&np.any(physics[indices,5:9]>.01,axis=1)
    peak=json.loads((run/'contact_force_peaks_500hz.json').read_text(encoding='utf-8'))
    result=dict(case=run.parent.name,source='saved qpos -> kinematics, original 100 Hz CSV and runtime 500 Hz peak monitor',
        stop_reason=report['stop_reason'],minimum_arm_table_clearance_mm=report['minimum_moving_arm_table_clearance_mm'],
        maximum_palm_height_m=float(palms[:,2].max()),last_palm_height_m=float(palms[-1,2]),
        maximum_horizontal_palm_excursion_mm=float(np.linalg.norm(palms[:,:2]-palms[0,:2],axis=1).max()*1000),
        missing_pinch_samples=int((~healthy).sum()),
        tracking_pause_seconds=report.get('tracking_pause_seconds',0.),
        close_finger_peaks_N={name:peak['stage_maxima_N'].get('close',{}).get(name) for name in ('thumb','index','middle')},
        lift_other_contact_normal_sum_peak_N=peak['stage_maxima_N'].get('lift',{}).get('other'),
        force_definition='sum of contact normal magnitudes per group, not net wrench magnitude',hardware_ready=False)
    (run/'pickup_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(run/'pickup_trajectory.npz',time_since_lift_s=times-times[0],palm_world_m=palms,
        nib_world_m=physics[indices,11:14],support_normal_sum_N=physics[indices,9],healthy_pinch=healthy)
    return result


def plot(cases,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,(ax,bars)=plt.subplots(2,1,figsize=(11,8))
    values=[];labels=[];colors=[]
    for label,case in cases:
        result=audit(case);run=Path(case)/'run'
        with np.load(run/'pickup_trajectory.npz') as tr:
            line=ax.plot(tr['time_since_lift_s'],tr['palm_world_m'][:,2],label=label,lw=2)[0]
        values.append(result['lift_other_contact_normal_sum_peak_N']);labels.append(label);colors.append(line.get_color())
    ax.set_ylabel('手掌控制点高度 / m',fontproperties=font)
    ax.set_xlabel('提笔开始后 / s',fontproperties=font);ax.grid(alpha=.2);ax.legend(prop=font)
    patches=bars.bar(np.arange(len(labels)),values,color=colors,width=.55)
    bars.set_xticks(np.arange(len(labels)),labels,fontproperties=font)
    bars.set_ylabel('提笔时其他接触法向力之和峰值 / N',fontproperties=font)
    for p,value in zip(patches,values):bars.text(p.get_x()+p.get_width()/2,value+2,f'{value:.2f}',ha='center')
    bars.set_ylim(0,max(values)*1.18);bars.grid(axis='y',alpha=.2)
    fig.suptitle('提笔路径与掉笔保护：保存的物理试验',fontproperties=font,fontsize=16)
    fig.text(.06,.02,'上图为 100 Hz 姿态；下图为 500 Hz 记录峰值。其他接触在本组提笔场景主要来自笔架，非合力大小。',fontproperties=font,fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(output,dpi=160);plt.close(fig)
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',action='append',nargs=2,required=True);p.add_argument('--plot')
    args=p.parse_args()
    if args.plot:plot(args.case,args.plot)
    else:
        for label,case in args.case:print(json.dumps(audit(case),ensure_ascii=False))
