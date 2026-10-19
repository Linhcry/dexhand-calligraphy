"""Plot saved relative pose, not sparse velocity integrals or rendered text."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',action='append',nargs=2,metavar=('LABEL','CASE'),required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(2,1,figsize=(11,7.5),sharex=True)
    titles=set()
    for label,case in args.case:
        run=Path(case)/'run'
        with np.load(run/'relative_motion.npz') as a:motion={k:a[k] for k in a.files}
        with np.load(run/'replay.npz') as a:stages=a['stage'].copy()
        report=json.loads((run/'report.json').read_text(encoding='utf-8'))
        titles.add(report['recognized_title'])
        start=np.flatnonzero(stages=='to_ink')[0]
        time=motion['time_s'][start:]-motion['time_s'][start]
        displacement=np.linalg.norm(motion['center_in_palm_m'][start:]-motion['center_in_palm_m'][start],axis=1)*1000
        angle=np.degrees(np.arccos(np.clip(motion['axis_in_palm'][start:]@motion['axis_in_palm'][start],-1.,1.)))
        status='完成流程，质量未过' if report['stop_reason']=='completed' else '未完成'
        color=axes[0].plot(time,displacement,label=f'{label}：{status}',lw=1.5)[0].get_color()
        axes[1].plot(time,angle,color=color,lw=1.5)
        for ax,values in zip(axes,(displacement,angle)):
            ax.plot(time[-1],values[-1],marker='o' if report['stop_reason']=='completed' else 'x',color=color)
    axes[0].set_ylabel('笔心相对手掌的偏移 / mm',fontproperties=font)
    axes[1].set_ylabel('笔轴相对初始方向的变化 / °',fontproperties=font)
    axes[1].set_xlabel('抓稳移向砚台后经过时间 / s',fontproperties=font)
    axes[0].legend(prop=font,fontsize=9,loc='upper left')
    for ax in axes:
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    fig.suptitle('握持漂移诊断：直接由保存的相对姿态计算',fontproperties=font,fontsize=16)
    text=f'图中均为开发诗题《{next(iter(titles))}》；' if len(titles)==1 else '图中含不同开发诗题；'
    fig.text(.08,.015,text+'配置或启动时序不同，不能估计成功率；100 Hz 记录不包含采样间运动。',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.96))
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160);plt.close(fig)
    print(destination)


if __name__=='__main__':main()
