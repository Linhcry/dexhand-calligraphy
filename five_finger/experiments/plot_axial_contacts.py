"""Plot saved measured contact migration; no scene rebuild or force solve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.grasp_geometry_audit import GROUPS
from fivefinger.paths import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',nargs='+',help='LABEL=case_directory')
    args=parser.parse_args()
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for item in args.cases:
        label,directory=item.split('=',1)
        run=Path(directory)/'run'
        samples=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))['samples']
        hold=[s for s in samples if s['stage']=='hold']
        if not hold:continue
        start=hold[-1]['time_s'];times=[];spans=[];angles=[]
        for sample in samples:
            if sample['time_s']<start:continue
            center={}
            for group in GROUPS:
                active=[c for c in sample['contacts'] if c['group']==group and c['normal_N']>.05]
                if active:
                    center[group]=np.average([c['position_brush_m'] for c in active],axis=0,
                        weights=[c['normal_N'] for c in active])
            times.append(sample['time_s']-start)
            spans.append((center['index'][2]-center['middle'][2])*1000
                if 'index' in center and 'middle' in center else np.nan)
            angles.append(np.arctan2(center['thumb'][1],center['thumb'][0])
                if 'thumb' in center else np.nan)
        angles=np.asarray(angles);valid=np.isfinite(angles)
        if valid.any():
            angles[valid]=np.unwrap(angles[valid]);angles[valid]-=angles[valid][0]
        axes[0].plot(times,spans,label=label,linewidth=1)
        axes[1].plot(times,np.rad2deg(angles),label=label,linewidth=1)
    axes[0].set(ylabel='Index minus middle axial span (mm)',title='Measured contact migration in brush coordinates')
    axes[1].set(ylabel='Thumb contact azimuth change (degrees)',xlabel='Time since end of acquired hold (s)')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('5 Hz runtime snapshots | gaps = missing contacts | geometry does not certify stability',fontsize=11)
    destination=ROOT/'outputs/robustness/axial_contact_migration_20260913.png'
    fig.savefig(destination,dpi=160);plt.close(fig)
    print(destination)


if __name__=='__main__':main()
