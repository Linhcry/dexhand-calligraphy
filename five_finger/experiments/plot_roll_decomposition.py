"""Plot saved thumb contact, material rotation and relative axis tilt separately."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fivefinger.paths import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cases',nargs='+')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for item in parser.parse_args().cases:
        label,directory=item.split('=',1)
        result=json.loads((Path(directory)/'run/contact_roll_decomposition.json').read_text(encoding='utf-8'))
        rows=[r for r in result['samples'] if r['finger']=='thumb']
        times=[];values=[];previous=None
        for row in rows:
            t=row['time_s']-result['source_time_s']
            if previous is not None and t-previous>.21:
                times.append((t+previous)/2);values.append([np.nan]*3)
            times.append(t);values.append(row['angle_changes_deg'][:2]+[row['axis_tilt_from_hold_deg']]);previous=t
        for j,ax in enumerate(axes):ax.plot(times,np.asarray(values)[:,j],label=label,linewidth=1.2)
    for ax,label in zip(axes,['Thumb contact migration\nin transported hand frame (deg)',
            'Tool material rotation\nabout transported axis (deg)','Tool-axis tilt from hold\nin hand frame (deg)']):
        ax.set_ylabel(label,fontsize=9);ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[-1].set_xlabel('Time since end of acquired hold (s)')
    fig.suptitle('Separate coordinates: contact position, material rotation, axis tilt\n5 Hz measured contacts + saved kinematics; no stability certification',fontsize=11)
    destination=ROOT/'outputs/robustness/contact_roll_decomposition_20260913.png'
    fig.savefig(destination,dpi=150);plt.close(fig);print(destination)


if __name__=='__main__':main()
