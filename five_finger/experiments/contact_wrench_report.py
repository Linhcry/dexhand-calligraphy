"""Summarize contacts captured during simulation, not recomputed from replay."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(run):
    run=Path(run)
    with np.load(run/'contact_trace.npz') as archive:trace={k:archive[k] for k in archive.files}
    with np.load(run/'replay.npz') as archive:stage=archive['stage'].copy();times=archive['time_s'].copy()
    phys=np.loadtxt(run/'physics.csv',delimiter=',',skiprows=1)
    if not np.array_equal(trace['time_s'],times) or not np.allclose(phys[:,0],times,atol=1e-9,rtol=0):raise ValueError('Mismatched contact timeline')
    if not np.allclose(trace['normal_N'][:,:5],phys[:,4:9],atol=1e-8,rtol=0) or not np.allclose(trace['normal_N'][:,5],phys[:,14],atol=1e-8,rtol=0):
        raise ValueError('Contact recording does not match original normal forces')
    normal=trace['normal_N'];torque=trace['wrench_world'][:,:,3:];axis=trace['brush_axis_world']
    hand=torque[:,:5].sum(axis=1);paper=torque[:,5]
    hand_bend=hand-(hand*axis).sum(axis=1)[:,None]*axis
    paper_bend=paper-(paper*axis).sum(axis=1)[:,None]*axis
    active=np.isin(stage,['write','touchdown','pen_up'])
    paired=active&(normal[:,0]>.01)&(normal[:,1]>.01)
    separation=np.abs(trace['position_brush_m'][:,0,2]-trace['position_brush_m'][:,1,2])*1000
    samples=[]
    for when in [30.,60.,100.,140.,180.,220.,float(times[-1])]:
        if when>times[-1]:continue
        i=min(np.searchsorted(times,when),len(times)-1)
        samples.append({'time_s':float(times[i]),'stage':str(stage[i]),'finger_normal_N':normal[i,:5].tolist(),
                        'contact_z_brush_mm':[None if not np.isfinite(v) else float(v*1000) for v in trace['position_brush_m'][i,:5,2]],
                        'hand_bending_moment_Nm':float(np.linalg.norm(hand_bend[i])),
                        'paper_bending_moment_Nm':float(np.linalg.norm(paper_bend[i]))})
    result={'source':'contact_trace.npz recorded at runtime, 100 Hz, simulation ground truth',
            'force_and_timeline_crosscheck_passed':True,'torque_reference':'world axes about brush COM',
            'middle_contact_fraction_during_execution':float(np.mean(normal[active,2]>.01)) if active.any() else None,
            'thumb_index_axial_separation_median_mm':float(np.median(separation[paired])) if paired.any() else None,
            'paper_bending_moment_peak_Nm':float(np.max(np.linalg.norm(paper_bend[active],axis=1))) if active.any() else None,
            'contact_span_is_not_force_closure_proof':True,'hardware_ready':False,'samples':samples}
    if 'friction_utilization_max' in trace:
        result['sliding_diagnostics']={}
        for group,name in enumerate(('thumb','index','middle','ring','little','paper','other')):
            mask=active&(normal[:,group]>.01)
            if not mask.any():continue
            usage=trace['friction_utilization_max'][mask,group]
            speed=trace['tangential_relative_speed_m_s'][mask,group]
            result['sliding_diagnostics'][name]={
                'contact_samples':int(mask.sum()),
                'friction_cone_utilization_p95':float(np.quantile(usage,.95)),
                'friction_cone_utilization_above_0_9_fraction':float(np.mean(usage>.9)),
                'tangential_speed_median_mm_s':float(np.median(speed)*1000),
                'tangential_speed_p95_mm_s':float(np.quantile(speed,.95)*1000)}
        result['sliding_diagnostics_note']='Instantaneous 100 Hz samples, force-weighted point speed and maximum per-contact cone ratio. Do not integrate sparse velocities as reliable slip distance.'
    (run/'contact_wrench_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.run),ensure_ascii=False,indent=2))
