"""Measure acquired contact geometry from saved runtime contact snapshots.

No scene construction, contact force reconstruction, or robot state mutation.
Weighted contact centroids are descriptive measurements, not force-closure tests.
"""
import argparse
import json
from pathlib import Path

import numpy as np


GROUPS=('thumb','index','middle')


def summarize_window(samples):
    points={group:[] for group in GROUPS}
    spans=[]
    for sample in samples:
        centroids={}
        for group in GROUPS:
            active=[c for c in sample['contacts'] if c['group']==group and c['normal_N']>.05]
            if not active:continue
            weights=np.array([c['normal_N'] for c in active])
            centroid=np.average([c['position_brush_m'] for c in active],axis=0,weights=weights)*1000
            points[group].append(centroid)
            centroids[group]=centroid
        if 'index' in centroids and 'middle' in centroids:
            spans.append(float(centroids['index'][2]-centroids['middle'][2]))
    return {
        'sample_count':len(samples),
        'median_force_weighted_contact_center_brush_mm':{
            group:np.median(values,axis=0).tolist() if values else None for group,values in points.items()},
        'contact_present_sample_count':{group:len(values) for group,values in points.items()},
        'median_index_minus_middle_axial_span_mm':float(np.median(spans)) if spans else None,
    }


def analyze(case):
    run=Path(case).resolve()/'run'
    if not (run/'report.json').exists():raise ValueError('Only completed runs can be audited')
    samples=json.loads((run/'contact_geometry_samples.json').read_text(encoding='utf-8'))['samples']
    hold=[s for s in samples if s['stage']=='hold']
    writing=[s for s in samples if s['stage']=='write']
    windows={
        'last_hold_0_5s':[s for s in hold if s['time_s']>=hold[-1]['time_s']-.5] if hold else [],
        'first_write_0_5s':[s for s in writing if s['time_s']<=writing[0]['time_s']+.5] if writing else [],
        'last_write_0_5s':[s for s in writing if s['time_s']>=writing[-1]['time_s']-.5] if writing else [],
    }
    result={'case':run.parent.name,'windows':{k:summarize_window(v) for k,v in windows.items()},
        'hardware_ready':False,
        'limitations':'Runtime contact centroids in brush body coordinates, normal-force weighted above 0.05 N, 5 Hz samples. These differ from scratch capsule centerline points. Axial span is not a stability or hardware acceptance metric.'}
    (run/'grasp_geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',nargs='+')
    for case in parser.parse_args().cases:
        result=analyze(case)
        print(json.dumps({'case':result['case'],'windows':result['windows']}))
