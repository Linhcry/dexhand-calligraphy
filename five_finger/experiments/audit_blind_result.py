"""Independent replay audit. Metrics are engineering diagnostics, never contest scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import cKDTree
from experiments.view_blind_competition import load,apply
from experiments.competition_scene import TableClearance
from fivefinger.trajectory import make_path
from fivefinger.paths import ASSETS
from fivefinger.ink_quality import measure_ink_precision
from fivefinger.robot_surface_monitor import surface_trace_passes


def audit(case):
    run,cfg,m,d,tr,phys,report,events=load(case);s=cfg['scene'];guard=TableClearance(m)
    for i in range(len(tr['time_s'])):
        d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);guard.require_clear(d)
    path=make_path(report['recognized_poem'],origin_xy=s['origin_xy_m'],character_size=s['character_size_m'],gap=s['character_gap_m'],
        landing_speed=s.get('landing_speed_m_s'),landing_band=s.get('landing_band_m',.002),
        paper_z=.0019,clearance=s['clearance_m'],speed=s['writing_speed_m_s'],travel_speed=s['travel_speed_m_s'],columns=s['columns'],continuous_strokes=s.get('continuous_strokes',False),stroke_timing=s.get('stroke_timing','quintic'))
    change=np.diff(np.r_[False,path.pen_down,False].astype(int));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)
    strokes=[];labels=[]
    for char in report['recognized_poem']:
        n=len(json.loads((ASSETS/'hanzi'/f'{char}.json').read_text(encoding='utf-8'))['medians']);labels.extend([char]*n)
    for number,(a,b) in enumerate(zip(starts,ends)):
        mask=(tr['path_index']>=a)&(tr['path_index']<=b)&(tr['ink'][:,0]>0)
        points=tr['ink'][mask,1:3];coverage=0.
        if len(points):coverage=float(np.mean(cKDTree(points).query(path.xyz[a:b,:2])[0]<=.0025))
        strokes.append({'character':labels[number],'stroke':number+1,'completed':bool(tr['path_index'][-1]>=b),
                        'sample_coverage_within_2_5mm':coverage,'actual_contact_samples':len(points)})
    first_ink=tr['time_s'][np.flatnonzero(tr['ink'][:,0]>0)[0]] if (tr['ink'][:,0]>0).any() else None
    withdrawal=next((x['sim_s'] for x in events if x['event']=='to_paper'),None)
    surface_path=run/'robot_surface_contacts_500hz.json'
    surface_trace=json.loads(surface_path.read_text(encoding='utf-8')) if surface_path.exists() else None
    checks={'finished_before_300s':report['stop_reason']=='completed' and report['elapsed_sim_s']<=300,
        'hold_and_dip_gates':report['hold_3s_passed'] and report['ink_loaded'] and report['ink_withdrawn'],
        'all_stroke_paths_finished':all(x['completed'] for x in strokes),
        'every_stroke_has_measured_ink':all(x['actual_contact_samples']>0 for x in strokes),
        'marks_require_real_positive_paper_force':bool(np.all(phys[tr['ink'][:,0]>0,14]>.01)),
        'no_ink_before_withdrawal':first_ink is not None and withdrawal is not None and first_ink>=withdrawal,
        'robot_meshes_above_table_at_saved_100hz':guard.minimum>=0,
        'per_step_500hz_runtime_guard_passed':report['arm_table_collision_enabled'] and report['minimum_moving_arm_table_clearance_mm']>=1,
        'non_tool_robot_surface_contacts_absent_500hz':surface_trace_passes(surface_trace,len(tr['time_s']),report.get('robot_non_tool_surface_contact_free_500hz')),
        'pen_still_gripped':report['pen_held_at_end']}
    checks={key:bool(value) for key,value in checks.items()}
    result={'checks':checks,'all_process_checks_passed':all(checks.values()),'strokes':strokes,'process_gate_version':'v3-robot-surfaces',
        'mean_stroke_coverage_2_5mm':float(np.mean([x['sample_coverage_within_2_5mm'] for x in strokes])),
        'minimum_arm_mesh_clearance_mm':guard.minimum_arm*1000,'minimum_including_fixed_base_mm':guard.minimum*1000,
        'official_score':None,'quality_note':'Spatial sample coverage is diagnostic; it is not official calligraphy/font similarity or OCR correctness.'}
    result.update(measure_ink_precision(tr['ink'],path.xyz[path.pen_down,:2]))
    (run/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='strokes'},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();r=audit(a.case)
    raise SystemExit(0 if r['all_process_checks_passed'] else 2)
