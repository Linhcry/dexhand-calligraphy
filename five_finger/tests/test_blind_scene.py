import json
import mujoco
import numpy as np
import pytest
from experiments.blind_competition import InkState,nib_point
from experiments.competition_scene import make_scene,load_settings,TableClearance
from fivefinger.trajectory import make_path
from fivefinger.paths import ROOT


def test_ink_requires_depth_dwell_withdrawal_and_real_contact():
    s=load_settings();ink=InkState(s);p=np.r_[s['ink_center_xy_m'],s['ink_surface_z_m']-.002]
    for _ in range(100):ink.update(p,.01)
    assert not ink.loaded and not ink.mark(1,p)
    p[2]-=.003
    for _ in range(40):ink.update(p,.01)
    ink.update(p+[1,0,0],.01)
    assert ink.dwell==0 and not ink.loaded
    for _ in range(51):ink.update(p,.01)
    assert ink.loaded and not ink.mark(1,p)
    p[2]+=.025;ink.update(p,.01)
    assert ink.withdrawn and not ink.mark(0,p) and ink.mark(.1,p)
    ink.mark(.1,p+[.01,0,0]);assert ink.remaining==pytest.approx(s['ink_capacity_m']-.01)
    ink.remaining=0;assert not ink.mark(1,p)


def test_arm_colliders_and_mesh_guard_reject_below_table_pose():
    m,d,*_=make_scene();guard=TableClearance(m)
    assert guard.require_clear(d)>.01
    for k in range(7):
        g=m.geom(f'collision_r_a_link{k}');assert g.contype[0]&2 and g.conaffinity[0]&4
    assert m.geom('table').conaffinity[0]&2
    # Known formerly rendered elbow-down pose: independent mesh guard must reject it.
    d.qpos[:6]=[-.539,3.18,2.57,-.608,2.11,0];mujoco.mj_forward(m,d)
    with pytest.raises(RuntimeError,match='clearance violation'):guard.require_clear(d)


def test_nib_point_is_capsule_lowest_point_not_fixed_site():
    m,d,*_=make_scene();qadr=m.joint('brush_free').qposadr[0] if 'brush_free' in [m.joint(i).name for i in range(m.njnt)] else 26
    d.qpos[qadr+3:qadr+7]=[np.cos(.3),np.sin(.3),0,0];mujoco.mj_forward(m,d)
    g=m.geom('brush_tip_approximation');pose=d.geom(g.id)
    expected=pose.xpos[2]-abs(pose.xmat.reshape(3,3)[2,2])*g.size[1]-g.size[0]
    assert nib_point(m,d)[2]==pytest.approx(expected)


def test_all_example_poems_fit_a3_with_margin_and_no_missing_strokes():
    s=load_settings();bank=json.loads((ROOT/'config/example_poems.json').read_text(encoding='utf-8'))
    low=np.array(s['paper_center_xy_m'])-np.array(s['paper_size_m'])/2+.005
    high=np.array(s['paper_center_xy_m'])+np.array(s['paper_size_m'])/2-.005
    for poem in bank.values():
        assert len(poem)==20
        p=make_path(poem,origin_xy=s['origin_xy_m'],character_size=s['character_size_m'],gap=s['character_gap_m'],
                    speed=s['writing_speed_m_s'],travel_speed=s['travel_speed_m_s'],clearance=s['clearance_m'])
        assert (p.xyz[:,:2]>=low).all() and (p.xyz[:,:2]<=high).all()
        assert p.time_s[-1]+60<300
