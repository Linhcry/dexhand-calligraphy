import json
import pytest
from experiments.competition_scene import make_scene
from fivefinger.grasp_posture import apply_grasp_offsets
from fivefinger.paths import ROOT


def test_offsets_are_bounded_and_applied_only_once_in_saved_scene_replay():
    parent=json.loads((ROOT/'config/profile_palm_opposition_v35_nominal.json').read_text(encoding='utf-8'))
    parent['grip_target_offsets_rad']={'r_f_joint2_2':.01}
    *_,resolved=make_scene(settings=parent)
    *_,replayed=make_scene(settings=parent,cfg=resolved)
    assert resolved['grip_targets_rad']==replayed['grip_targets_rad']
    before={'r_f_joint1_2':.5}
    assert apply_grasp_offsets(before,{'r_f_joint1_2':.1})['r_f_joint1_2']==.6
    assert before['r_f_joint1_2']==.5
    with pytest.raises(ValueError):apply_grasp_offsets(before,{'r_f_joint1_2':.11})
    with pytest.raises(ValueError):apply_grasp_offsets(before,{'unknown':0.})
