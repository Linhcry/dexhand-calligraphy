import pytest
from fivefinger.grasp_posture import unused_finger_posture


def test_only_selected_non_grasping_flexion_targets_change():
    original={f'r_f_joint{i}_{j}':.12 for i in range(1,6) for j in range(1,5)}
    target=unused_finger_posture(original,0.)
    for name,value in original.items():
        i,j=map(int,name.removeprefix('r_f_joint').split('_'))
        assert target[name]==(0. if i in (4,5) and j in (2,3,4) else value)
    assert all(x==.12 for x in original.values())
    for bad in [-.01,.31,float('nan')]:
        with pytest.raises(ValueError):unused_finger_posture(original,bad)
    with pytest.raises(ValueError):unused_finger_posture({},0.)
