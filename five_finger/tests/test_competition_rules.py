import yaml
import pytest
from experiments.judge import HoldGate,circle_score,round_score
from fivefinger.paths import ROOT


def test_hold_is_continuous_and_needs_three_seconds():
    gate=HoldGate()
    for _ in range(299):gate.update(True,.01)
    assert not gate.passed
    gate.update(False,.01)
    assert gate.continuous_s==0
    for _ in range(299):gate.update(True,.01)
    assert not gate.passed
    assert gate.update(True,.01)


@pytest.mark.parametrize('radius',[.008,.005,.003])
def test_ink_radius_strict_minimum_and_hit_geometry(radius):
    assert circle_score(radius,radius/3,0,50)['points']==0
    assert circle_score(radius,radius*.5,0,50)['points']==75
    assert circle_score(radius,radius*.5,radius*2,50)['points']==0


def test_unobserved_ink_and_late_events_do_not_score():
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    attempt={'paper':'A5','touch':{'passed':True,'time_s':2},'hold':{'passed':True,'time_s':181},
             'dot':{'time_s':5,'eye_radius':.003,'ink_radius':.002,'offset':0,'physical_ink_observed':False}}
    assert round_score([attempt],10,rules)['proxy_points']==10


def test_preliminary_cap_and_max_three_dots():
    rules=yaml.safe_load((ROOT/'experiments/rules.yaml').read_text(encoding='utf-8'))
    attempt={'paper':'A5',**{n:{'passed':True,'time_s':10} for n in ['touch','hold','dip','overhead']},
             'dot':{'time_s':10,'eye_radius':.003,'ink_radius':.002,'offset':0,'physical_ink_observed':True}}
    assert round_score([attempt]*4,179,rules)['proxy_points']==500
    assert not round_score([attempt],20,rules)['official_score']
