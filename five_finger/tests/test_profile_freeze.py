import json
import pytest
from experiments.blind_competition import run_case

def test_unknown_profile_is_rejected_before_physics(tmp_path):
    public=tmp_path/'public';public.mkdir()
    (public/'controller_manifest.json').write_text(json.dumps({}),encoding='utf-8')
    with pytest.raises(ValueError,match='frozen'):
        run_case(public,'config/not_frozen.json')

def test_profile_outside_config_is_rejected(tmp_path):
    public=tmp_path/'public';public.mkdir()
    (public/'controller_manifest.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='inside project config'):
        run_case(public,str(tmp_path/'untrusted.json'))
