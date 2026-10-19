"""Read-only local readiness check. Never imports an actual hardware driver."""
import hashlib
import importlib.util
import json
from pathlib import Path
from fivefinger.paths import ROOT


def main():
    config=json.loads((ROOT/'config/official_hardware.template.json').read_text(encoding='utf-8'))
    bank=json.loads((ROOT/'config/adaptive_poems.json').read_text(encoding='utf-8'))
    glyphs={c for t in bank.values() for c in t}
    missing=[c for c in sorted(glyphs) if not (ROOT/'assets/hanzi'/f'{c}.json').exists()]
    files=sorted((ROOT/'fivefinger/autonomy').glob('*.py'))
    result=dict(runtime_modules={n:importlib.util.find_spec(n) is not None for n in ('mujoco','numpy','cv2','scipy')},
        glyph_count=len(glyphs),missing_glyphs=missing,official_poem_bank=False,
        hardware_ready=False,missing_commissioning=[k for k,v in config['commissioning'].items() if not v],
        official_submission_format_confirmed=False,internet_required_by_new_modules=False,
        network_isolation_verified=False,offline_wheelhouse_prepared=False,
        physical_scene_variation_tasks_completed=False,
        code_hashes={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    out=ROOT/'outputs/adaptive_scenes/preflight.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
