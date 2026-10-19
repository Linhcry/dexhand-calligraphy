"""Reproduce an import-only port of MIT dex-retargeting to our MuJoCo wrapper."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/dex-retargeting'
COMMIT = '3f56141bc8bd2760d5e452e382937269554ebb21'

def main():
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Expected upstream {COMMIT}, found {actual}')
    target = ROOT / 'fivefinger/third_party'
    target.mkdir(exist_ok=True)
    (target / '__init__.py').write_text('', encoding='utf-8')
    manifest = {'repository': 'https://github.com/dexsuite/dex-retargeting', 'commit': COMMIT,
                'changes': 'Only two module import paths replaced; optimization code unchanged.', 'files': {}}
    for original, output in [('optimizer.py', 'dex_optimizer.py'), ('kinematics_adaptor.py', 'kinematics_adaptor.py')]:
        content = (SOURCE / 'src/dex_retargeting' / original).read_text(encoding='utf-8')
        derived = content.replace('from dex_retargeting.robot_wrapper import RobotWrapper',
                                  'from ..mujoco_kinematics import MujocoHandKinematics as RobotWrapper')
        derived = derived.replace('from dex_retargeting.kinematics_adaptor import (', 'from .kinematics_adaptor import (')
        (target / output).write_text(derived, encoding='utf-8')
        manifest['files'][output] = {'original_sha256': hashlib.sha256((SOURCE / 'src/dex_retargeting' / original).read_bytes()).hexdigest(),
                                     'derived_sha256': hashlib.sha256((target / output).read_bytes()).hexdigest()}
    (target / 'DEX_RETARGETING_LICENSE').write_bytes((SOURCE / 'LICENSE').read_bytes())
    (target / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
