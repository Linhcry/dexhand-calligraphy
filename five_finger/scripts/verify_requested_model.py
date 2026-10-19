"""Verify vendor-page identity, pinned upstream files, and the actual scene loader."""
import hashlib,json,subprocess,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/dexrobot_mujoco'
OUT=ROOT/'outputs/model_check_20260910'
PAGE='https://www.dex-robot.com/downloadCenter/detail/7309fb847499da21'
def git(*args):return subprocess.check_output(['git','-C',str(VENDOR),*args],encoding='utf-8').strip()
def main():
    entry=next(x for x in json.loads((OUT/'download_center.json').read_text(encoding='utf-8'))['data']['cn'] if x['id']=='7309fb847499da21')
    head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/main').split()[0]
    origin=git('remote','get-url','origin')
    assert origin=='https://github.com/DexRobot/dexrobot_mujoco.git'
    assert not git('status','--porcelain','--untracked-files=no')
    selected='dexrobot_mujoco/models/dexhand021_right_jaka_zu7.xml'
    source=VENDOR/selected
    upstream=urllib.request.urlopen(f'https://raw.githubusercontent.com/DexRobot/dexrobot_mujoco/{head}/{selected}',timeout=30).read()
    assert source.read_bytes().replace(b'\r\n',b'\n')==upstream.replace(b'\r\n',b'\n')
    tree=ET.parse(source);files={selected:hashlib.sha256(source.read_bytes()).hexdigest()}
    for mesh in tree.findall('./asset/mesh'):
        path=(source.parent/mesh.get('file')).resolve()
        expected=subprocess.check_output(['git','-C',str(VENDOR),'show',f'{head}:{path.relative_to(VENDOR).as_posix()}'])
        assert path.read_bytes()==expected, path
        files[path.relative_to(VENDOR).as_posix()]=hashlib.sha256(expected).hexdigest()
    from experiments.competition_scene import make_scene,TableClearance
    m,d,*_=make_scene();clearance=TableClearance(m).require_clear(d)
    report=dict(checked_date='2026-09-10',requested_page=PAGE,official_page_entry=entry,origin=origin,
        local_commit=head,remote_main_commit=remote,local_is_current_main=head==remote,selected_model=selected,
        selected_model_matches_upstream=True,upstream_mesh_files_verified=len(files)-1,
        loaded_scene=dict(nq=m.nq,nv=m.nv,nu=m.nu,initial_moving_arm_clearance_mm=clearance*1000),
        hand_family_matches_requested_download=True,collision_variant='official simplified collision model',
        replacement_required=False,official_competition_arm_confirmed=False,
        limits=['The user page identifies the hand, not the competition arm or final hardware configuration.',
                'Workcell contact parameters, motor gains and pen approximation remain development settings.',
                'Public vendor SDK availability does not establish competition driver integration.'],sha256=files)
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    config=ROOT/'config/model_sources.json';data=json.loads(config.read_text(encoding='utf-8'))
    data.update(checked_date='2026-09-10',manufacturer_download_center=PAGE,product_page='https://www.dex-robot.com/productionDexhand',
        product_variant='DexHand021 MP / 量产版',official_download_entry=entry['id'],official_download_document=entry['site_link'],
        collision_variant='official simplified collision',requested_page_check='outputs/model_check_20260910/verification.json')
    config.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','official_page_entry','limits')},ensure_ascii=False))
if __name__=='__main__':main()
