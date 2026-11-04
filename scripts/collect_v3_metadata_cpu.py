"""Collect unique P13 CPU evidence and remove only verified stale cache/duplicates."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from shift_full.control import Session,endpoints_from_env


def main():
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket=f'ssh-v3-meta-collect-{os.getpid()}.sock')
    out=s.root/'results/v3_p_20261103'
    archive=out/'p13_metadata-evidence.tgz'
    assert not archive.exists()
    remote='/tmp/shift-v3-p13-metadata-evidence.tgz'
    try:
        s.ensure_connection(e)
        code=r'''
from pathlib import Path
import hashlib,json,subprocess,tarfile
root=Path('/workspace/SHIFT/results/v3_p_20261103')
runs=[root/('p13_metadata'+n) for n in ('01','02','03')]
assert all((p/'status.json').is_file() for p in runs)
for n in ('01','02','03'):
 unit='shift-v3-p13-metadata'+n
 active=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True)
 assert active.stdout.strip()!='active',unit
manifest=[]
selected=[]
for directory in runs:
 for p in sorted(directory.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink()
  cache=any(x.startswith('compiler_cache_') for x in p.relative_to(directory).parts)
  keep=not cache or p.suffix in ('.cpp','.h','.py','.json','.txt')
  manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cache=cache,archived=keep))
  if keep:selected.append(p)
manifest_path=root/'p13_metadata_collect_manifest.json'
assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=Path('/tmp/shift-v3-p13-metadata-evidence.tgz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in selected+[manifest_path]:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(selected))))
'''
        evidence=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        subprocess.run(['scp','-q','-o','ControlPath='+str(s.root/'tmp'/e.socket),
            'test@'+e.ip+':'+remote,str(archive)],check=True,capture_output=True,timeout=30)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==evidence['sha256']
        archive.with_suffix('.tgz.sha256').write_text(evidence['sha256']+'  '+archive.name+'\n')
        with tarfile.open(archive,'r:gz') as t:
            members=t.getmembers()
            for m in members:
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                target=out/m.name
                assert not target.exists()
            t.extractall(out)
        manifest=json.loads((out/'p13_metadata_collect_manifest.json').read_text())
        for item in manifest:
            if item['archived']:
                local=out/Path(item['path']).relative_to('/workspace/SHIFT/results/v3_p_20261103')
                assert hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256']
        # Cache binaries are disposable compiler outputs; retain all source/manifest
        # and both failed reference probes in the unique archive.
        code=r'''
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/workspace/SHIFT/results/v3_p_20261103')
items=json.loads((root/'p13_metadata_collect_manifest.json').read_text())
for n in ('01','02','03'):
 assert subprocess.run(['systemctl','is-active','shift-v3-p13-metadata'+n],capture_output=True,text=True).stdout.strip()!='active'
for item in items:
 p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
removed=[]
for n in ('01','02','03'):
 for directory in (root/('p13_metadata'+n)).glob('compiler_cache_*'):
  assert directory.is_dir() and not directory.is_symlink()
  size=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
  shutil.rmtree(directory)
  removed.append(dict(path=str(directory),bytes=size,kind='completed disposable CPU compiler cache; source/manifest retained'))
p=Path('/tmp/shift-v3-p13-metadata-evidence.tgz')
assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_SHA
removed.append(dict(path=str(p),bytes=p.stat().st_size,kind='SHA-identical archive copy'))
p.unlink()
print(json.dumps(dict(removed=removed,bytes=sum(x['bytes'] for x in removed))))
'''.replace('EXPECTED_SHA',repr(evidence['sha256']))
        cleanup=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        cleanup['archive']=dict(path=str(archive.relative_to(s.root)),**evidence)
        (out/'cleanup_p13metadata.json').write_text(json.dumps(cleanup,indent=2)+'\n')
        print(json.dumps(dict(archive_bytes=evidence['bytes'],recovered_bytes=cleanup['bytes'])))
    finally:
        subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
