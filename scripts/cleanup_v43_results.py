"""Delete explicitly exported, inactive session results after saving every file hash."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.v41_control import command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sessions',nargs='+');a=p.parse_args()
    evidence=Path(__file__).resolve().parents[1]/'evidence/v4_3'
    exported={r['name'] for f in ('summary.json','rejected.json')
              for r in json.loads((evidence/f).read_text())['runs']}
    assert all(s.startswith('session_') and s in exported and
               all(c.isalnum() or c in '_-' for c in s) for s in a.sessions)
    dest=Path(__file__).resolve().parents[1]/'runtime/v3-reproduce/v43_cleanup'
    dest.mkdir(exist_ok=True)
    totals=[]
    for node in ('node0','node1'):
        for name in a.sessions:
            # Two calls deliberately separate durable local evidence from deletion.
            common='''from pathlib import Path
import hashlib,json,subprocess,shutil
root=Path('/workspace/SHIFT/results')
p=root/NAME
assert not p.is_symlink() and p.resolve().parent==root.resolve()
prefix='shift-session-'+NAME.removeprefix('session_')+'-'
units=json.loads(subprocess.check_output(['systemctl','list-units','--all','--output=json','--no-pager'],text=True))
assert not any(u['unit'].startswith(prefix) and u['active'] in ('active','activating','deactivating','reloading') for u in units),'session still active'
manifest={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(p.rglob('*')) if f.is_file()}
'''.replace('NAME',repr(name))
            manifest=json.loads(command(common+'print(json.dumps(manifest))',True,node))
            (dest/(node+'_'+name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
            digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            code=common+'assert hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()=='+repr(digest)+'\nif p.exists():shutil.rmtree(p)\nprint(json.dumps(dict(deleted=not p.exists())))'
            assert json.loads(command(code,True,node))['deleted']
            totals.append(dict(node=node,session=name,files=len(manifest),
                bytes=sum(x['bytes'] for x in manifest.values()),manifest_sha256=digest,deleted=True))
    public=evidence/'cleanup.json'
    previous=json.loads(public.read_text()) if public.exists() else dict(results=[])
    keys={(r['node'],r['session']) for r in totals}
    previous['results']=[r for r in previous['results'] if (r['node'],r['session']) not in keys]+totals
    public.write_text(json.dumps(previous,indent=2)+'\n')
    print(json.dumps(dict(deleted_sessions=len(a.sessions),files=sum(x['files'] for x in totals),bytes=sum(x['bytes'] for x in totals))))


if __name__=='__main__':main()
