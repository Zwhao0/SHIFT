"""Remove regenerable experiment data only after all supplied audits pass."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from shift_full.control import Session,endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=Path('evidence/v4/quick_validation.json'))
    p.add_argument('--output',type=Path,default=Path('evidence/v4/quick_cleanup.json'))
    a=p.parse_args();saved=json.loads(a.evidence.read_text());runs=saved['runs']
    assert runs and all(r['functional_pass'] and r['resources_clean'] for r in runs)
    root=Path(__file__).resolve().parents[1]
    baseline=json.loads((root/'runtime/v3-reproduce/company_gpu_baseline.json').read_text())['nodes']
    s=Session.__new__(Session);s.root=root;s._sudo_password=None
    original=endpoints_from_env();eps={n:replace(e,socket='ssh-quick-clean-'+n+'.sock') for n,e in original.items()}
    report=dict(nodes={},business_health_tested=False)
    try:
        for e in eps.values():s.ensure_connection(e)
        for name,e in eps.items():
            expected=sorted({line.split(',')[0].strip() for line in baseline[name]['gpu'] if 'VLLM::' in line})
            code='''from pathlib import Path
import fcntl,json,os,pwd,shutil,subprocess
r=Path('/workspace/SHIFT');lock=(r/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
entries=[x.split(',',1) for x in raw.splitlines() if x.strip()]
company={p.strip() for p,n in entries if 'VLLM::' in n}
assert company==set(EXPECTED) and len(entries)==len(company)==8,'company PID baseline mismatch or extra GPU work'
active=subprocess.check_output(['systemctl','list-units','--no-legend','--plain','--state=active,activating,deactivating','shift-*'],text=True).strip()
assert not active,'active SHIFT units'
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
 except OSError:continue
 assert not any(x.startswith('shift_full.') for x in args),'SHIFT worker remains'
ledger=r/'results/v1_acceptance_20261103/formal_injection_ledger.json'
if CONTROLLER:assert ledger.is_file()
# Historical injection accounting must survive deletion of controller JSONL.
count=0
if ledger.is_file():
 x=json.loads(ledger.read_text());archived=set(x.get('archived_sessions',[]));new=[];count=0
 for f in (r/'results').glob('*/controller.jsonl'):
  if f.parent.name in archived:continue
  n=sum(json.loads(line).get('kind')=='injected' for line in f.read_text().splitlines() if line.strip())
  if n:new.append(f.parent.name);count+=n
 x['archived_formal_injections']=x.get('archived_formal_injections',0)+count
 x['archived_sessions']=sorted(archived|set(new));ledger.write_text(json.dumps(x,indent=2)+'\\n')
removed=0;paths=[]
def delete(p):
 global removed
 if not p.exists() or p.is_symlink():return
 paths.append(str(p.relative_to(r)))
 if p.is_dir():removed+=sum(f.stat().st_size for f in p.rglob('*') if f.is_file());shutil.rmtree(p)
 else:removed+=p.stat().st_size;p.unlink()
for p in list((r/'results').iterdir()):
 if p==ledger.parent:
  for child in list(p.iterdir()):
   if child!=ledger:delete(child)
 else:delete(p)
for pattern in ('v1_quick_*-logs.tgz','v2_q2_*-logs.tgz'):
 for p in r.glob(pattern):delete(p)
for p in (r/'runtime/v3-reproduce').iterdir():
 if p.suffix in ('.log','.jsonl','.tgz','.trace'):delete(p)
for p in (r/'tmp').iterdir():
 if p.is_file() and p.suffix in ('.log','.tgz','.trace'):delete(p)
for directory in ('scripts','shift_full'):
 delete(r/directory/'__pycache__')
for p in (r/'shift_full').glob('quick_v1_entry_*.py'):delete(p)
for socket,target in SOCKETS:subprocess.run(['ssh','-S',str(r/'tmp'/socket),'-O','exit',target],capture_output=True,timeout=5)
marker='shift-v3-controller-20261103';u=pwd.getpwnam('test');p=Path(u.pw_dir)/'.ssh/authorized_keys'
if p.exists():
 old=p.read_text().splitlines();keep=[line for line in old if marker not in line]
 if old!=keep:p.write_text('\\n'.join(keep)+'\\n' if keep else '');os.chown(p,u.pw_uid,u.pw_gid);p.chmod(0o600)
private=r/'runtime/v1-controller';pub=private/'id_ed25519.pub'
if pub.exists() and marker in pub.read_text():
 for p in (pub,private/'id_ed25519',private/'credentials'):delete(p)
assert marker not in (Path(u.pw_dir)/'.ssh/authorized_keys').read_text()
assert (not CONTROLLER or ledger.is_file()) and (r/'runtime/torch214/bin/python').is_file() and (r/'runtime/python-headers').is_dir() and Path('/workspace/FLASH1022/models/qwen_moe').is_dir()
anchors=list((r/'runtime/v3-reference-num01').glob('anchor_r*.pt'))
print(json.dumps(dict(deleted_file_bytes=removed,removed_paths=paths,company_pid_count=8,company_pids_unchanged=True,extra_gpu=0,active_shift_units=0,shift_workers=0,temporary_authorization_markers=0,environment_headers_model_preserved=True,initial_anchor_files_preserved=len(anchors),formal_injection_ledger_preserved=ledger.is_file() if CONTROLLER else None,new_injections_archived=count,disk_free_bytes=shutil.disk_usage(r).free)))
'''.replace('CONTROLLER',repr(name=='node0')).replace('EXPECTED',repr(expected)).replace('SOCKETS',repr([(v.socket,'test@'+v.ip) for v in original.values()] if name=='node0' else []))
            report['nodes'][name]=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        report.update(all_resources_clean=True,deleted_file_bytes=sum(n['deleted_file_bytes'] for n in report['nodes'].values()),retained='source, installed runtime/headers/model, private configs/seeds/manifests, initial anchor roles, comparison signatures and compact audited timing data')
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(all_resources_clean=True,deleted_file_bytes=report['deleted_file_bytes'])))
    finally:
        for e in eps.values():subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
