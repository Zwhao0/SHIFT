"""Remove exact audited V4 raw runs and this project's temporary authorization."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',nargs='+',required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(not x or not all(c.isalnum() or c in '_-' for c in x) for x in a.suffix):p.error('invalid suffix')
    runs=['v4_'+x for x in a.suffix]
    evidence={r['name']:r for r in json.loads(a.evidence.read_text())['runs']}
    if any(name not in evidence or not evidence[name]['independent_audit']['pass_all'] for name in runs):
        raise ValueError('every deleted run requires a saved independent passing audit')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    baseline=json.loads((s.root/'runtime/v3-reproduce/company_gpu_baseline.json').read_text())['nodes']
    eps=endpoints_from_env();connections={n:replace(e,socket='ssh-v4-clean-'+n+'.sock') for n,e in eps.items()}
    report=dict(runs=runs,nodes={},business_health_tested=False)
    try:
        for e in connections.values():s.ensure_connection(e)
        for name,e in connections.items():
            code='''from pathlib import Path
import fcntl,hashlib,json,os,pwd,shutil,subprocess
root=Path('/workspace/SHIFT');f=(root/'runtime/v3-training.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
entries=[x.split(',',1) for x in raw.splitlines() if x.strip()]
company={p.strip():n.strip() for p,n in entries if 'VLLM::' in n}
assert set(company)==set(EXPECTED) and len(company)==8 and all('VLLM::' in n for _,n in entries),'GPU resource mismatch'
assert not subprocess.check_output(['systemctl','list-units','--no-legend','--plain','--state=active,activating,deactivating','shift-*'],text=True).strip(),'active SHIFT units'
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')
 except OSError:continue
 assert not any(x.startswith('shift_full.') or x=='scripts.v4_remote' for x in args),'SHIFT worker remains'
removed=0
for p in [root/'results'/r for r in RUNS]+[root/'runtime/v3-reproduce'/(r+'.controller.log') for r in RUNS]+[root/'scripts/__pycache__',root/'shift_full/__pycache__']:
 if not p.exists():continue
 assert not p.is_symlink()
 if p.is_dir():removed+=sum(x.stat().st_size for x in p.rglob('*') if x.is_file());shutil.rmtree(p)
 else:removed+=p.stat().st_size;p.unlink()
for socket,target in SOCKETS:subprocess.run(['ssh','-S',str(root/'tmp'/socket),'-O','exit',target],capture_output=True,timeout=5)
marker='shift-v3-controller-20261103';u=pwd.getpwnam('test');p=Path(u.pw_dir)/'.ssh/authorized_keys'
if p.exists():
 old=p.read_text().splitlines();keep=[x for x in old if marker not in x]
 if old!=keep:p.write_text('\\n'.join(keep)+'\\n' if keep else '');os.chown(p,u.pw_uid,u.pw_gid);p.chmod(0o600)
private=root/'runtime/v1-controller';pub=private/'id_ed25519.pub'
if pub.exists() and marker in pub.read_text():
 for p in (pub,private/'id_ed25519',private/'credentials'):
  if p.exists():removed+=p.stat().st_size;p.unlink()
assert marker not in (Path(u.pw_dir)/'.ssh/authorized_keys').read_text()
assert (root/'runtime/torch214/bin/python').is_file() and (root/'runtime/python-headers').is_dir() and Path('/workspace/FLASH1022/models/qwen_moe').is_dir()
site=next((root/'runtime/torch214/lib').glob('python*/site-packages'))
files=[site/'torchft/manager.py',site/'torchft/process_group.py',site/'torch/distributed/distributed_c10d.py',*site.glob('torchft/_torchft*.so')]
sources={str(p.relative_to(site)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
print(json.dumps(dict(deleted_file_bytes=removed,company_pid_count=8,company_pids_unchanged=True,
 extra_gpu=0,shift_processes=0,active_shift_units=0,temporary_authorization_markers=0,
 environment_preserved=True,headers_preserved=True,model_preserved=True,installed_source_sha256=sources)))
'''.replace('EXPECTED',repr(sorted({x.split(',')[0].strip() for x in baseline[name]['gpu'] if 'VLLM::' in x}))).replace('RUNS',repr(runs)).replace('SOCKETS',repr([(x.socket,'test@'+x.ip) for x in eps.values()] if name=='node0' else []))
            report['nodes'][name]=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        report.update(all_resources_clean=True,raw_retention='saved audited summaries and signatures only')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(all_resources_clean=True,deleted_file_bytes=sum(n['deleted_file_bytes'] for n in report['nodes'].values()),report=str(a.output))))
    except subprocess.CalledProcessError as error:
        message=(error.stdout or '')+(error.stderr or '')
        for e in eps.values():message=message.replace(e.ip,'<private>')
        print(message);raise SystemExit(1)
    finally:
        for e in connections.values():subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
