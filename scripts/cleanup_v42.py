"""Delete exact exported v4.2 trial data, preserving environments and anchors."""
import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
from shift_full.control import Session,endpoints_from_env


def clean_remote(names,expected,sockets,controller,expected_anchors):
    from pathlib import Path
    import fcntl,hashlib,json,os,pwd,shutil,subprocess
    root=Path('/workspace/SHIFT');lock=(root/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries=[x.split(',',1) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines() if x.strip()]
    assert {p.strip() for p,n in entries if 'VLLM::' in n}==set(expected) and len(entries)==8,'GPU baseline mismatch'
    assert not subprocess.check_output(['systemctl','list-units','--no-legend','--plain','--state=active,activating,deactivating','shift-*'],text=True).strip(),'SHIFT units active'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
        except OSError:continue
        assert not any(x.startswith('shift_full.') or x in ('scripts.session_remote','scripts.v4_remote','scripts.v4_transaction_remote') for x in args),'SHIFT process remains'
    removed=0;paths=[]
    def delete(p):
        nonlocal removed
        if not p.exists() or p.is_symlink():return
        paths.append(str(p.relative_to(root)))
        if p.is_dir():
            removed+=sum(x.stat().st_size for x in p.rglob('*') if x.is_file() and not x.is_symlink());shutil.rmtree(p)
        else:removed+=p.stat().st_size;p.unlink()
    # Preserve cumulative formal-injection accounting before removing JSONL.
    ledger=root/'results/v1_acceptance_20261103/formal_injection_ledger.json'
    if controller:
        assert ledger.is_file();data=json.loads(ledger.read_text());archived=set(data.get('archived_sessions',[]));count=0;added=[]
        for name in names:
            p=root/'results'/name/'controller.jsonl'
            if p.is_file() and name not in archived:
                n=sum(json.loads(line).get('kind')=='injected' for line in p.read_text().splitlines() if line.strip())
                if n:count+=n;added.append(name)
        data['archived_formal_injections']=data.get('archived_formal_injections',0)+count
        data['archived_sessions']=sorted(archived|set(added));ledger.write_text(json.dumps(data,indent=2)+'\n')
    for name in names:
        delete(root/'results'/name)
        for p in root.glob(name+'*-logs.tgz'):delete(p)
        for p in (root/'results').glob(name+'-*.tgz'):delete(p)
    delete(root/'results/v3_a1_20261103')  # regenerated deployment report, copied to private manifest
    for directory in ('runtime/v3-reproduce','tmp'):
        for p in (root/directory).iterdir():
            if p.is_file() and p.suffix in ('.log','.jsonl','.trace','.tgz'):delete(p)
    for directory in ('shift_full','scripts','shift_min'):
        for p in (root/directory).rglob('__pycache__'):delete(p)
    for socket,target in sockets:subprocess.run(['ssh','-S',str(root/'tmp'/socket),'-O','exit',target],capture_output=True,timeout=5)
    marker='shift-v3-controller-20261103';user=pwd.getpwnam('test');auth=Path(user.pw_dir)/'.ssh/authorized_keys'
    if auth.exists():
        old=auth.read_text().splitlines();keep=[line for line in old if marker not in line]
        if old!=keep:auth.write_text('\n'.join(keep)+'\n' if keep else '');os.chown(auth,user.pw_uid,user.pw_gid);auth.chmod(0o600)
        assert marker not in auth.read_text()
    private=root/'runtime/v1-controller';pub=private/'id_ed25519.pub'
    if pub.exists() and marker in pub.read_text():
        for p in (pub,private/'id_ed25519',private/'credentials'):delete(p)
    assert (root/'runtime/torch214/bin/python').is_file() and (root/'runtime/python-headers').is_dir() and Path('/workspace/FLASH1022/models/qwen_moe').is_dir()
    anchors=list((root/'runtime/v3-reference-num01').glob('anchor_r*.pt'));assert len(anchors)==expected_anchors,'initial anchor roles missing'
    site=next((root/'runtime/torch214/lib').glob('python*/site-packages'))
    installed={str(p.relative_to(site)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [site/'torchft/manager.py',site/'torchft/process_group.py',site/'torch/distributed/distributed_c10d.py',*site.glob('torchft/_torchft*.so')]}
    return dict(deleted_file_bytes=removed,removed_paths=paths,company_pids_unchanged=True,company_pid_count=8,extra_gpu=0,active_shift_units=0,shift_processes=0,temporary_authorization_markers=0,environment_headers_model_preserved=True,initial_anchor_files_preserved=len(anchors),formal_injection_ledger_preserved=ledger.is_file() if controller else None,installed_source_sha256=installed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];names=set()
    for folder in ('v4_recovery_20261105','v4_2'):
        for file in ('summary.json','rejected.json'):
            path=root/'evidence'/folder/file
            if not path.exists():continue
            data=json.loads(path.read_text());runs=data if isinstance(data,list) else data['runs']
            for run in runs:
                if file=='summary.json':assert run['independent_audit']['pass_all']
                else:assert run.get('accepted') is False and (run.get('error') is not None or run.get('audit_errors') or run.get('diagnosis') or run.get('done'))
                names.add(run['name'])
    for name,audit in json.loads((root/'evidence/v4_2/compatibility.json').read_text()).items():
        assert audit['pass_all'];names.add(name)
    assert names and all(n.startswith(('v4_r42_','session_s42_','v1_quick_20261104_r42_','v2_q2_r42_')) and all(c.isalnum() or c in '_-' for c in n) for n in names)
    if not a.execute:print(json.dumps(dict(exported_runs=sorted(names),dry_run=True)));return
    baseline=json.loads((root/'runtime/v3-reproduce/company_gpu_baseline.json').read_text())['nodes']
    s=Session.__new__(Session);s.root=root;s._sudo_password=None;original=endpoints_from_env()
    endpoints={n:replace(e,socket='ssh-v42-clean-'+n+'.sock') for n,e in original.items()};report=dict(runs=sorted(names),nodes={},business_health_tested=False)
    try:
        for e in endpoints.values():s.ensure_connection(e)
        for node,e in endpoints.items():
            expected=sorted({x.split(',')[0].strip() for x in baseline[node]['gpu'] if 'VLLM::' in x})
            sockets=[(x.socket,'test@'+x.ip) for x in original.values()] if node=='node0' else []
            code=inspect.getsource(clean_remote)+'\nimport json\nprint(json.dumps(clean_remote('+repr(sorted(names))+','+repr(expected)+','+repr(sockets)+','+repr(node=='node0')+','+repr(8 if node in ('node0','node3') else 0)+')))'
            report['nodes'][node]=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        report['all_resources_clean']=True;report['deleted_file_bytes']=sum(n['deleted_file_bytes'] for n in report['nodes'].values())
        (root/'evidence/v4_2/server_cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(all_resources_clean=True,deleted_file_bytes=report['deleted_file_bytes'])))
    except subprocess.CalledProcessError as error:
        message=(error.stderr or '')[-3000:]
        for e in endpoints.values():message=message.replace(e.ip,'<private>')
        raise RuntimeError(message) from None
    finally:
        for e in endpoints.values():subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
