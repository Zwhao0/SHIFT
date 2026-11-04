"""Compact read-only status, and collect completed B1 raw evidence locally."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import tarfile
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True)
    p.add_argument('--stage',choices=['b1','b2','c1'],default='b1')
    p.add_argument('--collect',action='store_true')
    p.add_argument('--cpu-only',action='store_true',help='C1 protocol/payload regression status')
    a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v3-inspect-'+str(os.getpid())+'.sock')
    name='v3_'+a.stage+'_20261103_'+a.suffix
    group='v3_'+a.stage+'_20261103'
    status_name='cpu_status.json' if a.cpu_only else 'status.json'
    prefix='v3b1_'+a.suffix+'_*' if a.stage=='b1' else 'v3c1_'+a.suffix if a.stage=='c1' else 'v1_accept_20261103_v3cold_'+a.suffix
    root='/workspace/SHIFT'
    code='''from pathlib import Path
import json,subprocess
r=Path('/workspace/SHIFT');name='''+repr(name)+'''
p=r/'results'/name/'''+repr(status_name)+'''
x=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps(dict(done=x.get('done'),success=x.get('success'),error=x.get('error'),completed=[dict(session=y['session'],mode=y['mode'],fixed=y['windows'].get('fixed11_80')) for y in x.get('rows',[]) if 'session' in y],digest={k:x[k] for k in ('tests','means','saved_seconds','hashes_exact','training') if k in x},comparison=x.get('comparison'),comparisons=x.get('comparisons'),checks={k:x[k] for k in ('nativeft_exact','flat_exact','flat_tolerance_pass','tolerance_unit_exit','prior_shift_exact','validate_against','compare_existing','functional_pass','process_start_to_joint_seconds','formal_injections_after') if k in x})))
log=r/'results'/'''+repr(group)+'''/'''+repr(a.suffix+'.log')+'''
if log.exists():print('\\n'.join(line[:400] for line in log.read_text().splitlines()[-3:]))
print(subprocess.run(['systemctl','show','''+repr('shift-v3-'+a.stage+'-'+a.suffix)+''','--property=ActiveState,SubState,Result'],capture_output=True,text=True).stdout)
'''
    try:
        s.ensure_connection(e)
        print(s.ssh(e,['python3','-c',code]))
        if a.collect:
            done=s.ssh(e,['python3','-c','import json; print(json.load(open('+repr(root+'/results/'+name+'/'+status_name)+'))["done"])']).strip()
            if done!='True':
                print('Runner still active; collection deferred.')
                return
            archive='/tmp/shift-v3-b1-'+a.suffix+'-evidence.tgz'
            code=('from pathlib import Path; import json,tarfile; r=Path('+repr(root)+'); '
                  'out=r/"results"/'+repr(name)+'; x=json.loads((out/'+repr(status_name)+').read_text()); '
                  'assert x["done"],"runner still active"; '
                  't=tarfile.open('+repr(archive)+',"w:gz"); '
                  't.add(out,arcname='+repr(name)+'); '
                  '[t.add(p,arcname=p.name,filter=lambda m: None if m.name.endswith(".pt") else m) for p in (r/"results").glob('+repr(prefix)+') if p.is_dir()]; '
                  't.close(); Path('+repr(archive)+').chmod(0o644)')
            s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True,timeout=60)
            dest=s.root/'results'/group;dest.mkdir(exist_ok=True)
            local=dest/(a.suffix+'-evidence.tgz')
            sp.run(['scp','-q','-o','ControlPath='+str(s.root/'tmp'/e.socket),'test@'+e.ip+':'+archive,str(local)],check=True,timeout=90)
            expected=s.ssh(e,['sha256sum',archive]).split()[0]
            actual=hashlib.sha256(local.read_bytes()).hexdigest()
            if actual!=expected:raise RuntimeError('evidence archive checksum mismatch')
            local.with_suffix(local.suffix+'.sha256').write_text(actual+'  '+local.name+'\n')
            with tarfile.open(local) as t:
                target=(s.root/'results').resolve()
                for member in t.getmembers():
                    path=(target/member.name).resolve()
                    if target not in path.parents or not (member.isfile() or member.isdir()):
                        raise ValueError('unsafe evidence archive entry')
                t.extractall(target)
            print(json.dumps(dict(collected=str(local),bytes=local.stat().st_size)))
    finally:
        sp.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
