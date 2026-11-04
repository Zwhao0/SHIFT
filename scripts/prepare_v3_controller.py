"""Restore scoped server control and current runners without starting training.

Run locally as `python3 -m scripts.prepare_v3_controller [--smoke]`.
Reads server.md only in memory; no environment or model is rebuilt.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import tarfile
import tempfile
import time

from shift_full.control import Session, endpoints_from_env, server_password

ROOT = Path(__file__).resolve().parents[1]
FILES = ['scripts/recheck_v1_host.py', 'scripts/v1_acceptance_remote.py',
         'scripts/v1_cold_remote.py', 'scripts/v3_controller_smoke.py']
FILES = sorted(set(FILES + [str(p.relative_to(ROOT)) for d in ('shift_full', 'scripts')
                            for p in (ROOT/d).glob('*.py')]))
MARKER = 'shift-v3-controller-20261103'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    eps = {n: replace(e, socket='ssh-v3-prepare-'+str(os.getpid())+'-'+n+'.sock')
           for n, e in endpoints_from_env().items()}
    s = Session.__new__(Session)
    s.root, s._sudo_password = ROOT, None
    private = eps['node0'].host_root+'/runtime/v1-controller'
    report = dict(runner_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                for p in FILES}, nodes={})

    def sudo(e, code):
        return s.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True)

    def copy(e, source, dest):
        sp.run(['scp', '-q', '-o', 'ControlPath='+str(ROOT/'tmp'/e.socket),
                str(source), 'test@'+e.ip+':'+dest], check=True,
               capture_output=True, timeout=30)

    try:
        for e in eps.values():
            s.ensure_connection(e)
        # The owner protocol remains authoritative; also catch an accidental
        # deployment while a bounded v3 controller is still using these files.
        sudo(eps['node0'], 'import fcntl; from pathlib import Path; '
             'p=Path('+repr(eps['node0'].host_root+'/runtime/v3-training.lock')+'); '
             'f=p.open("a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
        code = ('from pathlib import Path; import subprocess; p=Path('+repr(private)+'); '
                'p.mkdir(mode=0o700,parents=True,exist_ok=True); p.chmod(0o700); '
                'k=p/"id_ed25519";\n'
                'if not k.exists(): subprocess.run(["ssh-keygen","-q","-t","ed25519",'
                '"-N","","-C",'+repr(MARKER)+',"-f",str(k)],check=True)\n'
                'print(k.with_suffix(".pub").read_text().strip())')
        pub = sudo(eps['node0'], code).strip()
        assert pub.startswith('ssh-ed25519 ')
        line = ('from="'+eps['node0'].ip+'",no-agent-forwarding,no-port-forwarding,'
                'no-X11-forwarding,no-pty '+pub)
        for name, e in eps.items():
            code = ('from pathlib import Path; import os,pwd; u=pwd.getpwnam("test"); '
                    'd=Path(u.pw_dir)/".ssh"; d.mkdir(mode=0o700,exist_ok=True); '
                    'p=d/"authorized_keys"; old=p.read_text() if p.exists() else ""; '
                    'lines=[x for x in old.splitlines() if '+repr(MARKER)+' not in x]; '
                    'p.write_text('+repr('\n')+'.join(lines+['+repr(line)+'])+'+repr('\n')+'); '
                    'os.chown(d,u.pw_uid,u.pw_gid); os.chown(p,u.pw_uid,u.pw_gid); '
                    'p.chmod(0o600)')
            sudo(e, code)
            report['nodes'][name] = dict(scoped_key_installed=True)
        with tempfile.TemporaryDirectory(prefix='shift-v3-') as tmp:
            bundle = Path(tmp)/'runners.tgz'
            manifest = Path(tmp)/'runner-manifest.json'
            manifest.write_text(json.dumps(report['runner_sha256']))
            with tarfile.open(bundle, 'w:gz') as archive:
                for p in FILES:
                    archive.add(ROOT/p, arcname=p)
            for name, e in eps.items():
                path = '/tmp/shift-v3-runners-'+str(os.getpid())+'.tgz'
                manifest_path = '/tmp/shift-v3-manifest-'+str(os.getpid())+'.json'
                copy(e, bundle, path)
                copy(e, manifest, manifest_path)
                code = ('from pathlib import Path; import tarfile,hashlib,json; '
                        'r=Path('+repr(e.host_root)+'); p=Path('+repr(path)+'); '
                        't=tarfile.open(p); t.extractall(r,filter="data"); t.close(); p.unlink(); '
                        'm=Path('+repr(manifest_path)+'); expected=json.loads(m.read_text()); m.unlink(); '
                        'assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h '
                        'for p,h in expected.items()); print("verified")')
                assert sudo(e, code).strip() == 'verified'
                report['nodes'][name]['runners_verified'] = True
            # Preserve the old ledger as evidence; migrate only the active copy.
            code = '''from pathlib import Path
import json
r=Path('''+repr(eps['node0'].host_root)+''')
p=r/'results/v1_acceptance_20261103/formal_injection_ledger.json'
x=json.loads(p.read_text())
out=r/'results/v3_a1_20261103';out.mkdir(parents=True,exist_ok=True)
before=out/'formal_injection_ledger_before.json'
if not before.exists():before.write_bytes(p.read_bytes())
x.pop('limit',None);x.pop('remaining',None)
x.setdefault('historical_diagnostic_injections',5)
x.update(injection_limit=None,limit_policy='user removed cap 2026-11-03')
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\\n');tmp.replace(p)
print(json.dumps(x))
'''
            report['ledger'] = json.loads(sudo(eps['node0'], code))
            cred = Path(tmp)/'credential'
            cred.write_text('密码: '+server_password(ROOT/'server.md')+'\n')
            cred.chmod(0o600)
            path = '/tmp/shift-v3-credential-'+str(os.getpid())
            try:
                copy(eps['node0'], cred, path)
                code = ('from pathlib import Path; import os; p=Path('+repr(path)+'); '
                        'dest=Path('+repr(private+'/credentials')+'); '
                        'dest.write_bytes(p.read_bytes()); p.unlink(); '
                        'dest.chmod(0o400); os.chown(dest,0,0)')
                sudo(eps['node0'], code)
            finally:
                cred.unlink(missing_ok=True)
        if args.smoke:
            unit = 'shift-v3-a1-smoke-'+str(int(time.time()))
            code = ('from pathlib import Path; import subprocess,json; '
                    'r=Path('+repr(eps['node0'].host_root)+'); '
                    'out=r/"results/v3_a1_20261103"; '
                    'cmd=["systemd-run","--quiet","--collect","--unit='+unit+'",'
                    '"--property=RuntimeMaxSec=120","--property=KillMode=control-group",'
                    '"--property=StandardOutput=append:"+str(out/"controller-smoke.log"),'
                    '"--property=StandardError=inherit","--working-directory="+str(r),'
                    '"--setenv=PYTHONPATH="+str(r),"--setenv=PYTHONDONTWRITEBYTECODE=1",'
                    '"/usr/bin/python3","-m","scripts.v3_controller_smoke"]; '
                    'subprocess.run(cmd,check=True); '
                    'pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+','
                    '"--property=MainPID","--value"],text=True)); '
                    'print(json.dumps(dict(unit='+repr(unit)+',pid=pid)))')
            report['smoke_owner'] = json.loads(sudo(eps['node0'], code))
        out = ROOT/'results/v3_a1_20261103'
        out.mkdir(parents=True, exist_ok=True)
        (out/'controller_prepare.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        for e in eps.values():
            sp.run(['ssh', '-S', str(ROOT/'tmp'/e.socket), '-O', 'exit', 'test@'+e.ip],
                   capture_output=True, timeout=5)


if __name__ == '__main__':
    main()
