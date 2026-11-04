"""Local scoped controller launcher/reader; never prints private endpoints."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from shift_full.control import Session, endpoints_from_env


def command(code, sudo=False, node='node0'):
    root=Path(__file__).resolve().parents[1]
    s=Session.__new__(Session);s.root=root;s._sudo_password=None
    e=replace(endpoints_from_env()[node],socket=f'ssh-v41-{os.getpid()}-{node}.sock')
    try:
        s.ensure_connection(e)
        args=['python3','-c',code]
        if sudo:args=['sudo','-S','-p','',*args]
        return s.ssh(e,args,sudo_input=sudo)
    finally:
        subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','read'))
    p.add_argument('name')
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    assert a.name and all(c.isalnum() or c in '_-' for c in a.name)
    if a.action=='read':
        path='/workspace/SHIFT/results/'+a.name+'/summary.json'
        code='from pathlib import Path;import json;p=Path('+repr(path)+');print(p.read_text() if p.exists() else json.dumps(dict(pending=True)))'
        text=command(code)
        out=Path(__file__).resolve().parents[1]/'tmp'/f'{a.name}.summary.json';out.write_text(text)
        x=json.loads(text)
        print(json.dumps({k:v for k,v in x.items() if k not in ('source_sha256','resources_before','resources_after')}))
    else:
        unit='shift-v41-'+a.name+'-controller'
        cmd=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=700',
             '--property=KillMode=control-group','--property=TimeoutStopSec=20','--working-directory=/workspace/SHIFT',
             '--property=StandardOutput=append:/workspace/SHIFT/runtime/v3-reproduce/'+a.name+'.log',
             '--property=StandardError=inherit','--setenv=PYTHONPATH=/workspace/SHIFT','--setenv=PYTHONDONTWRITEBYTECODE=1',
             '/usr/bin/python3','-m','scripts.unified_baseline_remote','--suffix',a.name,*a.args]
        print(command('import subprocess,json;subprocess.run('+repr(cmd)+',check=True);print(json.dumps(dict(unit='+repr(unit)+')))',True))


if __name__=='__main__':main()
