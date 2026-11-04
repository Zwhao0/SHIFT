"""Launch one prepared, bounded V1/V2 quick controller on node0."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('version',choices=['v1','v2'])
    p.add_argument('--suffix',required=True)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--trigger',required=True,type=int)
    p.add_argument('--post-steps',default=5,type=int)
    a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    if min(a.trigger,a.post_steps)<1:p.error('positive step counts required')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket=f'ssh-quick-start-{os.getpid()}.sock')
    root='/workspace/SHIFT';unit='shift-quick-'+a.suffix+'-controller'
    args=['--suffix',a.suffix,'--seed',str(a.seed),'--trigger',str(a.trigger),'--post-steps',str(a.post_steps)]
    command=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=1020',
        '--property=KillMode=control-group','--property=TimeoutStopSec=20','--working-directory='+root,
        '--property=StandardOutput=append:'+root+'/runtime/v3-reproduce/'+a.suffix+'.controller.log',
        '--property=StandardError=inherit','--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/python3','-m','scripts.quick_'+a.version+'_remote',*args]
    code='from pathlib import Path;import subprocess,json;assert Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file(),"prepare required";subprocess.run('+repr(command)+',check=True);print(json.dumps(dict(unit='+repr(unit)+')))'
    try:
        s.ensure_connection(e)
        print(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
    finally:
        subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
