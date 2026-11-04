"""Start one server-owned pure-cold recovery; never preload a standby."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

from shift_full.control import Session,endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suffix',required=True)
    p.add_argument('--digest-engine',choices=['reference','staged'],default='reference')
    p.add_argument('--overlap-cpu-model-load',action='store_true')
    p.add_argument('--overlap-cpu-imports',action='store_true')
    p.add_argument('--sync-quorum',action='store_true')
    p.add_argument('--eager-groups',action='store_true');p.add_argument('--seed',type=int,default=20261106);p.add_argument('--cold-launcher',action='store_true');a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v3-cold-'+str(os.getpid())+'.sock')
    root='/workspace/SHIFT';unit='shift-v3-b2-'+a.suffix;out=root+'/results/v3_b2_20261103'
    cmd=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=900',
         '--property=KillMode=control-group','--property=TimeoutStopSec=20',
         '--property=StandardOutput=append:'+out+'/'+a.suffix+'.log','--property=StandardError=inherit',
         '--working-directory='+root,'--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=SHIFT_DIGEST_ENGINE='+a.digest_engine,
         '/usr/bin/python3','-m','scripts.v3_cold_remote','--suffix',a.suffix,'--seed',str(a.seed)]
    cmd.insert(cmd.index('/usr/bin/python3'), '--setenv=SHIFT_OVERLAP_CPU_MODEL_LOAD='+('1' if a.overlap_cpu_model_load else '0'))
    cmd.insert(cmd.index('/usr/bin/python3'), '--setenv=SHIFT_OVERLAP_CPU_IMPORTS='+('1' if a.overlap_cpu_imports else '0'))
    if a.cold_launcher:cmd.append('--cold-launcher')
    if a.sync_quorum:cmd.append('--sync-quorum')
    if a.eager_groups:cmd.append('--eager-groups')
    code='from pathlib import Path; import subprocess,json; assert Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file(), "Run prepare_v3_controller after the previous controller ends (single-use credential)"; Path('+repr(out)+').mkdir(exist_ok=True); subprocess.run('+repr(cmd)+',check=True); print(json.dumps(dict(unit='+repr(unit)+',pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))'
    try:
        s.ensure_connection(e);result=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        dest=s.root/'results/v3_b2_20261103';dest.mkdir(exist_ok=True)
        (dest/(a.suffix+'_owner.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
        provenance=s.root/'results/v3_a1_20261103/controller_prepare.json'
        if provenance.exists():
            (dest/(a.suffix+'_source_provenance.json')).write_bytes(provenance.read_bytes())
    finally:subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
