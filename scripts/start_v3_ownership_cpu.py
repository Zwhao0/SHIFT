"""Launch one bounded server-owned CPU early-quorum probe after prepare."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tarfile
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True);a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    root=Path(__file__).resolve().parents[1]
    s=Session.__new__(Session);s.root=root;s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-p10b4-start-'+a.suffix+'.sock')
    unit='shift-v3-p10b4-'+a.suffix;r=e.host_root
    cmd=['systemd-run','--quiet','--unit='+unit,'--property=RuntimeMaxSec=600',
        '--property=KillMode=control-group','--property=TimeoutStopSec=15',
        '--property=StandardOutput=append:'+r+'/results/p10b4_'+a.suffix+'.log',
        '--property=StandardError=inherit','--working-directory='+r,'--setenv=PYTHONPATH='+r,
        '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=TOKIO_WORKER_THREADS=2','--setenv=SHIFT_DIGEST_ENGINE=staged',
        '--setenv=PYTHONDONTWRITEBYTECODE=1',r+'/runtime/torch214/bin/python','-m',
        'scripts.v3_ownership_native_cross_cpu','--suffix',a.suffix,'--early']
    code='from pathlib import Path;import subprocess,json;assert Path('+repr(r+'/runtime/v1-controller/credentials')+').is_file();assert not Path('+repr(r+'/results/v3_p10b4_'+a.suffix)+').exists();subprocess.run('+repr(cmd)+',check=True);print(json.dumps(dict(unit='+repr(unit)+',pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))'
    try:
        s.ensure_connection(e)
        owner=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        out=root/'results/v3_p_20261103';prefix='p10b4_'+a.suffix
        (out/(prefix+'_owner.json')).write_text(json.dumps(owner,indent=2)+'\n')
        manifest=json.loads((root/'results/v3_a1_20261103/controller_prepare.json').read_text())
        (out/(prefix+'_source.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        with tarfile.open(out/(prefix+'_source.tgz'),'w:gz') as t:
            for f in manifest['runner_sha256']:t.add(root/f,arcname=f)
            t.add(out/(prefix+'_source.json'),arcname='manifest.json')
        print(json.dumps(owner))
    finally:
        subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
