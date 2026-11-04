"""Launch a bounded, CPU-only fresh-import admission; never starts training."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tarfile
from shift_full.control import Session, endpoints_from_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--use-helper', action='store_true')
    p.add_argument('--profile-imports', action='store_true')
    a = p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):
        p.error('invalid suffix')
    s = Session.__new__(Session)
    s.root, s._sudo_password = Path(__file__).resolve().parents[1], None
    e = replace(endpoints_from_env()['node0'], socket=f'ssh-v3-import-{os.getpid()}.sock')
    r, unit = e.host_root, 'shift-v3-p12-'+a.suffix
    prefix = 'p12_'+a.suffix
    out = s.root/'results/v3_p_20261103'
    assert not (out/(prefix+'_owner.json')).exists()
    cmd = ['systemd-run', '--quiet', '--collect', '--unit='+unit,
           '--property=RuntimeMaxSec=180', '--property=KillMode=control-group',
           '--property=TimeoutStopSec=10',
           '--property=StandardOutput=append:'+r+'/results/v3_p_20261103/'+prefix+'.log',
           '--property=StandardError=inherit', '--working-directory='+r, '--setenv=PYTHONPATH='+r,
           '--setenv=CUDA_VISIBLE_DEVICES=', '--setenv=OMP_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1',
           '--setenv=TOKIO_WORKER_THREADS=2', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           r+'/runtime/torch214/bin/python', '-m', 'scripts.v3_import_cpu_probe', '--suffix', a.suffix]
    if a.use_helper:
        cmd.append('--use-helper')
    if a.profile_imports:
        cmd.append('--profile-imports')
    try:
        s.ensure_connection(e)
        code = ('from pathlib import Path;import subprocess,json;assert not Path('+repr(r+'/results/v3_p_20261103/'+prefix)+
                ').exists();subprocess.run('+repr(cmd)+',check=True);print(json.dumps(dict(unit='+repr(unit)+
                ',pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))')
        owner = json.loads(s.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True))
        owner['command'] = cmd
        (out/(prefix+'_owner.json')).write_text(json.dumps(owner, indent=2)+'\n')
        source = json.loads((s.root/'results/v3_a1_20261103/controller_prepare.json').read_text())
        (out/(prefix+'_source.json')).write_text(json.dumps(source, indent=2)+'\n')
        with tarfile.open(out/(prefix+'_source.tgz'), 'w:gz') as t:
            for f in source['runner_sha256']:
                t.add(s.root/f, arcname=f)
            t.add(out/(prefix+'_source.json'), arcname='manifest.json')
        print(json.dumps(dict(unit=unit, pid=owner['pid'])))
    finally:
        subprocess.run(['ssh', '-S', str(s.root/'tmp'/e.socket), '-O', 'exit', 'test@'+e.ip],
                       capture_output=True, timeout=5)


if __name__ == '__main__':
    main()
