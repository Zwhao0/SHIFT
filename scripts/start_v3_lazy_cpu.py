"""Deploy and start the bounded P13 CPU diagnostic on node0 only."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from shift_full.control import Session, endpoints_from_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    a = p.parse_args()
    assert a.suffix and all(c.isalnum() or c in '_-' for c in a.suffix)
    s = Session.__new__(Session)
    s.root, s._sudo_password = Path(__file__).resolve().parents[1], None
    e = replace(endpoints_from_env()['node0'], socket=f'ssh-v3-p13cpu-{os.getpid()}.sock')
    prefix = 'p13_'+a.suffix
    out = s.root/'results/v3_p_20261103'
    assert not (out/(prefix+'_owner.json')).exists()
    files = ['scripts/v3_lazy_cpu_probe.py', 'scripts/start_v3_lazy_cpu.py',
             'shift_full/cold_imports.py', 'shift_full/train.py', 'shift_full/compat.py', 'shift_full/state_digest.py']
    manifest = {f:hashlib.sha256((s.root/f).read_bytes()).hexdigest() for f in files}
    unit = 'shift-v3-'+prefix.replace('_', '-')
    r = e.host_root
    cmd = ['systemd-run', '--quiet', '--collect', '--unit='+unit, '--property=RuntimeMaxSec=400',
           '--property=KillMode=control-group', '--property=TimeoutStopSec=10',
           '--property=StandardOutput=append:'+r+'/results/v3_p_20261103/'+prefix+'.log',
           '--property=StandardError=inherit', '--working-directory='+r, '--setenv=PYTHONPATH='+r,
           '--setenv=CUDA_VISIBLE_DEVICES=', '--setenv=OMP_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1',
           '--setenv=TOKIO_WORKER_THREADS=2', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           '--setenv=FLASH_PARALLEL_LAYOUT=legacy', '--setenv=TORCHFT_USE_OTEL=false',
           r+'/runtime/torch214/bin/python', '-m', 'scripts.v3_lazy_cpu_probe', '--suffix', a.suffix]
    try:
        s.ensure_connection(e)
        code = ('import fcntl;from pathlib import Path;f=Path('+repr(r+'/runtime/v3-training.lock')+').open("a");'
                'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not Path('+repr(r+'/results/v3_p_20261103/'+prefix)+').exists()')
        s.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True)
        subprocess.run(['scp', '-q', '-o', 'ControlPath='+str(s.root/'tmp'/e.socket),
                        str(s.root/'scripts/v3_lazy_cpu_probe.py'), 'test@'+e.ip+':'+r+'/scripts/v3_lazy_cpu_probe.py'],
                       check=True, capture_output=True, timeout=30)
        code = ('from pathlib import Path;import hashlib,json,subprocess;r=Path('+repr(r)+');m='+repr(manifest)+';'
                'assert all(hashlib.sha256((r/f).read_bytes()).hexdigest()==h for f,h in m.items() if f!="scripts/start_v3_lazy_cpu.py");'
                'subprocess.run('+repr(cmd)+',check=True);print(json.dumps(dict(unit='+repr(unit)+',pid=int(subprocess.check_output('
                '["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))')
        owner = json.loads(s.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True))
        owner.update(command=cmd, source_sha256=manifest)
        (out/(prefix+'_owner.json')).write_text(json.dumps(owner, indent=2)+'\n')
        with tarfile.open(out/(prefix+'_source.tgz'), 'w:gz') as archive:
            for f in files:
                archive.add(s.root/f, arcname=f)
        print(json.dumps(dict(unit=unit, pid=owner['pid'])))
    finally:
        subprocess.run(['ssh', '-S', str(s.root/'tmp'/e.socket), '-O', 'exit', 'test@'+e.ip],
                       capture_output=True, timeout=5)


if __name__ == '__main__':
    main()
