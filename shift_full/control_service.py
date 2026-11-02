"""Session-owned Lighthouse, independent of training replica placement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--seconds', type=int, required=True)
    a = p.parse_args()
    out = Path(a.directory)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(a.config).read_text())
    cmd = [cfg['python'], str(Path(a.root)/'vendor/bin/torchft_lighthouse'),
           '--bind', f"[::]:{cfg.get('lighthouse_port', 29771)}", '--min_replicas', str(cfg.get('min_replicas',1))]
    for key, default in [('heartbeat_timeout_ms', 500), ('join_timeout_ms', 200), ('quorum_tick_ms', 50)]:
        cmd += ['--'+key, str(cfg.get(key, default))]
    env = dict(os.environ, TORCHFT_USE_OTEL='false', RUST_LOG='warn')
    with (out/'lighthouse.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        (out/'owner.json').write_text(json.dumps(dict(pid=os.getpid(), lighthouse_pid=proc.pid, command=cmd)))
        try:
            deadline = time.monotonic()+a.seconds
            while not (out/'STOP').exists() and time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError('Lighthouse exited')
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            (out/'cleanup.json').write_text(json.dumps(dict(returncode=proc.returncode)))


if __name__ == '__main__':
    main()
