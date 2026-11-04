"""Server-owned, CPU-only transport smoke; never starts training or injects faults."""
import hashlib
import json
from pathlib import Path
import subprocess as sp
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env, server_password


def main():
    remote.PASSWORD = server_password(remote.PRIVATE / 'credentials')
    (remote.PRIVATE / 'credentials').unlink()
    root = remote.ROOT
    out = root / 'results/v3_a1_20261103'
    out.mkdir(parents=True, exist_ok=True)
    probe = remote.ServerRotation.__new__(remote.ServerRotation)
    probe.root = root
    probe._sudo_password = remote.PASSWORD
    eps = endpoints_from_env()
    result = dict(done=False, passed=False, pid=__import__('os').getpid(),
                  started=time.time(), training_started=False, injections=0)
    try:
        result['transport'] = {}
        for name, e in eps.items():
            probe.ensure_connection(e)
            payload = 'SHIFT v3 CPU-only transport smoke: '+name+'\n'
            path = e.host_root+'/results/v3_a1_20261103/transport_'+name+'.txt'
            code = ('from pathlib import Path; p=Path('+repr(path)+'); '
                    'p.parent.mkdir(parents=True,exist_ok=True); '
                    'p.write_text('+repr(payload)+'); p.chmod(0o644)')
            probe.ssh(e, ['sudo', '-S', '-p', '', 'python3', '-c', code], sudo_input=True)
            dest = out/('collected_'+name+'.txt')
            sp.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'ControlPath='+str(root/'tmp'/e.socket),
                    'test@'+e.ip+':'+path, str(dest)],
                   check=True, capture_output=True, timeout=25)
            assert dest.read_text() == payload, 'collection payload differs'
            result['transport'][name] = dict(ssh=True, scp=True,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        result['formal_injections'] = remote.injections()
        result['passed'] = True
    except BaseException as error:
        # Commands and credentials are deliberately excluded from the report.
        result['error'] = type(error).__name__
        raise
    finally:
        result.update(done=True, finished=time.time())
        (out/'controller_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
