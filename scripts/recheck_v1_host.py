"""Bounded V1 fail-stop regression using the current code and a small host venv.

The venv and /workspace aliases must be prepared first. Only the controller's
container-exec/copy transport is replaced; training, supervisor, and recovery
use the current shared NCCL recovery adapter. See results/v1_recheck_20261103.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shift_full.rotation import Rotation
from shift_full.verify_rotation import verify

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results/v1_recheck_20261103'


class HostRotation(Rotation):
    """Translate this controller's small Docker exec/cp surface into host SSH."""

    def docker(self, endpoint, cmd, timeout=45):
        endpoint = self.coerce_endpoint(endpoint)
        if cmd[0] == 'cp':
            source, destination = cmd[1:]
            prefix = endpoint.container + ':'
            if not source.startswith(prefix):
                raise ValueError('only collection from this endpoint is supported')
            code = ('import shutil; from pathlib import Path; Path(' + repr(destination) + ').parent.mkdir(parents=True,exist_ok=True); shutil.copyfile('
                    + repr(source[len(prefix):]) + ',' + repr(destination) + ')')
        elif cmd[0] == 'exec':
            i, detached, cwd, env = 1, False, '/workspace/SHIFT', {}
            while cmd[i].startswith('-'):
                if cmd[i] == '-d':
                    detached, i = True, i + 1
                elif cmd[i] in ('-w', '-e'):
                    option, value = cmd[i:i+2]
                    if option == '-w':
                        cwd = value
                    else:
                        key, value = value.split('=', 1)
                        env[key] = value
                    i += 2
                else:
                    raise ValueError('unsupported exec option')
            if cmd[i] != endpoint.container:
                raise ValueError('unexpected endpoint')
            args = list(cmd[i+1:])
            # Supervisor must use system Python so V1 selects its isolated
            # training-runtime branch when it compares sys.executable.
            if args[0] == 'python':
                args[0] = '/usr/bin/python3'
            code = ('import os,subprocess,sys,resource; from pathlib import Path; '
                    'soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); resource.setrlimit(resource.RLIMIT_NOFILE,(hard,hard)); '
                    'resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); '
                    'cache=Path("/workspace/SHIFT/runtime/v1-recheck-bootstrap/cache"); cache.mkdir(parents=True,exist_ok=True); '
                    'env=dict(os.environ,TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),HF_HOME=str(cache/"hf"),TRITON_CACHE_DIR=str(cache/"triton"),PYTHONDONTWRITEBYTECODE="1",CPATH="/workspace/SHIFT/runtime/python-headers"); env.update(' + repr(env) + '); ')
            if detached:
                code += ('p=Path("/workspace/SHIFT/tmp/v1-recheck-transport.log"); '
                         'p.parent.mkdir(parents=True,exist_ok=True); '
                         'log=p.open("a"); '
                         'proc=subprocess.Popen(' + repr(args) + ',cwd=' + repr(cwd)
                         + ',env=env,stdin=subprocess.DEVNULL,stdout=log,'
                           'stderr=subprocess.STDOUT,start_new_session=True); '
                           'print(proc.pid)')
            else:
                code += ('r=subprocess.run(' + repr(args) + ',cwd=' + repr(cwd)
                         + ',env=env,stdin=subprocess.DEVNULL); sys.exit(r.returncode)')
        else:
            raise ValueError('unsupported runtime operation')
        return self.ssh(endpoint, ['sudo', '-S', '-p', '', '/usr/bin/python3', '-c', code],
                        timeout=timeout, sudo_input=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--config', default='configs/shift_rotation_verify.json')
    parser.add_argument('--warmup-commits', type=int, default=3)
    parser.add_argument('--post-steps', type=int, default=6)
    parser.add_argument('--report-name', default='runs.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261106, 20261107, 20261110])
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, seed in enumerate(args.seeds, 1):
        name = f'v1_recheck_20261103_{index:02d}_{seed}' + args.suffix
        config = json.loads((ROOT/args.config).read_text())
        config['experiment'] = 'post-cleanup V1 cold fail-stop regression; host transport; seeded balanced placements'
        config['warmup_joint_commits'] = args.warmup_commits
        s = HostRotation(name, config, 600, seed, backup_mode='cold')
        try:
            s.start()
            s.await_('warmup', lambda d: bool(s.common_commits(d, args.warmup_commits)[0]))
            began = time.monotonic()
            s.event('fault_window_start', duration=0, requested_faults=1)
            s.fault(0)
            restored = s.fault_results[0]['first_joint_step']
            s.await_('post_recovery_progress', lambda d: bool(s.common_commits(d, restored + args.post_steps - 1)[0]))
            s.event('complete', faults=1, fault_window_seconds=time.monotonic()-began)
        except BaseException as error:
            s.event('failed', error=repr(error))
            raise
        finally:
            s.finish()
        report = verify(s.out, 1, require_audit=True)
        (REPORT/(name+'_verify.json')).write_text(json.dumps(report, indent=2)+'\n')
        reports.append(dict(session=name, seed=seed, passed=report['pass'], errors=report['errors'],
                            faults=s.fault_results))
        (REPORT/args.report_name).write_text(json.dumps(reports, indent=2)+'\n')
        print(json.dumps(dict(session=name, passed=report['pass'], errors=report['errors'])), flush=True)
        if not report['pass']:
            raise RuntimeError('V1 regression verification failed')

    if args.seeds == [20261106, 20261107, 20261110]:
        assert {r['faults'][0]['old_node'] for r in reports} == {'node0', 'node1', 'node3'}
        assert {r['faults'][0]['new_node'] for r in reports} == {'node0', 'node1', 'node3'}


if __name__ == '__main__':
    main()
