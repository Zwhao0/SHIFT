"""One bounded two-node experiment; faults target only our supervised workers."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import sys
import time


NODES = [dict(ip='192.0.2.10', container='flash1022-run', sock='ssh-h2.sock', sudo=False),
         dict(ip='192.0.2.11', container='flash1022-data16', sock='ssh-h1.sock', sudo=True)]
SERVER_ROOT = '/workspace/SHIFT'
CONTAINER_ROOT = '/workspace/SHIFT'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--seconds', type=int, default=900)
    a = p.parse_args()
    if not all(c.isalnum() or c in '_-' for c in a.session):
        p.error('Session must contain only letters, digits, underscores, hyphens')
    root = Path(__file__).resolve().parents[1]
    out = root / 'results' / a.session
    out.mkdir(parents=True, exist_ok=False)
    journal = (out / 'controller.jsonl').open('w', buffering=1)
    started = time.monotonic()
    launched = False
    injected = False
    success = False

    def event(kind, **fields):
        row = dict(kind=kind, time=time.time(), rid=-1, rank=-1, generation=-1,
                   pid=os.getpid())
        row.update(fields)
        journal.write(json.dumps(row) + '\n')
        if kind not in ('poll',):
            print(json.dumps(row), flush=True)

    def ssh(node, command, timeout=40):
        n = NODES[node]
        return sp.check_output(['ssh', '-S', str(root / 'tmp' / n['sock']),
                                f'test@{n["ip"]}', shlex.join(command)],
                               text=True, stderr=sp.PIPE, timeout=timeout)

    def docker(node, args, timeout=40):
        command = (['sudo', '-n', 'env'] if NODES[node]['sudo'] else ['env'])
        command += ['DOCKER_HOST=unix:///run/greyhound-docker.sock',
                    '/opt/greyhound-docker/docker'] + args
        return ssh(node, command, timeout)

    def execute(node, args, timeout=40):
        return docker(node, ['exec', NODES[node]['container']] + args, timeout)

    def mark(node, filename):
        execute(node, ['touch', f'{CONTAINER_ROOT}/results/{a.session}/{filename}'])

    def read(node):
        code = """import json,pathlib
p=pathlib.Path(%r)
r={}
if p.exists():
 for f in p.glob('*.jsonl'):
  rows=[]
  for line in f.read_text().splitlines():
   try: rows.append(json.loads(line))
   except json.JSONDecodeError: pass
  r[f.name]=rows
print(json.dumps(r))
""" % f'{CONTAINER_ROOT}/results/{a.session}'
        return json.loads(execute(node, ['python', '-c', code]))

    def await_condition(label, predicate):
        deadline = started + a.seconds
        while time.monotonic() < deadline:
            rows = [read(0), read(1)]
            for node in range(2):
                errors = [e for e in rows[node].get('supervisor.jsonl', [])
                          if e['kind'] in ('unexpected_exit', 'supervisor_error')]
                if errors:
                    raise RuntimeError(f'node{node}: {errors[-1]}')
            value = predicate(rows)
            if value:
                event(label)
                return rows
            time.sleep(0.5)
        raise TimeoutError(label)

    def events(rows, node, generation, rank, kind):
        return [e for e in rows[node].get(f'events_g{generation}_r{rank}.jsonl', [])
                if e['kind'] == kind]

    try:
        for node in range(2):
            docker(node, ['exec', '-d', NODES[node]['container'], 'sh', '-c',
                         f'cd {CONTAINER_ROOT} && PYTHONPATH={CONTAINER_ROOT} '
                         f'python -m shift_min.supervisor --session {a.session} '
                         f'--replica {node} --host-ip {NODES[node]["ip"]} --seconds {a.seconds}'])
            launched = True
        event('launched')
        rows = await_condition('all_16_ready', lambda rows: all(
            events(rows, n, 0, r, 'ready') for n in range(2) for r in range(8)))
        for rank in range(8):
            left, right = [events(rows, n, 0, rank, 'ready')[-1] for n in range(2)]
            assert left['hashes'] == right['hashes'], f'Initial state mismatch rank{rank}'
            assert left['local_parameters'] == right['local_parameters']
        event('initial_states_equal')
        for node in range(2):
            mark(node, 'START')
        await_condition('three_joint_commits_before_failure', lambda rows: all(
            len([e for e in events(rows, n, 0, r, 'commit') if e['participants'] == 2]) >= 3
            for n in range(2) for r in range(8)))
        mark(1, 'KILL')
        event('kill_requested', target_replica=1)
        rows = await_condition('fault_observed', lambda rows: any(
            e['kind'] == 'injected' for e in rows[1].get('supervisor.jsonl', [])))
        fault = next(e for e in rows[1]['supervisor.jsonl'] if e['kind'] == 'injected')
        event('injected', time=fault['time'], rid=1, generation=0,
              pid=fault['supervisor_pid'], signal=fault['signal'], pids=fault['pids'])
        injected = True
        await_condition('three_survivor_commits', lambda rows: all(
            len([e for e in events(rows, 0, 0, r, 'commit')
                 if e['participants'] == 1 and e['time'] > fault['time']]) >= 3
            for r in range(8)))
        mark(1, 'RESTART')
        event('restart_requested', target_replica=1)
        await_condition('all_8_restored', lambda rows: all(
            any(e['hashes_match'] and e['optimizer_tensors'] > 0 and e['step'] > 0
                for e in events(rows, 1, 1, r, 'restored')) for r in range(8)))

        def rejoined(rows):
            common = None
            for rank in range(8):
                restored = events(rows, 1, 1, rank, 'restored')
                if not restored:
                    return False
                step = restored[-1]['step']
                sets = [{e['step'] for e in events(rows, node, generation, rank, 'commit')
                         if e['participants'] == 2 and e['step'] > step}
                        for node, generation in ((0, 0), (1, 1))]
                both = sets[0] & sets[1]
                common = both if common is None else common & both
            return len(common) >= 3

        await_condition('three_joint_commits_after_recovery', rejoined)
        event('complete')
        success = True
    except BaseException as error:
        event('failed', error=repr(error), injected=injected)
        raise
    finally:
        if launched:
            for node in range(2):
                try:
                    mark(node, 'STOP')
                except Exception as error:
                    event('stop_error', node=node, error=repr(error))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if all(any(e['kind'] == 'cleanup' for e in read(node).get('supervisor.jsonl', []))
                           for node in range(2)):
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            for node in range(2):
                try:
                    ssh(node, ['mkdir', '-p', f'{SERVER_ROOT}/results'])
                    docker(node, ['cp', f'{NODES[node]["container"]}:{CONTAINER_ROOT}/results/{a.session}',
                                  f'{SERVER_ROOT}/results/'])
                    sp.run(['scp', '-q', '-r', '-o',
                            f'ControlPath={root / "tmp" / NODES[node]["sock"]}',
                            f'test@{NODES[node]["ip"]}:{SERVER_ROOT}/results/{a.session}',
                            str(out / f'node{node}')], check=True, timeout=60)
                except Exception as error:
                    event('collection_error', node=node, error=repr(error))
        event('controller_stopped', completed=success, seconds=time.monotonic() - started)
        journal.close()
    sp.run([sys.executable, '-m', 'shift_min.verify', str(out)], check=True, cwd=root)


if __name__ == '__main__':
    main()
