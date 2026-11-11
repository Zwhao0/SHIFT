"""Manage a session-owned process group inside a Lab runner Pod."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return dict(state=fields[0], group=int(fields[2]), started=int(fields[19]))


def group_members(group):
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            pid = int(path.name)
            state = process(pid)
            if state and state['group'] == group and state['state'] not in ('Z', 'X'):
                members.append(pid)
    return sorted(members)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def start(directory, command, env, log):
    directory.mkdir(mode=0o700)
    write_json(directory / 'launch.json', dict(command=command, env=env, log=log))
    with open(log, 'ab', buffering=0) as output:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise', str(directory)],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True)
    state = process(child.pid)
    if state is None or state['group'] != child.pid:
        raise RuntimeError('actor supervisor exited before registration')
    write_json(directory / 'process.json', dict(pid=child.pid, started=state['started']))


def supervise(directory):
    spec = json.loads((directory / 'launch.json').read_text())
    child = None
    try:
        child = subprocess.Popen(spec['command'], env=os.environ | spec['env'])
        returncode = child.wait()
    except OSError as error:
        print(error, file=sys.stderr, flush=True)
        returncode = 127
    write_json(directory / 'exit.json', dict(returncode=returncode, pid=child.pid if child else None))


def status(directory):
    identity = json.loads((directory / 'process.json').read_text())
    pid = identity['pid']
    state = process(pid)
    owned = bool(state and state['group'] == pid and state['started'] == identity['started'])
    result = directory / 'exit.json'
    return dict(pid=pid, live=owned and state['state'] not in ('Z', 'X'),
                remaining=group_members(pid),
                exit=json.loads(result.read_text()) if result.exists() else None)


def stop(directory):
    info = status(directory)
    if info['live']:
        try:
            os.killpg(info['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while status(directory)['remaining']:
        if time.monotonic() >= deadline:
            raise RuntimeError('owned process group did not drain before Pod cleanup')
        time.sleep(.05)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'supervise':
        supervise(Path(sys.argv[2]))
        return
    operation, directory, *payload = sys.argv[1:]
    directory = Path(directory)
    if operation == 'start':
        start(directory, **json.loads(payload[0]))
    elif operation == 'status':
        print(json.dumps(status(directory)))
    elif operation == 'stop':
        stop(directory)
    else:
        raise ValueError('unsupported actor operation')


if __name__ == '__main__':
    main()
