"""Probe GPU allocation at low frequency from the current acceptance task."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from scripts.lab_spare_pool import job


def watch(kube, node, slots, interval, evidence, *, once=False):
    while True:
        run = 'shift-watch-' + uuid.uuid4().hex[:10]
        selector = f'experiment={run}'
        spec = job(run, node, False, slots)
        spec['spec']['activeDeadlineSeconds'] = 155
        pod = spec['spec']['template']['spec']
        pod.pop('volumes')
        container = pod['containers'][0]
        container.pop('volumeMounts')
        container.pop('args')
        container['command'] = ['sleep','5']
        result = dict(run=run, node=node, slots=slots)
        try:
            kube('create', '-f', '-', payload=json.dumps(spec))
            pod_name = ''
            for _ in range(8):
                pod_name = kube('get', 'pod', '-l', selector, '-o', 'name').stdout.strip()
                if pod_name:
                    break
                time.sleep(2)
            check = None
            if pod_name and '\n' not in pod_name:
                check = kube('wait', '--for=condition=PodScheduled', pod_name, '--timeout=120s', check=False)
            pods = json.loads(kube('get', 'pod', '-l', selector, '-o', 'json').stdout)['items']
            result['scheduled'] = bool(check and check.returncode == 0 and len(pods) == 1
                                       and pods[0]['spec'].get('nodeName') == node)
            if not result['scheduled']:
                reason = 'FailedScheduling' if pods else 'FailedCreate'
                events = json.loads(kube('get', 'events', '--field-selector', f'reason={reason}', '-o', 'json').stdout)
                result['reasons'] = sorted({row['message'] for row in events['items']
                                            if row['involvedObject']['name'].startswith(run + '-')})
        finally:
            kube('delete', 'job', '-l', selector, '--cascade=foreground', '--ignore-not-found', '--timeout=30s')
            remaining = kube('get', 'job,pod', '-l', selector, '-o', 'name').stdout.strip()
            result['resources_clean'] = not remaining
            with evidence.open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
            if remaining:
                raise RuntimeError('capacity probe resources were not removed')
        if result['scheduled'] or once:
            return result['scheduled']
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', default='gpu-02', choices=('gpu-01','gpu-02','gpu-03'))
    parser.add_argument('--slots', type=int, default=4, choices=range(1,9))
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--ssh-target')
    parser.add_argument('--ssh-socket')
    args = parser.parse_args()
    if args.interval < 60 or bool(args.ssh_socket) != bool(args.ssh_target):
        parser.error('require an interval of at least 60 seconds and either both SSH options or neither')

    def kube(*words, payload=None, check=True):
        command = ['kubectl', '-n', 'lab', *words]
        if args.ssh_socket:
            command = ['ssh','-o','BatchMode=yes','-S',args.ssh_socket,args.ssh_target,shlex.join(command)]
        return subprocess.run(command, input=payload, text=True, capture_output=True, check=check, timeout=145)

    if not watch(kube, args.node, args.slots, args.interval, args.evidence, once=args.once):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
