"""Emit bounded Lab Jobs for a staged SHIFT training session."""

import argparse
import json
from pathlib import Path
import re

from scripts.lab_spare_pool import job
from scripts.lab_session_backend import POD_NODES


def resources(experiment, stage, *, prepare, seconds, two_node=False, whole_dp=False, anchor_node='gpu-01',
              colocated_spare=False):
    if not all(re.fullmatch(r'shift-[a-z0-9-]{1,42}', name) for name in (experiment, stage)):
        raise ValueError('expected session-owned SHIFT names')
    base = Path('/data/lab') / stage
    root = base / 'SHIFT'
    python = root / 'runtime/torch214/bin/python'
    items = []
    if not prepare:
        agent = Path(__file__).resolve().parents[1] / 'shift_full/spare_agent.py'
        items.append(dict(apiVersion='v1', kind='ConfigMap',
                          metadata=dict(name=experiment, labels={'experiment': experiment}),
                          data={'spare_agent.py': agent.read_text()}))
    if colocated_spare and (prepare or two_node):
        raise ValueError('colocated spare requires three training runners')
    if two_node:
        peer = 'gpu-01' if anchor_node == 'gpu-03' else 'gpu-03'
        runners = [('node0', anchor_node, 8 if whole_dp else 6), ('node1', peer, 4)]
    else:
        placement = dict(POD_NODES, node0='gpu-03', node1='gpu-01') if colocated_spare else POD_NODES
        runners = [(alias, node, 4) for alias, node in placement.items()]
    for alias, node, slots in runners:
        item = job(experiment, node, False, 0 if prepare else slots)
        if colocated_spare:
            item['metadata']['name'] = f'{experiment}-{alias}'
        if not prepare:
            item['metadata']['labels']['shift-runner'] = alias
            item['spec']['template']['metadata']['labels']['shift-runner'] = alias
        item['spec']['activeDeadlineSeconds'] = seconds
        pod = item['spec']['template']['spec']
        container = pod['containers'][0]
        pod['volumes'].append({'name':'stage', 'hostPath':{'path':'/data/lab', 'type':'Directory'}})
        container['volumeMounts'].append({'name':'stage','mountPath':str(base),'subPath':stage})
        if prepare:
            pod['volumes'] = [volume for volume in pod['volumes'] if volume['name'] != 'agent']
            container['volumeMounts'] = [mount for mount in container['volumeMounts'] if mount['name'] != 'agent']
            container['command'] = ['bash','-euc']
            container['args'] = [
                'root=$1; python3 "$root/scripts/lab_prepare_runtime.py" "$root"; '
                'python3 -m venv "$root/runtime/torch214"; '
                'p="$root/runtime/torch214/bin/python"; '
                '"$p" -m pip install --no-index --find-links "$root/runtime/lab-wheels" '
                '--no-cache-dir --disable-pip-version-check --require-hashes '
                '-r "$root/runtime/torch214-manifest/requirements-linux-cp312.lock"; '
                '"$p" -c "import torch,torchft; print(torch.__version__)"; '
                'touch "$root/runtime/lab-runtime-ready"', 'prepare', str(root)]
            container['resources'] = {'requests':{'cpu':'2','memory':'3Gi','ephemeral-storage':'1Gi'},
                                      'limits':{'memory':'8Gi','ephemeral-storage':'12Gi'}}
        else:
            pod.update(hostNetwork=True, dnsPolicy='ClusterFirstWithHostNet')
            pod['volumes'].append({'name':'shm', 'emptyDir':{'medium':'Memory','sizeLimit':'2Gi'}})
            container['volumeMounts'].append({'name':'shm','mountPath':'/dev/shm'})
            container['command'] = [str(python), '/agent/spare_agent.py']
            container['args'] = ['--gpus',str(slots),'--profile','shift-session','--model-dir',str(base/'FLASH1022/models/qwen_moe')]
            container['readinessProbe'] = {
                'exec': {'command': ['python3', '/agent/spare_agent.py', '--check-ready']}, 'periodSeconds': 5}
            container['securityContext'] = {'capabilities':{'add':['IPC_LOCK','SYS_RESOURCE']}}
            container['resources'] = {'requests':{'cpu':'4','memory':'8Gi','ephemeral-storage':'1Gi'},
                                      'limits':{'memory':'40Gi' if slots >= 6 else '24Gi','ephemeral-storage':'12Gi',
                                                'nvidia.com/gpu':slots,'rdma/rdma_shared_device_a':1}}
        items.append(item)
    return dict(apiVersion='v1',kind='List',items=items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--two-node', action='store_true', help='allocate only gpu-01 and gpu-03')
    parser.add_argument('--whole-dp', action='store_true', help='use eight GPUs on gpu-01 in a two-node v2/v1 session')
    parser.add_argument('--colocated-spare', action='store_true',
                        help='three four-GPU runners on gpu-01 and gpu-03')
    parser.add_argument('--anchor-node', choices=('gpu-01','gpu-03'), default='gpu-01')
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 60 <= args.seconds <= 3600:
        parser.error('job deadline must be between 60 and 3600 seconds')
    if args.whole_dp and not args.two_node:
        parser.error('--whole-dp requires --two-node')
    if args.colocated_spare and (args.prepare or args.two_node):
        parser.error('--colocated-spare requires three training runners')
    print(json.dumps(resources(args.experiment, args.stage, prepare=args.prepare,
                               seconds=args.seconds, two_node=args.two_node, whole_dp=args.whole_dp,
                               anchor_node=args.anchor_node, colocated_spare=args.colocated_spare)))


if __name__ == '__main__':
    main()
