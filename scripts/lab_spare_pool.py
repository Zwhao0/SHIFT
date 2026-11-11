"""Validate the spare pool on three lab nodes with one shared GPU per node."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from shift_full.spare_pool import SparePool
from shift_full.spare_selection import SpareReport


IMAGE = 'registry.reinfra.cn/reinfra/infrastructure/images/vllm-openai:v0.32.0'


def job(run, node, warm, gpus):
    name = f'{run}-{node}'
    labels = {'experiment': run}
    limits = {'memory': '4Gi', 'ephemeral-storage': '2Gi'}
    if gpus:
        limits['nvidia.com/gpu'] = gpus
    container = dict(
        name='agent', image=IMAGE, command=['python3', '/agent/spare_agent.py'],
        args=['--gpus', str(gpus)] + (['--warm-cuda'] if warm and gpus else []),
        env=[{'name': name, 'valueFrom': {'fieldRef': {'fieldPath': field}}}
             for name, field in [('SHIFT_NODE_NAME', 'spec.nodeName'), ('SHIFT_POD_UID', 'metadata.uid')]],
        resources={'requests': {'cpu': '1', 'memory': '2Gi', 'ephemeral-storage': '1Gi'},
                   'limits': limits},
        volumeMounts=[{'name': 'agent', 'mountPath': '/agent', 'readOnly': True}],
    )
    if not gpus:
        container['env'].append({'name': 'CUDA_VISIBLE_DEVICES', 'value': ''})
    spec = dict(restartPolicy='Never', terminationGracePeriodSeconds=5,
                serviceAccountName='lab', automountServiceAccountToken=False,
                nodeSelector={'kubernetes.io/hostname': node}, containers=[container],
                volumes=[{'name': 'agent', 'configMap': {'name': run}}])
    return dict(apiVersion='batch/v1', kind='Job', metadata=dict(name=name, labels=labels),
                spec=dict(backoffLimit=0, activeDeadlineSeconds=300, ttlSecondsAfterFinished=300,
                          template=dict(metadata=dict(labels=labels), spec=spec)))


def pool_run(kube, nodes, evidence, warm, cpu_only, *, colocated_spare=False):
    run = 'shift-spare-' + uuid.uuid4().hex[:10]
    agent = Path(__file__).resolve().parents[1] / 'shift_full' / 'spare_agent.py'
    config = dict(apiVersion='v1', kind='ConfigMap',
                  metadata=dict(name=run, labels={'experiment': run}),
                  data={'spare_agent.py': agent.read_text()})
    placement = {'node0':'gpu-03', 'node1':'gpu-01', 'node3':'gpu-03'} if colocated_spare else {n:n for n in nodes}
    resources = [config]
    for runner, node in placement.items():
        item = job(run, node, warm, int(node not in cpu_only))
        if colocated_spare:
            item['metadata']['name'] = f'{run}-{runner}'
        item['spec']['template']['metadata']['labels']['shift-runner'] = runner
        resources.append(item)
    result = dict(run=run, scope='one-GPU agent and selection smoke', warm_cuda=warm,
                  physical_nodes=placement, cpu_only=sorted(cpu_only), reports=[], decisions=[], reentries=[])
    selector = f'experiment={run}'

    def snapshot(pod):
        before = time.monotonic()
        row = json.loads(kube('exec', pod['metadata']['name'], '--', 'python3', '/agent/spare_agent.py', '--read'))
        if (row['node'] != pod['spec']['nodeName'] or row['instance'] != pod['metadata']['uid']
                or row['age_seconds'] < 0):
            raise RuntimeError('standby report identity or age is invalid')
        return row, before - row['age_seconds']

    try:
        kube('create', '-f', '-', payload=json.dumps(dict(apiVersion='v1', kind='List', items=resources)))
        for _ in range(12):
            pods = json.loads(kube('get', 'pod', '-l', selector, '-o', 'json'))['items']
            if len(pods) == len(placement):
                break
            time.sleep(1)
        else:
            raise RuntimeError('incomplete standby Pod set')
        kube('wait', '--for=condition=Ready', 'pod', '-l', selector, '--timeout=120s')
        pods = json.loads(kube('get', 'pod', '-l', selector, '-o', 'json'))['items']
        pods_by_node = {pod['metadata']['labels']['shift-runner']: pod for pod in pods}
        if len(pods) != len(placement) or {runner: pod['spec']['nodeName']
                                          for runner, pod in pods_by_node.items()} != placement:
            raise RuntimeError('standby Pod placement differs from the requested runners')
        pool = SparePool(placement)

        def refresh(node, *, after=0):
            pod = pods_by_node[node]
            for attempt in range(8):
                try:
                    row, observed = snapshot(pod)
                except subprocess.CalledProcessError:
                    if attempt == 7:
                        raise
                else:
                    if row['sequence'] > after:
                        pool.record(SpareReport(
                            node, row['instance'], observed, frozenset({row['profile']}),
                            tuple((d['uuid'], d['free_mib']) for d in row['devices']), row['ready']))
                        return row
                time.sleep(1)
            raise RuntimeError(f'no new report from {node}')

        def choose(excluded, *, attempt=None, required=True, minimum=512, now=None):
            started = time.monotonic()
            options = dict(profile='cuda-smoke', ranks=1, min_gpu_mib=minimum, now=started if now is None else now,
                           max_age=15, excluded_nodes=excluded)
            decision = pool.decide(**options) if attempt is None else pool.claim(attempt, **options)
            elapsed = time.monotonic() - started
            result['decisions'].append(dict(excluded=sorted(excluded), decision=asdict(decision),
                                            selection_seconds=elapsed))
            if required and decision.target is None:
                raise RuntimeError('no eligible spare')
            return decision

        eligible = {runner for runner, node in placement.items() if node not in cpu_only}
        result['reports'] = [refresh(runner) for runner in placement]
        all_spares = choose(())
        if {candidate.node for candidate in all_spares.ranked} != eligible:
            raise RuntimeError('GPU eligibility differs from the provisioned node set')
        if len({candidate.slots[0] for candidate in all_spares.ranked}) != len(eligible):
            raise RuntimeError('standby Pods received overlapping GPU UUIDs')
        devices = [device for report in result['reports'] for device in report['devices']]
        over_capacity = max(device['total_mib'] for device in devices) + 1
        exhausted = choose((), required=False, minimum=over_capacity)
        stale = choose((), required=False, now=time.monotonic()+16)
        if (any(dict(exhausted.rejected).get(node) != 'insufficient_slots' for node in eligible)
                or any(dict(stale.rejected).get(node) != 'stale_report' for node in eligible)):
            raise RuntimeError('a stale or memory-ineligible report was accepted')
        result['health_checks'] = dict(required_mib=over_capacity, aged_report_seconds=16,
                                       low_memory_rejected=asdict(exhausted), stale_rejected=asdict(stale))

        workload = (
            'import json,time,torch;start=time.monotonic();'
            'x=torch.arange(128,device="cuda");value=int(x.sum().item());'
            'p=torch.cuda.get_device_properties(0);'
            'print(json.dumps(dict(value=value,uuid=str(p.uuid),work_seconds=time.monotonic()-start)))'
        )
        for node in sorted(eligible):
            for candidate in placement:
                refresh(candidate)
            excluded = set(placement) - {node}
            chosen = choose(excluded, attempt=node).target
            if choose(excluded, required=False).target is not None:
                raise RuntimeError('claimed GPU can still be assigned')
            started = time.monotonic()
            pod = pods_by_node[chosen.node]['metadata']['name']
            task = json.loads(kube('exec', pod, '--', 'python3', '-c', workload))
            if task['value'] != sum(range(128)) or task['uuid'] != chosen.slots[0]:
                raise RuntimeError('standby workload result or GPU allocation mismatch')
            task_seconds = time.monotonic() - started
            pool.release_after_exit(node, now=time.monotonic())
            if choose(excluded, required=False).target is not None:
                raise RuntimeError('released GPU was eligible before it reported again')
            at_exit = refresh(node)
            after_exit = refresh(node, after=at_exit['sequence'])
            if (not after_exit['ready'] or after_exit['instance'] != chosen.instance
                    or choose(excluded).target.node != node):
                raise RuntimeError('node did not return to standby after workload exit')
            result['reentries'].append(dict(node=node, task_seconds=task_seconds, task=task,
                                            new_report=after_exit))
        for node in placement:
            refresh(node)
        if {candidate.node for candidate in choose(()).ranked} != eligible:
            raise RuntimeError('pool membership did not recover after the rotation')
        result['passed'] = True
    except Exception as error:
        result.update(passed=False, error=str(error))
        if isinstance(error, subprocess.CalledProcessError):
            result['stderr'] = error.stderr
        try:
            result['pods'] = kube('get', 'pod', '-l', selector, '-o', 'wide')
        except subprocess.SubprocessError:
            pass
        raise
    finally:
        evidence.write_text(json.dumps(result, indent=2) + '\n')
        try:
            kube('delete', 'job', '-l', selector, '--cascade=foreground', '--ignore-not-found',
                 '--wait=true', '--timeout=45s')
            kube('delete', 'pod,configmap', '-l', selector, '--ignore-not-found', '--wait=true', '--timeout=45s')
            remaining = kube('get', 'job,pod,configmap', '-l', selector, '-o', 'name')
            result['resources_clean'] = not remaining.strip()
            if remaining.strip():
                raise RuntimeError('standby resources still exist')
        finally:
            evidence.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', nargs=3, default=['gpu-01', 'gpu-02', 'gpu-03'])
    parser.add_argument('--ssh-target')
    parser.add_argument('--ssh-socket')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--warm-cuda', action='store_true')
    parser.add_argument('--colocated-spare', action='store_true')
    parser.add_argument('--cpu-only-node', action='append', default=[])
    args = parser.parse_args()
    if len(set(args.nodes)) != 3 or bool(args.ssh_socket) != bool(args.ssh_target):
        parser.error('use three different nodes and supply both SSH options or neither')
    cpu_only = set(args.cpu_only_node)
    if not cpu_only < set(args.nodes):
        parser.error('CPU-only nodes must be in the pool, with at least one node requesting GPUs')
    if args.colocated_spare and (cpu_only or args.nodes != ['gpu-01', 'gpu-02', 'gpu-03']):
        parser.error('--colocated-spare uses the standard Lab mapping without CPU-only runners')
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    if args.evidence.exists():
        parser.error('evidence path already exists')

    def kube(*words, payload=None):
        command = ['kubectl', '-n', 'lab', *words]
        if args.ssh_target:
            command = ['ssh', '-o', 'BatchMode=yes', '-S', args.ssh_socket, args.ssh_target, shlex.join(command)]
        return subprocess.run(command, input=payload, capture_output=True, text=True,
                              check=True, timeout=150).stdout

    pool_run(kube, args.nodes, args.evidence, args.warm_cuda, cpu_only, colocated_spare=args.colocated_spare)
    print(args.evidence)


if __name__ == '__main__':
    main()
