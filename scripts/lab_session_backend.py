"""Run the session controller against session-owned, bounded Lab runner Pods."""

import json
import re
import subprocess
import time
from types import SimpleNamespace

from shift_full.spare_selection import SpareReport


POD_NODES = {'node0': 'gpu-01', 'node1': 'gpu-02', 'node3': 'gpu-03'}
CHECK_POD = (
    'import os,sys; '
    'assert os.environ.get("SHIFT_POD_UID") == sys.argv[1], "runner Pod changed"; '
    'os.execvp(sys.argv[2], sys.argv[2:])'
)


class LabSessionBackend:
    def __init__(self, experiment, nodes, root, session, *, node0='gpu-01', node1='gpu-02', colocated_spare=False):
        if not re.fullmatch(r'shift-[a-z0-9][a-z0-9-]{0,55}', experiment):
            raise ValueError('expected a scoped SHIFT experiment label')
        self.root = root
        self.session = session
        self.pods = {}
        self.units = {}
        self.endpoints = {}
        self._uuids = {}
        self.released_at = {}
        mapping = dict(POD_NODES, node0=node0, node1=node1)
        self.physical_nodes = {node: mapping[node] for node in nodes}
        by_name = {physical: node for node, physical in self.physical_nodes.items()}
        if len(by_name) != len(nodes) and not colocated_spare:
            raise ValueError('each runner alias needs a different physical node')
        jobs = json.loads(self.kube('get', 'job', '-l', f'experiment={experiment}', '-o', 'json'))['items']
        job_uids = {job['metadata']['uid'] for job in jobs
                    if type(job['spec'].get('activeDeadlineSeconds')) is int
                    and 0 < job['spec']['activeDeadlineSeconds'] <= 3600}
        if len(job_uids) != len(nodes):
            raise RuntimeError('expected bounded session-owned runner Jobs')
        pods = json.loads(self.kube('get', 'pod', '-l', f'experiment={experiment}', '-o', 'json'))['items']
        for pod in pods:
            meta, spec, status = (pod[key] for key in ('metadata', 'spec', 'status'))
            alias = meta.get('labels', {}).get('shift-runner')
            if alias is None and not colocated_spare:
                alias = by_name.get(spec.get('nodeName'))
            if (alias not in self.physical_nodes or self.physical_nodes[alias] != spec.get('nodeName')
                    or alias in self.pods or status.get('phase') != 'Running' or meta.get('deletionTimestamp')):
                raise RuntimeError('expected one running runner per configured node')
            if not spec.get('hostNetwork') or spec.get('hostPID') or spec.get('shareProcessNamespace'):
                raise RuntimeError('runner must use host networking and isolated PIDs')
            if not any(owner.get('kind') == 'Job' and owner.get('controller') and owner.get('uid') in job_uids
                       for owner in meta.get('ownerReferences', [])):
                raise RuntimeError('runner must be owned by a bounded Job')
            self.pods[alias] = (meta['name'], meta['uid'])
            self.endpoints[alias] = SimpleNamespace(ip=status['hostIP'])
        if set(self.pods) != set(nodes):
            raise RuntimeError('incomplete Lab runner set')

    @staticmethod
    def kube(*words, timeout=45):
        return subprocess.run(['kubectl', '-n', 'lab', *words], check=True, capture_output=True,
                              text=True, timeout=timeout).stdout

    def exec(self, node, *command, timeout=45):
        name, uid = self.pods[node]
        return self.kube('exec', name, '-c', 'agent', '--', 'python3', '-c', CHECK_POD, uid, *command,
                         timeout=timeout)

    def python(self, node, code):
        return self.exec(node, 'python3', '-c', code)

    def verify_sources(self, node, manifest):
        paths = {str(self.root / path): digest for path, digest in manifest.items()}
        paths['/agent/spare_agent.py'] = manifest['shift_full/spare_agent.py']
        mismatches = json.loads(self.python(node,
            'import hashlib,json; from pathlib import Path; files='+repr(paths)+'; '
            'print(json.dumps([path for path, sha in files.items() if not Path(path).is_file() '
            'or hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha]))'))
        if mismatches:
            names = ', '.join(self.root.joinpath(path).name for path in mismatches)
            raise RuntimeError(f'{node} staged source mismatch: {names}')

    def inventory(self, node):
        now = time.monotonic()
        row = json.loads(self.exec(node, 'python3', '/agent/spare_agent.py', '--read', timeout=4))
        if (row['node'] != self.physical_nodes[node] or row['instance'] != self.pods[node][1]
                or not 0 <= row['age_seconds'] <= 15):
            raise RuntimeError('Lab agent report has stale or inconsistent identity')
        devices = tuple(device['uuid'] for device in row['devices'])
        previous = self._uuids.setdefault(node, devices)
        if devices != previous or tuple(d['index'] for d in row['devices']) != tuple(range(len(devices))):
            raise RuntimeError('runner GPU allocation changed')
        observed_at = now - row['age_seconds']
        ready = row['ready'] and observed_at > self.released_at.get(node, float('-inf'))
        return SpareReport(node, row['instance'], observed_at, frozenset({row['profile']}),
                           tuple((d['uuid'], d['free_mib']) for d in row['devices']), ready)

    def candidate_inventory(self, node):
        """Exclude a runner whose snapshot cannot be verified during selection."""
        try:
            return self.inventory(node)
        except (subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError):
            return SpareReport(node, self.pods[node][1], time.monotonic(), frozenset(), (), False)

    def indices(self, node):
        return range(len(self._uuids[node]))

    def device_ids(self, node, indices):
        return tuple(self._uuids[node][index] for index in indices)

    def actor(self, node, unit, operation, *payload):
        directory = self.root / 'runtime' / f'lab-{self.session}' / unit
        return self.exec(node, 'python3', str(self.root / 'scripts' / 'lab_actor.py'), operation,
                         str(directory), *payload)

    def launch(self, node, unit, command, env, log, devices):
        uuids = self.device_ids(node, devices) if devices else ()
        env = dict(env, CUDA_VISIBLE_DEVICES=','.join(uid if uid.startswith('GPU-') else f'GPU-{uid}' for uid in uuids))
        self.actor(node, unit, 'start', json.dumps(dict(command=command, env=env, log=str(log))))
        self.units[unit] = node

    def status(self, node, unit):
        return json.loads(self.actor(node, unit, 'status'))

    def stop(self, node, unit):
        self.actor(node, unit, 'stop')

    def resources(self):
        result = {node: dict(scope='lab_pod', pod_uid=uid, extra_gpu=[])
                  for node, (_, uid) in self.pods.items()}
        for unit, node in self.units.items():
            result[node]['extra_gpu'].extend(self.status(node, unit)['remaining'])
        return result
