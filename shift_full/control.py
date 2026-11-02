"""Bounded experiment controller; saves raw evidence and never kills by pattern."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess as sp
import tarfile
import time
from dataclasses import dataclass
from shift_full.event_reader import merge, remote_code

CONTAINER_ROOT = '/workspace/SHIFT'
HOST_ROOT = '/home/campus/zhangweihao/SHIFT'


@dataclass(frozen=True)
class Endpoint:
    physical_node: str
    ip: str
    container: str
    socket: str
    runtime: str = 'greyhound'
    namespace: str = ''
    sudo: str = 'none'
    host_root: str = HOST_ROOT


def endpoints_from_env(env=None):
    env = os.environ if env is None else env
    return {
        'node0': Endpoint('node0', env.get('SHIFT_NODE0_IP', '10.10.4.2'),
                          env.get('SHIFT_NODE0_CONTAINER', 'flash1022-run'), 'ssh-h2.sock',
                          host_root=env.get('SHIFT_NODE0_HOST_ROOT', HOST_ROOT)),
        'node1': Endpoint('node1', env.get('SHIFT_NODE1_IP', '10.10.4.1'),
                          env.get('SHIFT_NODE1_CONTAINER', 'shift-rdma-node1'), 'ssh-h1.sock',
                          sudo='noninteractive', host_root=env.get('SHIFT_NODE1_HOST_ROOT', HOST_ROOT)),
        'node3': Endpoint('node3', env.get('SHIFT_NODE3_IP', '10.10.4.3'),
                          env.get('SHIFT_NODE3_CONTAINER', 'shift-rdma-node3'), 'ssh-h3.sock',
                          runtime='nerdctl', namespace=env.get('SHIFT_NODE3_NAMESPACE', 'shift'),
                          sudo='password', host_root=env.get('SHIFT_NODE3_HOST_ROOT', HOST_ROOT)),
    }


def route_endpoint(rid, generation, endpoints, takeover_node3=False):
    """Resolve a logical replica/generation to its physical execution endpoint."""
    if (isinstance(rid, bool) or rid not in (0, 1) or isinstance(generation, bool)
            or not isinstance(generation, int) or generation < 0):
        raise ValueError('rid must be 0 or 1 and generation must be nonnegative')
    name = 'node0' if rid == 0 else 'node1'
    if rid == 1 and takeover_node3 and generation >= 1:
        name = 'node3'
    return endpoints[name]


def runtime_command(endpoint, command):
    """Build a container-runtime command without putting credentials in argv."""
    if endpoint.runtime == 'nerdctl':
        return ['sudo', '-S', '-p', '', '/usr/local/bin/nerdctl',
                '--namespace', endpoint.namespace, *command]
    prefix = ['sudo', '-n', 'env'] if endpoint.sudo == 'noninteractive' else ['env']
    return prefix + ['DOCKER_HOST=unix:///run/greyhound-docker.sock',
                     '/opt/greyhound-docker/docker', *command]


def server_password(path):
    """Read the lab password in memory for sudo -S; callers must never log it."""
    text = Path(path).read_text(encoding='utf-8')
    match = re.search(r'密码[：: ]+(\S+)', text)
    if not match:
        raise RuntimeError('server.md does not contain the expected password field')
    return match.group(1)


# Retain the old tuple surface for benchmark/archive helpers that import NODES.
_default_endpoints = endpoints_from_env()
NODES = [( _default_endpoints[f'node{i}'].ip, _default_endpoints[f'node{i}'].container,
           _default_endpoints[f'node{i}'].socket,
           _default_endpoints[f'node{i}'].sudo == 'noninteractive') for i in (0, 1)]

class Session:
    def __init__(self, name, config, seconds, takeover_node3=False, endpoints=None):
        self.root = Path(__file__).resolve().parents[1]
        self.name, self.config, self.seconds = name, config, seconds
        self.endpoints = endpoints or endpoints_from_env()
        self.takeover_node3 = takeover_node3
        self.out = self.root / 'results' / name
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        self.log = (self.out / 'controller.jsonl').open('w', buffering=1)
        self.gen = [0, 0]
        self.deadline = time.monotonic() + seconds
        self.launched = False
        self.started_endpoints = {}
        self._endpoint_generations = {}
        self._read_offsets = {}
        self._read_cache = {}
        self._logical_cache = [{}, {}]
        self._sudo_password = None

    def endpoint_for(self, rid, generation=None):
        generation = getattr(self, 'gen', [0, 0])[rid] if generation is None else generation
        return route_endpoint(rid, generation, getattr(self, 'endpoints', _default_endpoints),
                              getattr(self, 'takeover_node3', False))

    def coerce_endpoint(self, endpoint_or_rid):
        if isinstance(endpoint_or_rid, Endpoint):
            return endpoint_or_rid
        if isinstance(endpoint_or_rid, int) and not isinstance(endpoint_or_rid, bool):
            return getattr(self, 'endpoints', _default_endpoints)[f'node{endpoint_or_rid}']
        raise TypeError('endpoint must be an Endpoint or logical replica id')

    def event(self, kind, **fields):
        row = dict(kind=kind, time=time.time())
        row.update(fields)
        self.log.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'pids'}), flush=True)

    def ssh(self, endpoint, cmd, timeout=45, sudo_input=False):
        endpoint = self.coerce_endpoint(endpoint)
        args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
                '-S', str(self.root / 'tmp' / endpoint.socket),
                'test@' + endpoint.ip, shlex.join(cmd)]
        stdin = None
        if sudo_input:
            if self._sudo_password is None:
                self._sudo_password = server_password(self.root / 'server.md')
            stdin = self._sudo_password + '\n'
        result = sp.run(args, input=stdin, text=True, stdout=sp.PIPE,
                        stderr=sp.PIPE, timeout=timeout)
        if result.returncode:
            raise sp.CalledProcessError(result.returncode, args,
                                        output=result.stdout, stderr=result.stderr)
        return result.stdout

    def ensure_connection(self, endpoint, force=False):
        endpoint = self.coerce_endpoint(endpoint)
        socket_path = str(self.root / 'tmp' / endpoint.socket)
        target = 'test@' + endpoint.ip
        if force:
            sp.run(['ssh', '-S', socket_path, '-O', 'exit', target],
                   stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=5)
        check = sp.run(['ssh', '-S', socket_path, '-O', 'check', target],
                       stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        if check.returncode:
            (self.root/'tmp').mkdir(exist_ok=True)
            sp.run(['/usr/bin/expect', str(self.root / 'scripts/remote-auth.exp'),
                    'ssh', '-M', '-S', socket_path, '-o', 'ControlPersist=yes',
                    '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
                    '-o', 'ConnectTimeout=8', '-fnNT', target], check=True,
                   timeout=30, stdout=sp.DEVNULL)

    def docker(self, endpoint, cmd, timeout=45):
        endpoint = self.coerce_endpoint(endpoint)
        return self.ssh(endpoint, runtime_command(endpoint, cmd), timeout,
                        sudo_input=endpoint.sudo == 'password')

    def execute_endpoint(self, endpoint, cmd, timeout=45):
        endpoint = self.coerce_endpoint(endpoint)
        return self.docker(endpoint, ['exec', endpoint.container, *cmd], timeout)

    def execute(self, rid, cmd, timeout=45):
        if not hasattr(self, 'gen') and isinstance(rid, int):
            return self.execute_endpoint(self.coerce_endpoint(rid), cmd, timeout)
        return self.execute_endpoint(self.endpoint_for(rid), cmd, timeout)

    def mark_endpoint(self, endpoint, name, text=''):
        endpoint = self.coerce_endpoint(endpoint)
        path = f'{CONTAINER_ROOT}/results/{self.name}/{name}'
        code = (f'from pathlib import Path; p=Path({path!r}); '
                f't=p.with_name(p.name+".tmp"); t.write_text({text!r}); t.replace(p)')
        self.execute_endpoint(endpoint, ['python', '-c', code])

    def mark(self, rid, name, text=''):
        if not hasattr(self, 'gen') and isinstance(rid, int):
            return self.mark_endpoint(self.coerce_endpoint(rid), name, text)
        self.mark_endpoint(self.endpoint_for(rid), name, text)

    def read_endpoint(self, endpoint, generation, rid=None):
        offsets = self._read_offsets.setdefault(endpoint.physical_node, {})
        code = remote_code(f'{CONTAINER_ROOT}/results/{self.name}', generation, offsets)
        try:
            raw = self.execute_endpoint(endpoint, ['python', '-c', code])
        except (sp.CalledProcessError, sp.TimeoutExpired) as error:
            if isinstance(error, sp.CalledProcessError) and error.returncode != 255:
                raise
            self.event('ssh_read_reconnect', physical_node=endpoint.physical_node,
                       rid=rid, generation=generation, error=type(error).__name__)
            self.ensure_connection(endpoint, force=True)
            raw = self.execute_endpoint(endpoint, ['python', '-c', code])
        packet = json.loads(raw)
        self._read_offsets[endpoint.physical_node] = packet['offsets']
        merged = merge(self._read_cache.get(endpoint.physical_node, {}), packet)
        self._read_cache[endpoint.physical_node] = merged
        return packet

    def read(self, rid):
        generation = self.gen[rid]
        endpoint = self.endpoint_for(rid, generation)
        physical = self.read_endpoint(endpoint, generation, rid)
        logical = merge(self._logical_cache[rid], physical)
        self._logical_cache[rid] = logical
        return logical

    def rows(self, data, n, r, kind):
        return [e for e in data[n].get(f'events_g{self.gen[n]}_r{r}.jsonl', [])
                if e['kind'] == kind]

    def await_(self, label, predicate, timeout=120):
        phase_deadline = min(self.deadline, time.monotonic() + timeout)
        while time.monotonic() < phase_deadline:
            data = [self.read(0), self.read(1)]
            for n in range(2):
                bad = [e for e in data[n].get('supervisor.jsonl', [])
                       if e['kind'] in ('unexpected_exit', 'supervisor_error')]
                if bad:
                    raise RuntimeError(f'{self.endpoint_for(n).physical_node}: {bad[-1]}')
            if predicate(data):
                self.event(label, generations=list(self.gen),
                           physical_nodes=[self.endpoint_for(n).physical_node for n in range(2)])
                return data
            time.sleep(.5)
        raise TimeoutError(label)

    def start_supervisor(self, rid, endpoint, generation):
        config_path = f'{CONTAINER_ROOT}/configs/{self.name}.json'
        config_code = (f'import pathlib; p=pathlib.Path({config_path!r}); '
                       f'p.parent.mkdir(exist_ok=True); p.write_text({json.dumps(self.config)!r})')
        self.execute_endpoint(endpoint, ['python', '-c', config_code])
        cmd = ['python', '-m', 'shift_full.supervisor', '--session', self.name,
               '--replica', str(rid), '--physical-node', endpoint.physical_node,
               '--initial-generation', str(generation), '--host-ip', endpoint.ip,
               '--root', CONTAINER_ROOT, '--seconds', str(self.seconds),
               '--config', config_path]
        self.docker(endpoint, ['exec', '-d', '-w', CONTAINER_ROOT,
                               '-e', f'PYTHONPATH={CONTAINER_ROOT}',
                               '-e', f'SHIFT_PHYSICAL_NODE={endpoint.physical_node}',
                               endpoint.container, *cmd])
        self.launched = True
        self.started_endpoints.setdefault(endpoint.physical_node, endpoint)
        self._endpoint_generations.setdefault(endpoint.physical_node, set()).add(generation)
        self.event('endpoint_launched', physical_node=endpoint.physical_node,
                   physical_ip=endpoint.ip, rid=rid, generation=generation,
                   container=endpoint.container, runtime=endpoint.runtime,
                   namespace=endpoint.namespace or None)

    def physical_endpoints(self):
        """Return every endpoint launched during this session, including retired ones."""
        return list(self.started_endpoints.values())

    def request_stop_all(self, endpoints, stopped, boundary):
        for endpoint in endpoints:
            try:
                self.read_endpoint(endpoint, max(self._endpoint_generations[endpoint.physical_node]))
                physical = self._read_cache[endpoint.physical_node]
                already_clean = any(e.get('kind') == 'cleanup'
                                    for e in physical.get('supervisor.jsonl', []))
                self.mark_endpoint(endpoint, 'STOP', '' if stopped or already_clean else str(boundary))
            except Exception as error:
                self.event('stop_error', physical_node=endpoint.physical_node, error=repr(error))

    def start(self):
        # Reopen idle multiplexed sessions before any experiment mutation.
        for rid in range(2):
            self.ensure_connection(self.endpoint_for(rid, 0))
        for rid in range(2):
            self.start_supervisor(rid, self.endpoint_for(rid, 0), 0)
        self.event('launched', physical_endpoints=[e.physical_node for e in self.started_endpoints.values()])
        data = self.await_('all_ready', lambda d: all(
            self.rows(d, n, r, 'ready') for n in range(2) for r in range(8)))
        if self.config.get('audit_initial'):
            for r in range(8):
                assert self.rows(data, 0, r, 'ready')[-1]['hashes'] == self.rows(data, 1, r, 'ready')[-1]['hashes']
        for rid in range(2):
            endpoint = self.endpoint_for(rid)
            self.mark_endpoint(endpoint, 'START')
            self.event('start_released', physical_node=endpoint.physical_node,
                       rid=rid, generation=self.gen[rid])

    def replace_rid1_with_node3(self, index, policy, previous_generation):
        old = self.endpoint_for(1, previous_generation)
        generation = previous_generation + 1
        new = self.endpoint_for(1, generation)
        if old.physical_node == new.physical_node:
            raise RuntimeError('node3 takeover route did not change the physical endpoint')
        self.mark_endpoint(old, 'STOP')
        self.await_(f'old_endpoint_cleanup_{index}', lambda d: any(
            row['kind'] == 'cleanup' and row.get('physical_node') == old.physical_node
            for row in d[1].get('supervisor.jsonl', [])))
        self.gen[1] = generation
        self.ensure_connection(new)
        self.event('restart_requested', index=index, target=1, rid=1,
                   generation=generation, previous_generation=previous_generation,
                   previous_physical_node=old.physical_node,
                   physical_node=new.physical_node, physical_ip=new.ip,
                   restart_policy=policy, endpoint_takeover=True)
        self.start_supervisor(1, new, generation)
        data = self.await_(f'replacement_ready_{index}', lambda d: all(
            self.rows(d, 1, r, 'ready') for r in range(8)))
        self.mark_endpoint(new, 'START')
        self.event('start_released', physical_node=new.physical_node,
                   rid=1, generation=generation)
        return data
    def fault(self,index,kind='replica'):
        target=1-index%2; healthy=1-target; generation=self.gen[target]
        endpoint = self.endpoint_for(target, generation)
        self.await_(f'prefault_{index}',lambda d:all(any(e['participants']==2 for e in self.rows(d,n,r,'commit')) for n in range(2) for r in range(8)))
        marker=f'WORKER_KILL_g{generation}' if kind=='worker' else f'KILL_g{generation}'
        self.mark_endpoint(endpoint,marker,str(index%8) if kind=='worker' else '')
        data=self.await_(f'fault_{index}_ack',lambda d:any(e['kind']=='injected' and e['generation']==generation for e in d[target].get('supervisor.jsonl',[])))
        fault=next(e for e in data[target]['supervisor.jsonl'] if e['kind']=='injected' and e['generation']==generation)
        self.event('injected',index=index,target=target,rid=target,generation=generation,
                   physical_node=endpoint.physical_node,physical_ip=endpoint.ip,
                   time=fault['time'],fault_kind=kind,pids=fault.get('pids',[]))
        policy=self.config.get('restart_policy','after_solo')
        if policy not in ('immediate','after_solo'):raise ValueError('invalid restart_policy')
        if policy=='after_solo':
            self.await_(f'fault_{index}_survivor_progress',lambda d:all(len([e for e in self.rows(d,healthy,r,'commit') if e['participants']==1 and e['time']>fault['time']])>=2 for r in range(8)))
        if self.takeover_node3 and target == 1 and generation == 0:
            self.replace_rid1_with_node3(index, policy, generation)
        else:
            self.gen[target] += 1
            restarted = self.endpoint_for(target)
            self._endpoint_generations.setdefault(restarted.physical_node, set()).add(self.gen[target])
            self.mark_endpoint(restarted, f'RESTART_g{self.gen[target]}')
            self.event('restart_requested', index=index, target=target, rid=target,
                       generation=self.gen[target], physical_node=restarted.physical_node,
                       physical_ip=restarted.ip, restart_policy=policy)
        data=self.await_(f'fault_{index}_restored',lambda d:all(any(e['hashes_match'] and e['step']>0 and e['optimizer_tensors']>0 for e in self.rows(d,target,r,'restored')) for r in range(8)))
        restored={r:self.rows(data,target,r,'restored')[-1]['step'] for r in range(8)}
        assert len(set(restored.values()))==1
        endpoint = self.endpoint_for(target)
        self.event('restore_verified',index=index,target=target,rid=target,
                   physical_node=endpoint.physical_node,generation=self.gen[target],step=restored[0],
                   verification='full_hash' if '--audit-state' in self.config.get('train_args',[]) else 'metadata_only')
        self.await_(f'fault_{index}_rejoined',lambda d:all(len([e for e in self.rows(d,n,r,'commit') if e['participants']==2 and e['step']>restored[r]])>=3 for n in range(2) for r in range(8)))
    def checkpoint_restart(self):
        self.await_('checkpoint_available', lambda d: all(self.rows(d,n,r,'checkpoint_loaded') for n in range(2) for r in range(8)))
        data=[self.read(0),self.read(1)]
        checkpoint_step=min(self.rows(data,n,r,'checkpoint_loaded')[-1]['step'] for n in range(2) for r in range(8))
        self.await_('past_checkpoint',lambda d:all(any(e['step']>=checkpoint_step+2 for e in self.rows(d,n,r,'commit')) for n in range(2) for r in range(8)))
        for n in range(2): self.mark(n,'EXPECT_PEER_FAILURE_g0')
        for n in range(2): self.mark(n,'KILL_g0')
        self.await_('both_jobs_exited',lambda d:all(any(e['kind'] in ('injected','peer_fault_exit') for e in d[n].get('supervisor.jsonl',[])) for n in range(2)))
        self.event('checkpoint_restart_requested',checkpoint_step=checkpoint_step)
        for n in range(2):
            self.gen[n]=1
            endpoint=self.endpoint_for(n)
            self._endpoint_generations.setdefault(endpoint.physical_node,set()).add(1)
            self.mark_endpoint(endpoint,'RESTART_g1')
        self.await_('all_checkpoint_resumed',lambda d:all(self.rows(d,n,r,'checkpoint_resumed') for n in range(2) for r in range(8)))
        self.await_('checkpoint_three_commits',lambda d:all(len(self.rows(d,n,r,'commit'))>=3 for n in range(2) for r in range(8)))
    def finish(self):
        if not self.launched:
            self.log.close()
            return
        endpoints = self.physical_endpoints()
        for endpoint in endpoints:
            try:
                self.ensure_connection(endpoint)
            except Exception as error:
                self.event('connection_error', physical_node=endpoint.physical_node,
                           error=repr(error))
        try:
            data = [self.read(0), self.read(1)]
            stopped = all(self.rows(data, n, r, 'stopped') for n in range(2) for r in range(8))
            boundary = max([e['step'] for n in range(2) for r in range(8)
                            for e in self.rows(data, n, r, 'commit')] or [0]) + 3
        except Exception:
            stopped, boundary = True, 0
        self.request_stop_all(endpoints, stopped, boundary)
        stop_deadline = time.monotonic() + 60
        while time.monotonic() < stop_deadline:
            try:
                done = True
                for endpoint in endpoints:
                    gen = max(self._endpoint_generations[endpoint.physical_node])
                    self.read_endpoint(endpoint, gen)
                    physical = self._read_cache[endpoint.physical_node]
                    done &= any(e.get('kind') == 'cleanup' for e in physical.get('supervisor.jsonl', []))
                if done:
                    break
            except Exception:
                pass
            time.sleep(.5)

        manifest = {'endpoints': []}
        for endpoint in endpoints:
            generations = sorted(self._endpoint_generations[endpoint.physical_node])
            manifest['endpoints'].append({
                'physical_node': endpoint.physical_node, 'ip': endpoint.ip,
                'logical_rids': sorted({rid for rid in range(2)
                                        if any(self.endpoint_for(rid, g).physical_node == endpoint.physical_node
                                               for g in range(max(generations) + 1))}),
                'generations': generations, 'container': endpoint.container,
                'runtime': endpoint.runtime, 'namespace': endpoint.namespace or None,
                'host_root': endpoint.host_root,
            })
        (self.out / 'physical_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        collected = []
        for endpoint in endpoints:
            try:
                archive = f'{CONTAINER_ROOT}/{self.name}-logs.tgz'
                self.execute_endpoint(endpoint, ['tar', '-czf', archive, '--exclude=*.pt',
                                                 '-C', f'{CONTAINER_ROOT}/results/{self.name}', '.'])
                self.ssh(endpoint, ['mkdir', '-p', endpoint.host_root + '/results'])
                remote = f'{endpoint.host_root}/results/{self.name}-{endpoint.physical_node}.tgz'
                self.docker(endpoint, ['cp', endpoint.container + ':' + archive, remote])
                local = self.out / f'{endpoint.physical_node}.tgz'
                sp.run(['scp', '-q', '-o', f'ControlPath={self.root / "tmp" / endpoint.socket}',
                        f'test@{endpoint.ip}:{remote}', str(local)], check=True, timeout=90)
                physical_dir = self.out / f'physical_{endpoint.physical_node}'
                physical_dir.mkdir()
                with tarfile.open(local) as tar:
                    tar.extractall(physical_dir)
                collected.append(endpoint)
            except Exception as error:
                self.event('collection_error', physical_node=endpoint.physical_node,
                           error=repr(error))
        self.build_logical_views(collected)
        self.event('collected', physical_nodes=[endpoint.physical_node for endpoint in collected])
        self.log.close()

    def build_logical_views(self, endpoints):
        for rid in range(2):
            logical = self.out / f'node{rid}'
            sources = [self.out / f'physical_{endpoint.physical_node}' for endpoint in endpoints
                       if any(self.endpoint_for(rid, g).physical_node == endpoint.physical_node
                              for g in range(max(self._endpoint_generations[endpoint.physical_node]) + 1))]
            if not sources:
                continue
            if rid == 0:
                shutil.copytree(sources[0], logical, dirs_exist_ok=True)
                continue
            logical.mkdir(exist_ok=True)
            for source in sources:
                for path in sorted(source.rglob('*')):
                    relative = path.relative_to(source)
                    target = logical / relative
                    if path.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif path.name.endswith('.jsonl') and target.exists():
                        with target.open('ab') as out_stream, path.open('rb') as in_stream:
                            out_stream.write(in_stream.read())
                    elif not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--config',required=True)
    p.add_argument('--seconds',type=int,default=900);p.add_argument('--faults',type=int,default=0)
    p.add_argument('--interval',type=float,default=60);p.add_argument('--duration',type=float,default=0)
    p.add_argument('--worker-first',action='store_true');p.add_argument('--checkpoint-restart',action='store_true')
    p.add_argument('--takeover-node3',action='store_true',help='replace logical rid1 with node3 from generation 1')
    a=p.parse_args()
    if not all(c.isalnum() or c in '_-' for c in a.session):p.error('bad session name')
    if a.takeover_node3 and a.faults < 1:p.error('--takeover-node3 requires at least one fault')
    if a.takeover_node3 and a.checkpoint_restart:p.error('--takeover-node3 cannot be combined with --checkpoint-restart')
    s=Session(a.session,json.loads(Path(a.config).read_text()),a.seconds,
              takeover_node3=a.takeover_node3)
    try:
        s.start()
        actual_faults=0
        if a.checkpoint_restart:
            s.checkpoint_restart()
        elif a.faults:
            warmup_commits=s.config.get('warmup_joint_commits',3)
            if not isinstance(warmup_commits,int) or isinstance(warmup_commits,bool) or warmup_commits<1:
                raise ValueError('warmup_joint_commits must be a positive integer')
            s.await_('warmup_three_joint',lambda d:all(any(
                e['participants']==2 and e.get('step',0)>=warmup_commits
                for e in s.rows(d,n,r,'commit')) for n in range(2) for r in range(8)))
            began=time.monotonic();last_fault_seconds=0;s.event('fault_window_start')
            for i in range(a.faults):
                due=began+i*a.interval
                if a.duration and (time.monotonic() >= began+a.duration or began+a.duration-time.monotonic() < max(30,1.25*last_fault_seconds)):
                    s.event('injection_schedule_closed',requested_faults=a.faults,actual_faults=actual_faults,reason='remaining_window_budget')
                    break
                while time.monotonic()<due:time.sleep(min(1,due-time.monotonic()))
                fault_started=time.monotonic()
                s.fault(i,'worker' if a.worker_first and i==0 else 'replica')
                last_fault_seconds=time.monotonic()-fault_started
                actual_faults+=1
            end=began+a.duration
            while time.monotonic()<end:time.sleep(min(1,end-time.monotonic()))
        else:
            s.await_('all_stopped',lambda d:all(s.rows(d,n,r,'stopped') for n in range(2) for r in range(8)),timeout=a.seconds)
        s.event('complete',faults=actual_faults,requested_faults=a.faults)
    except BaseException as e:
        s.event('failed',error=repr(e));raise
    finally:s.finish()
if __name__=='__main__':main()
