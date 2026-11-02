"""Three physical nodes, two active replicas, seeded fail-stop rotation."""
import argparse
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess as sp
import tarfile
import time

from .control import Session, CONTAINER_ROOT
from .event_reader import merge, remote_code
from .resource_probe import snapshot


class Rotation(Session):
    def __init__(self, name, config, seconds, seed, backup_mode='cold'):
        config = dict(config, external_lighthouse=True)
        super().__init__(name, config, seconds)
        self.rng = random.Random(seed)
        placement = self.rng.sample(list(self.endpoints), 2)
        self.bindings = {(rid, 0): self.endpoints[node] for rid, node in enumerate(placement)}
        self.config['master_addr'] = self.endpoint_for(0).ip
        (self.out/'config.json').write_text(json.dumps(self.config,indent=2)+'\n')
        self.seed = seed
        self.control_endpoint = self.endpoints['node0']
        self.control_started = False
        self.fault_results = []
        self.backup_mode = backup_mode

    def endpoint_for(self, rid, generation=None):
        return self.bindings[rid, self.gen[rid] if generation is None else generation]

    def subdir(self, rid, generation=None):
        return f'rid{rid}_g{self.gen[rid] if generation is None else generation}'

    def directory(self, rid, generation=None):
        return f'{CONTAINER_ROOT}/results/{self.name}/{self.subdir(rid, generation)}'

    def mark(self, rid, name, text=''):
        self.write_marker(self.endpoint_for(rid), self.directory(rid), name, text)

    def write_marker(self, endpoint, directory, name, text=''):
        code = (f'from pathlib import Path; p=Path({directory!r})/{name!r}; '
                f't=p.with_name(p.name+".tmp"); t.write_text({text!r}); t.replace(p)')
        self.execute_endpoint(endpoint, ['python', '-c', code])

    def read_binding(self, rid, generation):
        endpoint = self.endpoint_for(rid, generation)
        key = (rid, generation)
        cmd=['python','-c',remote_code(self.directory(rid,generation),generation,self._read_offsets.get(key,{}))]
        try:
            raw=self.execute_endpoint(endpoint,cmd)
        except (sp.CalledProcessError,sp.TimeoutExpired) as error:
            if isinstance(error,sp.CalledProcessError) and error.returncode!=255:
                raise
            self.event('ssh_read_reconnect',rid=rid,generation=generation,physical_node=endpoint.physical_node)
            self.ensure_connection(endpoint,force=True)
            raw=self.execute_endpoint(endpoint,cmd)
        packet=json.loads(raw)
        self._read_offsets[key] = packet['offsets']
        self._read_cache[key] = merge(self._read_cache.get(key, {}), packet)
        return self._read_cache[key]

    def read(self, rid):
        return self.read_binding(rid, self.gen[rid])

    def write_config(self, endpoint, config=None):
        path = f'{CONTAINER_ROOT}/configs/{self.name}.json'
        code = f'from pathlib import Path; Path({path!r}).write_text({json.dumps(self.config if config is None else config)!r})'
        self.execute_endpoint(endpoint, ['python', '-c', code])
        return path

    def start_supervisor(self, rid, endpoint, generation, standby=None):
        config = dict(self.config)
        if standby:
            config['train_args'] = [*config['train_args'], '--standby', standby]
        path = self.write_config(endpoint, config)
        cmd = ['python', '-m', 'shift_full.supervisor', '--session', self.name,
               '--replica', str(rid), '--physical-node', endpoint.physical_node,
               '--initial-generation', str(generation), '--result-subdir', self.subdir(rid, generation),
               '--host-ip', endpoint.ip, '--root', CONTAINER_ROOT, '--seconds', str(self.seconds),
               '--config', path]
        self.docker(endpoint, ['exec', '-d', '-w', CONTAINER_ROOT, '-e', f'PYTHONPATH={CONTAINER_ROOT}',
                              '-e', f'SHIFT_PHYSICAL_NODE={endpoint.physical_node}', endpoint.container, *cmd])
        self.launched = True
        self.started_endpoints[endpoint.physical_node] = endpoint
        self._endpoint_generations.setdefault(endpoint.physical_node, set()).add(generation)
        self.event('endpoint_launched', rid=rid, generation=generation,
                   physical_node=endpoint.physical_node, directory=self.directory(rid, generation))

    def start(self):
        for endpoint in self.endpoints.values():
            self.ensure_connection(endpoint)
        endpoint = self.control_endpoint
        path = self.write_config(endpoint)
        directory = f'{CONTAINER_ROOT}/results/{self.name}/control'
        self.docker(endpoint, ['exec', '-d', '-w', CONTAINER_ROOT, '-e', f'PYTHONPATH={CONTAINER_ROOT}',
                              endpoint.container, 'python', '-m', 'shift_full.control_service',
                              '--config', path, '--directory', directory, '--root', CONTAINER_ROOT,
                              '--seconds', str(self.seconds)])
        self.control_started = True
        time.sleep(1)
        self.control_owner = json.loads(self.execute_endpoint(endpoint, ['cat', directory+'/owner.json']))
        self.event('placement', seed=self.seed, active={str(r): self.endpoint_for(r).physical_node for r in range(2)},
                   control_physical_node=endpoint.physical_node, control_owner=self.control_owner)
        super().start()

    def mark_endpoint(self, endpoint, name, text=''):
        # Session.start uses mark_endpoint for START; resolve the unique active binding.
        matches = [rid for rid in range(2) if self.endpoint_for(rid) == endpoint]
        if len(matches) != 1:
            raise RuntimeError('ambiguous active endpoint')
        self.mark(matches[0], name, text)

    def common_commits(self, data, after=0, before_time=float('inf')):
        steps = None
        by_role = {}
        for rid in range(2):
            for rank in range(8):
                rows = {e['step']: e for e in self.rows(data, rid, rank, 'commit')
                        if e['participants'] == 2 and e.get('contributed') and e['step'] > after
                        and e['time'] <= before_time}
                by_role[rid, rank] = rows
                steps = set(rows) if steps is None else steps & rows.keys()
        return sorted(steps or []), by_role

    def fault(self, index, kind='replica'):
        target = self.rng.randrange(2)
        healthy = 1-target
        generation = self.gen[target]
        old = self.endpoint_for(target)
        candidates = [e for e in self.endpoints.values() if e not in [self.endpoint_for(r) for r in range(2)]]
        if len(candidates) != 1:
            raise RuntimeError('expected exactly one idle compatible endpoint')
        new = candidates[0]
        idle_resources=snapshot(self,new)
        prepare_started = time.time()
        standby_rows = []
        if self.backup_mode != 'cold':
            self.bindings[target,generation+1] = new
            self.start_supervisor(target,new,generation+1,self.backup_mode)
            until = min(self.deadline,time.monotonic()+120)
            while time.monotonic()<until:
                standby_data = self.read_binding(target,generation+1)
                standby_rows = [e for rank in range(8)
                                for e in standby_data.get(f'events_g{generation+1}_r{rank}.jsonl',[])
                                if e['kind']=='standby_ready']
                if len(standby_rows)==8:
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('standby preparation')
            self.event('standby_prepared',index=index,rid=target,generation=generation+1,
                       physical_node=new.physical_node,mode=self.backup_mode,
                       preparation_seconds=time.time()-prepare_started,resources=standby_rows)
        prepare_seconds = time.time()-prepare_started
        prepared_resources=snapshot(self,new)
        before = self.await_(f'prefault_{index}', lambda d: bool(self.common_commits(d)[0]))
        committed = self.common_commits(before)[0][-1]
        healthy_pids = [self.rows(before, healthy, r, 'ready')[-1]['pid'] for r in range(8)]
        self.event('candidate_selected', index=index, target=target, candidates=[e.physical_node for e in candidates],
                   previous_physical_node=old.physical_node, physical_node=new.physical_node,
                   previous_generation=generation, generation=generation+1, committed_step=committed)
        self.mark(target, f'KILL_g{generation}')
        data = self.await_(f'fault_{index}_ack', lambda d: any(e['kind']=='injected' for e in d[target].get('supervisor.jsonl', [])))
        injected = next(e for e in data[target]['supervisor.jsonl'] if e['kind']=='injected')
        fault_at = injected['time']
        at_fault = self.common_commits(data, before_time=fault_at)[0]
        committed = max([committed, *at_fault])
        self.event('injected', index=index, target=target, generation=generation,
                   physical_node=old.physical_node, time=fault_at, pids=injected['pids'])
        self.mark(target, 'STOP')
        data = self.await_(f'old_cleanup_{index}', lambda d: any(e['kind']=='cleanup' for e in d[target].get('supervisor.jsonl', [])))
        cleanup = next(e for e in data[target]['supervisor.jsonl'] if e['kind']=='cleanup')
        if cleanup.get('own_children'):
            raise RuntimeError(f'unreaped children: {cleanup["own_children"]}')
        self.gen[target] += 1
        self.bindings[target, self.gen[target]] = new
        request_at = time.time()
        self.event('restart_requested', index=index, target=target, generation=self.gen[target],
                   previous_physical_node=old.physical_node, physical_node=new.physical_node)
        if self.backup_mode=='cold':
            self.start_supervisor(target, new, self.gen[target])
        else:
            self.mark(target, 'ACTIVATE')
        self.await_(f'replacement_ready_{index}', lambda d: all(self.rows(d,target,r,'ready') for r in range(8)))
        ready_data = [self.read(r) for r in range(2)]
        for rank in range(8):
            new_role=self.rows(ready_data,target,rank,'ready')[-1].get('role_manifest')
            healthy_role=self.rows(ready_data,healthy,rank,'ready')[-1].get('role_manifest')
            if new_role != healthy_role:
                raise RuntimeError(f'incompatible standby role {rank}')
        self.mark(target, 'START')
        data = self.await_(f'restored_{index}', lambda d: all(self.rows(d,target,r,'restored') for r in range(8)))
        restores = [self.rows(data,target,r,'restored')[-1] for r in range(8)]
        loaded = [self.rows(data,target,r,'state_loaded')[-1] for r in range(8)]
        recovery_requests = [self.rows(data,target,r,'recovery_request')[-1] for r in range(8)]
        audit = '--audit-state' in self.config['train_args']
        if not all(e['hashes_match'] and e['optimizer_tensors']>0 and e['step']>=committed for e in restores):
            raise RuntimeError('state restore audit or RPO failed')
        restored_steps = {e['step'] for e in restores}
        if len(restored_steps) != 1:
            raise RuntimeError('inconsistent restored rank progress')
        restored_step = restored_steps.pop()
        data = self.await_(f'rejoined_{index}', lambda d: bool(self.common_commits(d, restored_step)[0]))
        steps, roles = self.common_commits(data, restored_step)
        first_step = steps[0]
        if audit:
            for rank in range(8):
                a, b = roles[0,rank][first_step]['hashes'], roles[1,rank][first_step]['hashes']
                if len(a) != 7 or a != b:
                    raise RuntimeError(f'first joint state mismatch rank={rank} step={first_step}')
        healthy_commits = [e for e in self.rows(data,healthy,0,'commit') if e['time']>fault_at]
        if not healthy_commits or any(e['pid'] != healthy_pids[r] for r in range(8)
                                      for e in self.rows(data,healthy,r,'commit')):
            raise RuntimeError('healthy replica PID changed or did not progress')
        control_alive = self.execute_endpoint(self.control_endpoint, ['python', '-c',
            f'import os; os.kill({self.control_owner["lighthouse_pid"]},0); print("alive")']).strip()=='alive'
        result = dict(index=index, target=target, generation=self.gen[target], old_node=old.physical_node,
                      new_node=new.physical_node, committed_before_fault=committed, restored_step=restored_step,
                      first_joint_step=first_step, full_state_audit=audit, first_joint_hashes_match=True if audit else None,
                      healthy_pids_unchanged=True, healthy_progress=True, lighthouse_alive=control_alive,
                      retired_children=[], fault_time=fault_at, request_time=request_at,
                      backup_mode=self.backup_mode,preparation_seconds=prepare_seconds,
                      standby_resources=standby_rows,loaded_time=max(e['time'] for e in restores),
                      idle_resources=idle_resources,prepared_resources=prepared_resources,
                      recovered_resources=snapshot(self,new),
                      ready_time=max(self.rows(ready_data,target,r,'ready')[-1]['time'] for r in range(8)),
                      recovery_request_time=min(e['time'] for e in recovery_requests),
                      state_loaded_time=max(e['time'] for e in loaded),
                      joint_time=max(roles[r,k][first_step]['time'] for r in range(2) for k in range(8)))
        result['fault_to_joint_seconds'] = result['joint_time']-fault_at
        result['request_to_loaded_seconds'] = result['loaded_time']-request_at
        result['recovery_request_to_loaded_seconds'] = result['state_loaded_time']-result['recovery_request_time']
        self.fault_results.append(result)
        (self.out/'fault_results.json').write_text(json.dumps(self.fault_results, indent=2)+'\n')
        self.event('rotation_verified', **result)

    def finish(self):
        # Stop current generations at the same committed boundary, then collect every binding.
        try:
            data = [self.read(r) for r in range(2)] if self.launched else [{},{}]
            boundary = max([e['step'] for r in range(2) for k in range(8) for e in self.rows(data,r,k,'commit')] or [0])+3
            for r in range(2):
                if (r,self.gen[r]) in self.bindings and self.endpoint_for(r).physical_node in self.started_endpoints:
                    self.mark(r, 'STOP', str(boundary))
            for (rid,generation),endpoint in self.bindings.items():
                if generation != self.gen[rid] and endpoint.physical_node in self.started_endpoints:
                    self.write_marker(endpoint,self.directory(rid,generation),'STOP')
            until = time.monotonic()+60
            while self.launched and time.monotonic()<until:
                if all(any(e['kind']=='cleanup' for e in self.read_binding(r,g).get('supervisor.jsonl', []))
                       for r,g in self.bindings):
                    break
                time.sleep(.5)
            else:
                if self.launched:
                    self.event('training_cleanup_timeout')
        finally:
            if self.control_started:
                self.write_marker(self.control_endpoint, f'{CONTAINER_ROOT}/results/{self.name}/control', 'STOP')
                self.started_endpoints[self.control_endpoint.physical_node] = self.control_endpoint
                control_dir = f'{CONTAINER_ROOT}/results/{self.name}/control'
                until = time.monotonic()+10
                while time.monotonic()<until:
                    done = self.execute_endpoint(self.control_endpoint, ['python','-c',
                        f'from pathlib import Path; print(Path({control_dir!r},"cleanup.json").exists())'])
                    if done.strip()=='True':
                        break
                    time.sleep(.1)
                else:
                    self.event('control_cleanup_failed')
            manifest = [dict(rid=r,generation=g,physical_node=e.physical_node,directory=self.directory(r,g))
                        for (r,g),e in sorted(self.bindings.items())]
            (self.out/'physical_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            for endpoint in self.started_endpoints.values():
                remote = f'{endpoint.host_root}/results/{self.name}-{endpoint.physical_node}.tgz'
                archive = f'{CONTAINER_ROOT}/{self.name}-logs.tgz'
                self.execute_endpoint(endpoint, ['tar','-czf',archive,'--exclude=*.pt','-C',f'{CONTAINER_ROOT}/results/{self.name}','.'])
                self.docker(endpoint,['cp',endpoint.container+':'+archive,remote])
                local = self.out/(endpoint.physical_node+'.tgz')
                sp.run(['scp','-q','-o',f'ControlPath={self.root/"tmp"/endpoint.socket}',f'test@{endpoint.ip}:{remote}',str(local)],check=True,timeout=90)
                dest = self.out/('physical_'+endpoint.physical_node)
                dest.mkdir(exist_ok=True)
                with tarfile.open(local) as tar:
                    for member in tar.getmembers():
                        target=(dest/member.name).resolve()
                        if not str(target).startswith(str(dest.resolve())+'/') and target!=dest.resolve():
                            raise ValueError('unsafe evidence archive path')
                        if member.issym() or member.islnk():
                            raise ValueError('unexpected link in evidence archive')
                    tar.extractall(dest)
            for (rid,generation),endpoint in self.bindings.items():
                source = self.out/('physical_'+endpoint.physical_node)/self.subdir(rid,generation)
                if source.exists():
                    shutil.copytree(source,self.out/f'node{rid}',dirs_exist_ok=True,
                                    ignore=lambda d,names: [n for n in names if not (n.startswith('events_') or n.startswith('topology_'))])
            self.event('collected', physical_nodes=list(self.started_endpoints))
            self.log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--faults',type=int,default=10)
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('--duration',type=float,default=0)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--backup-mode',choices=('cold','cpu','gpu'),default='cold')
    a = p.parse_args()
    if not a.session or not all(c.isalnum() or c in '_-' for c in a.session):
        p.error('invalid session')
    if a.faults<0 or a.interval<0 or a.duration<0:
        p.error('invalid schedule')
    s = Rotation(a.session,json.loads(Path(a.config).read_text()),a.seconds,a.seed,a.backup_mode)
    try:
        s.start()
        if a.faults==0:
            s.await_('all_stopped',lambda d: all(s.rows(d,r,k,'stopped') for r in range(2) for k in range(8)),timeout=a.seconds)
            s.event('complete',faults=0)
            return
        warmup=s.config.get('warmup_joint_commits',3)
        s.await_('warmup',lambda d: bool(s.common_commits(d,warmup)[0]))
        began = time.monotonic()
        s.event('fault_window_start',duration=a.duration,requested_faults=a.faults)
        for i in range(a.faults):
            due = began+i*a.interval
            while time.monotonic()<due:
                time.sleep(min(1,due-time.monotonic()))
            late = time.monotonic()-due
            if late>2:
                s.event('schedule_late',index=i,seconds=late)
            if a.duration and time.monotonic()>=began+a.duration:
                raise RuntimeError('fault window expired before all injections')
            s.fault(i)
        while time.monotonic()<began+a.duration:
            time.sleep(min(1,began+a.duration-time.monotonic()))
        s.event('complete',faults=len(s.fault_results),fault_window_seconds=time.monotonic()-began)
    except BaseException as error:
        s.event('failed',error=repr(error))
        raise
    finally:
        s.finish()


if __name__ == '__main__':
    main()
