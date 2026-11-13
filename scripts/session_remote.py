"""One bounded persistent session controller; accepts live strategy commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env,server_password
from shift_full.session_control import Command
from shift_full.session_layout import SessionLayout, partial_roles

ROOT=Path('/workspace/SHIFT')


def main():
    global ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--audit',action='store_true')
    p.add_argument('--initial-steps',type=int,default=0,help='bounded reference run; zero accepts live commands')
    p.add_argument('--replica-world-size',type=int,choices=(4,8),default=8)
    p.add_argument('--seq-len',type=int,default=512)
    p.add_argument('--low-memory',action='store_true')
    p.add_argument('--nodes',nargs='+',choices=('node0','node1','node3'),default=['node0','node1','node3'])
    p.add_argument('--lab-experiment', help='experiment label of the session-owned Lab runner Jobs')
    p.add_argument('--lab-node0', choices=('gpu-01', 'gpu-03'), default='gpu-01')
    p.add_argument('--lab-node1', choices=('gpu-01', 'gpu-02', 'gpu-03'), default='gpu-02')
    p.add_argument('--lab-colocated-spare', action='store_true', help='allow three runners on two physical nodes')
    p.add_argument('--lab-keep-dp-spread', action='store_true',
                   help='keep whole replicas on different hosts when the spare shares a host')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--flash-root', type=Path, default=Path('/workspace/FLASH1022'))
    p.add_argument('--spare-min-mib', type=int, default=10000)
    p.add_argument('--normal-v1-exit', action='store_true', help='allow the Lab test source to exit at its committed boundary')
    p.add_argument('--prepare-v1', action='store_true', help='prepare the candidate before a controlled Lab source exit')
    p.add_argument('--lab-fixed-reference', type=Path, help='replay exact target GPU UUIDs from an accepted auto run')
    p.add_argument('--flash-redis-socket', help='private FLASH Redis socket on the Lab controller host')
    p.add_argument('--flash-slow-policy', choices=('v2', 'v4'))
    p.add_argument('--flash-max-age-steps', type=int, default=32)
    flash_mode = p.add_mutually_exclusive_group()
    flash_mode.add_argument('--flash-calibrate', action='store_true')
    flash_mode.add_argument('--flash-live-reference', type=Path)
    p.add_argument('--flash-inject-epochs', type=int, default=1)
    p.add_argument('--flash-inject-step', type=int, default=36)
    p.add_argument('--flash-inject-rank', type=int, help='Lab-only bounded cuBLAS delay on this logical rank')
    a=p.parse_args()
    layout=SessionLayout(a.replica_world_size)
    if a.seq_len<2:p.error("seq-len must be at least two")
    if not {'node0','node1'}<=set(a.nodes):p.error('node0 and node1 are required')
    if not a.session or not all(c.isalnum() or c in '_-' for c in a.session):p.error('invalid session')
    ROOT = a.root.resolve() if a.lab_experiment else a.root
    flash_root = a.flash_root.resolve() if a.lab_experiment else a.flash_root
    lab = None
    if a.lab_experiment:
        for path in (ROOT, flash_root):
            if not path.is_relative_to('/data/lab') or path == Path('/data/lab'):
                p.error('Lab code and model roots must be inside a session-owned /data/lab directory')
        from scripts.lab_session_backend import LabSessionBackend
        if a.lab_colocated_spare and (set(a.nodes) != {'node0', 'node1', 'node3'}
                                     or (a.lab_node0, a.lab_node1) != ('gpu-03', 'gpu-01')):
            p.error('colocated spare requires node0 and node3 on gpu-03 and node1 on gpu-01')
        lab = LabSessionBackend(a.lab_experiment, a.nodes, ROOT, a.session, node0=a.lab_node0, node1=a.lab_node1,
                                colocated_spare=a.lab_colocated_spare)
    elif a.lab_colocated_spare:
        p.error('--lab-colocated-spare requires a Lab experiment')
    if a.lab_keep_dp_spread and not a.lab_colocated_spare:
        p.error('--lab-keep-dp-spread requires a colocated spare')
    if a.spare_min_mib < 1:p.error('spare memory threshold must be positive')
    if a.normal_v1_exit and not lab:p.error('--normal-v1-exit requires a Lab experiment')
    if a.prepare_v1 and not a.normal_v1_exit:p.error('--prepare-v1 requires --normal-v1-exit')
    if bool(a.flash_redis_socket) != bool(a.flash_slow_policy) or (a.flash_redis_socket and not lab):
        p.error('FLASH integration requires a Lab experiment, Redis socket and slow policy')
    if a.flash_max_age_steps < 0:p.error('FLASH alert age must be nonnegative')
    live_profile = None
    live = a.flash_calibrate or a.flash_live_reference is not None
    if live and not a.flash_redis_socket:p.error('live PROFILE requires the FLASH bridge')
    if not 1 <= a.flash_inject_epochs <= 3 or (a.flash_inject_epochs > 1 and a.flash_inject_rank is None):
        p.error('invalid FLASH injection epoch count')
    if not 36 <= a.flash_inject_step <= 96 or (a.flash_inject_step != 36 and a.flash_inject_rank is None):
        p.error('invalid FLASH injection start step')
    if a.flash_inject_rank is not None and (not a.flash_live_reference or
                                          a.flash_inject_rank not in range(2 * layout.world)):
        p.error('compute delay requires a live reference and a valid logical rank')
    bridge = None
    if a.flash_redis_socket:
        from redis.exceptions import RedisError
        from shift_full.flash_bridge import Bridge, command as flash_command, connect, membership
        from shift_full.flash_monitor import EpochSamples
        bridge = Bridge(connect(a.flash_redis_socket), a.session)
    fixed_placements = {}
    if a.lab_fixed_reference:
        if not lab:p.error('--lab-fixed-reference requires a Lab experiment')
        reference = json.loads(a.lab_fixed_reference.read_text())
        fixed_placements = {case['command']['id']: case for case in reference['cases']}
    import torch.distributed as dist
    name='session_'+a.session;out=ROOT/'results'/name;out.mkdir(exist_ok=False)
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if lab:
        eps = lab.endpoints
    else:
        remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
        s=remote.ServerRotation.__new__(remote.ServerRotation);s.root=ROOT;s._sudo_password=remote.PASSWORD
        eps={n:e for n,e in endpoints_from_env().items() if n in a.nodes}
    units=[];actors={};offsets={};rows={};expected_stops=set();store=None;reserved_lights={};started_lights=set()
    report=dict(name=name,controller_pid=os.getpid(),session=a.session,epoch=0,command_count=0,commands=[],done=False,error=None,audit=a.audit,resources_clean=False,nodes=list(eps))
    if lab:report.update(backend='lab_pod', physical_nodes=lab.physical_nodes)
    requests=ROOT/'runtime/v3-reproduce'/name/'requests';requests.mkdir(parents=True,exist_ok=True)
    flash_snapshot = None
    flash_monitor = None
    flash_monitors = []
    rootcfg=ROOT/'runtime/v3-reproduce'/(name+'.json')
    def ssh(node,code):
        return lab.python(node, code) if lab else s.ssh(eps[node],['sudo','-S','-p','','python3','-c',code],sudo_input=True)
    def save():
        temp=out/'summary.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'summary.json')
    def resources():
        if lab:return lab.resources()
        result={}
        for node,e in eps.items():
            entries=[x.split(',',1) for x in s.ssh(e,['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).splitlines() if x.strip()]
            result[node]=dict(company={pid.strip():proc.strip() for pid,proc in entries if 'VLLM::' in proc},extra_gpu=[pid.strip() for pid,proc in entries if 'VLLM::' not in proc])
        return result
    def launch(node,label,command,devices=''):
        unit='shift-session-'+a.session+'-'+label;units.append((node,unit))
        env=dict(PYTHONPATH=f'{ROOT}:{flash_root}',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MALLOC_ARENA_MAX='2',TOKIO_WORKER_THREADS='2',
            CPATH=str(ROOT/'runtime/python-headers'),CUDA_VISIBLE_DEVICES=devices,
            CUDA_DEVICE_MAX_CONNECTIONS='1',NCCL_MAX_CTAS='4',NCCL_BUFFSIZE='1048576',NCCL_CUMEM_HOST_ENABLE='0',NCCL_CUMEM_ENABLE='0',
            NCCL_NVLS_ENABLE='0',NCCL_DEBUG='WARN',NCCL_NET='IB',NCCL_IB_DISABLE='0',NCCL_SOCKET_IFNAME='bond0',
            GLOO_SOCKET_IFNAME='bond0',NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1',NCCL_IB_GID_INDEX='3',
            NCCL_LAUNCH_ORDER_IMPLICIT='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='0',TORCH_NCCL_USE_COMM_NONBLOCKING='0',
            TORCHFT_USE_OTEL='false',SHIFT_DIGEST_ENGINE='staged',
            TMPDIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),XDG_CACHE_HOME=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            TRITON_CACHE_DIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache/triton'),NO_PROXY='*',no_proxy='*')
        if a.low_memory:
            env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
            env['SHIFT_CPU_ADAM']='1'
        if devices and live:
            libraries = ROOT / 'runtime/flash-libs'
            site = ROOT / 'runtime/torch214/lib/python3.12/site-packages/nvidia'
            cuda, nccl = site/'cu13/lib', site/'nccl/lib'
            preload = [str(libraries/'libobserver.so')]
            if a.flash_inject_rank is not None:
                preload.append(str(libraries/'libgemm-delay.so'))
            env.update(LD_PRELOAD=':'.join(preload), LD_LIBRARY_PATH=f'{cuda}:{nccl}',
                       FLASH_V4_SAMPLE_STRIDE='64', FLASH_V4_PROFILE_REARM='1',
                       FLASH_V4_CUBLAS_LIBRARY=str(cuda/'libcublas.so.13'),
                       FLASH_V4_CUBLASLT_LIBRARY=str(cuda/'libcublasLt.so.13'),
                       FLASH_V4_NCCL_LIBRARY=str(nccl/'libnccl.so.2'))
        if lab:
            indices = tuple(map(int, devices.split(','))) if devices else ()
            lab.launch(node, unit, command, env, out/(label+'.host.log'), indices)
            return
        args=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+str(a.seconds),
            '--property=KillMode=control-group','--property=TimeoutStopSec=10','--working-directory='+str(ROOT),
            '--property=StandardOutput=append:'+str(out/(label+'.host.log')),'--property=StandardError=inherit',
            *['--setenv='+k+'='+v for k,v in env.items()],*command]
        s.ssh(eps[node],['sudo','-S','-p','',*args],sudo_input=True)
    def lighthouse(service,*,planned_round=None):
        port=base['lighthouse_port']+service if planned_round is None else 31410+20*planned_round
        if port in started_lights:return
        if planned_round is None and service in reserved_lights:reserved_lights.pop(service).close()
        light_node='node0' if planned_round is None else state['replica_nodes']['0']
        if lab:
            ssh(light_node, 'import socket; probe=socket.socket(socket.AF_INET6); '
                f'probe.bind(("::", {port})); probe.close()')
        launch(light_node,f'light-{port}',[str(ROOT/'runtime/torch214/bin/python'),str(ROOT/'vendor/bin/torchft_lighthouse'),
            '--bind',f'[::]:{port}','--min_replicas','1' if planned_round is None else '2',
            '--heartbeat_timeout_ms','500','--join_timeout_ms','200','--quorum_tick_ms','50'])
        until=time.monotonic()+30
        while time.monotonic()<until:
            with socket.socket() as probe:
                if lab:probe.settimeout(.5)
                if probe.connect_ex((eps[light_node].ip,port))==0:started_lights.add(port);return
            time.sleep(.1)
        raise TimeoutError('session Lighthouse did not become reachable')
    def actor(label,node,role,replica,devices,*,generation=0,command=None):
        launch_started = time.monotonic()
        if lab:
            inventory = lab.inventory(node)
            available = dict(inventory.free_gpu_mib)
            chosen = lab.device_ids(node, devices)
            if not inventory.ready or any(available.get(device, 0) < a.spare_min_mib for device in chosen):
                raise RuntimeError('reserved Lab devices no longer pass the launch preflight')
            preflight_finished = time.monotonic()
        cfg=dict(base,actor_generation=generation,session_replica=replica,actor_devices=len(devices),
                 session_actor_host=eps[node].ip,
                 session_candidate=command is not None,candidate_full_dp=len(devices)==layout.world,
                 initial_round=report['epoch'],session_epoch=report['epoch'],manager_service=state['service'],manager_store_services=state['stores'],
                 channel_epochs=state['channels'],replica_hosts={k:eps[v].ip for k,v in state['replica_nodes'].items()},
                 partial_owner=state['partial_owner'],role=role)
        if command:cfg.update(session_command=command,command_sequence=command['sequence'])
        path=ROOT/'runtime/v3-reproduce'/(name+'_'+label+'.json')
        actors[label]=dict(node=node,role=role,replica=replica,devices=devices,generation=generation,unit='shift-session-'+a.session+'-'+label)
        if lab:
            cfg['session_gpu_uuids'] = list(chosen)
            actors[label].update(pod_uid=inventory.instance, gpu_uuids=list(chosen),
                                 physical_node=lab.physical_nodes[node])
        if command and command['command']['policy'] == 'v4':
            cfg['session_actor_ranks'] = command['command']['ranks']
            actors[label]['ranks'] = cfg['session_actor_ranks']
        ssh(node,'from pathlib import Path;Path('+repr(str(path))+').write_text('+repr(json.dumps(cfg))+')')
        rows[label]=[]
        launch(node,label,[str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.v4_train','--config',str(path),'--role','host','--host-role',role],','.join(map(str,devices)))
        if lab and command:
            command['target_launch'] = dict(
                request_to_preflight_start_seconds=launch_started-command['requested_monotonic'],
                preflight_seconds=preflight_finished-launch_started,
                config_and_launch_seconds=time.monotonic()-preflight_finished)
    def collect():
        for node in eps:
            selected={label:info for label,info in actors.items() if info['node']==node}
            if not selected:continue
            code='''from pathlib import Path
import json
out={}
for label,info in ACTORS.items():
 prefix='g'+str(info['generation'])+'_'+info['role']
 for p in Path(WORKERS).glob(prefix+'*.jsonl'):
  key=p.name;offset=OFFSETS.get(key,0)
  with p.open('rb') as f:f.seek(offset);data=f.read()
  end=data.rfind(b'\\n')+1
  if end:out[key]=dict(offset=offset+end,text=data[:end].decode())
print(json.dumps(out))
'''.replace('ACTORS',repr(selected)).replace('WORKERS',repr(str(out/'workers'))).replace('OFFSETS',repr(offsets.get(node,{})))
            data=json.loads(ssh(node,code));offsets.setdefault(node,{})
            dest=out/'raw'/node;dest.mkdir(parents=True,exist_ok=True)
            for filename,item in data.items():
                offsets[node][filename]=item['offset']
                with (dest/filename).open('a') as f:f.write(item['text'])
                for line in item['text'].splitlines():
                    e=json.loads(line)
                    label='g'+str(e['actor_round'])+'_'+e['role']
                    rows[label].append(e)
    def keys(kind,sequence):return [f'session/{kind}/{sequence}/{r}/{i}' for r in range(2) for i in range(layout.world)]
    def occupied(excluding=None):
        used={node:set() for node in eps}
        for logical,(label,device) in active.items():
            if excluding is not None and logical[0]==excluding:continue
            info=actors[label];used[info['node']].add(info['devices'][device])
        return used
    def device_indices(node):
        return lab.indices(node) if lab else range(8)

    def select_target(cmd):
        from dataclasses import replace
        from shift_full.spare_pool import SparePool
        from shift_full.spare_selection import Selection
        pool = SparePool(eps)
        with ThreadPoolExecutor(max_workers=len(eps)) as executor:
            for row in executor.map(lab.candidate_inventory, eps):
                pool.record(row)
        active_labels = {label for label, _ in active.values()}
        for label in active_labels:
            actor = actors[label];node = actor['node']
            pool.track_running(label, Selection(node, lab.pods[node][1], lab.device_ids(node, actor['devices'])))
        sources = {actors[active[cmd.source_replica, rank][0]]['node'] for rank in cmd.ranks}
        primary = actors[active[cmd.source_replica, cmd.ranks[0]][0]]['node']
        fault = {'v1':'process_exit', 'v2':'replica_slow', 'v4':'rank_slow'}[cmd.policy]
        plan = pool.claim_replacement(cmd.id, fault=fault, source_replica=cmd.source_replica, source_node=primary,
                                      world=layout.world, profile='shift-session', min_gpu_mib=a.spare_min_mib,
                                      now=time.monotonic(), max_age=15, excluded_nodes=sources,
                                      affected_rank=cmd.ranks[0] if cmd.policy == 'v4' else None)
        choice = plan.decision.target
        if choice is None:raise ValueError(f'no qualified Lab target: {plan.decision.rejected}')
        slots = lab.device_ids(choice.node, device_indices(choice.node))
        return replace(cmd, target=choice.node), [slots.index(uid) for uid in choice.slots], asdict(plan.decision)
    def terminal(label):
        return {e['device'] for e in rows[label] if e['kind'] in ('session_done','session_retired','retired','standby_done')}==set(range(len(actors[label]['devices'])))
    def check():
        allowed_exits = set()
        if lab and current and current['command']['policy'] == 'v1' and not current.get('killed'):
            source = current['command']['source_replica']
            allowed_exits = {active[source, rank][0] for rank in range(layout.world)}
        for label,events in rows.items():
            failures=[e for e in events if e['kind']=='session_error']
            if failures and not lab:raise RuntimeError('session worker failed: '+label+' '+failures[-1]['error_type'])
        for label,info in actors.items():
            if info['unit'] in expected_stops or terminal(label):continue
            if lab:
                status = lab.status(info['node'], info['unit'])
                if not status['live'] and label not in allowed_exits:
                    exit_info = status['exit']
                    if not status['remaining'] and exit_info and exit_info['pid']:
                        ranks = sorted(rank for (replica, rank), (actor_label, _) in active.items()
                                       if replica == info['replica'] and actor_label == label)
                        report.setdefault('fail_stop_events', []).append(dict(fault_type='process_exit',
                            node=info['node'], pod_uid=info['pod_uid'], actor=label,
                            replica=info['replica'], ranks=ranks, returncode=exit_info['returncode'],
                            pid=exit_info['pid'], observed_at=time.monotonic()))
                        raise RuntimeError('confirmed Lab actor exit outside a recovery boundary: '+label)
                    raise RuntimeError('Lab actor supervisor lost without complete exit evidence: '+label)
                continue
            values=dict(x.split('=',1) for x in s.ssh(eps[info['node']],['systemctl','show',info['unit'],'--property=ActiveState','--property=ExecMainStatus']).splitlines() if '=' in x)
            if values.get('ActiveState') not in ('active','activating','deactivating') or values.get('ExecMainStatus') not in (None,'0'):
                raise RuntimeError('session actor exited without proof: '+label)
    dp0_node, dp1_node = ('node1', 'node0') if a.lab_keep_dp_spread else ('node0', 'node1')
    before=None;active={};current=None;seen={};state=dict(service=0,stores={'0':0,'1':0},channels=[0]*layout.world,replica_nodes={'0':dp0_node,'1':dp1_node},partial_owner='B')
    # The legacy A endpoint names the session Store and Lighthouse host.
    base=dict(directory=str(out/'workers'),flash=str(flash_root),hosts={r:eps[n].ip for r,n in dict(A='node0',B=dp0_node,C='node3' if 'node3' in eps else 'node0',D=dp1_node,E=dp1_node).items()},
        node_hosts={n:e.ip for n,e in eps.items()},port=28000,timeout=120,communication_timeout=2,seconds=a.seconds,
        plan=dict(cut=2,post_steps=5,seq_len=a.seq_len,microbatches=2),seed=2026,unified=True,data='random',
        unified_session=True,session=a.session,replica_world_size=layout.world,session_services=16,outer_manager=True,transactional_switch=True,
        retain_outer_managers=True,overlap_peer=True,oracle=False,reference_only=False,
        performance_no_audit=not a.audit,boundary_validation='full' if a.audit else 'runtime',
        gpu_prepare=True,gpu_warmup=not a.low_memory,low_memory=a.low_memory,reuse_local_groups=True,preconnect_state=True,defer_source_local_close=True,
        slow_ms=0,cycles=1,between_steps=5,restore_deadline=45 if a.low_memory else 10,source_deadline=2,affected_ranks=list(layout.affected))
    report['configuration']=dict(batch=8,seq_len=a.seq_len,tp=layout.tp,pp=2,ep=2,replica_world_size=layout.world,seed=2026,memory_fraction=.12,gpu_warmup=not a.low_memory,low_memory=a.low_memory,optimizer='cpu_adamw' if a.low_memory else 'cuda_adamw')
    if a.lab_keep_dp_spread:
        report['configuration']['initial_replica_runners'] = [dp0_node, dp1_node]
    if live:
        from scripts.flash_live_controller import LiveProfile
        base.update(flash_live=True, flash_inject_rank=a.flash_inject_rank,
                    flash_inject_epochs=a.flash_inject_epochs, flash_inject_step=a.flash_inject_step)
        live_profile = LiveProfile(out, flash_root, a.flash_live_reference, ssh, bridge)
        report['flash_live_profile'] = live_profile.state
        report['flash_live_profiles'] = live_profile.history
    from shift_full.v4_reference import training_sources
    report['source_sha256']=training_sources(ROOT)
    for file in ['shift_full/offloaded_adam.py','shift_full/session_layout.py','shift_full/session_engine.py','shift_full/session_outer.py','shift_full/session_control.py','shift_full/training_loop.py','scripts/session_remote.py','scripts/sessionctl.py']:
        report['source_sha256'][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    if lab:
        for file in ('scripts/lab_acceptance.py', 'scripts/export_v42_session.py',
                     'scripts/lab_actor.py', 'scripts/lab_session_backend.py', 'shift_full/spare_agent.py',
                     'shift_full/spare_pool.py', 'shift_full/spare_selection.py'):
            report['source_sha256'][file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    if bridge:
        for file in ('shift_full/flash_bridge.py', 'shift_full/flash_monitor.py'):
            report['source_sha256'][file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        report['flash_alerts'] = []
        report['flash_memberships'] = []
        report['flash_monitor'] = []
    if live:
        for file in ('scripts/flash_live_controller.py', 'shift_full/flash_live.py'):
            report['source_sha256'][file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        libraries = ['runtime/flash-libs/libobserver.so']
        if a.flash_inject_rank is not None:
            libraries.append('runtime/flash-libs/libgemm-delay.so')
        for file in libraries:
            report['source_sha256'][file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        localizer = flash_root/'deploy/lab/shift_gemm_profile.py'
        report['flash_live_profile']['localizer_sha256'] = hashlib.sha256(localizer.read_bytes()).hexdigest()
        if a.flash_live_reference:
            report['flash_live_profile']['reference_sha256'] = hashlib.sha256(
                a.flash_live_reference.read_bytes()).hexdigest()

    def monitor_flash(snapshot):
        nonlocal flash_monitor
        if flash_monitor:
            flash_monitor['stop'].touch()
        directory = out / f"flash_epoch_{snapshot['epoch']}"
        directory.mkdir()
        training_log = directory / 'train.jsonl'
        training_log.touch()
        stop = directory / 'monitor_stop'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=f'{flash_root}:{os.environ.get("PYTHONPATH", "")}')
        with (directory / 'worker.log').open('w') as log:
            worker = subprocess.Popen([sys.executable, '-m', 'flash_v4.bocd_monitor',
                '--training-log', str(training_log), '--request', str(directory/'profile_request.json'),
                '--audit', str(directory/'bocd_audit.jsonl'), '--stop', str(stop),
                '--max-seconds', str(a.seconds),
                '--degradation', '0.1'], env=env, stdout=log, stderr=subprocess.STDOUT)
        evidence = dict(epoch=snapshot['epoch'], first_step=snapshot['first_step'], samples=0)
        report['flash_monitor'].append(evidence)
        flash_monitor = dict(feed=EpochSamples(snapshot), log=training_log, stop=stop,
                             worker=worker, evidence=evidence)
        flash_monitors.append(flash_monitor)
        if live_profile:
            live_profile.begin_epoch(snapshot, store)

    def collect_flash():
        samples = list(flash_monitor['feed'].collect(rows))
        if samples:
            with flash_monitor['log'].open('a') as log:
                log.writelines(json.dumps(sample) + '\n' for sample in samples)
            flash_monitor['evidence']['samples'] += len(samples)
        flash_monitor['evidence']['worker_returncode'] = flash_monitor['worker'].poll()

    def poll_flash(progress):
        try:
            bridge.publish(flash_snapshot)
            item = bridge.read()
        except RedisError as error:
            report['flash_redis_error'] = type(error).__name__
            return
        report.pop('flash_redis_error', None)
        if item is None:
            return
        cursor, raw = item
        try:
            event = json.loads(raw)
            payload = flash_command(flash_snapshot, event, a.flash_slow_policy,
                                    min(progress.values()), a.flash_max_age_steps)
        except (KeyError, TypeError, ValueError) as error:
            report['flash_alerts'].append(dict(stream_id=cursor, error=str(error)))
        else:
            path = requests / (payload['id'] + '.json')
            if not path.exists():
                temporary = path.with_suffix('.flash-tmp')
                temporary.write_text(json.dumps(payload))
                temporary.replace(path)
            report['flash_alerts'].append(dict(stream_id=cursor, id=payload['id'], event=event,
                                              received_monotonic=time.monotonic()))
        bridge.cursor = cursor

    save()
    try:
        if lab:
            from shift_full.spare_pool import SparePool
            inventory = SparePool(eps)
            for node in eps:
                lab.verify_sources(node, report['source_sha256'])
                report_row = lab.inventory(node)
                if (not report_row.ready or 'shift-session' not in report_row.profiles
                        or len(report_row.free_gpu_mib) < layout.world
                        or min(mib for _, mib in report_row.free_gpu_mib) < a.spare_min_mib):
                    raise RuntimeError('runner failed session GPU preflight: '+node)
                inventory.record(report_row)
                ssh(node, 'from pathlib import Path; paths='+repr([
                    str(ROOT/'runtime'/f'lab-{a.session}'), str(ROOT/'runtime/v3-reproduce'),
                    str(ROOT/'runtime/v1-recheck-bootstrap/cache'), str(out)])+
                    '; [Path(p).mkdir(parents=True, exist_ok=True) for p in paths]')
            report['verified_runner_sources'] = list(eps)
            pool = inventory.decide(profile='shift-session', ranks=layout.world,
                                    min_gpu_mib=a.spare_min_mib, now=time.monotonic(), max_age=15)
            if {selection.node for selection in pool.ranked} != set(eps):
                raise RuntimeError('initial Lab pool does not have all its runner nodes ready')
            report['initial_spare_pool'] = asdict(pool)
            report['runner_gpu_uuids'] = {node: lab.device_ids(node, device_indices(node)) for node in eps}
            all_devices = [uid for devices in report['runner_gpu_uuids'].values() for uid in devices]
            if len(all_devices) != len(set(all_devices)):
                raise RuntimeError('Lab runners have overlapping physical GPU UUIDs')
            if a.lab_colocated_spare:
                base['lab_distinct_runner_devices'] = True
        else:
            for e in eps.values():s.ensure_connection(e)
        before=resources();(out/'resources_before.json').write_text(json.dumps(before))
        if any(x['extra_gpu'] or (not lab and len(x['company'])!=8) for x in before.values()):raise RuntimeError('session requires idle SHIFT resources')
        for node in eps:ssh(node,'from pathlib import Path;Path('+repr(str(out/'workers'))+').mkdir(parents=True,exist_ok=True)')
        # Preflight a whole native-store block outside ephemeral RPC ports.
        for first in range(28000,29000,200):
            probes=[]
            try:
                for port in [first,*range(first+100,first+100+2*base['session_services'])]:
                    probe=socket.socket(socket.AF_INET6,socket.SOCK_STREAM);probes.append(probe);probe.bind(('::',port))
            except OSError:
                for probe in probes:probe.close()
                continue
            for probe in probes:probe.close()
            base['port']=first;break
        else:raise RuntimeError('no unused session native-store block')
        # Keep future control listeners outside Linux's ephemeral port range.
        # Reserve every native-service slot before workers open RPC sockets.
        for first in range(30200,30500,32):
            proposed={}
            try:
                for service in range(base['session_services']):
                    listener=socket.socket(socket.AF_INET6,socket.SOCK_STREAM)
                    proposed[service]=listener;listener.bind(('::',first+service))
            except OSError:
                for listener in proposed.values():listener.close()
                continue
            reserved_lights=proposed;base['lighthouse_port']=first;break
        else:raise RuntimeError('no unused session Lighthouse port block')
        rootcfg.write_text(json.dumps(base));launch('node0','store',[str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.v4_train','--config',str(rootcfg),'--role','store'])
        until=time.monotonic()+30
        while not (out/'workers/store_ready').exists():
            if time.monotonic()>until:raise TimeoutError('session store startup')
            time.sleep(.1)
        store=dist.TCPStore(eps['node0'].ip,base['port'],None,False,timeout=__import__('datetime').timedelta(seconds=120),wait_for_workers=False)
        store.set('session/command_count','0')
        if live_profile:
            store.set('session/flash_window', '0 1 0')
        lighthouse(0)
        for role,node,replica,devices in [('A',dp0_node,0,list(range(layout.width))),('B',dp0_node,0,list(layout.affected)),('D',dp1_node,1,list(range(layout.width))),('E',dp1_node,1,list(layout.affected))]:
            label='g0_'+role;actor(label,node,role,replica,devices)
            for d in range(layout.width):active[replica,d+(layout.width if role in 'BE' else 0)]=(label,d)
        deadline=time.monotonic()+a.seconds-30
        ready=False;ref_stopped=False
        while time.monotonic()<deadline:
            collect();check()
            if not ready and store.check([f'session/ready/{r}/{i}' for r in range(2) for i in range(layout.world)]):
                if bridge:
                    flash_snapshot = membership(a.session, 0, 1, layout, active, actors)
                    report['flash_memberships'].append(flash_snapshot)
                    monitor_flash(flash_snapshot)
                store.set('session/start','1');ready=True;report['ready']=True;save()
            progress={r:int(store.get(f'session/progress/{r}')) if store.check([f'session/progress/{r}']) else 0 for r in (0,1)}
            report['progress']=progress
            if current:
                seq=current['sequence'];cmd=current['command']
                if cmd['policy']=='v1' and not current.get('killed') and store.check(keys('arrival',seq)):
                    source=cmd['source_replica']
                    targets={active[source,i][0] for i in range(layout.world)}
                    if lab:
                        current.setdefault('observed_exit_wait_started', time.monotonic())
                        exited = {label: lab.status(actors[label]['node'], actors[label]['unit']) for label in targets}
                        if any(x['live'] or x['remaining'] or not x['exit'] or not x['exit']['pid'] for x in exited.values()):
                            if time.monotonic() - current['observed_exit_wait_started'] >= 90:
                                raise TimeoutError('no confirmed Lab source exit at the armed boundary')
                            save();time.sleep(.5);continue
                        if current.get('normal_source_exit') and any(x['exit']['returncode'] != 0 for x in exited.values()):
                            raise RuntimeError('Lab source did not exit normally at the committed boundary')
                    current['fault_monotonic']=time.monotonic()
                    current['fault_proof']=[]
                    for label in targets:
                        info=actors[label]
                        pids=sorted({e['pid'] for e in rows[label]})
                        if len(pids)!=len(info['devices']):raise RuntimeError('fault requires exact worker PID coverage')
                        code='''from pathlib import Path
import json,subprocess,time
unit=UNIT
pids=PIDS
before=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','show',unit,'--property=ActiveState','--property=MainPID'],text=True).splitlines())
assert before['ActiveState']=='active' and int(before['MainPID'])>0
result=subprocess.run(['systemctl','kill','--kill-who=all','--signal=SIGKILL',unit],capture_output=True,text=True)
def alive(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
until=time.monotonic()+5
while any(alive(pid) for pid in pids+[int(before['MainPID'])]) and time.monotonic()<until:time.sleep(.02)
assert not any(alive(pid) for pid in pids+[int(before['MainPID'])]),'owned fault workers survived SIGKILL'
print(json.dumps(dict(unit=unit,signal='SIGKILL',worker_pids=pids,main_pid=int(before['MainPID']),all_workers_dead=True,systemctl_returncode=result.returncode)))
'''.replace('UNIT',repr(info['unit'])).replace('PIDS',repr(pids))
                        if lab:
                            observed = exited[label]
                            current['fault_proof'].append(dict(unit=info['unit'], worker_pids=pids,
                                main_pid=observed['exit']['pid'], all_workers_dead=True, returncode=observed['exit']['returncode'],
                                pod_uid=lab.pods[info['node']][1]))
                        else:
                            current['fault_proof'].append(json.loads(ssh(info['node'],code)))
                        expected_stops.add(info['unit'])
                    label=f'g{seq}_D'
                    if current.get('prepare_v1'):
                        prepared = [f'session/native_prepared/{seq}/{rank}' for rank in range(layout.world)]
                        current['candidate_prepared_before_exit'] = store.check(prepared)
                        if not current['candidate_prepared_before_exit']:
                            raise RuntimeError('source exited before the warm candidate was prepared')
                    store.set(f'session/fault_released/{seq}','1')
                    if not current.get('prepare_v1'):
                        state['stores'][str(source)]=seq
                        actor(label,cmd['target'],'D',source,current['devices'],generation=seq,command=current)
                        dist.PrefixStore(f'round/{report["epoch"]}/',store).set('prepare','1')
                    current.update(killed=True,target_actor=label,fault_kind='observed_exit' if lab else 'SIGKILL');save()
                if current.get('target_actor') and 'first_common_observation' not in current:
                    wanted=dict(active)
                    target_label=current['target_actor']
                    for device, rank in enumerate(cmd['ranks']):
                        wanted[cmd['source_replica'],rank] = (target_label,device)
                    by_step={}
                    for (replica,rank),(label,device) in wanted.items():
                        for e in rows[label]:
                            if (e['kind']=='session_commit' and e['device']==device and
                                e.get('command_id')==cmd['id'] and e.get('participants')==2 and e.get('contributed')):
                                by_step.setdefault(e['step'],set()).add((replica,rank))
                    common=sorted(step for step,workers in by_step.items() if len(workers)==2*layout.world)
                    if common:
                        observed=time.monotonic()
                        current['first_common_observation']=dict(step=common[0],observed_monotonic=observed,
                            request_seconds=observed-current['requested_monotonic'],
                            fault_seconds=observed-current['fault_monotonic'] if current.get('fault_monotonic') else None)
                        save()
                if store.check(keys('done',seq)):
                    if cmd['policy']=='v4':
                        decision=json.loads(dist.PrefixStore(f'round/{cmd["expected_epoch"]}/',store).get('decision'))
                        if decision.get('decision')!='replace' or decision.get('owner')!=actors[current['target_actor']]['role']:
                            raise RuntimeError('v4 retained the source; migration was not accepted')
                    if 'first_common_observation' not in current:
                        current.setdefault('completion_observation_deadline',time.monotonic()+20)
                        if time.monotonic()>current['completion_observation_deadline']:raise RuntimeError('strategy completed without common commit evidence')
                        save();time.sleep(.2);continue
                    done_steps={int(store.get(k)) for k in keys('done',seq)}
                    if len(done_steps)!=1:raise RuntimeError('strategy completion is not a common boundary')
                    if lab:
                        retired = {active[cmd['source_replica'], rank][0] for rank in cmd['ranks']}
                        if any(lab.status(actors[label]['node'], actors[label]['unit'])['remaining']
                               for label in retired):
                            until = current.setdefault('retirement_deadline', time.monotonic() + 20)
                            if time.monotonic() >= until:
                                raise TimeoutError('old Lab actors still hold the transferred devices')
                            time.sleep(.2);continue
                        for label in retired:
                            lab.released_at[actors[label]['node']] = time.monotonic()
                        current['released_actors'] = [
                            dict(actor=label, node=actors[label]['node'], slots=actors[label]['gpu_uuids'])
                            for label in sorted(retired)]
                    if cmd['policy'] in ('v1','v2'):
                        for i in range(layout.world):active[cmd['source_replica'],i]=(current['target_actor'],i)
                        state['replica_nodes'][str(cmd['source_replica'])]=cmd['target']
                        if cmd['policy']=='v2':state.update(service=seq,stores={'0':seq,'1':seq},channels=[report['epoch']+1]*layout.world)
                    else:
                        for device, rank in enumerate(cmd['ranks']):
                            active[cmd['source_replica'],rank] = (current['target_actor'],device)
                        channel_ranks = range(layout.world) if 0 in cmd['ranks'] else cmd['ranks']
                        for rank in channel_ranks:
                            state['channels'][rank] = report['epoch'] + 1
                        if 0 in cmd['ranks']:
                            state['replica_nodes'][str(cmd['source_replica'])] = cmd['target']
                            state['stores'][str(cmd['source_replica'])] = current['outer_store_service']
                        state['partial_owner'] = current['target_role']
                    current.update(status='completed',completed_step=done_steps.pop(),controller_seconds=time.monotonic()-current['requested_monotonic'])
                    report['epoch']+=1
                    if bridge:
                        flash_snapshot = membership(a.session, report['epoch'], current['completed_step'] + 1,
                                                    layout, active, actors, cmd['id'])
                        report['flash_memberships'].append(flash_snapshot)
                        monitor_flash(flash_snapshot)
                    report['commands'].append(current);current=None;report['active_command']=None;save()
            if ready and current is None:
                if bridge:
                    collect_flash()
                    if live_profile:
                        try:
                            live_profile.poll(flash_snapshot, progress, flash_monitor['log'], rows, store)
                        except RedisError as error:
                            report['flash_redis_error'] = type(error).__name__
                    poll_flash(progress)
                if a.initial_steps and min(progress.values())>=a.initial_steps and not ref_stopped:
                    payload=dict(schema=1,session=a.session,id='reference-stop',expected_epoch=report['epoch'],action='stop',at_step=max(progress.values())+4,replica_world_size=layout.world)
                    (requests/'reference-stop.json').write_text(json.dumps(payload));ref_stopped=True
                for path in sorted(requests.glob('*.json')):
                    command_seen = time.monotonic()
                    payload=json.loads(path.read_text());identifier=payload.get('id');canonical=json.dumps(payload,sort_keys=True)
                    if identifier in seen:
                        if seen[identifier]!=canonical:
                            report.setdefault('rejected_commands',[]).append(dict(id=identifier,error='live command id payload conflict'));save()
                        continue
                    try:
                        cmd=Command.from_dict(payload)
                        placement = None
                        if cmd.replica_world_size!=layout.world:raise ValueError('command replica layout differs from the live session')
                        if cmd.session!=a.session or cmd.expected_epoch!=report['epoch']:raise ValueError('stale live command')
                        if cmd.action!='stop':
                            if report['command_count']+1>=base['session_services']:raise ValueError('session command budget reached; stop remains available')
                            if cmd.target not in eps and not (lab and cmd.target == 'auto'):
                                raise ValueError('unknown target node')
                            if cmd.policy=='v2' and cmd.source_replica!=1:
                                raise ValueError('planned whole-DP migration currently supports replica 1')
                            if cmd.policy=='v4':
                                scope = layout.partial_scope(cmd.source_replica, cmd.ranks)
                                labels = {active[scope.replica, rank][0] for rank in scope.ranks}
                                if len(labels) != 1 or len(actors[next(iter(labels))]['devices']) != layout.width:
                                    raise ValueError('partial migration requires an independent actor for the selected stage')
                                source_role, target_role = partial_roles(actors[next(iter(labels))]['role'])
                            if cmd.target == 'auto':
                                selection_started = time.monotonic()
                                cmd, selected_devices, placement = select_target(cmd)
                                placement['seconds'] = time.monotonic() - selection_started
                            else:
                                used=occupied(excluding=cmd.source_replica if cmd.policy=='v1' and not lab else None)
                                selected_devices = [i for i in device_indices(cmd.target) if i not in used[cmd.target]][:len(cmd.ranks)]
                                if fixed_placements:
                                    reference_case = fixed_placements[cmd.id]
                                    if any(reference_case['command'][field] != getattr(cmd, field)
                                           for field in ('target', 'policy', 'source_replica')):
                                        raise ValueError('fixed placement reference command differs')
                                    if tuple(reference_case['command']['ranks']) != cmd.ranks:
                                        raise ValueError('fixed placement reference rank scope differs')
                                    slots = lab.device_ids(cmd.target, device_indices(cmd.target))
                                    selected_devices = [slots.index(uid) for uid in reference_case['target_gpu_uuids']]
                                    if (len(set(selected_devices)) != len(selected_devices)
                                            or set(selected_devices) & used[cmd.target]):
                                        raise ValueError('fixed placement reference devices are already occupied')
                            if len(selected_devices) != len(cmd.ranks):
                                raise ValueError('target does not have enough free stage slots')
                    except (ValueError,KeyError) as error:
                        seen[identifier]=canonical
                        report.setdefault('rejected_commands',[]).append(dict(id=identifier,error=str(error)))
                        save();continue
                    seq=report['command_count']+1
                    # The controller admits commands before any candidate is launched.
                    payload=asdict(cmd);payload['ranks']=list(cmd.ranks) if cmd.ranks else None
                    if cmd.action=='stop':
                        payload={k:v for k,v in payload.items() if k not in ('policy','source_replica','target','ranks')}
                    devices = selected_devices if cmd.action != 'stop' else []
                    record=dict(sequence=seq,command=payload,devices=devices,
                                requested_monotonic=command_seen if lab else time.monotonic(),status='running')
                    if placement is not None:record['placement'] = placement
                    if a.normal_v1_exit and cmd.policy == 'v1':record['normal_source_exit'] = True
                    if cmd.policy == 'v2':
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            control = executor.submit(lighthouse, seq)
                            precopy = executor.submit(lighthouse, seq, planned_round=seq-1)
                            control.result()
                            precopy.result()
                    # Choose the boundary after service preparation and publish before
                    # remote actor launch: slow control RPCs cannot consume the cut.
                    latest=max(int(store.get(f'session/progress/{r}')) if store.check([f'session/progress/{r}']) else 0 for r in (0,1))
                    if payload['at_step'] is None:payload['at_step']=latest+4
                    elif payload['at_step']<=latest:
                        seen[identifier]=canonical
                        report.setdefault('rejected_commands',[]).append(dict(id=identifier,error='requested boundary already passed'));save();continue
                    if cmd.policy in ('v2','v4'):
                        role = 'D' if cmd.policy == 'v2' else target_role
                        label=f'g{seq}_{role}';record['target_actor']=label
                        if cmd.policy == 'v4':
                            record.update(source_role=source_role, target_role=target_role)
                            if 0 in cmd.ranks:record['outer_store_service'] = seq
                    if a.prepare_v1 and cmd.policy == 'v1':
                        label=f'g{seq}_D'
                        record.update(prepare_v1=True, target_actor=label)
                        state['stores'][str(cmd.source_replica)]=seq
                        dist.PrefixStore(f'round/{report["epoch"]}/',store).set('prepare','1')
                        actor(label,cmd.target,'D',cmd.source_replica,devices,generation=seq,command=record)
                    store.set(f'session/command/{seq}',json.dumps(record));store.set('session/command_count',str(seq))
                    if cmd.policy in ('v2','v4'):
                        dist.PrefixStore(f'round/{report["epoch"]}/',store).set('prepare','1')
                        actor(label,cmd.target,role,cmd.source_replica,devices,generation=seq,command=record)
                    report['command_count']=seq;seen[identifier]=canonical
                    if cmd.action!='stop':current=record;report['active_command']=record
                    save();break
            finished=[e for es in rows.values() for e in es if e['kind']=='session_done']
            if len(finished)==2*layout.world:
                if current is not None:raise RuntimeError('session stopped with an unfinished command')
                if len({(e['replica'],e['rank']) for e in finished})!=2*layout.world or len({e['step'] for e in finished})!=1:raise RuntimeError('incomplete final session coverage')
                report.update(done=True,final_step=finished[0]['step']);break
            save()
            time.sleep(.5)
        else:raise TimeoutError('session controller bounded deadline')
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        if isinstance(error,subprocess.CalledProcessError):report['error']+=' '+(error.stderr or '')[-1200:]
        for e in eps.values():report['error']=report['error'].replace(e.ip,'<private>')
    finally:
        cleanup=[]
        for monitor in flash_monitors:
            monitor['stop'].touch()
            worker = monitor['worker']
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=2)
            monitor['evidence']['worker_returncode'] = worker.returncode
        if bridge and bridge.published is not None:
            try:
                bridge.client.delete(bridge.membership_key)
            except RedisError as error:
                report['flash_redis_error'] = type(error).__name__
        for listener in reserved_lights.values():listener.close()
        for node,unit in reversed(units):
            try:
                if lab:
                    if unit in lab.units:lab.stop(node,unit)
                else:
                    ssh(node,'import subprocess;subprocess.run(["systemctl","stop",'+repr(unit)+'],capture_output=True,timeout=20);x=subprocess.run(["systemctl","show",'+repr(unit)+',"--property=ActiveState","--value"],capture_output=True,text=True);assert x.stdout.strip() not in ("active","activating","deactivating")')
            except BaseException as error:cleanup.append(unit+': '+type(error).__name__)
        try:
            collect();after=resources();(out/'resources_after.json').write_text(json.dumps(after))
            key = 'pod_uid' if lab else 'company'
            report['resources_clean']=before is not None and all(before[n][key]==after[n][key] and not after[n]['extra_gpu'] for n in eps)
        except BaseException as error:cleanup.append('collection: '+type(error).__name__)
        report.update(cleanup_errors=cleanup,actors=actors,active={f'{r}/{i}':list(v) for (r,i),v in active.items()})
        save();print(json.dumps({k:report.get(k) for k in ('name','done','error','resources_clean','cleanup_errors')}),flush=True)
    if report['error'] or not report['done'] or not report['resources_clean'] or cleanup:raise SystemExit(1)


if __name__=='__main__':main()
