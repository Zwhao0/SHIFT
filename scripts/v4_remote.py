"""One bounded server-owned V4 experiment; operates only exact owned units."""
import argparse
import fcntl
import hashlib
import json
import socket
from pathlib import Path
import subprocess
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env, server_password
from shift_full.v4_protocol import Plan, state_channel_matches

ROOT = Path('/workspace/SHIFT')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--post-steps', type=int, default=10)
    p.add_argument('--cut', type=int, default=4)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--no-oracle', action='store_true')
    p.add_argument('--gpu-prepare', action='store_true')
    p.add_argument('--gpu-warmup', action='store_true')
    p.add_argument('--slow-ms', type=float, default=0)
    p.add_argument('--reuse-local-groups', action='store_true')
    p.add_argument('--preconnect-state', action='store_true')
    p.add_argument('--defer-source-local-close', action='store_true')
    p.add_argument('--outer-manager', action='store_true')
    p.add_argument('--outer-audit', action='store_true')
    p.add_argument('--outer-timeout', type=float, default=120)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--reference-run')
    p.add_argument('--source-stall-round',type=int,default=-1)
    p.add_argument('--source-deadline',type=float,default=2)
    p.add_argument('--cross-node-tp',action='store_true')
    p.add_argument('--transactional-switch',action='store_true')
    p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--candidate-request-step',type=int,default=0)
    p.add_argument('--between-steps',type=int,default=4)
    p.add_argument('--cancel-round',type=int,default=-1)
    p.add_argument('--stall-restore-rank',type=int,default=-1)
    p.add_argument('--restore-deadline',type=float,default=10)
    p.add_argument('--fault-round',type=int,default=0)
    p.add_argument('--overlap-peer', action='store_true')
    p.add_argument('--retain-outer-managers', action='store_true')
    p.add_argument('--fail-restore-rank', type=int, default=-1)
    a = p.parse_args()
    if a.transactional_switch:
        a.outer_manager=True;a.overlap_peer=True
        if not a.reference_only:
            a.retain_outer_managers=True;a.gpu_prepare=True;a.gpu_warmup=True
            a.reuse_local_groups=True;a.preconnect_state=True;a.defer_source_local_close=True
    if a.cross_node_tp and not a.transactional_switch:p.error('cross-node TP requires transactional switching')
    if a.cross_node_tp:
        a.reuse_local_groups=False;a.defer_source_local_close=False;a.gpu_warmup=False
    if a.cycles not in range(1,4) or a.between_steps<1 or a.restore_deadline<=0 or a.source_deadline<=0:
        p.error('bounded cycles 1..3, positive between steps and restore deadline required')
    if a.candidate_request_step<0 or (a.candidate_request_step and (not a.transactional_switch or a.reference_only)):p.error('candidate request step requires migrating transactional mode')
    affected=[1,3,5,7] if a.cross_node_tp else [4,5,6,7]
    if a.transactional_switch and (a.source_stall_round not in range(-1,a.cycles) or a.cancel_round not in range(-1,a.cycles) or a.fault_round not in range(a.cycles) or
        (a.stall_restore_rank!=-1 and a.stall_restore_rank not in affected)):
        p.error('fault round/rank must belong to this bounded switching plan')
    if sum((a.source_stall_round!=-1,a.cancel_round!=-1,a.stall_restore_rank!=-1,a.fail_restore_rank!=-1))>1:
        p.error('use one injected fault per controlled run')
    if a.gpu_warmup and not a.gpu_prepare:p.error('GPU warmup requires isolated GPU preparation')
    if a.reuse_local_groups and not a.gpu_prepare:p.error('local group reuse requires GPU preparation')
    if a.defer_source_local_close and not a.reuse_local_groups:p.error('deferred source close requires local group preservation')
    if a.outer_audit and not a.outer_manager:p.error('outer gradient audit requires outer Manager')
    if a.fail_restore_rank!=-1 and (not a.outer_manager or (not a.no_oracle and not a.transactional_switch) or a.fail_restore_rank not in ([1,3,5,7] if a.cross_node_tp else [4,5,6,7])):
        p.error('restore failure requires outer manager, no oracle, and rank 4..7')
    if (a.reference_only or a.reference_run or a.retain_outer_managers) and not a.outer_manager:
        p.error('independent reference and Manager retention require outer Manager')
    if a.overlap_peer and (not a.outer_manager or (not a.reference_only and not a.retain_outer_managers)):
        p.error('peer overlap requires outer Manager retention (or reference-only)')
    if a.reference_only and (a.reference_run or a.fail_restore_rank != -1 or a.retain_outer_managers):
        p.error('reference-only cannot migrate, inject failure or consume a reference')
    if a.reference_run and (a.no_oracle or not all(x.isalnum() or x in '_-' for x in a.reference_run)):
        p.error('reference-run requires a valid suffix and oracle enabled')
    if a.outer_manager and not (a.no_oracle or a.reference_only or a.reference_run):
        p.error('outer oracle requires --reference-run; create it with --reference-only first')
    if not a.suffix or not all(x.isalnum() or x in '_-' for x in a.suffix):
        p.error('invalid suffix')
    if a.transactional_switch and not a.reference_only:
        from scripts.v4_transaction_remote import main as transaction_main
        return transaction_main(a)
    plan = Plan(cut=a.cut, post_steps=a.post_steps, seq_len=a.seq_len)
    name = 'v4_' + a.suffix
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=False)
    lock = (ROOT / 'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote.PASSWORD = server_password(remote.PRIVATE / 'credentials')
    (remote.PRIVATE / 'credentials').unlink()
    s = remote.ServerRotation.__new__(remote.ServerRotation)
    s.root, s._sudo_password = ROOT, remote.PASSWORD
    eps = endpoints_from_env()
    roles = dict(zip('ABC', (eps['node0'], eps['node1'], eps['node3'])))
    if a.outer_manager:
        roles.update(D=eps['node0'],E=eps['node1'])
    if a.reference_only:roles.pop('C')
    units = []
    rows = {}
    report = dict(name=name, functional_pass=False, oracle_exact=None, error=None,
        topology=dict(tp=2, pp=2, ep=2, ranks=8, devices_per_node=4,
                      A='node0/PP0', B='node1/PP1', C='node3/standby_PP1'),
        plan=vars(plan), oracle=not (a.no_oracle or a.reference_only), gpu_prepare=a.gpu_prepare,gpu_warmup=a.gpu_warmup,slow_ms=a.slow_ms,
        state_transport='packed_gpu',
        reuse_local_groups=a.reuse_local_groups, preconnect_state=a.preconnect_state,
        defer_source_local_close=a.defer_source_local_close,
        reference_only=a.reference_only,reference_run=a.reference_run,retain_outer_managers=a.retain_outer_managers,overlap_peer=a.overlap_peer,transactional_switch=a.transactional_switch,cycles=a.cycles,cross_node_tp=a.cross_node_tp,
        source_stall_round=a.source_stall_round,source_deadline=a.source_deadline,
        between_steps=a.between_steps,cancel_round=a.cancel_round,stall_restore_rank=a.stall_restore_rank,restore_deadline=a.restore_deadline,fault_round=a.fault_round,affected_ranks=[1,3,5,7] if a.cross_node_tp else [4,5,6,7],
        outer_manager=a.outer_manager,outer_audit=a.outer_audit or (a.outer_manager and not a.no_oracle),
        outer_timeout=a.outer_timeout,fail_restore_rank=a.fail_restore_rank,
        limitations=['single replica; outer TorchFT DP isolation not exercised',
            'default and cross-host groups rebuilt; local reuse/preconnect are opt-in; extra snapshot memory',
            'adapter-specific standby-local rehearsal; not a general communication replay sandbox',
            'state hashing and diagnostics perturb timing'],
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [ROOT/'shift_full/v4_train.py',ROOT/'shift_full/v4_protocol.py',ROOT/'shift_full/transport.py',
                      ROOT/'shift_full/compat.py',ROOT/'shift_full/v4_groups.py',ROOT/'shift_full/v4_reference.py',
                      ROOT/'scripts/v4_remote.py',ROOT/'scripts/verify_v4.py']})
    if a.outer_manager:
        report['topology'].update(replicas=2,ranks_per_replica=8,ranks=16,global_batch=16,
            D='node0/GPU4..7/replica1_PP0',E='node1/GPU4..7/replica1_PP1',independent_fault_domains=False)
        report['limitations'][0]='two DP replicas share hosts; both pause and rebuild outer Managers at boundary'
        if a.reference_only:
            report['limitations'][0]='independent no-migration reference; two replicas share hosts'
        elif a.retain_outer_managers:
            report['limitations'][0]='two replicas share hosts; both pause; healthy Managers/services retained'
        for filename in ('shift_full/v4_outer.py','scripts/verify_v4_outer.py','scripts/verify_v4_reference.py'):
            report['source_sha256'][filename]=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    if a.transactional_switch:
        from shift_full.v4_reference import training_sources
        report['source_sha256'].update(training_sources(ROOT))
    def save():
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def run(e, code):
        return s.ssh(e, ['sudo','-S','-p','','python3','-c',code], sudo_input=True)
    def resources():
        result = {}
        for n,e in eps.items():
            raw = s.ssh(e,['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
            entries = [x.split(',',1) for x in raw.splitlines() if x.strip()]
            company = {pid.strip():proc.strip() for pid,proc in entries if 'VLLM::' in proc}
            extra = [pid.strip() for pid,proc in entries if 'VLLM::' not in proc]
            result[n] = dict(company=company, extra_gpu=extra)
        return result
    def launch(e, suffix, role, config, host_role=None):
        unit = 'shift-v4-'+a.suffix+'-'+suffix
        units.append((e, unit))  # Track before a possibly lost launch acknowledgment.
        args = ['systemd-run','--quiet','--collect','--unit='+unit,
            '--property=RuntimeMaxSec='+str(a.seconds),'--property=KillMode=control-group',
            '--property=TimeoutStopSec=15','--working-directory='+str(ROOT),
            '--property=StandardOutput=append:'+str(out/(suffix+'.host.log')),'--property=StandardError=inherit']
        env = dict(PYTHONPATH=str(ROOT)+':/workspace/FLASH1022',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MALLOC_ARENA_MAX='2',
            CPATH=str(ROOT/'runtime/python-headers'),CUDA_VISIBLE_DEVICES='4,5,6,7' if host_role in ('D','E') else '0,1,2,3',
            NCCL_MAX_CTAS='4',NCCL_BUFFSIZE='1048576',NCCL_CUMEM_HOST_ENABLE='0',
            NCCL_CUMEM_ENABLE='0',NCCL_NVLS_ENABLE='0',NCCL_DEBUG='WARN',
            NCCL_NET='IB',NCCL_IB_DISABLE='0',NCCL_SOCKET_IFNAME='bond0',GLOO_SOCKET_IFNAME='bond0',
            NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1',NCCL_IB_GID_INDEX='3',
            NCCL_LAUNCH_ORDER_IMPLICIT='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='0',
            TORCH_NCCL_USE_COMM_NONBLOCKING='0',TORCHFT_USE_OTEL='false',
            TMPDIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            XDG_CACHE_HOME=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            TRITON_CACHE_DIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache/triton'),
            NO_PROXY='*',no_proxy='*')
        args += ['--setenv='+k+'='+v for k,v in env.items()]
        if role == 'lighthouse':
            args += [str(ROOT/'runtime/torch214/bin/python'),str(ROOT/'vendor/bin/torchft_lighthouse'),
                '--bind','[::]:'+str(32810+int(host_role)),'--min_replicas','2',
                '--join_timeout_ms','120000','--heartbeat_timeout_ms','300000','--quorum_tick_ms','100']
        else:
            args += [str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.v4_train',
                     '--config',str(config),'--role',role]
            if host_role:
                args += ['--host-role',host_role]
        s.ssh(e,['sudo','-S','-p','',*args],sudo_input=True)
    def collect():
        data = {}
        for role,e in roles.items():
            code = ('from pathlib import Path;import json;p=Path('+repr(str(out))+');'
                'print(json.dumps({f.name:f.read_text(errors="replace") for f in p.glob('+repr(role+'*.jsonl')+')}))')
            files = json.loads(s.ssh(e,['python3','-c',code]))
            data[role] = [json.loads(line) for content in files.values() for line in content.splitlines() if line.strip()]
            if role not in ('A','D'):  # These workers already write on this host.
                for filename,content in files.items():
                    (out/filename).write_text(content)
        return data
    save()
    try:
        for e in eps.values():
            s.ensure_connection(e)
        before = resources()
        (out/'resources_before.json').write_text(json.dumps(before,indent=2)+'\n')
        if any(len(x['company']) != 8 or x['extra_gpu'] for x in before.values()):
            raise RuntimeError('expected idle SHIFT resources and eight existing company PIDs')
        config = dict(directory=str(out), flash='/workspace/FLASH1022',
            hosts={role:e.ip for role,e in roles.items()}, port=32750, timeout=120,
            seconds=a.seconds, seed=20261106, plan=vars(plan), oracle=report['oracle'],
            reference_only=a.reference_only,reference_run=a.reference_run,retain_outer_managers=a.retain_outer_managers,overlap_peer=a.overlap_peer,transactional_switch=a.transactional_switch,cycles=a.cycles,cross_node_tp=a.cross_node_tp,
        source_stall_round=a.source_stall_round,source_deadline=a.source_deadline,
        between_steps=a.between_steps,cancel_round=a.cancel_round,stall_restore_rank=a.stall_restore_rank,restore_deadline=a.restore_deadline,fault_round=a.fault_round,affected_ranks=[1,3,5,7] if a.cross_node_tp else [4,5,6,7],
            gpu_prepare=a.gpu_prepare,gpu_warmup=a.gpu_warmup,slow_ms=a.slow_ms,
            reuse_local_groups=a.reuse_local_groups,preconnect_state=a.preconnect_state,
            defer_source_local_close=a.defer_source_local_close,outer_manager=a.outer_manager,
            outer_audit=report['outer_audit'],outer_timeout=a.outer_timeout,fail_restore_rank=a.fail_restore_rank)
        if a.reference_run:
            from shift_full.v4_reference import validate_compatibility
            refpath=ROOT/'results'/('v4_'+a.reference_run)/'reference_signatures.json'
            from scripts.verify_v4_reference import verify as verify_reference
            report['reference_audit']=verify_reference(refpath.parent)
            if not report['reference_audit']['pass_all']:raise ValueError(report['reference_audit']['errors'])
            reference=json.loads(refpath.read_text())
            reference_summary=json.loads((refpath.parent/'summary.json').read_text())
            if not reference_summary.get('resources_clean') or reference_summary.get('cleanup_errors'):
                raise ValueError('reference requires clean completed run')
            # Controllers/auditors have separate provenance; require identical
            # training/state/communication code for numerical comparison.
            validate_compatibility(reference,config,{k:v for k,v in report['source_sha256'].items()
                if k.startswith('shift_full/')})
            config['reference_file']=str(out/'reference_signatures.json')
            report['reference_sha256']=hashlib.sha256(refpath.read_bytes()).hexdigest()
            for e in eps.values():
                temporary='/tmp/shift-v4-'+a.suffix+'-reference.json'
                subprocess.run(['scp','-q','-o','ControlPath='+str(ROOT/'tmp'/e.socket),
                    str(refpath),'test@'+e.ip+':'+temporary],capture_output=True,timeout=30,check=True)
                run(e,'from pathlib import Path;import hashlib;p=Path('+repr(temporary)+');data=p.read_bytes();'
                    'assert hashlib.sha256(data).hexdigest()=='+repr(report['reference_sha256'])+';'
                    'dest=Path('+repr(config['reference_file'])+');dest.parent.mkdir(parents=True,exist_ok=True);'
                    'dest.write_bytes(data);p.unlink()')
        path = ROOT/'runtime/v3-reproduce'/('v4_'+a.suffix+'.json')
        for e in eps.values():
            code = ('from pathlib import Path;p=Path('+repr(str(path))+');p.parent.mkdir(parents=True,exist_ok=True);'
                'p.write_text('+repr(json.dumps(config))+');Path('+repr(str(out))+').mkdir(parents=True,exist_ok=True)')
            run(e,code)
        launch(roles['A'],'store','store',path)
        end = time.monotonic()+30
        while not (out/'store_ready').exists():
            if time.monotonic()>end:
                raise TimeoutError('control store did not start')
            time.sleep(.2)
        if a.outer_manager:
            for epoch in range(1 if a.reference_only or a.retain_outer_managers else 2):
                launch(roles['A'],'lighthouse'+str(epoch),'lighthouse',path,str(epoch))
                until=time.monotonic()+30
                while True:
                    with socket.socket() as probe:
                        if probe.connect_ex((roles['A'].ip,32810+epoch))==0:break
                    if time.monotonic()>until:raise TimeoutError('lighthouse startup timed out')
                    time.sleep(.2)
        for role,e in roles.items():
            launch(e,role,'host',path,role)
        until = time.monotonic()+a.seconds-30
        last_progress = None
        while time.monotonic()<until:
            rows = collect()
            progress = {r:dict(commits=sum(x['kind']=='commit' for x in es),
                done=sum(x['kind'] in ('done','retired') for x in es)) for r,es in rows.items()}
            if progress != last_progress:
                print(json.dumps(dict(progress=progress)),flush=True); last_progress=progress
            if a.fail_restore_rank!=-1 and any(x['kind']=='restore_failure_injected' for x in rows.get('C',[])):
                raise RuntimeError('pre-activation restore rejection observed; stopping owned job')
            if all(progress[r]['done']==4 for r in roles):
                break
            for e,u in units:
                status = s.ssh(e,['systemctl','show',u,'--property=ActiveState','--property=ExecMainStatus'])
                values = dict(line.split('=',1) for line in status.splitlines() if '=' in line)
                if values.get('ActiveState')=='failed' or values.get('ExecMainStatus') not in (None,'0'):
                    raise RuntimeError('owned unit failed: '+u)
            time.sleep(2)
        else:
            raise TimeoutError('bounded V4 run timed out')
        if a.reference_only:
            from shift_full.v4_reference import build_signatures
            report['functional_pass']=True
            signatures=build_signatures(rows,report)
            (out/'reference_signatures.json').write_text(json.dumps(signatures,sort_keys=True)+'\n')
        else:
            active_roles='ACDE' if a.outer_manager else 'AC'
            old_roles='ABDE' if a.outer_manager else 'AB'
            done = [x for r in active_roles for x in rows[r] if x['kind']=='done']
            selected=[x['step'] for r in old_roles for x in rows[r] if x['kind']=='boundary_selected']
            if len(selected)!=(16 if a.outer_manager else 8) or len(set(selected))!=1:
                raise RuntimeError('boundary selection disagreement')
            report['requested_cut']=plan.cut
            plan=Plan(cut=selected[0],post_steps=plan.post_steps,seq_len=plan.seq_len,microbatches=plan.microbatches)
            report['plan']=vars(plan)
            expected={(replica,rank) for replica in range(2 if a.outer_manager else 1) for rank in range(8)}
            if len(done)!=len(expected) or {(x.get('replica',0),x['rank']) for x in done}!=expected:
                raise RuntimeError('incomplete final role coverage')
            if any(x['step']!=plan.cut+plan.post_steps or not x['identity_preserved'] for x in done):
                raise RuntimeError('bad final step or changed retained identities')
            for role in active_roles:
                for device in range(4):
                    events = [x for x in rows[role] if x['device']==device]
                    commits = [x for x in events if x['kind']=='commit' and x['phase']=='after']
                    if [x['step'] for x in commits]!=list(range(plan.cut+1,plan.cut+plan.post_steps+1)):
                        raise RuntimeError('missing/duplicate migrated commits')
                    if len({x['pid'] for x in events})!=1:
                        raise RuntimeError('worker PID changed')
            for device in range(4):
                sent = next(x for x in rows['B'] if x['device']==device and x['kind']=='state_sent')
                received = next(x for x in rows['C'] if x['device']==device and x['kind']=='state_received')
                if not state_channel_matches(sent, received):
                    raise RuntimeError('state-channel integrity mismatch')
            if any(x['kind']=='commit' and x['epoch']==1 for x in rows['B']):
                raise RuntimeError('retired source committed')
            report['functional_pass'] = True
            report['oracle_exact'] = (all(all(x['comparison']['exact'].values()) and x['comparison']['trajectory_exact'] and x['comparison'].get('manager_exact',True) for x in done) if config['oracle'] else None)
            report['comparisons'] = {f"{x.get('replica',0)}/{x['rank']}":x['comparison'] for x in done}
            aa = [x for x in rows['A'] if x['device']==0]
            freeze = next(x for x in aa if x['kind']=='freeze')
            activated = next(x for x in aa if x['kind']=='activated')
            first = next(x for x in aa if x['kind']=='commit' and x['phase']=='after')
            boundary = next(x for x in aa if x['kind']=='cut_boundary')
            before_commit = [x for x in aa if x['kind']=='commit' and x['phase']=='before'][-1]
            report['diagnostic_timing'] = dict(freeze_to_active=activated['monotonic']-freeze['monotonic'],
                freeze_to_first_commit=first['monotonic']-freeze['monotonic'],
                boundary_to_active=activated['monotonic']-boundary['monotonic'],
                boundary_to_first_commit=first['monotonic']-boundary['monotonic'],
                commit_gap=first['monotonic']-before_commit['monotonic'],
                includes_oracle_replay=config['oracle'] and not bool(config.get('reference_run')),
                includes_reference_hashing=bool(config.get('reference_run')),performance_evidence=False)
            report['healthy_pids'] = sorted({x['pid'] for r in ('ADE' if a.outer_manager else 'A') for x in rows[r]})
            report['state_channel_bytes'] = sum(x['bytes'] for x in rows['B'] if x['kind']=='state_sent')
            report['transport_totals'] = {k:sum(x[k] for x in rows['B'] if x['kind']=='state_sent')
                                          for k in ('gpu_bytes','control_bytes','cpu_tensor_bytes')}
            report['phase_seconds'] = {r+str(d):[
                {k:x[k] for k in ('kind','step','epoch','seconds') if k in x}
                for x in rows[r] if x['device']==d and 'seconds' in x and x['kind']!='commit']
                for r in roles for d in range(4)}
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        cleanup = []
        for e,u in reversed(units):
            try:
                code=('import subprocess;subprocess.run(["systemctl","stop",'+repr(u)+'],capture_output=True,timeout=25);'
                    'r=subprocess.run(["systemctl","show",'+repr(u)+',"--property=ActiveState","--value"],capture_output=True,text=True);'
                    'assert r.stdout.strip() not in ("active","activating","deactivating"),r.stdout')
                run(e,code)
            except Exception as error:
                cleanup.append(u+': '+type(error).__name__)
        try:
            rows = collect()
            for role,e in roles.items():
                code = ('from pathlib import Path;import json;p=Path('+repr(str(out))+');'
                    'print(json.dumps({f.name:f.read_text(errors="replace")[-12000:] for f in p.glob('+repr(role+'*.log')+')}))')
                logs = json.loads(s.ssh(e,['python3','-c',code]))
                dest = out/('logs_'+role); dest.mkdir(exist_ok=True)
                for filename,content in logs.items(): (dest/filename).write_text(content)
            after = resources()
            (out/'resources_after.json').write_text(json.dumps(after,indent=2)+'\n')
            report['resources_clean'] = all(after[n]['company']==before[n]['company'] and not after[n]['extra_gpu'] for n in eps)
        except Exception as error:
            cleanup.append('collection: '+type(error).__name__)
        report['cleanup_errors'] = cleanup
        report['raw_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}
        selected=[x['step'] for r in ('ABDE' if a.outer_manager else 'AB')
                  for x in rows.get(r,[]) if x['kind']=='boundary_selected']
        if len(selected)==(16 if a.outer_manager else 8) and len(set(selected))==1:
            report['requested_cut']=a.cut; report['plan']['cut']=selected[0]
        if a.fail_restore_rank!=-1:
            injected=[x for x in rows.get('C',[]) if x['kind']=='restore_failure_injected']
            after=[x for es in rows.values() for x in es if x['kind']=='commit' and x['phase']=='after']
            active=[x for es in rows.values() for x in es if x['kind']=='activated']
            report['expected_failure_pass']=len(injected)==1 and not after and not active and bool(report['error'])
        save()
        print(json.dumps(report),flush=True)
    if (report['error'] and not report.get('expected_failure_pass')) or not report.get('resources_clean') or report['cleanup_errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
