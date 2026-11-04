"""Server-owned bounded transparent switching; every mutation names an owned unit."""
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env,server_password
from shift_full.v4_protocol import Plan
from shift_full.v4_reference import training_sources,validate_compatibility

ROOT=Path('/workspace/SHIFT')


def unfinished_actors(actors, rows):
    """Active completion alone cannot authorize cancelling candidate I/O."""
    return [label for label,info in actors.items() if not info.get('expected_stop') and
        {x['device'] for x in rows[label] if x['kind'] in ('done','retired','standby_done')} != set(range(4))]


def main(a):
    name='v4_'+a.suffix;out=ROOT/'results'/name;out.mkdir(exist_ok=False)
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    session=remote.ServerRotation.__new__(remote.ServerRotation);session.root=ROOT;session._sudo_password=remote.PASSWORD
    eps=endpoints_from_env();roles=dict(A=eps['node0'],B=eps['node1'],C=eps['node3'],D=eps['node0'],E=eps['node1'])
    if a.unified:roles.update(B=eps['node0'],D=eps['node1'])
    sources=training_sources(ROOT)
    report=dict(role_hosts={r:e.physical_node for r,e in roles.items()},unified=a.unified, performance_no_audit=a.performance_no_audit, seed=a.seed, initialization_seed=2026 if a.unified else a.seed, data='random' if a.unified else 'pattern', name=name,transactional_switch=True,overlap_peer=True,outer_manager=True,outer_audit=not a.performance_no_audit,
        retain_outer_managers=True,oracle=not a.no_oracle,reference_run=a.reference_run,
        cycles=a.cycles,between_steps=a.between_steps,cross_node_tp=a.cross_node_tp,plan=vars(Plan(cut=a.cut,post_steps=a.post_steps,seq_len=a.seq_len,microbatches=2 if a.unified else 4)),
        outer_timeout=a.outer_timeout,fail_restore_rank=a.fail_restore_rank,stall_restore_rank=a.stall_restore_rank,
        cancel_round=a.cancel_round,restore_deadline=a.restore_deadline,fault_round=a.fault_round,
        source_stall_round=a.source_stall_round,source_deadline=a.source_deadline,
        candidate_request_step=a.candidate_request_step,slow_ms=a.slow_ms,
        functional_pass=False,oracle_exact=None,error=None,source_sha256=sources,
        trace_step=a.trace_step,deterministic=a.deterministic,
        topology=dict(tp=2,pp=2,ep=2,replicas=2,ranks_per_replica=8,global_batch=8 if a.unified else 16,replica_hosts_distinct=a.unified,independent_fault_domains=False,
            physical_layout='complete_replica_per_host_then_partial_PP' if a.unified else 'even_odd_cross_host_tp' if a.cross_node_tp else 'pp_stage_per_host'))
    units=[];actors={};rows={};before=None;expected_stops=set();handled_recoveries=set()
    report['source_recovery_stops']=[]
    report['source_recovery_scope']=dict(failure_window='after_freeze_before_ready',transport_failure='fail_stop')
    until=time.monotonic()+a.seconds-30
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(endpoint,code):return session.ssh(endpoint,['sudo','-S','-p','','python3','-c',code],sudo_input=True)
    def resources():
        result={}
        for node,endpoint in eps.items():
            raw=session.ssh(endpoint,['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
            entries=[line.split(',',1) for line in raw.splitlines() if line.strip()]
            result[node]=dict(company={pid.strip():proc.strip() for pid,proc in entries if 'VLLM::' in proc},
                             extra_gpu=[pid.strip() for pid,proc in entries if 'VLLM::' not in proc],gpu_load=session.ssh(endpoint,['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits']).splitlines())
        return result
    def launch(endpoint,label,role,config,actor_role=None):
        unit='shift-v4-'+a.suffix+'-'+label;units.append((endpoint,unit))
        command=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+str(a.seconds),
            '--property=KillMode=control-group','--property=TimeoutStopSec=15','--working-directory='+str(ROOT),
            '--property=StandardOutput=append:'+str(out/(label+'.host.log')),'--property=StandardError=inherit']
        env=dict(PYTHONPATH=str(ROOT)+':/workspace/FLASH1022',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MALLOC_ARENA_MAX='2',TOKIO_WORKER_THREADS='2',
            CPATH=str(ROOT/'runtime/python-headers'),CUDA_VISIBLE_DEVICES='4,5,6,7' if actor_role in (('B','E') if a.unified else ('D','E')) else '0,1,2,3',
            CUDA_DEVICE_MAX_CONNECTIONS='1',NCCL_MAX_CTAS='4',NCCL_BUFFSIZE='1048576',NCCL_CUMEM_HOST_ENABLE='0',NCCL_CUMEM_ENABLE='0',
            NCCL_NVLS_ENABLE='0',NCCL_DEBUG='WARN',NCCL_NET='IB',NCCL_IB_DISABLE='0',NCCL_SOCKET_IFNAME='bond0',
            GLOO_SOCKET_IFNAME='bond0',NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1',NCCL_IB_GID_INDEX='3',
            NCCL_LAUNCH_ORDER_IMPLICIT='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='0',TORCH_NCCL_USE_COMM_NONBLOCKING='0',
            TORCHFT_USE_OTEL='false',TMPDIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            XDG_CACHE_HOME=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            TRITON_CACHE_DIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache/triton'),NO_PROXY='*',no_proxy='*')
        command+=['--setenv='+k+'='+v for k,v in env.items()]
        if role=='lighthouse':
            command+=[str(ROOT/'runtime/torch214/bin/python'),str(ROOT/'vendor/bin/torchft_lighthouse'),
                '--bind','[::]:32810','--min_replicas','2','--join_timeout_ms','120000',
                '--heartbeat_timeout_ms','300000','--quorum_tick_ms','100']
        else:
            command+=[str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.v4_train','--config',str(config),'--role',role]
            if actor_role:command+=['--host-role',actor_role]
        session.ssh(endpoint,['sudo','-S','-p','',*command],sudo_input=True)
    def actor(generation,role,source,target):
        label=f'g{generation}_{role}';config=dict(base,initial_round=generation,source_role=source,target_role=target)
        path=ROOT/'runtime/v3-reproduce'/f'{name}_{label}.json'
        endpoint=roles[role]
        run(endpoint,'from pathlib import Path;p=Path('+repr(str(path))+');p.write_text('+repr(json.dumps(config))+')')
        actors[label]=dict(generation=generation,role=role,endpoint=endpoint)
        launch(endpoint,label,'host',path,role)
    def collect():
        result={}
        for label,info in actors.items():
            endpoint=info['endpoint'];prefix=label
            code='from pathlib import Path;import json;p=Path('+repr(str(out))+');print(json.dumps({f.name:f.read_text(errors="replace") for f in p.glob('+repr(prefix+'*.jsonl')+')}))'
            files=json.loads(session.ssh(endpoint,['python3','-c',code]))
            result[label]=[json.loads(line) for content in files.values()
                for line in (content.splitlines() if content.endswith('\n') else content.splitlines()[:-1]) if line.strip()]
            if endpoint.physical_node != 'node0':
                for filename,content in files.items():(out/filename).write_text(content)
        return result
    def check_units():
        for endpoint,unit in units:
            if unit in expected_stops:continue
            values=dict(line.split('=',1) for line in session.ssh(endpoint,
                ['systemctl','show',unit,'--property=ActiveState','--property=ExecMainStatus']).splitlines() if '=' in line)
            if values.get('ActiveState')=='failed' or values.get('ExecMainStatus') not in (None,'0'):
                raise RuntimeError('owned unit failed: '+unit)

    save()
    try:
        for endpoint in eps.values():session.ensure_connection(endpoint)
        before=resources();(out/'resources_before.json').write_text(json.dumps(before,indent=2)+'\n')
        if any(len(v['company'])!=8 or v['extra_gpu'] for v in before.values()):raise RuntimeError('SHIFT resources are not idle')
        base=dict(directory=str(out),flash='/workspace/FLASH1022',hosts={r:e.ip for r,e in roles.items()},port=32750,
            timeout=120,seconds=a.seconds,slow_ms=a.slow_ms,slow_start_step=a.candidate_request_step or a.cut-1,seed=2026 if a.unified else a.seed, unified=a.unified, data='random' if a.unified else 'pattern', performance_no_audit=a.performance_no_audit,plan=report['plan'],oracle=report['oracle'],
            reference_only=False,reference_run=a.reference_run,outer_manager=True,outer_audit=not a.performance_no_audit,
            outer_timeout=a.outer_timeout,retain_outer_managers=True,overlap_peer=True,transactional_switch=True,
            trace_step=a.trace_step,deterministic=a.deterministic,
            gpu_prepare=True,gpu_warmup=not a.cross_node_tp,reuse_local_groups=not a.cross_node_tp,preconnect_state=True,defer_source_local_close=not a.cross_node_tp,
            cross_node_tp=a.cross_node_tp,affected_ranks=[1,3,5,7] if a.cross_node_tp else [4,5,6,7],
            cycles=a.cycles,between_steps=a.between_steps,fail_restore_rank=a.fail_restore_rank,
            stall_restore_rank=a.stall_restore_rank,cancel_round=a.cancel_round,restore_deadline=a.restore_deadline,
            fault_round=a.fault_round,source_stall_round=a.source_stall_round,source_deadline=a.source_deadline,initial_round=0,source_role='B',target_role='C')
        if a.reference_run:
            from scripts.verify_v4_reference import verify
            refpath=ROOT/'results'/('v4_'+a.reference_run)/'reference_signatures.json'
            report['reference_audit']=verify(refpath.parent)
            if not report['reference_audit']['pass_all']:raise ValueError(report['reference_audit']['errors'])
            reference=json.loads(refpath.read_text());validate_compatibility(reference,base,sources)
            report['reference_sha256']=hashlib.sha256(refpath.read_bytes()).hexdigest()
            base['reference_file']=str(out/'reference_signatures.json')
            for endpoint in eps.values():
                temporary='/tmp/shift-v4-'+a.suffix+'-reference.json'
                subprocess.run(['scp','-q','-o','ControlPath='+str(ROOT/'tmp'/endpoint.socket),
                    str(refpath),'test@'+endpoint.ip+':'+temporary],capture_output=True,timeout=30,check=True)
                run(endpoint,'from pathlib import Path;import hashlib;p=Path('+repr(temporary)+');data=p.read_bytes();'
                    'assert hashlib.sha256(data).hexdigest()=='+repr(report['reference_sha256'])+';'
                    'dest=Path('+repr(base['reference_file'])+');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);p.unlink()')
        for endpoint in eps.values():
            run(endpoint,'from pathlib import Path;p=Path('+repr(str(out))+');p.mkdir(parents=True,exist_ok=True)')
        service_config=ROOT/'runtime/v3-reproduce'/f'{name}.json'
        run(roles['A'],'from pathlib import Path;Path('+repr(str(service_config))+').write_text('+repr(json.dumps(base))+')')
        launch(roles['A'],'store','store',service_config)
        end=time.monotonic()+30
        while not (out/'store_ready').exists():
            if time.monotonic()>end:raise TimeoutError('store startup deadline')
            time.sleep(.1)
        launch(roles['A'],'lighthouse0','lighthouse',service_config)
        end=time.monotonic()+30
        while True:
            with socket.socket() as probe:
                if probe.connect_ex((roles['A'].ip,32810))==0:break
            if time.monotonic()>end:raise TimeoutError('Lighthouse startup deadline')
            time.sleep(.1)
        for role in ('ABDE' if a.candidate_request_step else 'ABCDE'):
            if role=='C':report['migration_request']=dict(monotonic=time.monotonic(),time=time.time(),observed_step=0)
            actor(0,role,'B','C')
        last=None
        while time.monotonic()<until:
            scan_started=time.monotonic()
            rows=collect();events=[event for es in rows.values() for event in es]
            mismatch=next((e for e in events if e.get('kind')=='reference_comparison' and not e.get('exact')),None)
            if mismatch:raise RuntimeError('independent reference mismatch: '+str((mismatch['role'],mismatch['rank'],mismatch['step'])))
            if 'migration_request' in report and 'first_common_observation' not in report:
                committed={}
                for event in events:
                    if event['kind']=='commit' and event.get('epoch',0)>0:
                        committed.setdefault(event['step'],set()).add((event['replica'],event['rank']))
                common=sorted(step for step,workers in committed.items() if len(workers)==16)
                if common:
                    report['first_common_observation']=dict(first_joint_step=common[0],
                        scan_started_monotonic=scan_started,observed_monotonic=time.monotonic(),
                        poll_seconds=1,semantics='node0 controller collected all sixteen worker commits; includes collection and polling latency')
            if 'g0_C' not in actors:
                observed=max((x['step'] for x in events if x['kind']=='commit' and x['role']=='A' and x['device']==0),default=0)
                if observed>=a.candidate_request_step:
                    report['migration_request']=dict(monotonic=time.monotonic(),time=time.time(),observed_step=observed)
                    actor(0,'C','B','C')
            for recovery in (x for x in events if x['kind']=='source_recovery_selected'):
                recovery_key=(recovery['round'],recovery['source'])
                if recovery_key in handled_recoveries:continue
                source_actors=[(label,info) for label,info in actors.items() if info['role']==recovery['source']
                    and info['generation']<=recovery['round']]
                source_label,info=max(source_actors,key=lambda item:item[1]['generation'])
                unit='shift-v4-'+a.suffix+'-'+source_label
                if unit in expected_stops:continue
                stop_proof=json.loads(run(info['endpoint'],'import subprocess,json;unit='+repr(unit)+';'
                    'cmd=["systemctl","show",unit,"--property=LoadState","--property=ActiveState","--property=SubState","--property=MainPID"];'
                    'before=dict(x.split("=",1) for x in subprocess.check_output(cmd,text=True).splitlines());'
                    'assert before["LoadState"]=="loaded" and before["ActiveState"]=="active" and before["SubState"]=="running" and int(before["MainPID"])>0;'
                    'r=subprocess.run(["systemctl","stop",unit],capture_output=True,timeout=25);assert r.returncode==0;'
                    'after=dict(x.split("=",1) for x in subprocess.check_output(cmd,text=True).splitlines());'
                    'assert after["ActiveState"]=="inactive" and after["SubState"]=="dead" and int(after["MainPID"])==0;'
                    'before["MainPID"]=int(before["MainPID"]);after["MainPID"]=int(after["MainPID"]);'
                    'print(json.dumps(dict(stop_returncode=r.returncode,pre=before,post=after)))'))
                expected_stops.add(unit)
                handled_recoveries.add(recovery_key)
                info['expected_stop']=True
                report['source_recovery_stops'].append(dict(round=recovery['round'],generation=info['generation'],
                    role=info['role'],unit_stopped=True,**stop_proof))
            decisions=sorted([x for x in events if x['kind']=='switch_decision'],key=lambda x:x['round'])
            for decision in decisions:
                generation=decision['round']+1
                if generation>=a.cycles:continue
                source=decision['owner'];target='C' if source=='B' else 'B';label=f'g{generation}_{target}'
                if label in actors:continue
                previous=[info for info in actors.values() if info['role']==target]
                completed=all(info.get('expected_stop',False) or sum(x['kind'] in ('retired','standby_done') for x in rows[f'g{info["generation"]}_{target}'])==4
                    for info in previous)
                if completed:
                    for info in previous:
                        old_unit=f'shift-v4-{a.suffix}-g{info["generation"]}_{target}'
                        run(info['endpoint'],'import subprocess;subprocess.run(["systemctl","stop",'+repr(old_unit)+'],capture_output=True,timeout=25)')
                    actor(generation,target,source,target)
            done=[x for x in events if x['kind']=='done']
            progress=dict(commits=sum(x['kind']=='commit' for x in events),done=len(done),decisions=len(decisions),actors=len(actors))
            if progress!=last:print(json.dumps(dict(progress=progress)),flush=True);last=progress
            if len(done)==16:
                if len(decisions)!=a.cycles:raise RuntimeError('missing switch decision')
                if len({(x['replica'],x['rank']) for x in done})!=16 or len({x['step'] for x in done})!=1:
                    raise RuntimeError('incomplete logical final coverage')
                if not all(x['identity_preserved'] for x in done):raise RuntimeError('retained model identity changed')
                # A retained candidate may still be draining a late state
                # receive after the decision deadline. Do not truncate its
                # transport/standby proof when active training finishes first.
                pending=unfinished_actors(actors,rows)
                if pending:
                    check_units()
                    time.sleep(.1)
                    continue
                report.update(functional_pass=True,migration_completed=all(x['decision']=='replace' for x in decisions),
                    final_step=done[0]['step'],decisions=decisions,
                    oracle_exact=all(all(x['comparison']['exact'].values()) and x['comparison']['trajectory_exact'] and
                                     x['comparison']['manager_exact'] for x in done) if a.reference_run else None)
                break
            check_units()
            time.sleep(1)
        else:raise TimeoutError('bounded transparent job deadline')
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        for endpoint in eps.values():report['error']=report['error'].replace(endpoint.ip,'<private>')
    finally:
        cleanup=[]
        for endpoint,unit in reversed(units):
            try:
                run(endpoint,'import subprocess;subprocess.run(["systemctl","stop",'+repr(unit)+'],capture_output=True,timeout=25);'
                    'r=subprocess.run(["systemctl","show",'+repr(unit)+',"--property=ActiveState","--value"],capture_output=True,text=True);'
                    'assert r.stdout.strip() not in ("active","activating","deactivating"),r.stdout')
            except Exception as error:cleanup.append(unit+': '+type(error).__name__)
        try:
            rows=collect();after=resources();(out/'resources_after.json').write_text(json.dumps(after,indent=2)+'\n')
            report['resources_clean']=before is not None and all(before[n]['company']==after[n]['company'] and not after[n]['extra_gpu'] for n in eps)
            for label,info in actors.items():
                code='from pathlib import Path;import json;p=Path('+repr(str(out))+');print(json.dumps({f.name:f.read_text(errors="replace")[-10000:] for f in p.glob('+repr(label+'*.log')+')}))'
                logs=json.loads(session.ssh(info['endpoint'],['python3','-c',code]))
                for filename,content in logs.items():(out/filename).write_text(content)
        except Exception as error:cleanup.append('collection: '+type(error).__name__)
        report['cleanup_errors']=cleanup
        report['actors']=[{k:v for k,v in info.items() if k!='endpoint'} for info in actors.values()]
        save();print(json.dumps({k:report.get(k) for k in ('name','functional_pass','oracle_exact','final_step','error','resources_clean','cleanup_errors')}),flush=True)
    if report['error'] or not report.get('resources_clean') or cleanup:raise SystemExit(1)
