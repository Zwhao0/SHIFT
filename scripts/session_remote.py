"""One bounded persistent session controller; accepts live strategy commands."""
import argparse
from dataclasses import asdict
import fcntl
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import time

from scripts import v1_acceptance_remote as remote
from shift_full.control import endpoints_from_env,server_password
from shift_full.session_control import Command

ROOT=Path('/workspace/SHIFT')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--audit',action='store_true')
    p.add_argument('--initial-steps',type=int,default=0,help='bounded reference run; zero accepts live commands')
    a=p.parse_args()
    if not a.session or not all(c.isalnum() or c in '_-' for c in a.session):p.error('invalid session')
    import torch.distributed as dist
    name='session_'+a.session;out=ROOT/'results'/name;out.mkdir(exist_ok=False)
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    s=remote.ServerRotation.__new__(remote.ServerRotation);s.root=ROOT;s._sudo_password=remote.PASSWORD
    eps=endpoints_from_env();units=[];actors={};offsets={};rows={};expected_stops=set();store=None;reserved_lights={};started_lights=set()
    report=dict(name=name,controller_pid=os.getpid(),session=a.session,epoch=0,command_count=0,commands=[],done=False,error=None,audit=a.audit,resources_clean=False)
    requests=ROOT/'runtime/v3-reproduce'/name/'requests';requests.mkdir(parents=True,exist_ok=True)
    rootcfg=ROOT/'runtime/v3-reproduce'/(name+'.json')
    def ssh(node,code):return s.ssh(eps[node],['sudo','-S','-p','','python3','-c',code],sudo_input=True)
    def save():
        temp=out/'summary.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'summary.json')
    def resources():
        result={}
        for node,e in eps.items():
            entries=[x.split(',',1) for x in s.ssh(e,['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).splitlines() if x.strip()]
            result[node]=dict(company={pid.strip():proc.strip() for pid,proc in entries if 'VLLM::' in proc},extra_gpu=[pid.strip() for pid,proc in entries if 'VLLM::' not in proc])
        return result
    def launch(node,label,command,devices=''):
        unit='shift-session-'+a.session+'-'+label;units.append((node,unit))
        env=dict(PYTHONPATH=str(ROOT)+':/workspace/FLASH1022',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MALLOC_ARENA_MAX='2',TOKIO_WORKER_THREADS='2',
            CPATH=str(ROOT/'runtime/python-headers'),CUDA_VISIBLE_DEVICES=devices,
            CUDA_DEVICE_MAX_CONNECTIONS='1',NCCL_MAX_CTAS='4',NCCL_BUFFSIZE='1048576',NCCL_CUMEM_HOST_ENABLE='0',NCCL_CUMEM_ENABLE='0',
            NCCL_NVLS_ENABLE='0',NCCL_DEBUG='WARN',NCCL_NET='IB',NCCL_IB_DISABLE='0',NCCL_SOCKET_IFNAME='bond0',
            GLOO_SOCKET_IFNAME='bond0',NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1',NCCL_IB_GID_INDEX='3',
            NCCL_LAUNCH_ORDER_IMPLICIT='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='0',TORCH_NCCL_USE_COMM_NONBLOCKING='0',
            TORCHFT_USE_OTEL='false',SHIFT_DIGEST_ENGINE='staged',
            TMPDIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),XDG_CACHE_HOME=str(ROOT/'runtime/v1-recheck-bootstrap/cache'),
            TRITON_CACHE_DIR=str(ROOT/'runtime/v1-recheck-bootstrap/cache/triton'),NO_PROXY='*',no_proxy='*')
        args=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+str(a.seconds),
            '--property=KillMode=control-group','--property=TimeoutStopSec=10','--working-directory='+str(ROOT),
            '--property=StandardOutput=append:'+str(out/(label+'.host.log')),'--property=StandardError=inherit',
            *['--setenv='+k+'='+v for k,v in env.items()],*command]
        s.ssh(eps[node],['sudo','-S','-p','',*args],sudo_input=True)
    def lighthouse(service,*,planned_round=None):
        port=base['lighthouse_port']+service if planned_round is None else 31410+20*planned_round
        if port in started_lights:return
        if planned_round is None and service in reserved_lights:reserved_lights.pop(service).close()
        launch('node0',f'light-{port}',[str(ROOT/'runtime/torch214/bin/python'),str(ROOT/'vendor/bin/torchft_lighthouse'),
            '--bind',f'[::]:{port}','--min_replicas','1' if planned_round is None else '2',
            '--heartbeat_timeout_ms','500','--join_timeout_ms','200','--quorum_tick_ms','50'])
        until=time.monotonic()+30
        while time.monotonic()<until:
            with socket.socket() as probe:
                if probe.connect_ex((eps['node0'].ip,port))==0:started_lights.add(port);return
            time.sleep(.1)
        raise TimeoutError('session Lighthouse did not become reachable')
    def actor(label,node,role,replica,devices,*,generation=0,command=None):
        cfg=dict(base,actor_generation=generation,session_replica=replica,actor_devices=len(devices),
                 session_candidate=command is not None,candidate_full_dp=len(devices)==8,
                 initial_round=report['epoch'],session_epoch=report['epoch'],manager_service=state['service'],manager_store_services=state['stores'],
                 channel_epochs=state['channels'],replica_hosts={k:eps[v].ip for k,v in state['replica_nodes'].items()},
                 partial_owner=state['partial_owner'],role=role)
        if command:cfg.update(session_command=command,command_sequence=command['sequence'])
        path=ROOT/'runtime/v3-reproduce'/(name+'_'+label+'.json')
        ssh(node,'from pathlib import Path;Path('+repr(str(path))+').write_text('+repr(json.dumps(cfg))+')')
        actors[label]=dict(node=node,role=role,replica=replica,devices=devices,generation=generation,unit='shift-session-'+a.session+'-'+label)
        rows[label]=[]
        launch(node,label,[str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.v4_train','--config',str(path),'--role','host','--host-role',role],','.join(map(str,devices)))
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
    def keys(kind,sequence):return [f'session/{kind}/{sequence}/{r}/{i}' for r in range(2) for i in range(8)]
    def occupied(excluding=None):
        used={node:set() for node in eps}
        for logical,(label,device) in active.items():
            if excluding is not None and logical[0]==excluding:continue
            info=actors[label];used[info['node']].add(info['devices'][device])
        return used
    def terminal(label):
        return {e['device'] for e in rows[label] if e['kind'] in ('session_done','session_retired','retired','standby_done')}==set(range(len(actors[label]['devices'])))
    def check():
        for label,info in actors.items():
            if info['unit'] in expected_stops or terminal(label):continue
            values=dict(x.split('=',1) for x in s.ssh(eps[info['node']],['systemctl','show',info['unit'],'--property=ActiveState','--property=ExecMainStatus']).splitlines() if '=' in x)
            if values.get('ActiveState') not in ('active','activating','deactivating') or values.get('ExecMainStatus') not in (None,'0'):
                raise RuntimeError('session actor exited without proof: '+label)
    before=None;active={};current=None;seen={};state=dict(service=0,stores={'0':0,'1':0},channels=[0]*8,replica_nodes={'0':'node0','1':'node1'},partial_owner='B')
    base=dict(directory=str(out/'workers'),flash='/workspace/FLASH1022',hosts={r:eps[n].ip for r,n in dict(A='node0',B='node0',C='node3',D='node1',E='node1').items()},
        node_hosts={n:e.ip for n,e in eps.items()},port=28000,timeout=120,communication_timeout=2,seconds=a.seconds,
        plan=dict(cut=2,post_steps=5,seq_len=512,microbatches=2),seed=2026,unified=True,data='random',
        unified_session=True,session=a.session,session_services=16,outer_manager=True,transactional_switch=True,
        retain_outer_managers=True,overlap_peer=True,oracle=False,reference_only=False,
        performance_no_audit=not a.audit,boundary_validation='full' if a.audit else 'runtime',
        gpu_prepare=True,gpu_warmup=True,reuse_local_groups=True,preconnect_state=True,defer_source_local_close=True,
        slow_ms=0,cycles=1,between_steps=5,restore_deadline=10,source_deadline=2,affected_ranks=[4,5,6,7])
    report['configuration']=dict(batch=8,seq_len=512,tp=2,pp=2,ep=2,seed=2026,memory_fraction=.12)
    from shift_full.v4_reference import training_sources
    report['source_sha256']=training_sources(ROOT)
    for file in ['shift_full/session_engine.py','shift_full/session_outer.py','shift_full/session_control.py','shift_full/training_loop.py','scripts/session_remote.py','scripts/sessionctl.py']:
        report['source_sha256'][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    save()
    try:
        for e in eps.values():s.ensure_connection(e)
        before=resources();(out/'resources_before.json').write_text(json.dumps(before))
        if any(len(x['company'])!=8 or x['extra_gpu'] for x in before.values()):raise RuntimeError('session requires idle SHIFT resources')
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
        lighthouse(0)
        for role,node,replica,devices in [('A','node0',0,list(range(4))),('B','node0',0,list(range(4,8))),('D','node1',1,list(range(4))),('E','node1',1,list(range(4,8)))]:
            label='g0_'+role;actor(label,node,role,replica,devices)
            for d in range(4):active[replica,d+(4 if role in 'BE' else 0)]=(label,d)
        deadline=time.monotonic()+a.seconds-30
        ready=False;ref_stopped=False
        while time.monotonic()<deadline:
            collect();check()
            if not ready and store.check([f'session/ready/{r}/{i}' for r in range(2) for i in range(8)]):
                store.set('session/start','1');ready=True;report['ready']=True;save()
            progress={r:int(store.get(f'session/progress/{r}')) if store.check([f'session/progress/{r}']) else 0 for r in (0,1)}
            report['progress']=progress
            if current:
                seq=current['sequence'];cmd=current['command']
                if cmd['policy']=='v1' and not current.get('killed') and store.check(keys('arrival',seq)):
                    targets={active[1,i][0] for i in range(8)}
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
                        current['fault_proof'].append(json.loads(ssh(info['node'],code)))
                        expected_stops.add(info['unit'])
                    store.set(f'session/fault_released/{seq}','1')
                    state['stores']['1']=seq
                    label=f'g{seq}_D';actor(label,cmd['target'],'D',1,current['devices'],generation=seq,command=current)
                    dist.PrefixStore(f'round/{report["epoch"]}/',store).set('prepare','1')
                    current.update(killed=True,target_actor=label,fault_kind='SIGKILL');save()
                if current.get('target_actor') and 'first_common_observation' not in current:
                    wanted=dict(active)
                    target_label=current['target_actor']
                    if cmd['policy'] in ('v1','v2'):
                        for i in range(8):wanted[1,i]=(target_label,i)
                    else:
                        for i in range(4):wanted[0,4+i]=(target_label,i)
                    by_step={}
                    for (replica,rank),(label,device) in wanted.items():
                        for e in rows[label]:
                            if (e['kind']=='session_commit' and e['device']==device and
                                e.get('command_id')==cmd['id'] and e.get('participants')==2 and e.get('contributed')):
                                by_step.setdefault(e['step'],set()).add((replica,rank))
                    common=sorted(step for step,workers in by_step.items() if len(workers)==16)
                    if common:
                        observed=time.monotonic()
                        current['first_common_observation']=dict(step=common[0],observed_monotonic=observed,
                            request_seconds=observed-current['requested_monotonic'],
                            fault_seconds=observed-current['fault_monotonic'] if current.get('fault_monotonic') else None)
                        save()
                if store.check(keys('done',seq)):
                    done_steps={int(store.get(k)) for k in keys('done',seq)}
                    if len(done_steps)!=1:raise RuntimeError('strategy completion is not a common boundary')
                    if cmd['policy'] in ('v1','v2'):
                        for i in range(8):active[1,i]=(current['target_actor'],i)
                        state['replica_nodes']['1']=cmd['target']
                        if cmd['policy']=='v2':state.update(service=seq,stores={'0':seq,'1':seq},channels=[report['epoch']+1]*8)
                    else:
                        for i in range(4):active[0,4+i]=(current['target_actor'],i)
                        state['partial_owner']='C' if state['partial_owner']=='B' else 'B'
                        state['channels'][4:8]=[report['epoch']+1]*4
                    current.update(status='completed',completed_step=done_steps.pop(),controller_seconds=time.monotonic()-current['requested_monotonic'])
                    report['epoch']+=1;report['commands'].append(current);current=None;report['active_command']=None;save()
            if ready and current is None:
                if a.initial_steps and min(progress.values())>=a.initial_steps and not ref_stopped:
                    payload=dict(schema=1,session=a.session,id='reference-stop',expected_epoch=report['epoch'],action='stop',at_step=max(progress.values())+4)
                    (requests/'reference-stop.json').write_text(json.dumps(payload));ref_stopped=True
                for path in sorted(requests.glob('*.json')):
                    payload=json.loads(path.read_text());identifier=payload.get('id');canonical=json.dumps(payload,sort_keys=True)
                    if identifier in seen:
                        if seen[identifier]!=canonical:
                            report.setdefault('rejected_commands',[]).append(dict(id=identifier,error='live command id payload conflict'));save()
                        continue
                    try:
                        cmd=Command.from_dict(payload)
                        if cmd.session!=a.session or cmd.expected_epoch!=report['epoch']:raise ValueError('stale live command')
                        if cmd.action!='stop':
                            if report['command_count']+1>=base['session_services']:raise ValueError('session command budget reached; stop remains available')
                            if cmd.target not in eps:raise ValueError('unknown target node')
                            if (cmd.policy in ('v1','v2') and cmd.source_replica!=1) or (cmd.policy=='v4' and (cmd.source_replica!=0 or tuple(cmd.ranks)!=(4,5,6,7))):raise ValueError('unsupported session source/scope')
                            used=occupied(excluding=1 if cmd.policy=='v1' else None)
                            if len(set(range(8))-used[cmd.target])<(4 if cmd.policy=='v4' else 8):raise ValueError('target does not have enough free stage slots')
                    except (ValueError,KeyError) as error:
                        seen[identifier]=canonical
                        report.setdefault('rejected_commands',[]).append(dict(id=identifier,error=str(error)))
                        save();continue
                    seq=report['command_count']+1
                    # The controller admits commands before any candidate is launched.
                    payload=asdict(cmd);payload['ranks']=list(cmd.ranks) if cmd.ranks else None
                    if cmd.action=='stop':
                        payload={k:v for k,v in payload.items() if k not in ('policy','source_replica','target','ranks')}
                    devices=[]
                    if cmd.action!='stop':
                        if (cmd.policy in ('v1','v2') and cmd.source_replica!=1) or (cmd.policy=='v4' and (cmd.source_replica!=0 or tuple(cmd.ranks)!=(4,5,6,7))):raise ValueError('unsupported session source/scope')
                        used=occupied(excluding=1 if cmd.policy=='v1' else None)
                        devices=[i for i in range(8) if i not in used[cmd.target]][:4 if cmd.policy=='v4' else 8]
                        if len(devices)!=(4 if cmd.policy=='v4' else 8):raise ValueError('target does not have enough free stage slots')
                    record=dict(sequence=seq,command=payload,devices=devices,requested_monotonic=time.monotonic(),status='running')
                    if cmd.policy=='v2':lighthouse(seq);lighthouse(seq,planned_round=seq-1)
                    # Choose the boundary after service preparation and publish before
                    # remote actor launch: slow control RPCs cannot consume the cut.
                    latest=max(int(store.get(f'session/progress/{r}')) if store.check([f'session/progress/{r}']) else 0 for r in (0,1))
                    if payload['at_step'] is None:payload['at_step']=latest+4
                    elif payload['at_step']<=latest:
                        seen[identifier]=canonical
                        report.setdefault('rejected_commands',[]).append(dict(id=identifier,error='requested boundary already passed'));save();continue
                    if cmd.policy in ('v2','v4'):
                        role='D' if cmd.policy=='v2' else ('C' if state['partial_owner']=='B' else 'B')
                        label=f'g{seq}_{role}';record['target_actor']=label
                    store.set(f'session/command/{seq}',json.dumps(record));store.set('session/command_count',str(seq))
                    if cmd.policy in ('v2','v4'):
                        dist.PrefixStore(f'round/{report["epoch"]}/',store).set('prepare','1')
                        actor(label,cmd.target,role,1 if cmd.policy=='v2' else 0,devices,generation=seq,command=record)
                    report['command_count']=seq;seen[identifier]=canonical
                    if cmd.action!='stop':current=record;report['active_command']=record
                    save();break
            finished=[e for es in rows.values() for e in es if e['kind']=='session_done']
            if len(finished)==16:
                if current is not None:raise RuntimeError('session stopped with an unfinished command')
                if len({(e['replica'],e['rank']) for e in finished})!=16 or len({e['step'] for e in finished})!=1:raise RuntimeError('incomplete final session coverage')
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
        for listener in reserved_lights.values():listener.close()
        for node,unit in reversed(units):
            try:ssh(node,'import subprocess;subprocess.run(["systemctl","stop",'+repr(unit)+'],capture_output=True,timeout=20);x=subprocess.run(["systemctl","show",'+repr(unit)+',"--property=ActiveState","--value"],capture_output=True,text=True);assert x.stdout.strip() not in ("active","activating","deactivating")')
            except BaseException as error:cleanup.append(unit+': '+type(error).__name__)
        try:
            collect();after=resources();(out/'resources_after.json').write_text(json.dumps(after))
            report['resources_clean']=before is not None and all(before[n]['company']==after[n]['company'] and not after[n]['extra_gpu'] for n in eps)
        except BaseException as error:cleanup.append('collection: '+type(error).__name__)
        report.update(cleanup_errors=cleanup,actors=actors,active={f'{r}/{i}':list(v) for (r,i),v in active.items()})
        save();print(json.dumps({k:report.get(k) for k in ('name','done','error','resources_clean','cleanup_errors')}),flush=True)
    if report['error'] or not report['done'] or not report['resources_clean'] or cleanup:raise SystemExit(1)


if __name__=='__main__':main()
