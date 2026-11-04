"""Server-owned planned replacement with three explicit physical bindings.

All workers and epoch services have bounded cgroups. This controller never
injects fail-stop faults or touches any unowned process.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess as sp
import tarfile
import tempfile
import time

from scripts import v1_acceptance_remote as remote
from scripts.v3_benchmark_remote import ServerOwnedRotation, SCHEMAS
from shift_full.control import CONTAINER_ROOT, server_password
from shift_full.event_reader import remote_code, merge

ROOT=Path(__file__).resolve().parents[1]


def complete(rows):
    return len(rows)==8 and {r['rank'] for r in rows}==set(range(8))


class PlannedRotation(ServerOwnedRotation):
    def __init__(self,name,config,seconds,nodes,baseline):
        super().__init__(name,config,seconds,20261106,backup_mode='cold')
        if len(nodes)!=3 or set(nodes)!=set(self.endpoints):raise ValueError('three distinct physical nodes required')
        self.roles={r:self.endpoints[n] for r,n in zip('ABC',nodes)}
        self.role_bindings={'A':(0,0),'B':(1,0),'C':(1,1)}
        self.bindings={self.role_bindings[r]:e for r,e in self.roles.items()}
        self.baseline=baseline;self.role_started=set();self.control_started=False
        self.config.update(master_addr=self.roles['A'].ip,lighthouse_ip=self.roles['A'].ip,external_lighthouse=True)
        self.config['train_args'] += ['--planned-hosts',json.dumps({r:e.ip for r,e in self.roles.items()})]
        (self.out/'config.json').write_text(json.dumps(self.config,indent=2)+'\n')
        (self.out/'role_map.json').write_text(json.dumps({r:e.physical_node for r,e in self.roles.items()},indent=2)+'\n')

    def fault(self,*args,**kwargs):raise RuntimeError('fail-stop injection forbidden in planned controller')

    def rows_role(self,data,role,kind,round_index=None,step=None):
        found=[e for n,rs in data.get(role,{}).items() if n.startswith('events_') for e in rs if e['kind']==kind]
        if round_index is not None:found=[e for e in found if e.get('planned_round')==round_index]
        if step is not None:found=[e for e in found if e.get('step')==step]
        return found

    def data(self,full=False):
        if full:
            return {r:json.loads(self.execute_endpoint(self.roles[r],['python','-c',remote_code(self.directory(*self.role_bindings[r]),self.role_bindings[r][1],{})]))['rows'] for r in self.role_started}
        return {r:self.read_binding(*self.role_bindings[r]) for r in self.role_started}

    def wait(self,label,predicate,timeout=180):
        end=min(self.deadline,time.monotonic()+timeout)
        while time.monotonic()<end:
            d=self.data()
            for r,files in d.items():
                bad=[e for e in files.get('supervisor.jsonl',[]) if e['kind'] in ('unexpected_exit','supervisor_error')]
                if bad:raise RuntimeError(r+': '+repr(bad[-1]))
                errors=self.rows_role(d,r,'planned_prepare_error')
                if errors:raise RuntimeError(r+': '+repr(errors[-1]))
            if predicate(d):self.event(label,monotonic=time.monotonic());return d
            time.sleep(.2)
        raise TimeoutError(label)

    def mark_role(self,role,key,text=''):
        rid,g=self.role_bindings[role]
        self.write_marker(self.roles[role],self.directory(rid,g),key,text)

    def set_key(self,key,value):
        code='from datetime import timedelta;import torch.distributed as d;s=d.TCPStore('+repr(self.roles['A'].ip)+',31402,None,False,timeout=timedelta(seconds=15));s.set('+repr(key)+','+repr(str(value))+')'
        self.execute_endpoint(self.roles['A'],[self.config['python'],'-c',code])

    def resource_check(self,phase):
        result={}
        for n,e in self.endpoints.items():
            expected={x.split(',')[0].strip() for x in self.baseline['nodes'][n]['gpu'] if 'VLLM::' in x}
            if len(expected)!=8:raise RuntimeError('expected eight company baseline PIDs')
            rows=self.ssh(e,['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).splitlines()
            company={x.split(',')[0].strip() for x in rows if 'VLLM::' in x}
            current={x.split(',')[0].strip() for x in rows}
            result[n]=dict(gpu=rows,company_unchanged=expected<=company,extra_gpu=sorted(current-expected))
        (self.out/f'resources_{phase}.json').write_text(json.dumps(result,indent=2)+'\n')
        if any(not x['company_unchanged'] or x['extra_gpu'] for x in result.values()):raise RuntimeError('company PID change or unknown GPU load')

    def start_services(self,cycles):
        for e in self.endpoints.values():self.ensure_connection(e)
        self.resource_check('before')
        if '--reference-anchor' in self.config['train_args']:
            anchor=self.config['train_args'][self.config['train_args'].index('--reference-anchor')+1]
            code='from pathlib import Path;p=Path('+repr(anchor)+');assert all((p/f"anchor_r{r}.pt").is_file() for r in range(8)),"complete existing anchor required on active host"'
            for role in 'AB':self.execute_endpoint(self.roles[role],['python','-c',code])
        e=self.roles['A'];path=self.write_config(e)
        unit='shift-v3-'+self.name+'-epochs'
        self.units.append((e,unit));self.control_started=True
        args=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+str(self.seconds+30),
              '--property=KillMode=control-group','--property=TimeoutStopSec=15','--working-directory='+CONTAINER_ROOT,
              '--setenv=PYTHONPATH='+CONTAINER_ROOT,'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=TOKIO_WORKER_THREADS=2',
              '--setenv=TORCHFT_USE_OTEL=false','/usr/bin/python3','-m','scripts.v3_planned_lights',
              '--config',path,'--directory',CONTAINER_ROOT+'/results/'+self.name+'/epochs','--cycles',str(cycles),'--seconds',str(self.seconds)]
        self.ssh(e,['sudo','-S','-p','',*args],sudo_input=True)
        end=time.monotonic()+20
        while time.monotonic()<end:
            code='from pathlib import Path;print(Path('+repr(CONTAINER_ROOT+'/results/'+self.name+'/epochs/owner.json')+').exists())'
            if self.execute_endpoint(e,['python','-c',code]).strip()=='True':return
            time.sleep(.2)
        raise TimeoutError('epoch services startup')

    def launch_role(self,role):
        rid,g=self.role_bindings[role];old=self.config
        train_args=list(old['train_args'])
        if role=='C' and '--reference-anchor' in train_args:
            index=train_args.index('--reference-anchor');del train_args[index:index+2]
        self.config=dict(old,train_args=train_args+['--planned-role',role])
        # Record before launch: lost SSH acknowledgments must still be cleaned.
        self.role_started.add(role);self.started_endpoints[self.roles[role].physical_node]=self.roles[role]
        try:self.start_supervisor(rid,self.roles[role],g)
        finally:self.config=old

    def finish(self):
        errors=[]
        def record(label,exc):
            errors.append(dict(action=label,error=type(exc).__name__+': '+str(exc)))
        try:
            for r in self.role_started:
                try:self.mark_role(r,'STOP')
                except Exception as exc:record('stop '+r,exc)
            until=time.monotonic()+20
            while time.monotonic()<until and self.role_started:
                try:
                    d=self.data()
                    if all(any(e['kind']=='cleanup' for e in d[r].get('supervisor.jsonl',[])) for r in self.role_started):break
                except Exception as exc:record('graceful cleanup read',exc);break
                time.sleep(.3)
            if self.control_started:
                try:self.write_marker(self.roles['A'],CONTAINER_ROOT+'/results/'+self.name+'/epochs','STOP')
                except Exception as exc:record('epoch stop',exc)
            # Exact cgroups are stopped even if all graceful markers fail.
            for e,u in reversed(self.units):
                try:
                    code='import subprocess,json;r=subprocess.run(["systemctl","stop",'+repr(u)+'],capture_output=True,timeout=25);s=subprocess.check_output(["systemctl","show",'+repr(u)+',"--property=ActiveState","--value"],text=True).strip();assert s not in ("active","activating","deactivating"),s;print(s)'
                    self.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True)
                except Exception as exc:record('unit '+u,exc)
            for e,u in self.units:
                try:
                    code='from pathlib import Path;import subprocess;p=Path('+repr(CONTAINER_ROOT+'/results/'+self.name)+');p.mkdir(parents=True,exist_ok=True);r=subprocess.run(["journalctl","--unit",'+repr(u)+',"--no-pager","--output=short-precise"],capture_output=True,text=True,timeout=25);(p/'+repr(u+'.journal')+').write_text(r.stdout+r.stderr);assert r.returncode==0'
                    self.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True)
                except Exception as exc:record('journal '+u,exc)
            (self.out/'physical_manifest.json').write_text(json.dumps([dict(role=r,rid=self.role_bindings[r][0],generation=self.role_bindings[r][1],physical_node=e.physical_node,directory=self.directory(*self.role_bindings[r])) for r,e in self.roles.items()],indent=2)+'\n')
            collect=dict(self.started_endpoints)
            if self.control_started:collect[self.roles['A'].physical_node]=self.roles['A']
            for n,e in collect.items():
                try:
                    archive=e.host_root+'/results/'+self.name+'-'+n+'.tgz'
                    self.execute_endpoint(e,['tar','-czf',archive,'--exclude=*.pt','-C',CONTAINER_ROOT+'/results/'+self.name,'.'])
                    local=self.out/(n+'.tgz')
                    sp.run(['scp','-q','-o','ControlPath='+str(self.root/'tmp'/e.socket),'test@'+e.ip+':'+archive,str(local)],check=True,timeout=90)
                    dest=self.out/('physical_'+n);dest.mkdir(exist_ok=True)
                    with tarfile.open(local) as t:t.extractall(dest,filter='data')
                    (self.out/(n+'.tgz.sha256')).write_text(hashlib.sha256(local.read_bytes()).hexdigest()+'\n')
                except Exception as exc:record('collect '+n,exc)
            try:self.resource_check('after')
            except Exception as exc:record('resource after',exc)
        finally:
            (self.out/'unit_cleanup.json').write_text(json.dumps(dict(units=[u for _,u in self.units],errors=errors),indent=2)+'\n')
            self.event('collected',cleanup_errors=len(errors));self.log.close()
            for e in self.endpoints.values():
                sp.run(['ssh','-S',str(self.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)
        if errors:raise RuntimeError('planned cleanup/collection failed: '+repr(errors))


def config(variant,cycles,scenario,slow_ms,anchor=False,shadow=False,ownership=False):
    c=json.loads((ROOT/'configs/v1_bench_shift.json').read_text())
    a=c['train_args'];a[a.index('--communication-timeout')+1]='30'
    a[a.index('--max-seconds')+1]='1600'
    a += ['--audit-state','--hash-every','1','--sync-quorum','--fixed-batch',
          '--planned-variant',variant,'--planned-source','B','--planned-cycles',str(cycles),
          '--planned-inject',scenario,'--planned-slow-ms',str(slow_ms)]
    if anchor:
        a.remove('--warmup-before-join')
        a += ['--reference-signatures','--audit-routing','--reference-anchor',str(ROOT/'runtime/v3-reference-num01'),'--reference-anchor-step','1']
    if shadow:a+=['--planned-shadow-recompute']
    if ownership:a+=['--planned-ownership-recompute']
    c.update(entry_module='shift_full.planned_train',target_steps=11 if anchor else 100000,lighthouse_port=31400,master_port=31500,
             min_replicas=2,heartbeat_timeout_ms=1000,join_timeout_ms=1000,quorum_tick_ms=100,recycle_after_worker_fault=False,
             fast_launch=False,persistent_cpu_launcher=False,single_gpu_visibility=False,audit_initial=True)
    c['env'].update(TORCH_NCCL_NONBLOCKING_TIMEOUT='60',CUDA_DEVICE_MAX_CONNECTIONS='1')
    return c


def audit_cycle(s,d,r,source,target,cut,goal):
    restored=s.rows_role(d,target,'restored',r,cut)
    if not complete(restored) or any(not e['hashes_match'] or e['optimizer_tensors']<=0 or set(e['actual_hashes'])!=SCHEMAS or e['actual_hashes']!=e['expected_hashes'] for e in restored):raise RuntimeError('incomplete full-state restoration')
    if '--planned-ownership-recompute' in s.config['train_args']:
        audit_ownership(s,d,cut)
    else:
        sent=s.rows_role(d,source,'planned_gradient_sent',r,cut);recv=s.rows_role(d,target,'planned_gradient_received',r,cut)
        if not complete(sent) or not complete(recv):raise RuntimeError('incomplete relay')
        for rank in range(8):
            x=next(e for e in sent if e['rank']==rank);y=next(e for e in recv if e['rank']==rank)
            if not (x['gradient_hash']==y['gradient_hash'] and x['sample_ids']==y['sample_ids'] and x['tail_hash']==y['applied_tail_hash'] and y['forward_calls']==0):raise RuntimeError('gradient/tail/sample relay mismatch')
    for step in range(cut+1,goal+1):
        aa=s.rows_role(d,'A','commit',r,step);bb=s.rows_role(d,target,'commit',r,step)
        if not complete(aa) or not complete(bb):raise RuntimeError('missing joint step')
        for rank in range(8):
            x=next(e for e in aa if e['rank']==rank);y=next(e for e in bb if e['rank']==rank)
            if any(e['participants']!=2 or not e['contributed'] or not e['finite_gradients'] or e['global_batch']!=8 or e['epoch']!=r+1 or set(e['hashes'])!=SCHEMAS for e in (x,y)):raise RuntimeError('invalid joint commit')
            if any(x['hashes'][k]!=y['hashes'][k] for k in ('model','optimizer','scheduler')):raise RuntimeError('shared state mismatch')


def audit_ownership(s,d,cut):
    def rows(role,kind):
        value=s.rows_role(d,role,kind,0,cut)
        if not complete(value):raise RuntimeError('ownership incomplete: '+role+'/'+kind)
        return {e['rank']:e for e in value}
    reserve=rows('B','planned_ownership_reserved')
    computed=rows('C','planned_ownership_computed')
    initial=rows('C','planned_ownership_initial_audit')
    late=rows('B','planned_ownership_late_exact')
    for role in 'AC':
        if any(e['members']!='AC' for e in rows(role,'planned_ownership_activated').values()):
            raise RuntimeError('ownership wrong activation')
    rows('B','planned_ownership_b_closed')
    for rank in range(8):
        if not initial[rank]['hashes_match'] or not initial[rank]['before_forward'] or set(initial[rank]['hashes'])!=SCHEMAS:
            raise RuntimeError('ownership initial audit missing/mismatch')
        if initial[rank]['monotonic']>=computed[rank]['monotonic']:
            raise RuntimeError('ownership initial audit after recompute')
        if reserve[rank]['descriptor']!=computed[rank]['descriptor'] or not late[rank]['exact']:
            raise RuntimeError('ownership descriptor/late full contribution mismatch')
        for key in ('sample_ids','gradient_hash','parameter_gradient_hash','tail_hash'):
            if late[rank][key]!=computed[rank][key]:raise RuntimeError('ownership '+key+' mismatch')
        def routes(role):
            groups={}
            for e in s.rows_role(d,role,'routing',0,cut):
                if e['rank']!=rank:continue
                if e.get('shadow'):raise RuntimeError('production ownership routing marked shadow')
                groups.setdefault(e['layer'],[]).append({k:e[k] for k in
                    ('topk_hash','sorted_topk_hash','min_boundary_margin','zero_margins','near_ties_1e6','tokens')})
            return groups
        if not routes('B') or routes('B')!=routes('C'):
            raise RuntimeError('ownership complete routing sequence mismatch')
    return dict(roles=8,late_full_contribution_exact=True,performance_pass=False)


def audit_shadow(s,d,cut):
    kinds=('planned_shadow_computed','planned_shadow_rollback','planned_shadow_compared')
    rows={k:s.rows_role(d,'C',k,0,cut) for k in kinds}
    descriptors=s.rows_role(d,'B','planned_shadow_descriptor',0,cut)
    if not complete(descriptors) or any(not complete(v) for v in rows.values()):
        raise RuntimeError('incomplete eight-role shadow probe')
    for rank in range(8):
        computed=next(e for e in rows['planned_shadow_computed'] if e['rank']==rank)
        rollback=next(e for e in rows['planned_shadow_rollback'] if e['rank']==rank)
        compared=next(e for e in rows['planned_shadow_compared'] if e['rank']==rank)
        descriptor=next(e['descriptor'] for e in descriptors if e['rank']==rank)
        if (not compared['exact'] or not rollback['hashes_match'] or not rollback['ddp_clean']
                or computed['descriptor']!=descriptor or computed['owner_thread']!=rollback['owner_thread']):
            raise RuntimeError('shadow correctness/owner/rollback mismatch')
        def routes(role,shadow):
            result={}
            for e in s.rows_role(d,role,'routing',0,cut):
                if e['rank']!=rank or bool(e.get('shadow'))!=shadow:continue
                result.setdefault(e['layer'],[]).append({k:e[k] for k in
                    ('topk_hash','sorted_topk_hash','min_boundary_margin','zero_margins','near_ties_1e6','tokens')})
            return result
        source=routes('B',False);target=routes('C',True)
        if not source or source!=target:raise RuntimeError('shadow routing mismatch')
    return dict(all_roles_exact=True,roles=8,production_contribution='original B relay',performance_evidence=False)


def audit_lifecycle(s,d):
    counts={};by_role={}
    for role in s.role_started:
        commits=s.rows_role(d,role,'commit');by_role[role]={}
        if s.rows_role(d,role,'skipped'):raise RuntimeError('planned run skipped an update')
        for rank in range(8):
            rows=[e for e in commits if e['rank']==rank]
            steps=[e['step'] for e in rows]
            if steps!=sorted(set(steps)):raise RuntimeError('duplicate/backward commit')
            if len({e['pid'] for e in rows})>1:raise RuntimeError('worker PID changed')
            ready=[e for e in s.rows_role(d,role,'ready') if e['rank']==rank]
            if ready and rows and ready[0]['pid']!=rows[-1]['pid']:raise RuntimeError('healthy PID changed')
            for e in rows:
                if set(e['hashes'])!=SCHEMAS or not e['finite_gradients'] or not e['contributed'] or e['participants']!=2 or e['global_batch']!=8:raise RuntimeError('invalid full-state joint update')
                by_role[role][rank,e['epoch'],e['step']]=e
        counts[role]=len(commits)
    for key,a in by_role['A'].items():
        peers=[by_role[r][key] for r in ('B','C') if key in by_role.get(r,{})]
        if len(peers)!=1:raise RuntimeError('missing/ambiguous logical peer')
        b=peers[0]
        if a['sample_ids']==b['sample_ids']:raise RuntimeError('duplicate replica samples')
        if any(a['hashes'][k]!=b['hashes'][k] for k in ('model','optimizer','scheduler')):raise RuntimeError('joint shared-state mismatch')
    return dict(rank_commits=counts,healthy_pids_unchanged=True,no_duplicate_or_backward_commit=True,no_skipped_steps=True)


def run(s,variant,cycles,scenario,post,anchor,status,save,steady_steps=12):
    s.start_services(cycles)
    for role in 'AB':s.launch_role(role)
    s.wait('ab_ready',lambda d:all(complete(s.rows_role(d,r,'ready')) for r in 'AB'))
    if anchor and variant=='p2':
        s.launch_role('C')
        s.wait('numerical_target_prepared',lambda d:complete(s.rows_role(d,'C','planned_local_ready',0)),timeout=300)
        if cycles==1 and scenario=='slow-compute':
            # Select before START, while progress is still the anchor boundary.
            # Both ownership and its fresh p2 control must use the same cut.
            s.set_key('fixed_cut',6)
            status['fixed_cut']=6
    for r in 'AB':s.mark_role(r,'START')
    if variant=='steady':
        goal=11 if anchor else steady_steps
        for r in 'AB':s.mark_role(r,'TRAIN_UNTIL',str(goal))
        if scenario=='slow-compute':
            # Same worker injection as migration, without launching C or arming a cut.
            s.wait('steady_two_steps',lambda d:all(complete(s.rows_role(d,r,'commit',step=2)) for r in 'AB'))
            s.event('REQUEST',monotonic=time.monotonic());s.set_key('request','1')
            status['slow_control_no_migration']=True
        d=s.wait('steady_complete',lambda d:all(complete(s.rows_role(d,r,'commit',step=goal)) for r in 'AB'),timeout=300)
        for r in 'AB':
            commits=s.rows_role(d,r,'commit')
            if any(e['kind']=='skipped' for files in d[r].values() for e in files):raise RuntimeError('steady skipped')
            if any(set(e['hashes'])!=SCHEMAS or not e['finite_gradients'] or e['participants']!=2 for e in commits):raise RuntimeError('steady invalid state')
        status['steady_steps']=goal
        return
    s.wait('ab_two_steps',lambda d:all(complete(s.rows_role(d,r,'commit',step=2)) for r in 'AB'))
    s.event('REQUEST',monotonic=time.monotonic())
    if 'C' not in s.role_started:s.launch_role('C')
    s.set_key('round/0/request' if cycles>1 else 'request','1')
    d=s.wait('c_ready',lambda d:complete(s.rows_role(d,'C','planned_local_ready',0)),timeout=300)
    if scenario=='cancel':
        latest=max(e['step'] for r in 'AB' for e in s.rows_role(d,r,'commit'));goal=latest+8
        for r in 'AB':s.mark_role(r,'TRAIN_UNTIL',str(goal))
        s.set_key('cancel','1')
        s.wait('cancel_all',lambda d:all(complete(s.rows_role(d,r,'planned_cancelled')) for r in 'ABC'))
        d=s.wait('cancel_ab_progress',lambda d:all(complete(s.rows_role(d,r,'commit',step=goal)) for r in 'AB'))
        if s.rows_role(d,'C','commit'):raise RuntimeError('cancel target committed')
        status['cancel_pass']=True;return
    s.mark_role('C','START')
    for r in range(cycles):
        source,target=('B','C') if r%2==0 else ('C','B')
        key=lambda k:f'round/{r}/{k}' if cycles>1 else k
        if r:
            s.event('round_prepare_requested',round=r,monotonic=time.monotonic())
            s.set_key(key('request'),'1')
            s.set_key('advance',r)
            s.wait(f'r{r}_ready',lambda d:complete(s.rows_role(d,target,'planned_local_ready',r)),timeout=300)
        s.set_key(key('arm'),'1')
        d=s.wait(f'r{r}_cut',lambda d:bool(s.rows_role(d,'A','planned_cut_selected',r)))
        cut=s.rows_role(d,'A','planned_cut_selected',r)[-1]['step'];goal=cut+(post if r==cycles-1 else 3)
        if anchor:
            goal=11
            if cut+3>goal:raise RuntimeError('anchor migration cut too late for three post-steps')
        if r==cycles-1:
            for role in 'ABC':s.mark_role(role,'TRAIN_UNTIL',str(goal))
        if scenario in ('stale-state','relay-reject'):
            kind='planned_new_epoch_ab_recovered' if scenario=='relay-reject' else 'planned_cancelled'
            s.wait('rejection_all',lambda d:all(complete(s.rows_role(d,role,kind)) for role in 'AB'))
            d=s.wait('rollback_progress',lambda d:all(complete(s.rows_role(d,role,'commit',step=goal)) for role in 'AB'))
            if s.rows_role(d,'C','commit'):raise RuntimeError('rejected target committed')
            for role in 'AB':
                final=s.rows_role(d,role,'commit',step=goal)
                if any(e['epoch']!=(1 if scenario=='relay-reject' else 0) or e['physical_members']!=['A','B'] for e in final):raise RuntimeError('invalid rollback membership')
            status['rejection_pass']=True;return
        s.wait(f'r{r}_first_joint',lambda d:all(complete(s.rows_role(d,role,'commit',r,cut+1)) for role in ('A',target)),timeout=600)
        d=s.wait(f'r{r}_joint',lambda d:all(complete(s.rows_role(d,role,'commit',r,goal)) for role in ('A',target)),timeout=600)
        d=s.wait(f'r{r}_standby',lambda d:complete(s.rows_role(d,source,'planned_standby_verified',r)))
        audit_cycle(s,d,r,source,target,cut,goal)
        status['cycles'].append(dict(round=r,cut=cut,goal=goal,source=source,target=target,pass_audit=True));save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True);p.add_argument('--nodes',default='node0,node1,node3')
    p.add_argument('--baseline',required=True);p.add_argument('--variant',choices=['steady','p2'],default='p2');p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--scenario',choices=['healthy','cancel','stale-state','relay-reject','slow-compute'],default='healthy')
    p.add_argument('--slow-ms',type=float,default=0);p.add_argument('--post-steps',type=int,default=10);p.add_argument('--anchor',action='store_true')
    p.add_argument('--steady-steps',type=int,default=12)
    p.add_argument('--shadow-recompute',action='store_true')
    p.add_argument('--ownership-recompute',action='store_true')
    a=p.parse_args()
    if a.shadow_recompute and not (a.anchor and a.variant=='p2' and a.cycles==1 and a.scenario=='slow-compute'):
        p.error('shadow requires one p2 slow-compute anchor diagnostic')
    if a.ownership_recompute and (a.shadow_recompute or not (a.anchor and a.variant=='p2' and a.cycles==1 and a.scenario=='slow-compute')):
        p.error('ownership first admission requires one original-anchor p2 slow-compute cycle')
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    if not 1<=a.cycles<=6 or a.post_steps<3 or (a.scenario not in ('healthy','slow-compute') and a.cycles!=1):p.error('invalid cycle/scenario')
    if a.anchor and a.cycles!=1:p.error('anchor diagnostic uses one cycle inside the existing ten-step reference')
    if a.steady_steps<12:p.error('fresh steady probe requires at least 12 steps')
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    out=ROOT/f'results/v3_c1_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    status=dict(done=False,success=False,cycles=[],variant=a.variant,scenario=a.scenario,anchor=a.anchor,shadow_recompute=a.shadow_recompute,ownership_recompute=a.ownership_recompute,started=time.time(),nodes=a.nodes.split(','))
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();s=None
    try:
        c=config(a.variant,a.cycles,'healthy' if a.scenario=='cancel' else a.scenario,a.slow_ms,a.anchor,a.shadow_recompute,a.ownership_recompute)
        s=PlannedRotation('v3c1_'+a.suffix,c,1700,a.nodes.split(','),json.loads(Path(a.baseline).read_text()))
        try:
            run(s,a.variant,a.cycles,a.scenario,a.post_steps,a.anchor,status,save,a.steady_steps)
            full=s.data(full=True)
            if a.shadow_recompute:status['shadow']=audit_shadow(s,full,status['cycles'][0]['cut'])
            status['lifecycle']=audit_lifecycle(s,full)
            for cycle in status['cycles']:audit_cycle(s,full,cycle['round'],cycle['source'],cycle['target'],cycle['cut'],cycle['goal'])
        finally:s.finish()
        if a.anchor:
            from scripts.v3_numerical_remote import compare_compact
            with tempfile.TemporaryDirectory(prefix='planned-anchor-') as tmp:
                candidate=Path(tmp)
                for rid,role in enumerate('AB'):
                    source=s.out/('physical_'+s.roles[role].physical_node)/s.subdir(*s.role_bindings[role])
                    target=candidate/f'node{rid}';target.mkdir()
                    for rank in range(8):
                        content=(source/f'events_g0_r{rank}.jsonl').read_text()
                        if role=='B' and a.variant=='p2':
                            incoming=s.out/('physical_'+s.roles['C'].physical_node)/s.subdir(*s.role_bindings['C'])/f'events_g1_r{rank}.jsonl'
                            content+=''.join(line+'\n' for line in incoming.read_text().splitlines()
                                if not json.loads(line).get('shadow'))
                        (target/f'events_g0_r{rank}.jsonl').write_text(content)
                report=compare_compact(ROOT/'results/v3b1_num01_01_shift',candidate)
            (out/'prior_shift_compact.json').write_text(json.dumps(report,indent=2)+'\n')
            status['anchor_exact']=report['exact_pass']
            if not report['exact_pass']:raise RuntimeError('planned steady anchor differs from original SHIFT')
        status['success']=True
    except BaseException as exc:status['error']=type(exc).__name__+': '+str(exc)
    finally:
        status.update(done=True,finished=time.time(),session=s.name if s else None);save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
