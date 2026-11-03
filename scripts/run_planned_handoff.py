"""Bounded full-load planned replacement; no fault injection or recovery wait."""
import argparse,concurrent.futures,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,CONTAINER_ROOT
from shift_full.remote_job import run_job
ROOT=Path(__file__).resolve().parents[1]

class PlannedSession(Session):
 def data(self):
  data={}
  for node,e in self.endpoints.items():
   if node not in self.started_endpoints:continue
   self.read_endpoint(e,1 if node=='node3' else 0);data[node]=self._read_cache[node]
  return data
 def events(self,data,node,kind):
  return [r for name,rs in data.get(node,{}).items() if name.startswith('events_') for r in rs if r['kind']==kind]
 def wait(self,label,predicate,timeout=150):
  end=time.monotonic()+timeout
  while time.monotonic()<min(end,self.deadline):
   d=self.data()
   for node,files in d.items():
    bad=[r for r in files.get('supervisor.jsonl',[]) if r['kind'] in ('unexpected_exit','supervisor_error')]
    if bad:raise RuntimeError(f'{node}: {bad[-1]}')
   if predicate(d):self.event(label,mono=time.monotonic());return d
   time.sleep(.15)
  raise TimeoutError(label)
 def launch_role(self,role,base):
  node={'A':'node0','B':'node1','C':'node3'}[role];rid=0 if role=='A' else 1;gen=1 if role=='C' else 0
  self.config=copy.deepcopy(base);
  if base['entry_module']=='shift_full.planned_train':self.config['train_args']+=['--planned-role',role]
  self.start_supervisor(rid,self.endpoints[node],gen)
 def set_key(self,key,value):
  code=f'from datetime import timedelta;import torch.distributed as d;s=d.TCPStore("10.10.4.2",31402,None,False,timeout=timedelta(seconds=15));s.set({key!r},str({value!r}))'
  self.execute_endpoint(self.endpoints['node0'],[self.config['python'],'-c',code])

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--variant',choices=['p0','p1','p1d','p1f','steady','native'],required=True);p.add_argument('--source',choices=['A','B'],default='B');p.add_argument('--audit',action='store_true');p.add_argument('--cancel',action='store_true');p.add_argument('--post-steps',type=int,default=3);a=p.parse_args()
 if a.variant in ('p1d','p1f') and a.source!='B':p.error('optimized variants require live B donor')
 if a.post_steps<3:p.error('post-steps must be >=3')
 base=json.loads((ROOT/'configs/v1_bench_shift.json').read_text());base.update(entry_module='shift_full.planned_train',target_steps=100000,lighthouse_port=31400,master_port=31500,min_replicas=2,heartbeat_timeout_ms=1000,join_timeout_ms=1000,quorum_tick_ms=100,recycle_after_worker_fault=False)
 args=base['train_args'];args[args.index('--communication-timeout')+1]='30';args += ['--sync-quorum','--fixed-batch','--data','random']
 if a.variant=='native':base['entry_module']='shift_full.train'
 else:args+=['--planned-variant',a.variant,'--planned-source',a.source]
 if a.audit:args+=['--audit-state','--hash-every','1']
 base['env']['TORCH_NCCL_NONBLOCKING_TIMEOUT']='60'
 base['audit_initial']=a.audit
 s=PlannedSession(a.name,base,600,takeover_node3=True)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);light=None
 try:
  for e in s.endpoints.values():s.ensure_connection(e)
  # Snapshot before mutations; refuse to overlap any GPU compute load.
  pre={}
  for name,e in s.endpoints.items():
   raw=s.execute_endpoint(e,['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']);pre[name]=raw.strip()
  (s.out/'preflight.json').write_text(json.dumps(pre,indent=2))
  if any(pre.values()):raise RuntimeError('GPU compute load present; refusing experiment')
  s.launch_role('A',base);s.launch_role('B',base)
  s.wait('ab_ready',lambda d:all(len(s.events(d,n,'ready'))==8 for n in ('node0','node1')))
  for n in ('node0','node1'):s.mark_endpoint(s.endpoints[n],'START')
  s.wait('ab_two_steps',lambda d:all(len([r for r in s.events(d,n,'commit') if r['step']>=2])>=8 for n in ('node0','node1')))
  if a.variant in ('steady','native'):
   for n in ('node0','node1'):s.mark_endpoint(s.endpoints[n],'TRAIN_UNTIL','50')
   s.wait('steady_50_steps',lambda d:all(len([r for r in s.events(d,n,'commit') if r['step']==50])==8 for n in ('node0','node1')))
   (s.out/'steady_status.json').write_text(json.dumps(dict(passed=True,steps=50),indent=2));return
  s.event('REQUEST',mono=time.monotonic(),variant=a.variant,source=a.source)
  s.launch_role('C',base)
  s.set_key('request','1')
  lightcode='''import subprocess,time
from pathlib import Path
p=subprocess.Popen(['/workspace/SHIFT/runtime/torch214/bin/python','/workspace/SHIFT/vendor/bin/torchft_lighthouse','--bind','[::]:31410','--min_replicas','2','--heartbeat_timeout_ms','1000','--join_timeout_ms','1000','--quorum_tick_ms','100'])
try:
 end=time.monotonic()+400
 while time.monotonic()<end and not Path(STOPFILE).exists():
  if p.poll() is not None:raise RuntimeError('new-epoch lighthouse died')
  time.sleep(.1)
finally:
 p.terminate()
 try:p.wait(timeout=5)
 except subprocess.TimeoutExpired:p.kill();p.wait()
'''.replace('STOPFILE',repr('/workspace/SHIFT/results/'+a.name+'/STOP_NEW_LIGHTHOUSE'))
  light=pool.submit(run_job,s,s.endpoints['node0'],['env','TOKIO_WORKER_THREADS=2','TORCHFT_USE_OTEL=false',base['python'],'-c',lightcode],450)
  d=s.wait('c_local_ready',lambda d:len(s.events(d,'node3','planned_local_ready'))==8)
  latest=max(r['step'] for n in ('node0','node1') for r in s.events(d,n,'commit'));cut=latest+8
  if a.cancel:
   for n in ('node0','node1'):s.mark_endpoint(s.endpoints[n],'TRAIN_UNTIL',str(cut))
   s.set_key('cancel','1')
   s.wait('cancel_all_roles',lambda d:all(len(s.events(d,n,'planned_cancelled'))==8 for n in ('node0','node1','node3')))
   d=s.wait('ab_progress_after_cancel',lambda d:all(len([r for r in s.events(d,n,'commit') if r['step']==cut])==8 for n in ('node0','node1')))
   assert not s.events(d,'node3','commit')
   assert all(r['epoch']==0 and r['participants']==2 and r['contributed'] and r['global_batch']==8 for n in ('node0','node1') for r in s.events(d,n,'commit'))
   (s.out/'cancel_status.json').write_text(json.dumps(dict(passed=True,ready_progress=latest,final_progress=cut,ab_steps_after_cancel=cut-latest,c_effective_commits=0),indent=2))
   s.event('CANCEL_PASS',step=cut,mono=time.monotonic());return
  if a.variant=='p1f':
   s.mark_endpoint(s.endpoints['node3'],'START')
   s.set_key('arm','1')
   selected=s.wait('worker_boundary_selected',lambda d:bool(s.events(d,'node0','planned_cut_selected')),timeout=20)
   cut=s.events(selected,'node0','planned_cut_selected')[-1]['step']
  for e in s.endpoints.values():s.mark_endpoint(e,'TRAIN_UNTIL',str(cut+a.post_steps))
  s.mark_endpoint(s.endpoints['node3'],'START')
  s.event('FINAL_BOUNDARY_SELECTED',step=cut,mono=time.monotonic());s.set_key('cut',cut)
  s.event('AB_PROGRESS_WHEN_C_READY',step=latest,mono=time.monotonic())
  s.wait('FIRST_AC_COMMIT',lambda d:all(len([r for r in s.events(d,n,'commit') if r.get('epoch')==1 and r['step']==cut+1])==8 for n in ('node0','node3')))
  d=s.wait('ac_three_commits',lambda d:all(len([r for r in s.events(d,n,'commit') if r.get('epoch')==1 and r['step']==cut+3])==8 for n in ('node0','node3')))
  d=s.wait('b_standby_verified',lambda d:len(s.events(d,'node1','planned_standby_verified'))==8)
  if a.audit:
   for rank in range(8):
    src=[r for r in s.events(d,'node1' if a.source=='B' else 'node0','snapshot') if r['rank']==rank and r['step']==cut][-1]
    restored=[r for r in s.events(d,'node3','restored') if r['rank']==rank][-1]
    assert restored['hashes_match'] and restored['optimizer_tensors']>0
    for step in range(cut+1,cut+4):
     rows=[[r for r in s.events(d,n,'commit') if r['rank']==rank and r['step']==step][-1] for n in ('node0','node3')]
     assert all(r['participants']==2 and r['global_batch']==8 and r['physical_members']==['A','C'] for r in rows)
     assert rows[0]['hashes']['model']==rows[1]['hashes']['model']
     assert rows[0]['hashes']['optimizer']==rows[1]['hashes']['optimizer']
     assert rows[0]['hashes']['scheduler']==rows[1]['hashes']['scheduler']
  if a.post_steps>3:
   s.wait('ac_post_steps',lambda d:all(len([r for r in s.events(d,n,'commit') if r.get('epoch')==1 and r['step']==cut+a.post_steps])==8 for n in ('node0','node3')),timeout=240)
  s.gen[1]=1
  (s.out/'planned_status.json').write_text(json.dumps(dict(passed=True,audit=a.audit,variant=a.variant,source=a.source,cut=cut,post_steps=a.post_steps),indent=2))
  s.event('PLANNED_PASS',step=cut,mono=time.monotonic())
 finally:
  if light:
   s.mark_endpoint(s.endpoints['node0'],'STOP_NEW_LIGHTHOUSE')
   try:light.result(timeout=30)
   except Exception as e:s.event('new_lighthouse_cleanup_error',error=str(e))
  pool.shutdown(wait=True)
  s.finish()
if __name__=='__main__':main()
