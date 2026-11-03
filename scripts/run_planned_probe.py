"""Three-node bounded microprobe, only owns subprocesses it starts."""
import concurrent.futures,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,endpoints_from_env
from shift_full.remote_job import run_job
ROOT=Path(__file__).resolve().parents[1]
def main():
 name=sys.argv[1];phase=sys.argv[2] if len(sys.argv)>2 else 'shadow'
 out=ROOT/'results/planned_handoff'/name;out.mkdir(parents=True,exist_ok=False)
 s=Session.__new__(Session);s.root=ROOT;s._sudo_password=None;eps=endpoints_from_env();python='/workspace/SHIFT/runtime/torch214/bin/python'
 remote='/workspace/SHIFT/results/'+name
 for e in eps.values():s.ensure_connection(e)
 env={'PYTHONPATH':'/workspace/SHIFT','OMP_NUM_THREADS':'1','TORCHFT_USE_OTEL':'false','RUST_LOG':'warn','TOKIO_WORKER_THREADS':'2','NCCL_DEBUG':'WARN','NCCL_NET':'IB','NCCL_IB_DISABLE':'0','NCCL_IB_HCA':'=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1','NCCL_IB_GID_INDEX':'3','NCCL_SOCKET_IFNAME':'bond0','GLOO_SOCKET_IFNAME':'bond0','NCCL_CUMEM_HOST_ENABLE':'0','NCCL_CUMEM_ENABLE':'0','NCCL_NVLS_ENABLE':'0','TORCH_NCCL_ASYNC_ERROR_HANDLING':'0','NO_PROXY':'*'}
 for e in eps.values():
  s.execute_endpoint(e,['python','-c',f'from pathlib import Path; p=Path({remote!r});p.mkdir(exist_ok=False)'])
 (out/'config.json').write_text(json.dumps(dict(phase=phase,env=env,python=python,seed=20261103),indent=2))
 lightcode='''import subprocess,time,os
from pathlib import Path
ps=[subprocess.Popen(['/workspace/SHIFT/runtime/torch214/bin/python','/workspace/SHIFT/vendor/bin/torchft_lighthouse','--bind','[::]:'+str(port),'--min_replicas','2','--heartbeat_timeout_ms','1000','--join_timeout_ms','1000','--quorum_tick_ms','100']) for port in PORTS]
try:
 end=time.monotonic()+200
 while time.monotonic()<end and not Path(REMOTE+'/STOP_LIGHTHOUSE').exists():
  if any(p.poll() is not None for p in ps):raise RuntimeError('lighthouse died')
  time.sleep(.1)
finally:
 for p in ps:
  p.terminate()
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
'''.replace('REMOTE',repr(remote)).replace('PORTS',repr((31300,31410) if phase=='relay' else (31300,31310)))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  light=pool.submit(run_job,s,eps['node0'],['env',*[f'{k}={v}' for k,v in env.items()],python,'-c',lightcode],240)
  time.sleep(1)
  futures={node:pool.submit(run_job,s,e,['env',*[f'{k}={v}' for k,v in env.items()],python,'-m','shift_full.planned_probe','--rid',str(i),'--host',e.ip,'--session',name,'--phase',phase],180) for i,(node,e) in enumerate(eps.items())}
  try:
   deadline=time.monotonic()+100;started=False;joined=False
   while time.monotonic()<deadline and not all(f.done() for f in futures.values()):
    info={}
    for node,e in eps.items():
     raw=s.execute_endpoint(e,['python','-c',f'from pathlib import Path; p=Path({remote!r}); print((p/"events.jsonl").read_text() if (p/"events.jsonl").exists() else "")'])
     (out/(node+'.jsonl')).write_text(raw);info[node]=[json.loads(l) for l in raw.splitlines() if l.strip()]
    if not started and all(any(r['kind']=='ready' for r in rs) for rs in info.values()):
     for e in eps.values():s.execute_endpoint(e,['touch',remote+'/START'])
     started=True;print('START',flush=True)
    if started and not joined and any(r['kind']=='commit' and r['step']>=2 for r in info['node0']):
     s.execute_endpoint(eps['node3'],['touch',remote+'/JOIN']);joined=True;print('JOIN C',flush=True)
    time.sleep(.1)
   statuses={}
   for n,f in futures.items():
    try:statuses[n]={'ok':True,'stdout':f.result()}
    except Exception as e:statuses[n]={'ok':False,'error':str(e)}
   (out/'status.json').write_text(json.dumps(statuses,indent=2));print(json.dumps(statuses),flush=True)
  finally:
   s.execute_endpoint(eps['node0'],['touch',remote+'/STOP_LIGHTHOUSE']);light.result()
   for node,e in eps.items():
    raw=s.execute_endpoint(e,['python','-c',f'from pathlib import Path; p=Path({remote!r})/"events.jsonl";print(p.read_text() if p.exists() else "")']);(out/(node+'.jsonl')).write_text(raw)
 print(out)
if __name__=='__main__':main()
