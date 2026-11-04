"""One bounded no-prewarm cold recovery, owned by server systemd."""
import argparse,json,time
from pathlib import Path
import scripts.v1_acceptance_remote as remote
from shift_full.control import server_password
from shift_full.verify_rotation import verify
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v1_cold_20261103'

def config():
 c=json.loads((ROOT/'configs/v1_bench_shift.json').read_text())
 c['train_args']=[x for x in c['train_args'] if x!='--warmup-before-join']
 c['train_args']+=['--audit-state','--hash-every','0']
 c.update(target_steps=1000000,fast_launch=False,persistent_cpu_launcher=False,
          external_cpu_launcher_socket=None,single_gpu_visibility=False,audit_initial=True)
 c['env'].update(CUDA_DEVICE_MAX_CONNECTIONS='1',SHIFT_GENERIC_PRELOAD='0',SHIFT_COMMON_GPU_WARMUP='0')
 return c

def rows(path):
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def summarize(s):
 f=s.fault_results[0]
 manifest=json.loads((s.out/'physical_manifest.json').read_text())
 b=next(x for x in manifest if x['rid']==f['target'] and x['generation']==f['generation'])
 d=s.out/('physical_'+b['physical_node'])/Path(b['directory']).name
 launch=next(x for x in rows(d/'supervisor.jsonl') if x['kind']=='launched')
 start=launch['process_started_time']
 events={r:rows(d/f"events_g{f['generation']}_r{r}.jsonl") for r in range(8)}
 critical=max(range(8),key=lambda r:next(e['time'] for e in events[r] if e['kind']=='ready'))
 phases=[dict(name=e['name'],seconds=e['seconds'],end_since_launch=e['time']-start)
         for e in events[critical] if e['kind']=='phase']
 first_phase=next(e for e in events[critical] if e['kind']=='phase')
 offsets={k:f[k]-start for k in ('fault_time','request_time','ready_time','recovery_request_time','state_loaded_time','loaded_time','joint_time')}
 return dict(backup=b['physical_node'],killed=f['old_node'],process_start_to_joint_seconds=f['joint_time']-start,
             process_start_semantics='supervisor timestamp immediately before Popen; launcher startup included',
             launch_to_worker_import_start_seconds=first_phase['time']-first_phase['seconds']-start,
             critical_ready_rank=critical,phases=phases,offsets_from_launch=offsets,
             phase_seconds_by_rank={str(r):{e['name']:e['seconds'] for e in es if e['kind']=='phase'} for r,es in events.items()},
             fault_to_joint_seconds=f['fault_to_joint_seconds'],request_to_joint_seconds=f['joint_time']-f['request_time'],
             ready_to_state_loaded_seconds=f['state_loaded_time']-f['ready_time'],
             state_loaded_to_joint_seconds=f['joint_time']-f['state_loaded_time'],
             first_joint_step=f['first_joint_step'],time_pass=f['joint_time']-start<=10)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261106);p.add_argument('--suffix',default='01');a=p.parse_args()
 remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
 OUT.mkdir(exist_ok=True,parents=True)
 status=dict(done=False,success=False,seed=a.seed,formal_injections_before=remote.injections(),error=None)
 path=OUT/f'status_{a.suffix}.json'
 def save():path.write_text(json.dumps(status,indent=2)+'\n')
 save();s=None
 try:
  name=f'v1_accept_20261103_cold_{a.suffix}'
  remote.ensure_unused_session(name)
  s=remote.ServerRotation(name,config(),600,a.seed,backup_mode='cold')
  try:
   s.start()
   # Initial training commits are the measured workload; no synthetic prewarm.
   s.await_('prefault_training',lambda d:bool(s.common_commits(d,40)[0]))
   fault_window_started=time.monotonic()
   s.event('fault_window_start',duration=0,requested_faults=1)
   s.fault(0)
   joint=s.fault_results[0]['first_joint_step']
   s.await_('post_recovery_progress',lambda d:bool(s.common_commits(d,joint+9)[0]))
   s.event('complete',faults=1,fault_window_seconds=time.monotonic()-fault_window_started)
  except BaseException as e:
   s.event('failed',error=repr(e));raise
  finally:s.finish()
  audit=verify(s.out,1,require_audit=True)
  (OUT/f'verify_{a.suffix}.json').write_text(json.dumps(audit,indent=2)+'\n')
  status.update(summarize(s),functional_pass=audit['pass'])
  status['success']=status['functional_pass'] and status['time_pass']
 except BaseException as e:status['error']=repr(e)
 finally:
  status.update(done=True,formal_injections_after=remote.injections(),session=s.name if s else None)
  save();print(json.dumps(status),flush=True)
 return 0 if status['success'] else 1
if __name__=='__main__':raise SystemExit(main())
