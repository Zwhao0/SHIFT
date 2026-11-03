"""Run the snapshot-stream component after full experiment cleanup."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,endpoints_from_env
from shift_full.remote_job import run_job

def main():
 root=Path(__file__).resolve().parents[1];s=Session.__new__(Session);s.root=root;s._sudo_password=None;eps=endpoints_from_env()
 for e in eps.values():
  s.ensure_connection(e)
  if s.execute_endpoint(e,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise RuntimeError('GPU workload exists')
 raw=run_job(s,eps['node0'],['env','PYTHONPATH=/workspace/SHIFT','OMP_NUM_THREADS=1','/workspace/SHIFT/runtime/torch214/bin/python','-m','shift_full.planned_overlap'],timeout=120)
 result=json.loads(raw.strip().splitlines()[-1]);(root/'results/planned_handoff/overlap_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 for e in eps.values():
  if s.execute_endpoint(e,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise RuntimeError('GPU cleanup incomplete')
if __name__=='__main__':main()
