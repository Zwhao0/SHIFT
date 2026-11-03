"""Serial bounded acceptance/timing jobs. Stop immediately on failure."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_planned_handoff import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','timing'],required=True);p.add_argument('--start',type=int,default=1);p.add_argument('--count',type=int,default=None);a=p.parse_args();out=ROOT/'results/planned_handoff';manifest=out/(a.phase+'_suite.jsonl')
 # Alternate donor and ABBA variants; no overlapping GPU experiments.
 jobs=[('p1','B' if i%2==0 else 'A') for i in range(10)] if a.phase=='audit' else [(v,s) for s in ('B','A') for v in ('p0','p1','p1','p0','p0','p1','p1','p0','p0','p1')]
 for i,(variant,source) in enumerate(jobs[a.start-1:a.start-1+(a.count if a.count is not None else len(jobs))],a.start):
  name=f'ph_{a.phase}_{i:02d}_{variant}_{source}';cmd=[sys.executable,'scripts/run_planned_handoff.py','--name',name,'--variant',variant,'--source',source]+(['--audit'] if a.phase=='audit' else [])
  log=out/(name+'.console.log');start=time.monotonic()
  with log.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  row=dict(independent_launch=True,same_A_continuous_rotation=False,name=name,variant=variant,source=source,exit_code=r.returncode,seconds=time.monotonic()-start)
  if not r.returncode:
   result=analyze(ROOT/'results'/name);row.update(checks_pass=result['all_checks_pass'],gap=result['last_ab_to_first_ac_seconds'],extra_gap=result['extra_gap_seconds'])
  with manifest.open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
  if r.returncode or not row['checks_pass']:raise SystemExit(1)
if __name__=='__main__':main()
