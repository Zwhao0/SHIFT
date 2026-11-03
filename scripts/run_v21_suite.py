"""Serial bounded v2.1 matrices; record errors and stop immediately."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_planned_handoff import analyze

def main():
 if sys.platform=='darwin':subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['scenarios','audit','timing'],required=True);p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=100);p.add_argument('--suffix',default='');a=p.parse_args();out=ROOT/'results/overlap_v21';out.mkdir(exist_ok=True)
 jobs=([('p2',s) for s in ('relay-reject','stale-state','prepare-fail','ready-timeout','slow-compute')] if a.phase=='scenarios' else [('p2','healthy')]*3 if a.phase=='audit' else [(v,'healthy') for v in ('p1f','p2','p2','p1f','p1f','p2','p2','p1f','p1f','p2')])
 for i,(v,scenario) in enumerate(jobs,1):
  if not a.start<=i<=a.end:continue
  name=f'v2_1_final_{a.phase}_{i:02}_{v}_{scenario.replace("-","_")}{a.suffix}'
  cmd=[sys.executable,'scripts/run_planned_handoff.py','--name',name,'--variant',v,'--source','B','--scenario',scenario,'--post-steps','50' if a.phase=='timing' else '10' if a.phase=='scenarios' else '6']
  if a.phase=='audit' or scenario in ('stale-state','relay-reject'):cmd+=['--audit']
  start=time.monotonic()
  with (out/(name+'.console.log')).open('w') as log:code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  r=dict(name=name,variant=v,scenario=scenario,exit_code=code,seconds=time.monotonic()-start,passed=False)
  if code==0:
   path=ROOT/'results'/name
   if (path/'scenario_status.json').exists():r['passed']=json.loads((path/'scenario_status.json').read_text())['passed']
   else:
    result=analyze(path);r.update(passed=result['all_checks_pass'],gap=result['last_ab_to_first_ac_seconds'],extra=result['extra_gap_seconds'],post50=result['post_ac_total_seconds'],relay_pause=result.get('relay_pause_seconds'))
  with (out/(a.phase+'.jsonl')).open('a') as log:log.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
  if not r['passed']:raise SystemExit(1)
 if a.phase=='timing':
  followups=[('cycles10',[sys.executable,'scripts/run_overlap_cycles.py','--name','v2_1_cycles10','--cycles','10','--post-steps','50']),('slow_baseline',[sys.executable,'scripts/run_planned_handoff.py','--name','v2_1_slow_baseline_p1f','--variant','p1f','--source','B','--scenario','slow-compute','--post-steps','10'])]
  for label,cmd in followups:
   with (out/(label+'.console.log')).open('w') as log:code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
   print(json.dumps(dict(followup=label,exit_code=code)),flush=True)
   if code:raise SystemExit(code)
if __name__=='__main__':main()
