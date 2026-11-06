"""Sample resources; stop only this owned controller on lost memory headroom."""
import argparse
import json
from pathlib import Path
import time
from scripts.sessionctl import read
from scripts.v41_control import command


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True)
    a=p.parse_args();assert a.session and all(c.isalnum() or c in '_-' for c in a.session)
    out=Path(__file__).resolve().parents[1]/'runtime/v3-reproduce'/('watch_'+a.session+'.json')
    report=dict(session=a.session,interval_seconds=5,stop_free_mib=512,samples=[],stopped=False)
    path='/workspace/SHIFT/results/session_'+a.session+'/resources_before.json'
    for _ in range(90):
        try:
            before=json.loads(command('from pathlib import Path;print(Path('+repr(path)+').read_text())',True));break
        except Exception:time.sleep(1)
    else:raise RuntimeError('resource baseline did not become available')
    for _ in range(360):
        state=read(a.session)
        if state.get('done') or state.get('error'):break
        sample=dict(time=time.time(),nodes={})
        for node,baseline in before.items():
            code='''from pathlib import Path
import json,subprocess
expected=EXPECTED
prefix=PREFIX
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
company={};unknown=[];test_mem=[]
for row in apps:
 pid,name,mem=[x.strip() for x in row.split(',',2)]
 if pid in expected:company[pid]=name
 else:
  try:owned=prefix in (Path('/proc')/pid/'cgroup').read_text()
  except FileNotFoundError:owned=True
  if not owned:unknown.append(pid)
  elif mem.isdigit():test_mem.append(int(mem))
gpus=[[int(x.strip()) for x in row.split(',')] for row in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
print(json.dumps(dict(company_unchanged=company==expected,unknown_gpu_count=len(unknown),
 min_free_mib=min(x[0] for x in gpus),max_utilization_percent=max(x[1] for x in gpus),
 max_test_process_mib=max(test_mem,default=0))))
'''.replace('EXPECTED',repr(baseline['company'])).replace('PREFIX',repr('shift-session-'+a.session+'-'))
            sample['nodes'][node]=json.loads(command(code,True,node))
        report['samples'].append(sample)
        violations=[n for n,x in sample['nodes'].items() if not x['company_unchanged'] or x['unknown_gpu_count'] or x['min_free_mib']<512]
        if violations:
            report.update(stopped=True,violations=violations)
            unit='shift-session-'+a.session+'-controller'
            command('import subprocess;subprocess.run(["systemctl","kill","--kill-who=main","--signal=SIGINT",'+repr(unit)+'],capture_output=True)',True)
        out.write_text(json.dumps(report,indent=2)+'\n')
        if report['stopped']:break
        time.sleep(5)
    report['finished']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
