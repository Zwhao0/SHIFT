"""Serial local driver; bounded remote cgroups and the existing training lock."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.v41_control import command

ROOT=Path(__file__).resolve().parents[1]


def run(task):
    name=task['name'];unit=task['unit'];path=task['path']
    print(json.dumps(dict(start=name)),flush=True)
    subprocess.run([sys.executable,'-m','scripts.prepare_v3_controller'],cwd=ROOT,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=120)
    prepared=ROOT/'results/v3_a1_20261103/controller_prepare.json'
    (ROOT/'tmp'/f'{name}.prepare.json').write_bytes(prepared.read_bytes())
    subprocess.run([sys.executable,'-m',*task['launch']],cwd=ROOT,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=40)
    deadline=time.monotonic()+1100
    while time.monotonic()<deadline:
        code='''from pathlib import Path
import json,subprocess
p=Path(PATH)
x=json.loads(p.read_text()) if p.exists() else {}
state=subprocess.run(['systemctl','show',UNIT,'--property=ActiveState','--value'],capture_output=True,text=True).stdout.strip()
finished=x.get('done') is True or 'resources_clean' in x or ('pass_all' in x and 'controller_seconds' in x)
print(json.dumps(dict(finished=finished,state=state,result=x if finished else {})))
'''.replace('PATH',repr(path)).replace('UNIT',repr(unit))
        packet=json.loads(command(code,True))
        if packet['finished']:
            report=packet['result']
            # Full private summaries remain local, never in the public evidence.
            (ROOT/'tmp'/f'{name}.summary.json').write_text(json.dumps(report,indent=2)+'\n')
            keys=('name','session','pass_all','success','functional_pass','resources_clean','error',
                  'steady_mean_seconds','controller_seconds','request_to_observed_joint_seconds','observed_cut','goal')
            print(json.dumps({k:report[k] for k in keys if k in report}),flush=True)
            if not (report.get('pass_all') or report.get('success') or report.get('functional_pass')):
                raise RuntimeError('trial failed: '+name)
            if report.get('error') or report.get('cleanup_errors') or report.get('finish_error'):
                raise RuntimeError('trial/cleanup failed: '+name)
            if report.get('transactional_switch') and not report.get('reference_only'):
                expected=task.get('expected_decision','replace')
                if not all(x['decision']==expected for x in report['decisions']):
                    raise RuntimeError('unexpected migration outcome: '+name)
            return report
        if packet['state'] not in ('active','activating'):
            raise RuntimeError('controller exited before a complete report: '+name+' '+packet['state'])
        time.sleep(5)
    raise TimeoutError(name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);a=p.parse_args()
    for task in json.loads(a.manifest.read_text())['tasks']:run(task)


if __name__=='__main__':main()
