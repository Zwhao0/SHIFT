"""Read a compact V4 snapshot through the retained local lab transport."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from shift_full.control import Session,endpoints_from_env


def main():
    p=argparse.ArgumentParser();p.add_argument('suffix');a=p.parse_args()
    if not a.suffix or not all(x.isalnum() or x in '_-' for x in a.suffix):p.error('invalid suffix')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v4-inspect.sock')
    code='''from pathlib import Path
import json
p=Path('/workspace/SHIFT/results/v4_'''+a.suffix+'''')
f=p/'summary.json';summary=json.loads(f.read_text()) if f.exists() else {}
print(json.dumps({k:v for k,v in summary.items() if k not in ('comparisons','source_sha256','raw_sha256','healthy_pids','phase_seconds')}))
f=Path('/workspace/SHIFT/runtime/v3-reproduce/v4_'''+a.suffix+'''.controller.log')
if f.exists():
 lines=[x for x in f.read_text().splitlines() if x.startswith('{"progress":')]
 print('controller:',lines[-1] if lines else 'starting')
names=sorted(f.stem[:-1] for f in p.glob('g*_?0.jsonl')) if summary.get('transactional_switch') and not summary.get('reference_only') else list('ABCDE' if summary.get('outer_manager') else 'ABC')
for role in names:
 f=p/(role+'0.jsonl')
 if f.exists():
  raw=f.read_text();lines=raw.splitlines() if raw.endswith('\\n') else raw.splitlines()[:-1]
  es=[json.loads(x) for x in lines if x.strip()]
  print(role,[{k:x[k] for k in ('kind','step','phase','epoch','seconds','exact') if k in x} for x in es[-3:]])
if summary.get('error'):
 f=p/'A0.log';print(f.read_text()[-1200:] if f.exists() else '')
'''
    try:
        s.ensure_connection(e)
        result=s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True)
        for endpoint in endpoints_from_env().values():result=result.replace(endpoint.ip,'<private>')
        print(result)
    finally:
        subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
