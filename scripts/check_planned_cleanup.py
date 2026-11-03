"""Read-only final audit of only this exploration's session names/processes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,endpoints_from_env

def main():
 root=Path(__file__).resolve().parents[1];s=Session.__new__(Session);s.root=root;s._sudo_password=None
 names=[p.name for pattern in ('ph_*','v2_*') for p in (root/'results').glob(pattern) if p.is_dir()]
 code='''import json,os,subprocess
from pathlib import Path
names=NAMES
left=[];ancestors={os.getpid()};parent=os.getppid()
while parent and parent not in ancestors:
 ancestors.add(parent)
 try:parent=int(next(l.split()[1] for l in Path(f'/proc/{parent}/status').read_text().splitlines() if l.startswith('PPid:')))
 except (OSError,StopIteration):break
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in ancestors:continue
 try:args=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
 except OSError:continue
 if any(n in args for n in names) and ('shift_full.' in args or 'torchft_lighthouse' in args):left.append(dict(pid=int(p.name),command=args[:400]))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
print(json.dumps(dict(experiment_processes_left=left,gpu_compute_apps=gpu,passed=not left and not gpu)))
'''.replace('NAMES',repr(names))
 nodes={}
 for n,e in endpoints_from_env().items():
  s.ensure_connection(e);nodes[n]=json.loads(s.execute_endpoint(e,['python','-c',code]))
 result=dict(passed=all(r['passed'] for r in nodes.values()),nodes=nodes,action='read-only; no signals or deletions')
 (root/'results/planned_handoff/cleanup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
