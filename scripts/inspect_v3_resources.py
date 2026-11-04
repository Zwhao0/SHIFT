"""Read-only resource evidence against a saved company GPU PID baseline."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time

from shift_full.control import Session, endpoints_from_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text())
    session = Session.__new__(Session)
    session.root, session._sudo_password = Path(__file__).resolve().parents[1], None
    result = dict(observed_at=time.time(), nodes={}, business_health_tested=False)
    code = '''import json,shutil,subprocess
from pathlib import Path
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines()
units=subprocess.check_output(['systemctl','list-units','shift-v3-*','--state=running','--plain','--no-legend','--no-pager'],text=True).splitlines()
processes=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\\x00')
  modules=[argv[i+1] for i,x in enumerate(argv[:-1]) if x=='-m']
  if any(m.startswith(('shift_full.','scripts.v3_','scripts.v1_')) for m in modules) or any(x.endswith('/vendor/bin/torchft_lighthouse') for x in argv):
   processes.append(dict(pid=int(proc.name),module=modules))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
keys=Path.home()/'.ssh/authorized_keys'
marker_count=sum('shift-v3-controller-20261103' in x for x in keys.read_text().splitlines()) if keys.exists() else 0
print(json.dumps(dict(gpu=gpu,units=units,shift_processes=processes,scoped_key_markers=marker_count,free_gib=shutil.disk_usage('/workspace/SHIFT').free/1024**3)))'''
    for node, original in endpoints_from_env().items():
        endpoint = replace(original, socket=f'ssh-v3-resources-{os.getpid()}-{node}.sock')
        try:
            expected_rows = baseline.get('nodes', baseline)[node]['gpu']
            expected = {row.split(',')[0].strip() for row in expected_rows if 'VLLM::' in row}
            if len(expected) != 8:
                raise ValueError(f'{node}: expected eight company GPU PIDs in baseline')
            session.ensure_connection(endpoint)
            row = json.loads(session.ssh(endpoint, ['python3', '-c', code]))
            current = {line.split(',')[0].strip() for line in row['gpu']}
            company = {line.split(',')[0].strip() for line in row['gpu'] if 'VLLM::' in line}
            row.update(company_pids_unchanged=expected <= company, company_count=len(expected),
                       extra_gpu_pids=sorted(current-expected), missing_company_pids=sorted(expected-company))
            result['nodes'][node] = row
        finally:
            subprocess.run(['ssh', '-S', str(session.root/'tmp'/endpoint.socket), '-O', 'exit',
                            'test@'+endpoint.ip], capture_output=True, timeout=5)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({node: {key: row[key] for key in
                     ('company_pids_unchanged', 'extra_gpu_pids', 'missing_company_pids', 'free_gib')}
                     for node, row in result['nodes'].items()}))


if __name__ == '__main__':
    main()
