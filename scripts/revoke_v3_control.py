"""Revoke only SHIFT's marked server-controller SSH authorization while idle.

Reusable environments/models and other authorized keys are untouched. The next
prepare_v3_controller call can re-authorize the retained root-owned scoped key.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time
from shift_full.control import Session, endpoints_from_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',nargs='+',choices=('node0','node1','node3'),default=['node0','node1','node3'])
    parser.add_argument('--evidence-dir',type=Path)
    args=parser.parse_args()
    if 'node0' not in args.nodes:parser.error('node0 is required for the training lock check')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    eps={n:replace(e,socket=f'ssh-v3-revoke-{os.getpid()}-{n}.sock') for n,e in endpoints_from_env().items() if n in args.nodes}
    report=dict(observed_at=time.time(),nodes={})
    try:
        for e in eps.values():s.ensure_connection(e)
        check="import fcntl;from pathlib import Path;p=Path('/workspace/SHIFT/runtime/v3-training.lock').open('a');fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not Path('/workspace/SHIFT/runtime/v1-controller/credentials').exists(),'prepared credential still present'"
        s.ssh(eps['node0'],['sudo','-S','-p','','python3','-c',check],sudo_input=True)
        code='''import json,pwd
from pathlib import Path
p=Path(pwd.getpwnam('test').pw_dir)/'.ssh/authorized_keys'
lines=p.read_text().splitlines() if p.exists() else []
kept=[x for x in lines if 'shift-v3-controller-20261103' not in x]
if kept!=lines:p.write_text('\\n'.join(kept)+'\\n')
print(json.dumps(dict(removed=len(lines)-len(kept),remaining_markers=sum('shift-v3-controller-20261103' in x for x in p.read_text().splitlines()) if p.exists() else 0)))
'''
        for n,e in eps.items():report['nodes'][n]=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        dest=args.evidence_dir or s.root/'results/v3_b2_20261103'
        dest.mkdir(parents=True,exist_ok=True)
        out=dest/('revoke-'+str(time.time_ns())+'.json')
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    finally:
        for e in eps.values():subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
