"""Bounded CPU protocol/payload regressions; no training or fault injection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True);a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=root/f'results/v3_c1_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    status=dict(done=False,success=False,training_started=False,injections=0,started=time.time())
    path=out/'cpu_status.json';path.write_text(json.dumps(status))
    # Preparation leaves one-use control material; CPU tests do not need it.
    (root/'runtime/v1-controller/credentials').unlink(missing_ok=True)
    try:
        cmd=[str(root/'runtime/torch214/bin/python'),'-m','unittest',
             'scripts.test_ownership_protocol','scripts.test_ownership_runtime','scripts.test_ownership_control','scripts.test_ownership_handoff',
             'scripts.test_planned_shadow','scripts.test_planned_state','scripts.test_planned_payload','scripts.test_v3_planned_controller','shift_full.test_planned_protocol',
             'scripts.test_v3_state','shift_full.test_reference_audit','-v']
        with (out/'cpu_tests.log').open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=100,
                env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
        status.update(exit_code=proc.returncode,success=proc.returncode==0)
    except BaseException as exc:status['error']=type(exc).__name__
    finally:
        status.update(done=True,finished=time.time())
        path.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)

if __name__=='__main__':main()
