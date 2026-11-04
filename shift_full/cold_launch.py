"""Launch fresh workers using only stdlib; no imports, model preload or fork pool."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nnodes',type=int,default=1)
    p.add_argument('--nproc_per_node',type=int,default=8)
    p.add_argument('--node_rank',type=int,default=0)
    p.add_argument('--max_restarts',type=int,default=0)
    p.add_argument('--master_addr',required=True)
    p.add_argument('--master_port',type=int,required=True)
    p.add_argument('--module',required=True)
    a,worker_args=p.parse_known_args()
    if a.max_restarts or a.nnodes<1 or a.nproc_per_node<1 or not 0<=a.node_rank<a.nnodes or not 0<a.master_port<65536:
        p.error('invalid fresh-worker topology or restart policy')
    stopped=False
    def stop(_signum,_frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    workers=[]
    result=1
    try:
        for local_rank in range(a.nproc_per_node):
            if stopped:break
            env=dict(os.environ, RANK=str(a.node_rank*a.nproc_per_node+local_rank),
                     LOCAL_RANK=str(local_rank), WORLD_SIZE=str(a.nnodes*a.nproc_per_node),
                     LOCAL_WORLD_SIZE=str(a.nproc_per_node), GROUP_RANK=str(a.node_rank),
                     GROUP_WORLD_SIZE=str(a.nnodes), ROLE_RANK=str(a.node_rank*a.nproc_per_node+local_rank),
                     ROLE_WORLD_SIZE=str(a.nnodes*a.nproc_per_node), ROLE_NAME='default',
                     MASTER_ADDR=a.master_addr, MASTER_PORT=str(a.master_port),
                     TORCHELASTIC_RESTART_COUNT='0', TORCHELASTIC_MAX_RESTARTS='0',
                     TORCHELASTIC_USE_AGENT_STORE='False')
            workers.append(subprocess.Popen([sys.executable,'-u','-m',a.module,*worker_args],env=env,
                                             start_new_session=True))
        print(json.dumps(dict(kind='fresh_launcher_started',pids=[w.pid for w in workers],
                              time=time.time(),preloaded=False)),flush=True)
        while not stopped and len(workers)==a.nproc_per_node:
            codes=[w.poll() for w in workers]
            failed=next((c for c in codes if c not in (None,0)),None)
            if failed is not None:
                result=failed if failed>0 else 128-failed
                break
            if all(c==0 for c in codes):
                result=0
                break
            time.sleep(.02)
        if stopped:result=143
    finally:
        # Each child created its own session. Only these owned groups are signaled.
        for w in workers:
            if w.poll() is None:
                try:os.killpg(w.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+3
        for w in workers:
            try:w.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(w.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                w.wait(timeout=3)
    return result


if __name__=='__main__':raise SystemExit(main())
