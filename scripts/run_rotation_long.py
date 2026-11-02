"""One-hour, 60-fault audited run, with bounded all-node resource sampling."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session, endpoints_from_env


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/v1';out.mkdir(parents=True,exist_ok=True)
    done=threading.Event()
    probe=Session.__new__(Session);probe.root=root;probe._sudo_password=None
    endpoints=list(endpoints_from_env().values())
    def sample(endpoint):
        try:
            gpu=probe.execute_endpoint(endpoint,['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw','--format=csv,noheader'])
            proc=probe.execute_endpoint(endpoint,['python','-c',
                'from pathlib import Path; from collections import Counter; '
                'rows=[p.joinpath("stat").read_text().split() for p in Path("/proc").glob("[0-9]*") if p.joinpath("stat").exists()]; '
                'print(dict(Counter(r[2] for r in rows)))'])
            return dict(time=time.time(),physical_node=endpoint.physical_node,gpu=gpu,process_states=proc)
        except Exception as e:
            return dict(time=time.time(),physical_node=endpoint.physical_node,error=str(e))
    def monitor():
        with (out/'long_resources.jsonl').open('w',buffering=1) as stream, concurrent.futures.ThreadPoolExecutor(3) as pool:
            while not done.is_set():
                for row in pool.map(sample,endpoints):
                    stream.write(json.dumps(row)+'\n')
                done.wait(30)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        subprocess.run([sys.executable,'-m','shift_full.rotation','--session','v1_rotation_long_20261102_01',
                        '--config','configs/shift_rotation_verify.json','--seconds','4800','--faults','60',
                        '--interval','60','--duration','3600','--seed','20261102'],cwd=root,check=True)
    finally:
        done.set();thread.join(timeout=45)


if __name__=='__main__':
    main()
