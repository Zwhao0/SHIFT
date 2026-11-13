"""Launch FLASH's frozen trainer with explicit ranks in a Lab runner Pod."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker():
    from flash_v4 import observer_runtime, train_entry

    train_entry.LOCAL_WORLD_SIZE = int(os.environ['LOCAL_WORLD_SIZE'])
    if train_entry.LOCAL_WORLD_SIZE == 4:
        from flash_v1 import rank_layout

        initialize = rank_layout.initialize_node_local

        def initialize_on_split_host(*args, **kwargs):
            # The physical PP grouping remains eight ranks across both Pods.
            os.environ['LOCAL_WORLD_SIZE'] = '8'
            try:
                return initialize(*args, **kwargs)
            finally:
                os.environ['LOCAL_WORLD_SIZE'] = '4'

        rank_layout.initialize_node_local = initialize_on_split_host
    target = os.getenv('FLASH_LAB_GEMM_DELAY_RANK')
    if target == os.environ['RANK']:
        install = observer_runtime.install
        log_path = Path(os.environ['FLASH_V4_ACTUATION_DIR_PRIVATE']) / 'gemm_delay.jsonl'

        def install_with_delay(*args):
            observer = install(*args)
            delay = ctypes.CDLL(None).flash_lab_gemm_delay
            delay.argtypes = [ctypes.c_int]
            delay.restype = ctypes.c_int
            begin, end = observer.begin, observer.end

            def start(iteration):
                begin(iteration)
                delay(int(36 <= iteration <= 67))

            def finish(iteration):
                count = delay(0)
                end(iteration)
                if 36 <= iteration <= 67:
                    with log_path.open('a') as log:
                        log.write(json.dumps(dict(iteration=iteration, delay_launches=count)) + '\n')

            observer.begin, observer.end = start, finish
            return observer

        observer_runtime.install = install_with_delay
    train_entry.main(sys.argv[2:])


def launch():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-base', type=int, choices=(0, 8, 12), required=True)
    parser.add_argument('--slots', type=int, choices=(4, 8), required=True)
    parser.add_argument('--master-addr', required=True)
    parser.add_argument('--master-port', type=int, required=True)
    args, training = parser.parse_known_args()
    children = []
    try:
        for local_rank in range(args.slots):
            env = dict(os.environ, MASTER_ADDR=args.master_addr, MASTER_PORT=str(args.master_port),
                       WORLD_SIZE='16', LOCAL_WORLD_SIZE=str(args.slots), LOCAL_RANK=str(local_rank),
                       RANK=str(args.rank_base + local_rank))
            children.append(subprocess.Popen([sys.executable, __file__, 'worker', *training], env=env))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                return 1
            time.sleep(0.2)
        return int(any(child.returncode for child in children))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    if sys.argv[1:2] == ['worker']:
        worker()
    else:
        sys.exit(launch())
