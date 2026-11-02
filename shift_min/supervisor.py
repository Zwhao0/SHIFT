"""Owns only this session's launcher/workers and optional Lighthouse."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import sys
import time


def descendants(parent):
    rows = sp.check_output(['ps', '-e', '-o', 'pid=,ppid='], text=True)
    pairs = [tuple(map(int, line.split())) for line in rows.splitlines() if line.strip()]
    result, frontier = [], [parent]
    while frontier:
        children = [pid for pid, ppid in pairs if ppid in frontier]
        result.extend(children)
        frontier = children
    return [pid for pid in result if Path(f'/proc/{pid}').exists()]


def terminate(proc, sig):
    if proc is None or proc.poll() is not None:
        return []
    victims = list(reversed(descendants(proc.pid))) + [proc.pid]
    for pid in victims:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    return victims


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--replica', type=int, required=True)
    p.add_argument('--host-ip', required=True)
    p.add_argument('--lighthouse-ip', default='10.10.4.2')
    p.add_argument('--root', default='/workspace/SHIFT')
    p.add_argument('--seconds', type=int, default=900)
    a = p.parse_args()
    root = Path(a.root)
    out = root / 'results' / a.session
    out.mkdir(parents=True, exist_ok=False)
    log = (out / 'supervisor.jsonl').open('a', buffering=1)
    handles = []

    def event(kind, **fields):
        log.write(json.dumps(dict(kind=kind, time=time.time(), rid=a.replica,
                                  supervisor_pid=os.getpid(), **fields)) + '\n')

    env = os.environ.copy()
    env.update(PYTHONPATH=f'{root}/vendor:{root}:/workspace/FLASH1022',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               MALLOC_ARENA_MAX='2', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               NCCL_MAX_NCHANNELS='4', NCCL_BUFFSIZE='1048576',
               NCCL_CUMEM_HOST_ENABLE='0', NCCL_CUMEM_ENABLE='0', NCCL_NVLS_ENABLE='0',
               NCCL_DEBUG='WARN', NCCL_NET='Socket', NCCL_IB_DISABLE='1',
               NCCL_SOCKET_IFNAME='bond0', GLOO_SOCKET_IFNAME='bond0',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TORCHFT_USE_OTEL='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               NO_PROXY='*', no_proxy='*', RUST_LOG='warn', TOKIO_WORKER_THREADS='2')
    env.pop('LD_PRELOAD', None)
    lighthouse = training = None
    generation = 0

    def launch():
        stream = (out / f'torchrun_g{generation}.log').open('w')
        handles.append(stream)
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
               '--nproc_per_node=8', '--max_restarts=0',
               f'--master_addr={a.host_ip}', f'--master_port={29860 + 10*a.replica + generation}',
               '--module', 'shift_min.train', '--session', a.session,
               '--replica', str(a.replica), '--generation', str(generation),
               '--host-ip', a.host_ip, '--lighthouse', f'http://{a.lighthouse_ip}:29771',
               '--max-seconds', str(a.seconds)]
        proc = sp.Popen(cmd, env=env, cwd=root, stdout=stream, stderr=sp.STDOUT,
                        start_new_session=True)
        event('launched', generation=generation, launcher_pid=proc.pid, command=cmd)
        return proc

    started = time.monotonic()
    killed = False
    stopping = False
    try:
        if a.replica == 0:
            stream = (out / 'lighthouse.log').open('w')
            handles.append(stream)
            lighthouse = sp.Popen(
                [f'{root}/vendor/bin/torchft_lighthouse', '--bind', '[::]:29771',
                 '--min_replicas', '1', '--heartbeat_timeout_ms', '1000',
                 '--join_timeout_ms', '1000', '--quorum_tick_ms', '100'],
                env=env, cwd=root, stdout=stream, stderr=sp.STDOUT,
                start_new_session=True)
            event('lighthouse_started', pid=lighthouse.pid)
            time.sleep(0.5)
            if lighthouse.poll() is not None:
                raise RuntimeError('Lighthouse failed to start')
        training = launch()
        event('supervisor_ready')
        while time.monotonic() - started < a.seconds:
            if (out / 'STOP').exists():
                stopping = True
                break
            if (out / 'KILL').exists() and not killed:
                assert a.replica == 1 and generation == 0
                if training.poll() is not None:
                    raise RuntimeError('Training exited before fault injection')
                victims = terminate(training, signal.SIGKILL)
                training.wait(timeout=15)
                event('injected', generation=generation, signal='SIGKILL', pids=victims)
                killed = True
            if (out / 'RESTART').exists() and generation == 0:
                assert killed and training.poll() is not None
                generation = 1
                training = launch()
                event('restart_started', generation=generation)
            if training.poll() is not None and not (killed and generation == 0):
                event('unexpected_exit', generation=generation, returncode=training.returncode)
                raise RuntimeError('Training exited unexpectedly')
            if lighthouse is not None and lighthouse.poll() is not None:
                raise RuntimeError('Lighthouse exited unexpectedly')
            time.sleep(0.1)
        event('stopping', requested=stopping)
    except BaseException as error:
        event('supervisor_error', error=repr(error))
        raise
    finally:
        if training is not None and training.poll() is None:
            (out / 'STOP').touch()
            try:
                training.wait(timeout=4)
            except sp.TimeoutExpired:
                terminate(training, signal.SIGTERM)
                try:
                    training.wait(timeout=4)
                except sp.TimeoutExpired:
                    terminate(training, signal.SIGKILL)
                    training.wait(timeout=10)
        if lighthouse is not None:
            terminate(lighthouse, signal.SIGTERM)
            try:
                lighthouse.wait(timeout=3)
            except sp.TimeoutExpired:
                terminate(lighthouse, signal.SIGKILL)
                lighthouse.wait(timeout=3)
        event('cleanup', trainer_returncode=training.poll() if training else None,
              lighthouse_returncode=lighthouse.poll() if lighthouse else None,
              own_children=descendants(os.getpid()))
        for stream in handles:
            stream.close()


if __name__ == '__main__':
    main()
