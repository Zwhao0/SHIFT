"""Small bounded real Manager/PG probe separating RPC and gradient deadlines."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import subprocess
import time


def worker(replica,cfg):
    if cfg['backend']=='gloo':os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import torch.distributed as dist
    from torchft import Manager,ProcessGroupGloo
    import torchft.process_group as groups
    from torchft.manager import WorldSizeMode
    import shift_full.compat as compat
    from shift_full.v4_outer import _ForbiddenCheckpointTransport
    torch.set_num_threads(1);compat.install_managed_callback_fix()
    if cfg['backend']=='nccl':
        torch.cuda.set_device(replica);torch.cuda.set_per_process_memory_fraction(.12,replica)
    log=Path(cfg['out'],f'R{replica}.jsonl').open('w',buffering=1);step=-1
    def emit(kind,**fields):
        if not log.closed:log.write(json.dumps(dict(kind=kind,replica=replica,step=step,
            pid=os.getpid(),monotonic=time.monotonic(),**fields))+'\n')
    native_context=compat.cancelling_timeout;native_stream=groups.stream_timeout
    @contextmanager
    def traced_context(callback,timeout):
        frame=sys._getframe(2);origin=frame.f_code.co_name
        started=time.monotonic()
        def fire():
            emit('timer_abort',origin='context/'+origin,timeout_seconds=timeout.total_seconds(),elapsed=time.monotonic()-started)
            callback()
        with native_context(fire,timeout):yield
    def traced_stream(callback,timeout):
        started=time.monotonic()
        def fire():
            emit('timer_abort',origin='cuda_stream',timeout_seconds=timeout.total_seconds(),elapsed=time.monotonic()-started)
            callback()
        return native_stream(fire,timeout)
    compat.cancelling_timeout=traced_context;groups.stream_timeout=traced_stream
    class CPUGroup(ProcessGroupGloo):
        def _create_pg(self,store,rank,world):
            old=self._global_ranks,self._group_rank
            try:
                self._global_ranks=self._group_rank=None
                return super()._create_pg(store,rank,world)
            finally:self._global_ranks,self._group_rank=old
    pg=(compat.BlockingNCCL if cfg['backend']=='nccl' else CPUGroup)(timeout=timedelta(seconds=cfg['pg_timeout']))
    manager=Manager(pg=pg,load_state_dict=lambda _:(_ for _ in ()).throw(RuntimeError('unexpected heal')),
        state_dict=lambda:(_ for _ in ()).throw(RuntimeError('unexpected export')),
        min_replica_size=2,world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,use_async_quorum=False,
        init_sync=False,max_retries=0,timeout=timedelta(seconds=90),quorum_timeout=timedelta(seconds=180),
        connect_timeout=timedelta(seconds=20),rank=0,world_size=1,store_addr='127.0.0.1',
        store_port=cfg['stores'][replica],hostname='127.0.0.1',lighthouse_addr=cfg['lighthouse'],
        replica_id=f'v4deadline{replica}',checkpoint_transport=_ForbiddenCheckpointTransport())
    control=dist.TCPStore('127.0.0.1',cfg['control'],None,False,
        timeout=timedelta(seconds=20),wait_for_workers=False)
    emit('ready',manager_timeout=manager._timeout.total_seconds(),quorum_timeout=manager._quorum_timeout.total_seconds(),
        pg_timeout=pg._timeout.total_seconds())
    try:
        for step in range(3):
            control.set(f'ready/{step}/{replica}','1');control.wait([f'ready/{step}/{r}' for r in range(2)])
            manager.start_quorum(allow_heal=False)
            assert manager.num_participants()==2
            emit('quorum_ready')
            if step==1 and cfg['cancel_timer']:
                try:
                    with compat.cancelling_timeout(pg.abort,timedelta(seconds=2)):
                        raise RuntimeError('injected guarded-operation rejection')
                except RuntimeError:emit('guard_exception_caught')
                time.sleep(2.5)
            if step==1 and replica==0 and cfg['delay']:
                emit('injected_delay',seconds=cfg['delay']);time.sleep(cfg['delay'])
            gradient=torch.full((128,),1. if replica==0 else 3.,device='cuda' if cfg['backend']=='nccl' else 'cpu')
            started=time.monotonic();emit('allreduce_begin')
            work=manager.allreduce(gradient,reduce_op=dist.ReduceOp.AVG)
            ok=work.wait()
            if cfg['backend']=='nccl':torch.cuda.synchronize()
            exact=bool(torch.equal(gradient,torch.full_like(gradient,2.)))
            emit('allreduce_end',seconds=time.monotonic()-started,wait_return=ok,
                error=repr(manager.errored()),native_pg_error=repr(pg.errored()),exact=exact)
            decision=manager.should_commit()
            emit('commit_decision',decision=decision)
            assert ok and exact and decision
            emit('commit',committed_step=manager.current_step())
        emit('done')
    except BaseException as error:
        emit('error',error=type(error).__name__+': '+str(error),
             manager_error=repr(manager.errored()),native_pg_error=repr(pg.errored()));raise
    finally:
        manager.shutdown();pg.shutdown();log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=('gloo','nccl'),default='nccl')
    p.add_argument('--pg-timeout',type=float,default=2);p.add_argument('--delay',type=float,default=0)
    p.add_argument('--cancel-timer',action='store_true');a=p.parse_args()
    if a.pg_timeout<=0 or a.delay<0:p.error('positive timeout and nonnegative delay required')
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.out.mkdir(exist_ok=False)
    def gpu_inventory():
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        entries=[line.split(',',1) for line in raw.splitlines() if line.strip()]
        assert len(entries)==8 and all('VLLM::' in name for _,name in entries),'expected idle SHIFT and eight company GPU PIDs'
        return {pid.strip():name.strip() for pid,name in entries}
    before=gpu_inventory() if a.backend=='nccl' else None
    import torch.distributed as dist
    from torchft._torchft import LighthouseServer
    control=dist.TCPStore('127.0.0.1',0,None,True,wait_for_workers=False)
    stores=[dist.TCPStore('127.0.0.1',0,None,True,wait_for_workers=False) for _ in range(2)]
    lighthouse=LighthouseServer(bind='[::]:0',min_replicas=2,join_timeout_ms=1000,
        quorum_tick_ms=100,heartbeat_timeout_ms=5000)
    cfg=dict(out=str(a.out),backend=a.backend,pg_timeout=a.pg_timeout,delay=a.delay,cancel_timer=a.cancel_timer,
        control=control.port,stores=[s.port for s in stores],lighthouse=lighthouse.address())
    processes=[mp.get_context('spawn').Process(target=worker,args=(r,cfg)) for r in range(2)]
    try:
        for process in processes:process.start()
        end=time.monotonic()+65
        for process in processes:process.join(max(0,end-time.monotonic()))
    finally:
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join(5)
        lighthouse.shutdown()
        events=[json.loads(line) for f in a.out.glob('R*.jsonl') for line in f.read_text().splitlines()]
        success=all(p.exitcode==0 for p in processes)
        aborts=[e for e in events if e['kind']=='timer_abort']
        result=dict(success=success,expected_timeout=a.delay>a.pg_timeout,
            exit_codes=[p.exitcode for p in processes],backend=a.backend,pg_timeout=a.pg_timeout,
            delay=a.delay,cancel_timer=a.cancel_timer,gpu_used=a.backend=='nccl',
            effective_deadlines=[{k:e[k] for k in ('manager_timeout','quorum_timeout','pg_timeout')} for e in events if e['kind']=='ready'],
            timers=[{k:e[k] for k in ('replica','step','origin','timeout_seconds','elapsed')} for e in aborts],
            errors=[{k:e[k] for k in ('replica','step','error','manager_error','native_pg_error')} for e in events if e['kind']=='error'],
            resources_clean=before is None or before==gpu_inventory(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        observed_timeout=(len(result['errors'])==2 and all(e['step']==1 for e in result['errors']) and
            (a.backend=='gloo' or any(e['step']==1 and e['timeout_seconds']==a.pg_timeout and
                                    a.pg_timeout<=e['elapsed']<a.pg_timeout+1 for e in aborts)))
        result['pass_expected']=(not success and observed_timeout) if result['expected_timeout'] else success and not aborts
        (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['pass_expected']:raise SystemExit(1)


if __name__=='__main__':main()
