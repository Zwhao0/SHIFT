"""Bounded real FT0.2 multi-process lifecycle probe; no CUDA or training.

Actual OwnershipHandoff methods and real native Manager/Store/Gloo. Only CPU
local-device synchronization and model contribution are CPU test seams. Small
tensors do not establish model guard/hash cost or GPU performance admission.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace


def worker(role, rank, cfg):
    import torch
    import torch.distributed as dist
    from torchft import Manager, ProcessGroupGloo
    from torchft.manager import WorldSizeMode
    from torchft.checkpointing.transport import CheckpointTransport
    from shift_full.compat import install_managed_callback_fix
    from shift_full.ownership_handoff import OwnershipHandoff
    from shift_full.ownership_runtime import ComputedContribution, forbid_quorum_recovery, completed_quorum
    from shift_full.planned_handoff import EpochGate

    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'CPU-only environment required'
    # Explicit CPU local-device seam; all native RPCs/PGs remain real.
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.set_device = lambda *a, **k: None
    install_managed_callback_fix()
    out = Path(cfg['out'])
    log = (out / f'{role}{rank}.jsonl').open('w', buffering=1)
    def emit(kind, **fields):
        log.write(json.dumps(dict(kind=kind, role=role, rank=rank, pid=os.getpid(),
                                  monotonic=time.monotonic(), **fields))+'\n')
    host=cfg.get('control_host','127.0.0.1')
    coord = dist.TCPStore(host, cfg['coord'], None, False,
                          timeout=timedelta(seconds=25), wait_for_workers=False)
    h = OwnershipHandoff.__new__(OwnershipHandoff)
    h.role, h.rank, h.local_world = role, rank, cfg['world']
    h.cut, h.new_epoch, h.base_epoch, h.round, h.offset = 6, 1, 0, 0, cfg['offset']
    h.coord, h.coord_prefix, h.control_host = coord, '', host
    h.hosts = cfg.get('hosts',dict(A='127.0.0.1', B='127.0.0.1', C='127.0.0.1'))
    h.args = SimpleNamespace(host_ip=h.hosts[role])
    h.progress, h.gate, h.protocol = dict(updates=6), EpochGate(), None
    h.errors, h.threads, h.retired_pgs, h.local_pgs, h.channels = [], [], [], [], {}
    h.cleanup_thread, h.cleanup_cancel, h.device = None, threading.Event(), 0
    h.fallback = h.relay_done = h.old_epoch_revoked = h.ownership_active = False
    h.building_manager, h.candidate, h.presend = False, None, None
    h.reservation = h.contribution = None
    h.deferred, h.cancelled = True, False
    h.control_ready, h.reserved_ready = threading.Event(), threading.Event()
    h.control_done, h.control_stop = threading.Event(), threading.Event()
    h.control_error = h.wire = h.listener = h.control_thread = None
    h.emit, h.control_client = emit, lambda seconds: coord
    gradient = torch.full((32,), 1.0 if role == 'A' else 3.0)
    version = gradient._version
    h.owner_guard = lambda: gradient._version == version
    c_guard = lambda: True
    if cfg['anchor_guards']:
        from scripts.ownership_anchor_guard_fixture import anchor_guards
        h.owner_guard,c_guard = anchor_guards(Path(__file__).resolve().parents[1],rank,h,emit)
    managers = []

    class NoTransfer(CheckpointTransport):
        # Production PackedGPUTransport metadata/shutdown/disallow are these
        # same no-ops. No checkpoint transfer is legal in this epoch probe.
        def metadata(self): return '<packed-gpu>'
        def shutdown(self, wait=True): pass
        def disallow_checkpoint(self): pass
        def send_checkpoint(self, *a, **k): raise RuntimeError('unexpected send')
        def recv_checkpoint(self, *a, **k): raise RuntimeError('unexpected recv')

    class CPUGroup(ProcessGroupGloo):
        # Same tracing compatibility seam as planned_train; real Gloo.
        def _create_pg(self, store, replica_rank, world):
            saved = self._global_ranks, self._group_rank
            try:
                self._global_ranks = self._group_rank = None
                return super()._create_pg(store, replica_rank, world)
            finally:
                self._global_ranks, self._group_rank = saved

    def make(epoch):
        started = time.monotonic()
        rid = 0 if role == 'A' else 1
        def forbidden(*a):
            raise RuntimeError('unexpected checkpoint state access')
        manager = Manager(pg=CPUGroup(timeout=timedelta(seconds=10)),
            load_state_dict=forbidden, state_dict=forbidden, min_replica_size=2,
            world_size_mode=WorldSizeMode.FIXED_WITH_SPARES, use_async_quorum=bool(cfg.get('early') and epoch=='AC'),
            init_sync=False, timeout=timedelta(seconds=10), quorum_timeout=timedelta(seconds=10),
            connect_timeout=timedelta(seconds=10), rank=rank, world_size=cfg['world'],
            hostname=h.args.host_ip, lighthouse_addr=cfg['lh'][epoch],
            store_addr=h.args.host_ip, store_port=cfg['stores'][epoch][role],
            checkpoint_transport=NoTransfer(),
            replica_id=f"shift-own-{cfg['nonce']}-{epoch}-{rid}", max_retries=1)
        mismatch = cfg.get('early_case')=='future_reject' and epoch=='AC' and role=='C'
        manager.load_state_dict(dict(step=7 if mismatch else 6, batches_committed=12))
        managers.append(manager)
        client = manager._client
        class TraceClient:
            def __getattr__(self, name): return getattr(client, name)
            def _quorum(self, *a, **kw):
                response = client._quorum(*a, **kw)
                fields = ('quorum_id','replica_rank','replica_world_size','max_step',
                          'max_replica_rank','max_world_size','heal','recover_dst_replica_ranks',
                          'store_address','replica_ids')
                emit('native_response', epoch=epoch,
                     response={f:getattr(response, f) for f in fields})
                return response
        manager._client = TraceClient()
        emit('manager_constructed', epoch=epoch, seconds=time.monotonic()-started)
        return manager

    active = None
    try:
        if role in 'AB':
            h.control_thread = threading.Thread(target=h._prepare_control)
            h.control_thread.start()
            old = make('old')
            forbid_quorum_recovery(old, 0 if role == 'A' else 1)
            old.start_quorum(allow_heal=False)
            completed_quorum(old, step=6, replica_index=0 if role == 'A' else 1)
            h.old_replica_index = old._participating_replica_rank
            coord.set(f'old_ready/{role}/{rank}', '1')
        else:
            old = None
        h.wait_keys([f'old_ready/{r}/{i}' for r in 'AB' for i in range(cfg['world'])])
        coord.set(f'cpu_ready/{role}/{rank}', '1')
        h.wait_keys([f'cpu_ready/{r}/{i}' for r in 'ABC' for i in range(cfg['world'])])
        started = time.monotonic()
        emit('control_begin')
        if role == 'B':
            h.reserve_descriptor(old, dict(replica_index=1), [])
            # Synthetic wait tests asynchronous fence and deferred cleanup;
            # it is not a simulated model workload or a performance result.
            time.sleep(.35)
            h._b_closed()
        elif role == 'A':
            h._a_closed(old)
        else:
            result = dict(descriptor=dict(replica_index=1))
            h.contribution = ComputedContribution(ddp=SimpleNamespace(), result=result,
                unchanged=lambda: c_guard() and not (cfg['failure'] and rank == 0), rollback=lambda: None)
            for proof in ('initial_audit','computed','gradient_audit'):
                if not cfg.get('early'):h.proof(proof, value=True)
        prospective = binding = None
        if cfg.get('early') and role in 'AC':
            from shift_full.ownership_future import ProspectiveQuorum
            boundary_version=[0]
            binding=(asdict(h.ticket()), role, rank, 6, boundary_version[0])
            current_binding=lambda:(asdict(h.ticket()),role,rank,h.progress['updates'],boundary_version[0])
            guard=h.owner_guard if role=='A' else c_guard
            m=make('AC')
            assert old is None or (m is not old and m._pg is not old._pg)
            prospective=ProspectiveQuorum(m,cut=h.ticket(),replica_index=0 if role=='A' else 1,
                binding=binding,unchanged=lambda b:b==current_binding() and guard())
            try:
                # Fresh nine hashes from actual trainer function, real tiny CPU
                # state; CUDA RNG entry is an explicitly CPU-only proxy.
                import ast, random
                from shift_full.planned_train import digest
                source=ast.parse((Path(__file__).resolve().parents[1]/'shift_full/planned_train.py').read_text())
                fn=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='current_hashes')
                model=torch.arange(262144,dtype=torch.float32)
                tiny_rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.zeros(16,dtype=torch.uint8),
                    python_rng=random.Random(7).getstate(),tracker_rng={'tiny':torch.zeros(16,dtype=torch.uint8)})
                scope=dict(torch=torch,digest=digest,stage=SimpleNamespace(state_dict=lambda:{'tiny':model},
                    named_buffers=lambda:[('tiny',model[:8])]),optimizer=SimpleNamespace(state_dict=lambda:{'adam':model.clone()}),
                    scheduler=SimpleNamespace(state_dict=lambda:{'step':6}),progress=h.progress,rng_snapshot=lambda:tiny_rng)
                torch.cuda.current_stream=lambda:SimpleNamespace(synchronize=lambda:None)
                exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_current_hashes_cpu','exec'),scope)
                expected_hashes=scope['current_hashes']()
                emit('prospective_start');prospective.start()
                future=m._quorum_future
                future.add_done_callback(lambda f:emit('prospective_future_done',error=repr(f.exception())))
                emit('early_initial_begin',cpu_tiny_state=True,cuda_rng_proxy=True)
                hashes=scope['current_hashes']()
                assert hashes==expected_hashes and len(hashes)==9
                emit('early_initial_done',hashes=hashes)
                if cfg.get('early_case')=='stale_k' and role=='C' and rank==0:
                    h.progress['updates']=7;boundary_version[0]+=1
                emit('prospective_wait_begin');prospective.before_forward()
                emit('prospective_before_forward',binding=binding)
                if role=='C':
                    for proof in ('initial_audit','computed','gradient_audit'):h.proof(proof,value=True)
            except Exception as exc:
                emit('prospective_rejected',error=repr(exc),state=prospective.state,
                    future_done=m._quorum_future.done(),configured=m._quorum_id)
                h.fail(exc)
                # Finish the actual executor before disposal. Never resurrect it.
                try:m.wait_quorum()
                except Exception:pass
                m.shutdown();h.retired_pgs.append(m._pg)
                prospective=None
        members = h._select()
        if members=='AC' and role in 'AC':
            active = h.prepare_manager(lambda: make('AC'), 'AC',prospective=prospective,
                                       binding=current_binding() if cfg.get('early') else binding)
            accepted = h.activate_manager(active, 'AC')
            if not accepted:
                active = None
                if role == 'C':
                    h.contribution.cancel()
                    h.proof('cancelled', value=True)
        elif members=='AC':
            h.wait_keys(['ownership_activation_AC'])
            accepted = coord.get('ownership_activation_AC') == b'AC'
        else:
            accepted=False
            if role=='C':
                h.contribution.cancel();h.proof('cancelled',value=True)
        if not accepted and role in 'AB':
            h.fallback = True
            active = h.prepare_manager(lambda: make('AB'), 'AB')
            assert h.activate_manager(active, 'AB')
        emit('control_complete', seconds=time.monotonic()-started,
             decision='AC' if accepted else 'AB', active=active is not None)
        if active is not None:
            active.allreduce(gradient).wait()
            assert torch.equal(gradient, torch.full_like(gradient, 2.0)), gradient
            assert active.should_commit()
            assert active.current_step() == 7 and active.batches_committed() == 14
            h.proof('cpu_commit', value=7)
        else:
            h.proof('cpu_retired', value=True)
        selected = 'AC' if accepted else 'AB'
        h.wait_keys(h.keys(selected, 'cpu_commit'))
        if role == 'B' and accepted:
            h.start_retired_cleanup()
        if h.cleanup_thread is not None:
            h.finish_retired_cleanup()
        if old is not None:
            for method in ('start_quorum','allreduce','should_commit','shutdown'):
                try:
                    getattr(old, method)()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('old Manager reused: '+method)
        emit('success', selected=selected)
    except BaseException as exc:
        emit('error', error=repr(exc), traceback=traceback.format_exc())
        coord.set('ownership_fatal', repr(exc))
        raise
    finally:
        emit('cleanup_begin')
        h.cancel_cleanup_on_exit()
        for manager in managers:
            if h.reservation is not None and manager is h.reservation.manager:
                continue
            manager.shutdown()
        # Native parent cleanup only after all A/B closure proofs.
        if coord.check(h.keys('AB', 'closed')):
            for manager in managers:
                manager._pg.abort(errored=False)
                manager._pg.shutdown()
        parents_gone=all(getattr(m._pg,'_pg',None) is None for m in managers)
        assert parents_gone, 'native parent remains after global closure'
        emit('owner_exit', threads=threading.active_count(),parents_gone=parents_gone)


def case(out, failure, world, anchor_guards):
    import torch.distributed as dist
    from torchft._torchft import LighthouseServer
    out.mkdir()
    stores, lights = [], []
    coord = dist.TCPStore('127.0.0.1', 0, None, True, wait_for_workers=False)
    cfg = dict(out=str(out), world=world, failure=failure, coord=coord.port,
               offset=500, nonce=out.name, lh={}, stores={}, anchor_guards=anchor_guards)
    for epoch, roles in [('old','AB'),('AC','AC'),('AB','AB')]:
        lh = LighthouseServer(bind='[::]:0', min_replicas=2, join_timeout_ms=1000,
                              quorum_tick_ms=100, heartbeat_timeout_ms=1000)
        lights.append(lh); cfg['lh'][epoch] = lh.address(); cfg['stores'][epoch] = {}
        for role in roles:
            store = dist.TCPStore('127.0.0.1', 0, None, True, wait_for_workers=False)
            stores.append(store); cfg['stores'][epoch][role] = store.port
    coord.set('cut','6'); coord.set('ownership_nonce', cfg['nonce'])
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    ctx = mp.get_context('spawn')
    processes = [ctx.Process(target=worker, args=(role, rank, cfg))
                 for role in 'ABC' for rank in range(world)]
    try:
        for p in processes: p.start()
        deadline = time.monotonic()+105
        for p in processes: p.join(max(0, deadline-time.monotonic()))
        return dict(success=all(p.exitcode == 0 for p in processes),
                    exit_codes=[p.exitcode for p in processes], pids=[p.pid for p in processes])
    finally:
        for p in processes:
            if p.is_alive(): p.terminate()
        for p in processes: p.join(5)
        for lh in lights: lh.shutdown()


def role_main(role, config):
    """Remote role supervisor; bounded by its own systemd cgroup and lock."""
    import torch.distributed as dist
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads(Path(config).read_text());out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    stores=[dist.TCPStore(cfg['hosts'][role],cfg['stores'][epoch][role],None,True,
                         wait_for_workers=False) for epoch in cfg['stores'] if role in cfg['stores'][epoch]]
    ctx=mp.get_context('spawn');processes=[ctx.Process(target=worker,args=(role,r,cfg)) for r in range(cfg['world'])]
    try:
        for p in processes:p.start()
        deadline=time.monotonic()+105
        for p in processes:p.join(max(0,deadline-time.monotonic()))
        result=dict(success=all(p.exitcode==0 for p in processes),exit_codes=[p.exitcode for p in processes])
        (out/f'role_{role}.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:p.join(5)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--suffix')
    parser.add_argument('--role',choices=['B','C']);parser.add_argument('--config')
    parser.add_argument('--world', type=int, default=8, choices=(1,8))
    parser.add_argument('--anchor-guards', action='store_true')
    args=parser.parse_args()
    if args.role:
        if not args.config:parser.error('role requires config')
        return role_main(args.role,args.config)
    if not args.suffix or not all(c.isalnum() or c in '_-' for c in args.suffix):
        parser.error('invalid suffix')
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'runtime/v1-controller/credentials').unlink(missing_ok=True)
    out=root/f'results/v3_p10b3_{args.suffix}';out.mkdir(exist_ok=False)
    status=dict(done=False, success=False, training_started=False, injections=0,
                started=time.time(), scope='real native FT CPU lifecycle, tiny tensor guards; no complete budget/GPU admission')
    path=out/'status.json';path.write_text(json.dumps(status))
    try:
        manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for d in ('shift_full','scripts') for p in (root/d).glob('*.py')}
        (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        status['cases']={}
        for name, failure in [('success',False),('guard_fallback',True)]:
            result=case(out/name, failure, args.world, args.anchor_guards)
            status['cases'][name]=result
            path.write_text(json.dumps(status,indent=2)+'\n')
            if not result['success']: break
        status['success']=len(status['cases'])==2 and all(r['success'] for r in status['cases'].values())
    except BaseException as exc:
        status['error']=repr(exc);status['traceback']=traceback.format_exc()
    finally:
        status.update(done=True,finished=time.time());path.write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)


if __name__=='__main__': main()
