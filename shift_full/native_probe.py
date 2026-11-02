"""Native torchFT communication and state-size probe (not a model benchmark)."""

import time

_MODULE_STARTED = time.monotonic()

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading

import torch
import torch.distributed as dist


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def digest(value):
    hasher = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda value: (type(value).__name__, str(value))):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(str(type(item)).encode())
            for child in item:
                add(child)
        else:
            hasher.update(repr(item).encode() + b"\0")

    add(value)
    return hasher.hexdigest()


def tensor_count(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(item) for item in value)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--replica", type=int, required=True, choices=(0, 1))
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--lighthouse", default="http://10.10.4.2:29771")
    parser.add_argument("--root", default="/workspace/SHIFT")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--target-steps", "--steps", dest="steps", type=int, default=1_000_000)
    parser.add_argument("--numel", type=int, default=125_000_000)
    parser.add_argument("--backend", "--pg", "--process-group", dest="pg_backend",
                        choices=("nccl", "baby"), default="nccl")
    parser.add_argument("--transport", choices=("gpu",), default="gpu")
    parser.add_argument("--audit-state", action="store_true")
    parser.add_argument("--hash-every", type=int, default=0)
    parser.add_argument("--sync-quorum", action="store_true")
    parser.add_argument("--measure-step", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.generation < 0 or args.max_seconds < 1 or args.steps < 1 or args.numel < 1:
        parser.error("generation must be nonnegative; max-seconds, target-steps, and numel must be positive")
    if args.hash_every < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("hash-every must be nonnegative and lr positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    if world_size != 8:
        raise RuntimeError(f"native_probe expects 8 local ranks, got WORLD_SIZE={world_size}")
    output_dir = Path(args.root) / "results" / args.session
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"events_g{args.generation}_r{local_rank}.jsonl"
    log = log_path.open("a", buffering=1, encoding="utf-8")
    log_lock = threading.Lock()

    def event(kind, **fields):
        row = dict(kind=kind, time=time.time(), monotonic=time.monotonic(),
                   rid=args.replica, rank=local_rank, generation=args.generation,
                   pid=os.getpid())
        row.update(fields)
        with log_lock:
            log.write(json.dumps(row, sort_keys=True) + "\n")

    def phase(name, started):
        event("phase", name=name, seconds=time.monotonic() - started)
        return time.monotonic()

    phase_started = phase("python_imports", _MODULE_STARTED)
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    phase_started = phase("runtime_cuda_setup", phase_started)

    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    if dist.get_world_size() != 8:
        raise RuntimeError(f"expected 8 local NCCL ranks, got {dist.get_world_size()}")
    phase_started = phase("default_process_group", phase_started)

    parameter = torch.nn.Parameter(torch.zeros(args.numel, dtype=torch.bfloat16, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=args.lr, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)
    progress = {"updates": 0}
    state_lock = threading.RLock()
    boundary_rng = None
    audit_joint_pending = False
    manager = None
    compute_stream = torch.cuda.current_stream(device)
    phase_started = phase("parameter_optimizer_init", phase_started)

    def rng_snapshot():
        return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                    python_rng=random.getstate())

    def current_hashes(rng=None):
        torch.cuda.current_stream(device).synchronize()
        rng = rng if rng is not None else rng_snapshot()
        return dict(model=digest({"weight": parameter.detach()}),
                    optimizer=digest(optimizer.state_dict()),
                    scheduler=digest(scheduler.state_dict()), progress=digest(progress),
                    cpu_rng=digest(rng["torch_rng"]), cuda_rng=digest(rng["cuda_rng"]),
                    python_rng=digest(rng["python_rng"]))

    def export_state():
        nonlocal audit_joint_pending
        with state_lock:
            compute_stream.synchronize()
            step = manager.current_step() if manager else progress["updates"]
            if step != progress["updates"]:
                raise RuntimeError(f"manager/model step mismatch: {step} != {progress}")
            audit_joint_pending = True
            rng = (boundary_rng["rng"] if boundary_rng and boundary_rng["step"] == step
                   else rng_snapshot())
            state = clone_tree(dict(model={"weight": parameter.detach()},
                                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                    progress=dict(progress), **rng, source_replica=args.replica,
                                    source_rank=local_rank, step=step))
            state["hashes"] = current_hashes(rng) if args.audit_state else {}
            torch.cuda.current_stream(device).synchronize()
            event("snapshot", step=step, hashes=state["hashes"],
                  optimizer_tensors=tensor_count(state["optimizer"].get("state", {})))
            return state

    def load_state(state):
        nonlocal audit_joint_pending
        load_started = time.monotonic()
        with state_lock:
            if state["source_rank"] != local_rank or state["step"] != state["progress"]["updates"]:
                raise RuntimeError("restored state identity or progress mismatch")
            if manager and state["step"] != manager.current_step():
                raise RuntimeError("restored step differs from Manager.current_step")
            with torch.no_grad():
                parameter.copy_(state["model"]["weight"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            progress.update(state["progress"])
            audit_joint_pending = True
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], device=device)
            random.setstate(state["python_rng"])
            loaded_at = time.monotonic()
            hashes = current_hashes() if args.audit_state else {}
            expected = state.get("hashes", {})
            matches = hashes == expected
            event("restored", step=state["step"], load_seconds=loaded_at - load_started,
                  audit_seconds=time.monotonic() - loaded_at, source_replica=state["source_replica"],
                  source_rank=state["source_rank"], hashes_match=matches,
                  expected_hashes=expected, actual_hashes=hashes,
                  optimizer_tensors=tensor_count(optimizer.state_dict().get("state", {})),
                  cuda_allocated_bytes=torch.cuda.memory_allocated(local_rank),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(local_rank))
            if not matches:
                raise RuntimeError("restored state hash mismatch")

    from shift_full.compat import install_baby_event_fix, install_managed_callback_fix
    install_managed_callback_fix()
    if args.pg_backend == "baby":
        install_baby_event_fix()

    from torchft import Manager, ProcessGroupBabyNCCL, ProcessGroupNCCL
    from shift_full.transport import PackedGPUTransport

    pg_class = ProcessGroupBabyNCCL if args.pg_backend == "baby" else ProcessGroupNCCL
    pg = pg_class(timeout=timedelta(seconds=30 if args.pg_backend == "baby" else 10))
    transport = PackedGPUTransport(pg, device, timedelta(seconds=90), event)
    manager = Manager(pg=pg, load_state_dict=load_state, state_dict=export_state,
                      min_replica_size=1, use_async_quorum=not args.sync_quorum,
                      init_sync=False, timeout=timedelta(seconds=90),
                      quorum_timeout=timedelta(seconds=180), connect_timeout=timedelta(seconds=20),
                      hostname=args.host_ip, lighthouse_addr=args.lighthouse,
                      replica_id=f"shift{args.replica}", checkpoint_transport=transport, max_retries=8)
    phase_started = phase("manager_transport_init", phase_started)

    torch.cuda.synchronize(device)
    phase_started = phase("initial_cuda_sync", phase_started)
    initial_hashes = current_hashes() if args.audit_state else {}
    phase_started = phase("initial_audit", phase_started)
    event("ready", physical_rank=args.replica * 8 + local_rank, layout="native_probe",
          tp=1, pp=1, ep=1, local_dense_dp=1, local_expert_dp=1,
          model_id="native_probe.single_bf16_parameter", model_revision="deterministic",
          full_parameters=args.numel, local_parameters=args.numel, hashes=initial_hashes,
          torch_version=torch.__version__, cuda_version=torch.version.cuda,
          nccl_version=torch.cuda.nccl.version(), global_rank=dist.get_rank(), world_size=8,
          transport="gpu", cross_gradient=args.pg_backend, mode="shift", fp32=False,
          optimizer_tensors=tensor_count(optimizer.state_dict().get("state", {})),
          cuda_allocated_bytes=torch.cuda.memory_allocated(local_rank),
          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(local_rank))

    deadline = time.monotonic() + args.max_seconds
    while not (output_dir / "START").exists():
        if (output_dir / "STOP").exists() or time.monotonic() >= deadline:
            raise RuntimeError("Stopped or timed out before START")
        time.sleep(0.1)

    def requested_stop():
        marker = output_dir / "STOP"
        if not marker.exists():
            return False
        text = marker.read_text().strip()
        return not text or progress["updates"] >= int(text)

    attempts = 0
    try:
        while not requested_stop() and time.monotonic() < deadline and progress["updates"] < args.steps:
            iteration_started = time.monotonic()
            step = progress["updates"]
            boundary_rng = dict(step=step, rng=rng_snapshot())
            manager.start_quorum()
            compute_started = time.monotonic()

            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            gradient_value = ((step * 17 + local_rank * 13) % 29 + 1) / 1024.0
            parameter.grad.fill_(gradient_value)
            reduce_work = manager.allreduce(parameter.grad)
            reduce_work.wait()
            finite_started = time.monotonic()
            finite = bool(torch.isfinite(parameter.grad).all().item())
            finite_done = time.monotonic()
            if not finite:
                error = RuntimeError("nonfinite deterministic gradient")
                manager.report_error(error)
                raise error

            decision_started = time.monotonic()
            commit = manager.should_commit()
            decision_done = time.monotonic()
            attempts += 1
            if commit:
                participants = manager.num_participants()
                contributed = manager.is_participating()
                with state_lock:
                    optimizer.step()
                    scheduler.step()
                    progress["updates"] += 1
                if args.measure_step:
                    torch.cuda.current_stream(device).synchronize()
                audit_joint = (args.audit_state and audit_joint_pending
                               and participants >= 2 and contributed)
                regular_hash = (args.hash_every
                                and progress["updates"] % args.hash_every == 0)
                hashes = current_hashes() if (
                    regular_hash or audit_joint
                ) else {}
                if audit_joint:
                    audit_joint_pending = False
                event("commit", step=progress["updates"], updates=progress["updates"],
                      participants=participants, contributed=contributed,
                      hashes=hashes, model_hash=hashes.get("model"),
                      optimizer_hash=hashes.get("optimizer"), scheduler_hash=hashes.get("scheduler"),
                      optimizer_tensors=tensor_count(optimizer.state_dict().get("state", {})),
                      cross_calls=1, finite_gradients=finite,
                      seconds=time.monotonic() - iteration_started,
                      compute_seconds=decision_started - compute_started,
                      schedule_seconds=decision_started - compute_started,
                      gradient_check_seconds=finite_done - finite_started,
                      decision_seconds=decision_done - decision_started, loss=[])
            else:
                event("skipped", step=step, error=str(manager.errored()),
                      seconds=time.monotonic() - iteration_started,
                      compute_seconds=decision_started - compute_started,
                      decision_seconds=decision_done - decision_started)
            if manager.current_step() != progress["updates"]:
                raise RuntimeError(f"Manager/model step mismatch: {manager.current_step()} != {progress}")

        event("stopped", step=progress["updates"], attempts=attempts)
        dist.barrier()
    finally:
        try:
            manager.shutdown()
            manager._pg.abort(errored=False)
            manager._pg.shutdown()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            log.flush()
            log.close()


if __name__ == "__main__":
    main()
