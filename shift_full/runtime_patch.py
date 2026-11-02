"""Version-bound runtime adapter for the unmodified FLASH train_pipeline.py.

This is intentionally narrow, not a generic Megatron integration. It binds to
one source SHA256 with exactly one ``main()`` world-size guard of the form
``dist.get_world_size() != 8 * args.tensor_parallel_size``. The adapter spoofs
only that call to 16; all other distributed world-size queries remain real.
It requires TP2/PP2/EP2 on eight local ranks, the original ``batches`` local
immediately before ``optimizer.zero_grad()``, an SGD factory call after
``megatron_adapter.wrap``, and the captured DDP bucket layout. It replaces the
original random batches with a deterministic pattern keyed by committed step,
replica, and local dense-DP rank. No source file is edited.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import threading
import time
import types


class StopTraining(Exception):
    """Internal orderly stop raised at the original optimizer boundary."""


PATCHES = [
    "isolated runtime loads the real legacy Core through its package namespace, bypassing unused full-training/RETRO eager imports",
    "FLASH_PARALLEL_LAYOUT=legacy for each independent TP2/PP2/EP2 replica",
    "spoof only the unique original main() group=None world-size guard as 16; require actual local world size 8",
    "capture megatron_adapter.wrap and all-reduce every dense and expert DDP bucket across replicas",
    "replace torch.optim.SGD with a commit-gated proxy over real SGD(momentum=0.9), including optimizer and StepLR recovery state",
    "at zero_grad, start synchronous torchft quorum and overwrite source-local batches with committed-step/replica/local-DP pattern tokens",
    "replace RedisWriter with an in-process no-op sink that preserves reader/connection close calls",
    "use the selected BabyNCCL, blocking NCCL, or HybridGloo process group; force native source init/new_group to blocking=True and TORCH_NCCL_USE_COMM_NONBLOCKING=0",
    "install Baby IPC event fix only for baby; install managed callback fix for baby and hybrid backends",
    "prepend worker_bootstrap/sitecustomize.py to child PYTHONPATH so Baby spawn workers install both fixes",
    "cache CPU Torch, CUDA, and Python RNG states at committed progress boundaries for state exports",
    "wait for START in the worker; empty STOP stops immediately and numeric STOP stops at its committed-step boundary",
    "STOP, target steps, and timeout raise StopTraining at zero_grad",
]
EXPECTED_SOURCE_SHA256 = "cb22f32feb0b4dd39535b0d1d818f0e6701eb9504b2a767c64111c47b05c9067"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_version_bound_guard(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "main"), None)
    if main is None:
        raise RuntimeError("source must define top-level main()")

    def is_target(node):
        return (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and isinstance(node.left.func.value, ast.Name)
            and node.left.func.value.id == "dist"
            and node.left.func.attr == "get_world_size"
            and not node.left.args and not node.left.keywords
            and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.BinOp)
            and isinstance(node.comparators[0].left, ast.Constant)
            and node.comparators[0].left.value == 8
            and isinstance(node.comparators[0].op, ast.Mult)
            and isinstance(node.comparators[0].right, ast.Attribute)
            and isinstance(node.comparators[0].right.value, ast.Name)
            and node.comparators[0].right.value.id == "args"
            and node.comparators[0].right.attr == "tensor_parallel_size"
        )

    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "dist"
             and node.func.attr == "get_world_size"
             and not node.args and not node.keywords]
    guards = [node for node in ast.walk(main) if is_target(node)]
    if len(calls) != 1 or len(guards) != 1:
        raise RuntimeError(
            "unsupported train_pipeline.py: expected exactly one zero-argument "
            "dist.get_world_size() main guard, comparing against 8*args.tensor_parallel_size"
        )
    return guards[0].lineno


def digest(value, torch_module) -> str:
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch_module.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch_module.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(str(type(item)).encode())
            for child in item:
                add(child)
        else:
            h.update(repr(item).encode() + b"\0")

    add(value)
    return h.hexdigest()


def tensor_count(value, torch_module) -> int:
    if isinstance(value, torch_module.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(item, torch_module) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_count(item, torch_module) for item in value)
    return 0


class NullReader:
    def close(self):
        pass


class NullRedisWriter:
    """RedisWriter-shaped sink: no Redis connection or external service."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.reader = NullReader()
        self.connection = NullReader()

    def write(self, *args, **kwargs):
        del args, kwargs


class OptimizerProxy:
    """Original-loop optimizer facade; inner is a real torch.optim.SGD."""

    def __init__(self, runtime, inner):
        self.runtime = runtime
        self.inner = inner

    def zero_grad(self, *args, **kwargs):
        self.runtime.before_zero_grad()
        return self.inner.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        return self.runtime.committed_step(self.inner, closure)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        return self.inner.load_state_dict(state)

    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state


class RuntimePatch:
    def __init__(self, *, script, root, session, replica, generation,
                 host_ip, lighthouse, target_steps, max_seconds, backend="hybrid"):
        self.script = Path(script).resolve()
        self.root = Path(root).resolve()
        self.session = session
        self.replica = replica
        self.generation = generation
        self.host_ip = host_ip
        self.lighthouse = lighthouse
        self.target_steps = target_steps
        self.max_seconds = max_seconds
        if backend not in {"blocking", "baby", "hybrid"}:
            raise ValueError("backend must be 'blocking', 'baby', or 'hybrid'")
        self.backend = backend
        self.guard_line = find_version_bound_guard(self.script)
        self.source_sha256 = sha256_file(self.script)
        if self.source_sha256 != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(
                "unsupported train_pipeline.py SHA256; runtime patches are bound to "
                f"{EXPECTED_SOURCE_SHA256}, got {self.source_sha256}"
            )
        self.rank = int(os.environ["LOCAL_RANK"])
        self.session_dir = self.root / "results" / session
        self.output_dir = self.session_dir / f"native_g{generation}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.session_dir / f"events_g{generation}_r{self.rank}.jsonl"
        self.event_log = self.event_path.open("a", buffering=1, encoding="utf-8")
        self.event_lock = threading.Lock()
        self.deadline = time.monotonic() + max_seconds
        self.progress = {"updates": 0}
        self.rng_boundary = None
        self.optimizer = None
        self.inner_optimizer = None
        self.scheduler = None
        self.stage = None
        self.ddp = None
        self.state = None
        self.manager = None
        self.transport = None
        self.worker_bootstrap_path = None
        self.cross_calls = 0
        self.attempts = 0
        self.guard_runtime_hits = 0
        self.closed = False
        self._installed = False

    def event(self, kind, **fields):
        row = dict(kind=kind, time=time.time(), rid=self.replica, rank=self.rank,
                   generation=self.generation, pid=os.getpid())
        row.update(fields)
        with self.event_lock:
            self.event_log.write(json.dumps(row, sort_keys=True) + "\n")

    def install(self):
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "0"
        import torch
        import torch.distributed as dist
        import torch._dynamo  # Initialize optimizer class references before the SGD factory patch.

        from shift_full.compat import bootstrap_legacy_core
        self.legacy_core_namespace = bootstrap_legacy_core(self.script.parents[1]) if os.environ.get("SHIFT_ISOLATED_RUNTIME") == "1" else False
        self.torch = torch
        self.dist = dist
        self.original_get_world_size = dist.get_world_size
        self.original_init_process_group = dist.init_process_group
        self.original_new_group = dist.new_group
        self.adapter = __import__("flash_v1.megatron_adapter", fromlist=["wrap"])
        self.original_wrap = self.adapter.wrap
        self.flash_train = __import__("flash_v1.train", fromlist=["RedisWriter"])
        self.original_redis_writer = self.flash_train.RedisWriter
        self.original_sgd = torch.optim.SGD

        def nccl_options(positional, keywords, *, new_group):
            positional = list(positional)
            backend_index = 2 if new_group else 0
            options_index = 3 if new_group else 7
            backend = keywords.get("backend", positional[backend_index]
                                   if len(positional) > backend_index else None)
            if backend not in (None, "nccl"):
                return tuple(positional), keywords
            options = keywords.get("pg_options")
            if options is None and len(positional) > options_index:
                options = positional[options_index]
            if options is None:
                options = dist.ProcessGroupNCCL.Options()
            options.config.blocking = True
            if len(positional) > options_index:
                positional[options_index] = options
            else:
                keywords["pg_options"] = options
            return tuple(positional), keywords

        def native_init_process_group(*positional, **keywords):
            positional, keywords = nccl_options(positional, keywords, new_group=False)
            return self.original_init_process_group(*positional, **keywords)

        def native_new_group(*positional, **keywords):
            positional, keywords = nccl_options(positional, keywords, new_group=True)
            return self.original_new_group(*positional, **keywords)

        def guarded_get_world_size(group=None):
            caller = inspect.currentframe().f_back
            if (group is None
                    and Path(caller.f_code.co_filename).resolve() == self.script
                    and caller.f_lineno == self.guard_line):
                actual = self.original_get_world_size(group)
                if actual != 8:
                    raise RuntimeError(f"version-bound main guard expected local world size 8, got {actual}")
                self.guard_runtime_hits += 1
                return 16
            return self.original_get_world_size(group)

        def wrapped_megatron_wrap(stage, *args, **kwargs):
            if self.ddp is not None:
                raise RuntimeError("version-bound adapter expected one Megatron DDP wrap")
            ddp = self.original_wrap(stage, *args, **kwargs)
            self.bind_model(stage, ddp)
            return ddp

        def sgd_factory(params, *args, **kwargs):
            if self.ddp is None or self.optimizer is not None:
                return self.original_sgd(params, *args, **kwargs)
            kwargs = dict(kwargs)
            kwargs["momentum"] = 0.9
            kwargs["foreach"] = False
            inner = self.original_sgd(params, *args, **kwargs)
            proxy = OptimizerProxy(self, inner)
            self.inner_optimizer = inner
            self.optimizer = proxy
            self.scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=1000, gamma=0.98)
            self.write_ready_and_wait()
            return proxy

        dist.init_process_group = native_init_process_group
        dist.new_group = native_new_group
        dist.get_world_size = guarded_get_world_size
        self.adapter.wrap = wrapped_megatron_wrap
        torch.optim.SGD = sgd_factory
        self.flash_train.RedisWriter = NullRedisWriter
        self._installed = True
        return self

    def bind_model(self, stage, ddp):
        self.stage = stage
        self.ddp = ddp
        self.state = self.adapter.core()[0]
        self.install_manager()

        original_finish = ddp.finish_grad_sync
        buffers = list(ddp.buffers) + list(ddp.expert_parallel_buffers)

        def finish_grad_sync(_ddp):
            result = original_finish()
            pending = []
            for buffer in buffers:
                for bucket in buffer.buckets:
                    tensor = bucket.grad_data
                    if self.backend == "hybrid":
                        cpu_tensor = tensor.detach().float().cpu()
                        pending.append((self.manager.allreduce(cpu_tensor), cpu_tensor, tensor))
                    else:
                        pending.append((self.manager.allreduce(tensor), None, None))
            for work, cpu_tensor, cuda_tensor in pending:
                work.wait()
                if cpu_tensor is not None:
                    with self.torch.no_grad():
                        cuda_tensor.copy_(cpu_tensor)
            self.cross_calls += len(pending)
            return result

        ddp.finish_grad_sync = types.MethodType(finish_grad_sync, ddp)

    def install_manager(self):
        from datetime import timedelta
        from torchft import Manager
        from shift_full.transport import PackedGPUTransport

        torch = self.torch
        if self.backend == "baby":
            from torchft import ProcessGroupBabyNCCL
            from shift_full.compat import install_baby_event_fix, install_managed_callback_fix

            os.environ["SHIFT_BABY_COMPAT"] = "1"
            bootstrap_dir = (self.root / "shift_full" / "worker_bootstrap").resolve()
            bootstrap_file = bootstrap_dir / "sitecustomize.py"
            if not bootstrap_file.is_file():
                raise FileNotFoundError(bootstrap_file)
            entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                       if entry and Path(entry).resolve() != bootstrap_dir]
            os.environ["PYTHONPATH"] = os.pathsep.join([str(bootstrap_dir), *entries])
            self.worker_bootstrap_path = str(bootstrap_file)
            install_baby_event_fix()
            install_managed_callback_fix()
            pg = ProcessGroupBabyNCCL(timeout=timedelta(seconds=30))
            transport_pg = pg
        elif self.backend == "hybrid":
            from shift_full.compat import HybridGloo, install_managed_callback_fix

            os.environ.pop("SHIFT_BABY_COMPAT", None)
            install_managed_callback_fix()
            pg = HybridGloo(timeout=timedelta(seconds=10))
            transport_pg = pg.gpu_pg
        else:
            os.environ["SHIFT_BABY_COMPAT"] = "0"
            from shift_full.compat import BlockingNCCL

            pg = BlockingNCCL(timeout=timedelta(seconds=10))
            transport_pg = pg
        self.transport = PackedGPUTransport(
            transport_pg, torch.device("cuda", self.rank), timedelta(seconds=90), self.event)
        self.manager = Manager(
            pg=pg,
            load_state_dict=self.load_state,
            state_dict=self.export_state,
            min_replica_size=1,
            use_async_quorum=False,
            init_sync=False,
            timeout=timedelta(seconds=90),
            quorum_timeout=timedelta(seconds=180),
            connect_timeout=timedelta(seconds=20),
            hostname=self.host_ip,
            lighthouse_addr=self.lighthouse,
            replica_id=f"native{self.replica}",
            checkpoint_transport=self.transport,
            max_retries=8,
        )

    def capture_rng(self):
        return {
            "torch_rng": self.torch.get_rng_state().clone(),
            "cuda_rng": self.torch.cuda.get_rng_state(self.rank).clone(),
            "python_rng": random.getstate(),
        }

    def cache_rng_boundary(self, step):
        self.rng_boundary = {"step": int(step), **self.capture_rng()}
        return self.rng_boundary

    def hashes(self, rng_override=None):
        torch = self.torch
        torch.cuda.current_stream().synchronize()
        rng = rng_override if rng_override is not None else self.capture_rng()
        return {
            "model": digest(self.stage.state_dict(), torch),
            "optimizer": digest(self.optimizer.state_dict(), torch),
            "scheduler": digest(self.scheduler.state_dict(), torch),
            "progress": digest(self.progress, torch),
            "torch_rng": digest(rng["torch_rng"], torch),
            "cuda_rng": digest(rng["cuda_rng"], torch),
            "python_rng": digest(rng["python_rng"], torch),
        }

    def export_state(self):
        torch = self.torch
        step = self.manager.current_step()
        if step != self.progress["updates"]:
            raise RuntimeError(f"snapshot step/progress mismatch: {step} != {self.progress}")
        boundary = self.rng_boundary
        if boundary is None or boundary["step"] != step:
            raise RuntimeError(f"snapshot RNG boundary mismatch for progress step {step}")
        payload = {
            "model": self.stage.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "progress": dict(self.progress),
            "torch_rng": boundary["torch_rng"].clone(),
            "cuda_rng": boundary["cuda_rng"].clone(),
            "python_rng": boundary["python_rng"],
            "rng_boundary_step": step,
            "source_replica": self.replica,
            "source_rank": self.rank,
            "step": step,
        }
        payload_rng = {key: payload[key] for key in ("torch_rng", "cuda_rng", "python_rng")}
        payload["hashes"] = self.hashes(rng_override=payload_rng)
        self.event("snapshot", step=step, hashes=payload["hashes"],
                   optimizer_tensors=tensor_count(payload["optimizer"].get("state", {}), torch))
        return payload

    def load_state(self, payload):
        if payload.get("source_rank") != self.rank:
            raise RuntimeError("torchft snapshot source rank does not match local rank")
        if payload.get("step") != payload.get("progress", {}).get("updates"):
            raise RuntimeError("snapshot step/progress mismatch")
        if payload["step"] != self.manager.current_step():
            raise RuntimeError("snapshot step does not match torchft manager step")
        if payload.get("rng_boundary_step") != payload["step"]:
            raise RuntimeError("snapshot RNG boundary does not match snapshot step")
        self.stage.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.progress.update(payload["progress"])
        self.torch.set_rng_state(payload["torch_rng"])
        self.torch.cuda.set_rng_state(payload["cuda_rng"], self.rank)
        random.setstate(payload["python_rng"])
        self.cache_rng_boundary(self.progress["updates"])
        self.rng_boundary = {
            "step": payload["rng_boundary_step"],
            "torch_rng": payload["torch_rng"].clone(),
            "cuda_rng": payload["cuda_rng"].clone(),
            "python_rng": payload["python_rng"],
        }
        actual = self.hashes()
        expected = payload["hashes"]
        matches = actual == expected
        self.event("restored", step=payload["step"],
                   source_replica=payload["source_replica"], source_rank=payload["source_rank"],
                   hashes_match=matches, expected_hashes=expected, actual_hashes=actual,
                   optimizer_tensors=tensor_count(self.optimizer.state_dict().get("state", {}), self.torch))
        if not matches:
            raise RuntimeError("restored full-state hashes differ from source snapshot")

    def write_ready_and_wait(self):
        hashes = self.hashes()
        hybrid = self.backend == "hybrid"
        self.event("ready", physical_rank=self.replica * 8 + self.rank,
                   backend=self.backend, layout="legacy", tp=2, pp=2, ep=2, local_world_size=8,
                   hashes={key: hashes[key] for key in ("model", "optimizer", "scheduler")},
                   optimizer_momentum=0.9,
                   transport=("PackedGPU(HybridGloo.gpu_pg)" if hybrid else "PackedGPU"),
                   state_transport=("GPU BlockingNCCL" if hybrid else "GPU NCCL"),
                   cross_gradient=("CPU Gloo" if hybrid else "NCCL"))
        while not (self.session_dir / "START").exists():
            if self.stop_requested(self.progress["updates"]):
                raise StopTraining("STOP before START")
            if time.monotonic() >= self.deadline:
                raise StopTraining("timeout before START")
            time.sleep(0.1)

    def _main_frame(self):
        frame = inspect.currentframe().f_back
        while frame is not None:
            if Path(frame.f_code.co_filename).resolve() == self.script and frame.f_code.co_name == "main":
                return frame
            frame = frame.f_back
        raise RuntimeError("optimizer.zero_grad was not called by the version-bound source main()")

    def _next_batches(self, frame, step):
        args = frame.f_locals.get("args")
        batches = frame.f_locals.get("batches")
        if args is None or not isinstance(batches, list):
            raise RuntimeError("version-bound batch adapter expected main().locals['batches'] list")
        microbatches = int(args.microbatches)
        if len(batches) != microbatches or microbatches != 2:
            raise RuntimeError("source batch structure changed; expected two microbatches")
        dp_rank = self.state.get_data_parallel_rank()
        data_rank = self.replica * 2 + dp_rank
        seq_len = int(args.seq_len)
        batch_size = int(args.batch_size)
        replacements = []
        for micro in range(microbatches):
            sample = step * (microbatches * 4) + micro * 4 + data_rank
            tokens = (self.torch.arange(seq_len + 1, device="cuda").view(1, -1)
                      + sample % 8) % 32 + 100
            if batch_size != 1:
                tokens = tokens.expand(batch_size, -1).clone()
            replacements.append({
                "input_ids": tokens[:, :-1].contiguous(),
                "next_token_labels": tokens[:, 1:].contiguous(),
            })
        batches[:] = replacements

    def stop_requested(self, step):
        marker = self.session_dir / "STOP"
        if not marker.is_file():
            return False
        content = marker.read_text(encoding="utf-8").strip()
        if not content:
            return True
        try:
            boundary = int(content)
        except ValueError as exc:
            raise ValueError("STOP must be empty or contain a numeric committed-step boundary") from exc
        if boundary < 0:
            raise ValueError("STOP committed-step boundary must be nonnegative")
        return step >= boundary

    def before_zero_grad(self):
        if self.stop_requested(self.progress["updates"]):
            raise StopTraining("STOP")
        if time.monotonic() >= self.deadline:
            raise StopTraining("timeout")
        if self.rng_boundary is None:
            self.cache_rng_boundary(self.progress["updates"])
        elif self.rng_boundary["step"] != self.progress["updates"]:
            raise RuntimeError("cached RNG boundary does not match committed progress")
        self.manager.start_quorum()
        # Restored/committed progress is the source of truth for batch IDs and
        # event steps; do not derive them from Manager.current_step().
        step = self.progress["updates"]
        if self.stop_requested(step):
            raise StopTraining("STOP")
        if step >= self.target_steps:
            raise StopTraining("target_steps")
        frame = self._main_frame()
        self._next_batches(frame, step)
        participants = self.manager.num_participants()
        self.event("begin", step=step, participants=participants)
        self.attempts += 1

    def committed_step(self, inner, closure=None):
        finite = True
        flags = [self.torch.isfinite(bucket.grad_data).all()
                 for buffer in list(self.ddp.buffers) + list(self.ddp.expert_parallel_buffers)
                 for bucket in buffer.buckets]
        if flags:
            finite = bool(self.torch.stack(flags).all().item())
        if not finite:
            self.manager.report_error(RuntimeError("nonfinite gradients"))
        commit = self.manager.should_commit()
        if not commit:
            self.event("skipped", step=self.progress["updates"],
                       error=str(self.manager.errored()))
            return None
        loss = inner.step(closure) if closure is not None else inner.step()
        self.scheduler.step()
        self.progress["updates"] += 1
        step = self.progress["updates"]
        self.cache_rng_boundary(step)
        hashes = self.hashes()
        self.event("commit", step=step, updates=step,
                   participants=self.manager.num_participants(),
                   model_hash=hashes["model"], optimizer_hash=hashes["optimizer"],
                   scheduler_hash=hashes["scheduler"],
                   optimizer_tensors=tensor_count(inner.state_dict().get("state", {}), self.torch),
                   cross_calls=self.cross_calls, finite_gradients=finite)
        return loss

    def verify_guard(self):
        if self.guard_runtime_hits != 1:
            raise RuntimeError(f"expected exactly one version-bound guard call, saw {self.guard_runtime_hits}")

    def write_patch_manifest(self, source_sha256_after):
        unchanged = self.source_sha256 == source_sha256_after
        process_group = (
            "ProcessGroupBabyNCCL(timeout=30s)" if self.backend == "baby"
            else "HybridGloo(timeout=10s; gpu_pg=BlockingNCCL)" if self.backend == "hybrid"
            else "BlockingNCCL(timeout=10s)"
        )
        compatibility_fixes = []
        if self.backend == "baby":
            compatibility_fixes = ["install_baby_event_fix", "install_managed_callback_fix", "spawn sitecustomize"]
        elif self.backend == "hybrid":
            compatibility_fixes = ["install_managed_callback_fix"]
        manifest = {
            "backend": self.backend,
            "torchft_process_group": process_group,
            "manager_use_async_quorum": False,
            "native_process_group_blocking": True,
            "compatibility_fixes_installed": compatibility_fixes,
            "gradient_and_state_transport": (
                "CPU Gloo gradients/control; GPU BlockingNCCL checkpoint state"
                if self.backend == "hybrid" else "GPU process group gradients and checkpoint state"
            ),
            "spawn_worker_bootstrap": self.worker_bootstrap_path,
            "source_path": str(self.script),
            "source_sha256": self.source_sha256,
            "expected_source_sha256": EXPECTED_SOURCE_SHA256,
            "source_sha256_after": source_sha256_after,
            "guard_line": self.guard_line,
            "guard_runtime_hits": self.guard_runtime_hits,
            "patches": PATCHES,
            "zero_source_file_edits": unchanged,
            "limitations": [
                "Bound to the recorded source SHA and its unique main() world-size guard.",
                "Requires the native source loop's SGD call, DDP buffer names, and batches local at zero_grad.",
                "The random input generator is replaced at runtime with deterministic pattern tokens; results do not represent the unmodified random-data workload.",
                "The adapter changes optimizer commit timing and adds cross-replica bucket all-reduces; it is for this fixed TP2/PP2/EP2 eight-rank topology only.",
                "STOP may be empty for immediate stop or contain a numeric committed-step boundary; the session controller must allow workers to reach numeric boundaries.",
                "No training source file is edited; this manifest records runtime monkey patches and their version binding.",
            ],
        }
        path = self.output_dir / f"patch_manifest_rank{self.rank}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not unchanged:
            raise RuntimeError("original training source SHA256 changed during runtime patch")

    def mark_stopped(self, reason):
        self.event("stopped", step=self.progress["updates"], attempts=self.attempts,
                   reason=reason)

    def mark_error(self, error):
        self.event("runtime_error", error=repr(error))

    def close(self, graceful=False):
        if self.closed:
            return
        self.closed = True
        manager_step = self.progress["updates"]
        if graceful and hasattr(self, "dist") and self.dist.is_initialized():
            try:
                self.dist.barrier()
            except Exception as error:
                self.event("local_barrier_error", error=repr(error))
        try:
            if self.manager is not None:
                self.manager.shutdown()
        except Exception as error:
            self.event("manager_shutdown_error", error=repr(error))
        if self.manager is not None:
            pg = getattr(self.manager, "_pg", None)
            if pg is not None:
                try:
                    pg.abort(errored=False)
                except Exception as error:
                    self.event("manager_pg_abort_error", error=repr(error))
                try:
                    pg.shutdown()
                except Exception as error:
                    self.event("manager_pg_shutdown_error", error=repr(error))
        try:
            if hasattr(self, "dist") and self.dist.is_initialized():
                self.dist.destroy_process_group()
        except Exception as error:
            self.event("process_group_shutdown_error", error=repr(error))
        if self._installed:
            self.dist.init_process_group = self.original_init_process_group
            self.dist.new_group = self.original_new_group
            self.dist.get_world_size = self.original_get_world_size
            self.adapter.wrap = self.original_wrap
            self.torch.optim.SGD = self.original_sgd
            self.flash_train.RedisWriter = self.original_redis_writer
            self._installed = False
        self.event("runtime_cleanup", manager_step=manager_step)
        self.event_log.close()
