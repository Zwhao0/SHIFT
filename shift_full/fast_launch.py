"""CPU-preload the FLASH model before fork-based torchrun worker startup."""

import os

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import json
import random
import sys
import time
from pathlib import Path

_MODULE_STARTED = time.perf_counter()
_CACHED_MODEL = None
_PRELOAD_COMPLETE = False
_PRELOAD_MODE = None


def _launcher_phase(name, seconds):
    print(json.dumps({"kind": "launcher_phase", "name": name, "seconds": seconds,
                      "time": time.time()}, sort_keys=True), flush=True)


def _require_parent_cuda_uninitialized(torch, phase):
    if torch.cuda.is_initialized():
        raise RuntimeError(f"fast_launch parent initialized CUDA during {phase}; fork preload is unsafe")


def acquire_model(required=True):
    """Return an inherited model once; optionally allow common-only preload."""
    global _CACHED_MODEL
    if not isinstance(required, bool):
        raise TypeError("required must be a bool")
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("acquire_model must run before the child initializes CUDA")
    if not _PRELOAD_COMPLETE:
        if not required:
            return None
        raise RuntimeError("fast_launch preload did not complete; start workers with fast_launch and fork")
    if _PRELOAD_MODE == "generic":
        if required:
            raise RuntimeError("generic fast_launch preload has no model cache")
        return None
    if _PRELOAD_MODE != "model":
        raise RuntimeError(f"unknown fast_launch preload mode: {_PRELOAD_MODE!r}")
    if _CACHED_MODEL is None:
        raise RuntimeError("fast_launch model cache is empty; start workers with fast_launch and fork")
    model = _CACHED_MODEL
    _CACHED_MODEL = None
    return model


def preload():
    """Import common CPU dependencies and optionally cache the configured model."""
    global _CACHED_MODEL, _PRELOAD_COMPLETE, _PRELOAD_MODE
    import torch

    mode = "generic" if os.environ.get("SHIFT_GENERIC_PRELOAD") == "1" else "model"
    if _PRELOAD_COMPLETE:
        if _PRELOAD_MODE != mode:
            raise RuntimeError(
                f"fast_launch already preloaded in {_PRELOAD_MODE!r} mode; cannot switch to {mode!r}"
            )
        _require_parent_cuda_uninitialized(torch, "cached preload")
        sys.modules["shift_full.fast_launch"] = sys.modules[__name__]
        return

    _launcher_phase("python_torch_imports", time.perf_counter() - _MODULE_STARTED)
    torch.set_num_threads(1)
    _require_parent_cuda_uninitialized(torch, "torch import / thread setup")

    flash_root = Path(os.environ.get(
        "FLASHPATH", os.environ.get("FLASH", "/workspace/FLASH1022")
    )).resolve()
    if not flash_root.is_dir():
        raise FileNotFoundError(f"FLASH source directory is missing: {flash_root}")
    sys.path.insert(0, str(flash_root))

    started = time.perf_counter()
    from shift_full.compat import bootstrap_legacy_core
    bootstrap_legacy_core(str(flash_root))
    _launcher_phase("legacy_core_bootstrap", time.perf_counter() - started)
    _require_parent_cuda_uninitialized(torch, "legacy Core bootstrap")

    started = time.perf_counter()
    import transformers
    _launcher_phase("transformers_import", time.perf_counter() - started)
    _require_parent_cuda_uninitialized(torch, "Transformers import")

    started = time.perf_counter()
    from flash_v1 import megatron_adapter
    import flash_v1.train
    _launcher_phase("flash_cpu_imports", time.perf_counter() - started)
    _require_parent_cuda_uninitialized(torch, "FLASH imports")

    started = time.perf_counter()
    megatron_adapter.core()
    _launcher_phase("megatron_core_import", time.perf_counter() - started)
    _require_parent_cuda_uninitialized(torch, "Megatron Core imports")

    if mode == "generic":
        # Common-only preload deliberately avoids model config/weights and all
        # distributed/CUDA initialization; the forked worker loads its model.
        if torch.distributed.is_initialized():
            raise RuntimeError("generic preload unexpectedly initialized torch.distributed")
        _CACHED_MODEL = None
        _launcher_phase("preload_status", 0.0)
        print(json.dumps({
            "kind": "launcher_preload_status", "mode": "generic",
            "common_only": True, "model_loaded": False,
            "cuda_initialized": torch.cuda.is_initialized(), "time": time.time(),
        }, sort_keys=True), flush=True)
    else:
        torch.manual_seed(2026)
        random.seed(2026)
        _require_parent_cuda_uninitialized(torch, "seed setup")
        started = time.perf_counter()
        from flash_v1.train import load_model
        model = load_model(flash_root / "models/qwen_moe")
        if any(tensor.device.type != "cpu"
               for tensor in list(model.parameters()) + list(model.buffers())):
            raise RuntimeError("CPU preload returned a model containing non-CPU tensors")
        _require_parent_cuda_uninitialized(torch, "CPU model load")
        _CACHED_MODEL = model
        del model
        _launcher_phase("model_load_cpu", time.perf_counter() - started)
        print(json.dumps({
            "kind": "launcher_preload_status", "mode": "model",
            "common_only": False, "model_loaded": True,
            "cuda_initialized": torch.cuda.is_initialized(), "time": time.time(),
        }, sort_keys=True), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError(f"fast_launch {mode} preload initialized CUDA")
    _PRELOAD_MODE = mode
    _PRELOAD_COMPLETE = True
    _launcher_phase("preload_total", time.perf_counter() - _MODULE_STARTED)

    # Forked workers import this alias and consume their copy of the model cache.
    sys.modules["shift_full.fast_launch"] = sys.modules[__name__]


def main():
    preload()
    import torch

    from torch.distributed.run import main as torchrun_main
    _require_parent_cuda_uninitialized(torch, "torchrun import")
    torchrun_main()


if __name__ == "__main__":
    main()
