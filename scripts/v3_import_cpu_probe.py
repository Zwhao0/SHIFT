"""Bounded fresh-process admission for the pinned Megatron import path; no GPU work."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def child(mode, output, use_helper):
    report = dict(mode=mode, success=False, performance_evidence=False, denied=[])
    try:
        # Match train.py's original top-level imports before its CUDA phase.
        import shift_full.train
        import numpy as np
        import random
        import torch
        import torch.distributed as dist
        from shift_full.compat import bootstrap_legacy_core, install_managed_callback_fix
        install_managed_callback_fix()
        torch.set_num_threads(1)
        bootstrap_legacy_core('/workspace/FLASH1022')
        os.environ['FLASH_PARALLEL_LAYOUT'] = 'legacy'
        os.environ['TORCHFT_USE_OTEL'] = 'false'
        sys.path.insert(0, '/workspace/FLASH1022')
        from flash_v1 import megatron_adapter
        from flash_v1.pipeline_adapter import StageModel, forward_step
        from flash_v1.tensor_parallel import patch_tensor_parallel
        from flash_v1.train import load_model, patch_moe, MODEL_ID, REVISION
        assert not torch.cuda.is_initialized() and not dist.is_initialized()
        torch.random.default_generator.manual_seed(2026)
        random.seed(2026)
        np.random.seed(2026)

        def state():
            return dict(cpu_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                        python_rng=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                        numpy_rng=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                        dtype=str(torch.get_default_dtype()), device=str(torch.get_default_device()),
                        queued_cuda_calls=len(torch.cuda._queued_calls),
                        num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
                        other_threads=sorted(t.name for t in threading.enumerate()
                                             if t.name not in ('cpu-import-admission', 'shift-cpu-imports')))

        before = state()
        before_modules = set(sys.modules)

        def forbid(obj, name):
            if not hasattr(obj, name):
                return
            def denied(*args, **kwargs):
                report['denied'].append(name)
                raise RuntimeError('CPU import admission forbids '+name)
            setattr(obj, name, denied)

        for name in ('_lazy_init', '_lazy_call', 'init', 'set_device', 'current_device',
                     'is_available', 'device_count', 'get_device_properties', 'current_stream',
                     'default_stream', 'Stream', 'Event'):
            forbid(torch.cuda, name)
        for name in ('_cuda_init', '_cuda_getDevice', '_cuda_getDeviceCount', '_cuda_setDevice',
                     '_cuda_getDeviceProperties', '_cuda_setStream', '_cuda_getCurrentStream'):
            forbid(torch._C, name)
        for name in ('init_process_group', 'new_group', 'all_gather_object', 'all_reduce',
                     'broadcast', 'barrier', 'all_gather', 'reduce', 'all_to_all_single'):
            forbid(dist, name)
        for name in ('manual_seed', 'seed', 'set_rng_state', 'set_default_dtype', 'set_default_device'):
            forbid(torch, name)
        for name in ('set_num_threads', 'set_num_interop_threads'):
            forbid(torch, name)
        for obj in (random, np.random):
            for name in ('seed', 'setstate', 'set_state'):
                forbid(obj, name)

        error = []
        result = []
        def imports():
            try:
                if use_helper:
                    from shift_full.cold_imports import CPUImports, megatron_components
                    if mode == 'threaded':
                        task = CPUImports(megatron_adapter)
                        components, timing = task.result()
                        assert not task.thread.is_alive()
                        report['helper_thread_exited'] = True
                        report['helper_timing'] = timing
                    else:
                        components = megatron_components(megatron_adapter)
                    result.extend(x.__module__ if hasattr(x, '__module__') else x.__name__ for x in components)
                else:
                    core = megatron_adapter.core()
                    from megatron.core.distributed import finalize_model_grads
                    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
                    result.extend([*(x.__module__ if hasattr(x, '__module__') else x.__name__ for x in core),
                                   finalize_model_grads.__module__, get_forward_backward_func.__module__])
            except BaseException as exc:
                error.append(exc)

        started = time.monotonic()
        if mode == 'threaded' and not use_helper:
            thread = threading.Thread(target=imports, name='cpu-import-admission', daemon=True)
            thread.start()
            thread.join(45)
            if thread.is_alive():
                raise TimeoutError('CPU import thread did not exit')
            report['thread_exited'] = True
        else:
            imports()
        report['import_seconds'] = time.monotonic()-started
        if error:
            raise error[0]
        after = state()
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
        constant = MoEAuxLossAutoScaler.main_loss_backward_scale
        report['moe_import_constant'] = dict(device=str(constant.device), dtype=str(constant.dtype),
                                             value=constant.item())
        assert constant.device.type == 'cpu' and constant.dtype == torch.float32 and constant.item() == 1.0
        report['synthetic_megatron_parent'] = sys.modules['megatron'].__spec__.loader is None
        report.update(state_before=before, state_after=after, imports=result)
        assert before == after, 'import changed process-global state'
        assert not report['denied']
        assert not torch.cuda.is_initialized() and not dist.is_initialized()
        report['success'] = True
    except BaseException as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        report['traceback'] = traceback.format_exc()
    finally:
        report['sources'] = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if not path:
                continue
            p = Path(path)
            if p.is_file() and (name not in locals().get('before_modules', set()) or name.startswith(('megatron', 'flash_v1'))):
                report['sources'][name] = dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        report.update(done=True, cuda_initialized=bool('torch' in locals() and torch.cuda.is_initialized()),
                      distributed_initialized=bool('dist' in locals() and dist.is_initialized()))
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    return 0 if report['success'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--mode', choices=('sequential', 'threaded'))
    p.add_argument('--use-helper', action='store_true')
    p.add_argument('--profile-imports', action='store_true')
    a = p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):
        p.error('invalid suffix')
    root = Path('/workspace/SHIFT')
    out = root/'results/v3_p_20261103'/('p12_'+a.suffix)
    if a.mode:
        return child(a.mode, out/(a.mode+'.json'), a.use_helper)
    lock = (root/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    report = dict(done=False, success=False, performance_evidence=False, children=[])
    for mode in ('sequential', 'threaded'):
        cmd = [sys.executable]
        if a.profile_imports:
            cmd += ['-X', 'importtime']
        cmd += ['-m', 'scripts.v3_import_cpu_probe', '--suffix', a.suffix, '--mode', mode]
        if a.use_helper:
            cmd.append('--use-helper')
        if a.profile_imports:
            with (out/(mode+'_importtime.log')).open('x') as log:
                run = subprocess.run(cmd, timeout=70, stdout=log, stderr=log)
        else:
            run = subprocess.run(cmd, timeout=70)
        result = json.loads((out/(mode+'.json')).read_text())
        report['children'].append(dict(mode=mode, exit=run.returncode, success=result['success'],
                                       error=result.get('error'), denied=result['denied']))
    report.update(done=True, success=all(x['success'] for x in report['children']))
    (out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
