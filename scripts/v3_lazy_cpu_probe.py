"""P13 diagnostic only: fresh CPU full-startup A/B, never trains or uses GPUs."""
import argparse
import ast
import fcntl
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path('/workspace/SHIFT')
TARGET = 'megatron.core.transformer.utils'
UTILS_SHA = 'fe89a3458286a036d8d36e9df9d191eaf0d048eb39a2172af96c4bb414b5deb9'


def lazy_decorator(decorator):
    def decorate(function):
        compiled = None
        lock = threading.Lock()

        @functools.wraps(function)
        def invoke(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                with lock:
                    if compiled is None:
                        compiled = decorator(function)
            return compiled(*args, **kwargs)
        return invoke
    return decorate


class ImportProbe(importlib.abc.MetaPathFinder):
    def __init__(self, candidate, report):
        self.candidate, self.report = candidate, report

    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch._dynamo', 'torch._inductor'):
            self.report.setdefault('first_import', {}).setdefault(fullname, time.monotonic())
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        source = Path(spec.origin).read_bytes()
        assert hashlib.sha256(source).hexdigest() == UTILS_SHA, 'utils source drift'
        self.report['utils_sha256'] = UTILS_SHA
        if not self.candidate:
            return None
        tree = ast.parse(source, filename=spec.origin)
        lines = source.decode().splitlines(keepends=True)
        changed = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ('gelu_impl', 'erf_gelu'):
                assert len(node.decorator_list) == 1
                d = node.decorator_list[0]
                assert isinstance(d, ast.Name) and d.id == 'jit_fuser'
                assert lines[d.lineno-1].strip() == '@jit_fuser'
                lines[d.lineno-1] = '@_shift_lazy_decorator(jit_fuser)\n'
                changed.append(node.name)
        assert sorted(changed) == ['erf_gelu', 'gelu_impl']
        code = compile(''.join(lines), spec.origin, 'exec')

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                module._shift_lazy_decorator = lazy_decorator
                exec(code, module.__dict__)
        spec.loader = Loader()
        self.report['changed_decorators'] = changed
        return spec


def child(candidate, output):
    report = dict(success=False, diagnostic_only=True, denied=[], optional_calls=[])
    try:
        import shift_full.train
        import numpy as np
        import random
        import torch
        import torch.distributed as dist
        from shift_full.compat import bootstrap_legacy_core, install_managed_callback_fix
        from shift_full.state_digest import digest_staged as digest
        install_managed_callback_fix()
        torch.set_num_threads(1)
        assert bootstrap_legacy_core('/workspace/FLASH1022')
        sys.path.insert(0, '/workspace/FLASH1022')
        from flash_v1 import megatron_adapter
        from flash_v1.pipeline_adapter import StageModel, forward_step
        from flash_v1.tensor_parallel import patch_tensor_parallel
        from flash_v1.train import load_model
        torch.random.default_generator.manual_seed(2026)
        random.seed(2026)
        np.random.seed(2026)

        def state():
            return dict(cpu_rng=digest(torch.get_rng_state()), python_rng=digest(random.getstate()),
                        numpy_rng=digest(np.random.get_state()), dtype=str(torch.get_default_dtype()),
                        device=str(torch.get_default_device()), queued_cuda_calls=len(torch.cuda._queued_calls),
                        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())

        saved = []
        def forbid(obj, names, temporary=False):
            for name in names:
                if not hasattr(obj, name):
                    continue
                original = getattr(obj, name)
                def denied(*args, _name=name, **kwargs):
                    report['denied'].append(_name)
                    raise RuntimeError('CPU probe forbids '+_name)
                setattr(obj, name, denied)
                if temporary:
                    saved.append((obj, name, original))

        forbid(torch.cuda, ('_lazy_init', '_lazy_call', 'init', 'set_device', 'current_device',
                            'get_device_properties', 'current_stream', 'default_stream', 'Stream', 'Event'))
        forbid(torch._C, ('_cuda_init', '_cuda_getDevice', '_cuda_setDevice', '_cuda_setStream',
                          '_cuda_getCurrentStream'))
        forbid(dist, ('init_process_group', 'new_group', 'all_gather_object', 'all_reduce',
                      'broadcast', 'barrier', 'all_gather', 'reduce', 'all_to_all_single'))
        forbid(torch, ('set_num_threads', 'set_num_interop_threads'))
        forbid(torch, ('manual_seed', 'seed', 'set_rng_state', 'set_default_dtype', 'set_default_device'), True)
        forbid(random, ('seed', 'setstate'), True)
        forbid(np.random, ('seed', 'set_state'), True)
        before = state()
        finder = ImportProbe(candidate, report)
        sys.meta_path.insert(0, finder)
        report['preexisting_compiler_modules'] = [x for x in ('torch._dynamo', 'torch._inductor') if x in sys.modules]
        assert not report['preexisting_compiler_modules']
        started = time.monotonic()
        from shift_full.cold_imports import megatron_components
        megatron_components(megatron_adapter)
        report['imports_seconds'] = time.monotonic()-started
        assert before == state(), 'imports changed global state'
        for obj, name, original in saved:
            setattr(obj, name, original)

        # Guard all currently exported aliases, including imports by value.
        utils = sys.modules[TARGET]
        targets = {name:getattr(utils, name) for name in ('gelu_impl', 'openai_gelu', 'erf_gelu')}
        report['wrapper_metadata'] = {name:dict(name=f.__name__, qualname=f.__qualname__,
            module=f.__module__, doc=f.__doc__, attributes=sorted(f.__dict__)) for name,f in targets.items()}
        for name, original in targets.items():
            @functools.wraps(original)
            def unused(*args, _name=name, **kwargs):
                report['optional_calls'].append(_name)
                raise RuntimeError('optional GELU actually called: '+_name)
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith(('megatron', 'flash_v1')):
                    for key, value in list(vars(module).items()):
                        if value is original:
                            setattr(module, key, unused)
        loaded_at = time.monotonic()
        model = load_model(Path('/workspace/FLASH1022/models/qwen_moe'))
        report['model_seconds'] = time.monotonic()-loaded_at
        optimizer_at = time.monotonic()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)
        ended = time.monotonic()
        report.update(total_seconds=ended-started, optimizer_seconds=ended-optimizer_at,
                      first_import={k:v-started for k,v in report.get('first_import', {}).items()},
                      compiler_modules_after=[x for x in ('torch._dynamo', 'torch._inductor') if x in sys.modules])
        after = state()
        report.update(state_before=before, state_after=after,
                      parameter_elements=sum(p.numel() for p in model.parameters()),
                      optimizer_state_entries=len(optimizer.state), training=all(m.training for m in model.modules()))
        assert report['parameter_elements'] == 494712832
        assert all(p.device.type == 'cpu' for p in (*model.parameters(), *model.buffers()))
        assert report['training'] and not report['optional_calls'] and not report['denied']
        for key in ('dtype', 'device', 'queued_cuda_calls', 'threads', 'interop_threads'):
            assert before[key] == after[key], key
        report['hashes'] = dict(model=digest(model.state_dict()), buffers=digest(dict(model.named_buffers())),
            optimizer=digest(optimizer.state_dict()), scheduler=digest(scheduler.state_dict()))
        assert not torch.cuda.is_initialized() and not dist.is_initialized()
        report['success'] = True
    except BaseException as exc:
        report.update(error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        report['sources'] = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_file() and name.startswith(('megatron', 'flash_v1', 'torch', 'transformers', 'shift_full', 'scripts.v3_lazy')):
                report['sources'][name] = dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        report.update(done=True, cuda_initialized=bool('torch' in locals() and torch.cuda.is_initialized()),
                      distributed_initialized=bool('dist' in locals() and dist.is_initialized()))
        with output.open('x') as stream:
            stream.write(json.dumps(report, indent=2)+'\n')
    return 0 if report['success'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--child', choices=('a0', 'b1', 'b2', 'a3'))
    a = p.parse_args()
    assert a.suffix and all(c.isalnum() or c in '_-' for c in a.suffix)
    out = ROOT/'results/v3_p_20261103'/('p13_'+a.suffix)
    if a.child:
        return child(a.child.startswith('b'), out/(a.child+'.json'))
    lock = (ROOT/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    report = dict(done=False, success=False, diagnostic_only=True, children=[])
    for name in ('a0', 'b1', 'b2', 'a3'):
        with (out/(name+'.log')).open('x') as log:
            run = subprocess.run([sys.executable, '-m', 'scripts.v3_lazy_cpu_probe', '--suffix', a.suffix,
                                  '--child', name], stdout=log, stderr=log, timeout=90)
        row = json.loads((out/(name+'.json')).read_text())
        report['children'].append(dict(name=name, exit=run.returncode, success=row['success'], error=row.get('error')))
        if not row['success']:
            break
    report.update(done=True, success=len(report['children']) == 4 and all(x['success'] for x in report['children']))
    (out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
