"""P13 conservative genuine-export semantics and full CPU budget; diagnostic only."""
import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import types
from scripts import v3_lazy_cpu_probe as base


class ExportProbe(base.ImportProbe):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != base.TARGET or not self.candidate:
            return super().find_spec(fullname, path, target)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        source = Path(spec.origin).read_bytes()
        assert hashlib.sha256(source).hexdigest() == base.UTILS_SHA
        self.report['utils_sha256'] = base.UTILS_SHA
        tree = ast.parse(source, filename=spec.origin)
        lines = source.decode().splitlines(keepends=True)
        changed = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ('gelu_impl', 'erf_gelu'):
                assert len(node.decorator_list) == 1
                d = node.decorator_list[0]
                assert isinstance(d, ast.Name) and d.id == 'jit_fuser'
                assert lines[d.lineno-1].strip() == '@jit_fuser'
                lines[d.lineno-1] = '@_shift_capture(jit_fuser)\n'
                changed.append(node.name)
        assert sorted(changed) == ['erf_gelu', 'gelu_impl']
        code = compile(''.join(lines), spec.origin, 'exec')
        report = self.report

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                namespace = module.__dict__
                pending = {}
                lock = threading.RLock()

                def capture(decorator):
                    def save(function):
                        pending[function.__name__] = (function, decorator)
                        return function
                    return save

                namespace['_shift_capture'] = capture
                exec(code, namespace)
                del namespace['_shift_capture']

                def materialize(reason):
                    with lock:
                        report.setdefault('materialization', []).append(dict(reason=reason, at=time.monotonic()))
                        for name, (function, decorator) in list(pending.items()):
                            # Exact original decorator, zero kwargs; no synthesized metadata.
                            namespace[name] = decorator(function)
                            del pending[name]
                        module.__class__ = types.ModuleType

                class DeferredExports(types.ModuleType):
                    def __getattribute__(self, name):
                        value = types.ModuleType.__getattribute__(self, name)
                        if name == '__dict__' or isinstance(value, types.FunctionType):
                            materialize(name)
                            return types.ModuleType.__getattribute__(self, name)
                        return value

                module.__class__ = DeferredExports
        spec.loader = Loader()
        report['changed_decorators'] = changed
        return spec


def semantics(candidate, output):
    report = dict(success=False, denied=[], diagnostic_only=True)
    try:
        import shift_full.train
        import torch
        import torch.distributed as dist
        from shift_full.compat import bootstrap_legacy_core, install_managed_callback_fix
        install_managed_callback_fix()
        torch.set_num_threads(1)
        assert bootstrap_legacy_core('/workspace/FLASH1022')
        sys.path.insert(0, '/workspace/FLASH1022')
        from flash_v1 import megatron_adapter
        torch.random.default_generator.manual_seed(2026)
        before = torch.get_rng_state().clone()

        cache_source = Path(torch.__file__).parent/'_inductor/codecache.py'
        report['cache_source_sha256'] = hashlib.sha256(cache_source.read_bytes()).hexdigest()
        assert report['cache_source_sha256'] == '1edadc1e1229adbd7c6f84cef43a04c031add6e1793be0f12b73f22798966156'
        report['expected_rejected_cuda_probe'] = []

        def deny(*args, **kwargs):
            report['denied'].append(dict(operation='native CUDA/PG initialization', stack=traceback.format_stack(limit=14)))
            raise RuntimeError('CPU-only probe')

        def deny_lazy_init(*args, **kwargs):
            stack = traceback.extract_stack()
            known = any(Path(f.filename).resolve() == cache_source.resolve() and
                        f.name == 'get_system' and f.lineno == 340 for f in stack)
            if known:
                report['expected_rejected_cuda_probe'].append(traceback.format_list(stack[-14:]))
            else:
                report['denied'].append(dict(operation='unexpected CUDA lazy init', stack=traceback.format_list(stack[-14:])))
            raise RuntimeError('CPU-only: CUDA cache device query rejected')
        torch.cuda._lazy_init = deny_lazy_init
        torch._C._cuda_init = deny
        dist.init_process_group = dist.new_group = deny
        finder = ExportProbe(candidate, report)
        sys.meta_path.insert(0, finder)
        from shift_full.cold_imports import megatron_components
        megatron_components(megatron_adapter)
        module = sys.modules[base.TARGET]
        # Concurrent ordinary reads must expose one genuine function per export.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            exports = list(pool.map(lambda _: (module.gelu_impl, module.erf_gelu), range(4)))
        assert all(a is exports[0][0] and b is exports[0][1] for a,b in exports)
        assert type(module) is types.ModuleType
        report['functions'] = {}
        for name in ('gelu_impl', 'erf_gelu'):
            fn = getattr(module, name)
            assert inspect.isfunction(fn) and fn.__wrapped__.__globals__ is vars(module)
            assert vars(module)[name] is fn
            assert fn._torchdynamo_wrapper_id == id(fn)
            assert fn._torchdynamo_orig_callable is fn.__wrapped__
            assert fn._torchdynamo_inline is fn.__wrapped__
            assert fn._is_torch_compile is True
            assert callable(fn.aot_compile) and callable(fn.get_compiler_config)
            config = fn.get_compiler_config()
            assert isinstance(config, dict)
            rows = []
            for size in (17, 17, 29):
                x = torch.linspace(-3, 3, size, requires_grad=True)
                started = time.monotonic()
                y = fn(x)
                y.sum().backward()
                rows.append(dict(size=size, seconds=time.monotonic()-started,
                                 output=y.detach().tolist(), gradient=x.grad.tolist()))
            try:
                fn(object())
                raise AssertionError('non-Tensor input unexpectedly succeeded')
            except TypeError as exc:
                error = dict(type=type(exc).__name__, text=str(exc))
            x = torch.ones(7, requires_grad=True)
            fn(x).sum().backward()
            assert torch.isfinite(x.grad).all()
            report['functions'][name] = dict(attributes=sorted(fn.__dict__),
                signature=str(inspect.signature(fn)), config_sha256=hashlib.sha256(
                    repr(sorted(config.items())).encode()).hexdigest(), samples=rows, invalid_input=error,
                recovery_output=fn(torch.ones(7)).tolist(), isolate_recompiles_id=fn._isolate_recompiles_id)
        report['compiler_metrics'] = {k:v for k,v in vars(torch._inductor.metrics).items()
                                      if k in ('generated_kernel_count', 'generated_cpp_vec_kernel_count')}
        assert report['compiler_metrics']['generated_kernel_count'] > 0, 'no real Inductor kernels'
        assert torch.equal(before, torch.get_rng_state())
        assert len(report['expected_rejected_cuda_probe']) == 1
        assert not report['denied'] and not torch.cuda.is_initialized() and not dist.is_initialized()
        report.update(success=True, concurrent_identity=True, rng_exact=True)
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    report.update(done=True, cuda_initialized=bool('torch' in locals() and torch.cuda.is_initialized()),
                  distributed_initialized=bool('dist' in locals() and dist.is_initialized()))
    output.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['success'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--child', choices=('s0','s1','a0','b1','b2','a3'))
    a=p.parse_args()
    assert a.suffix and all(c.isalnum() or c in '_-' for c in a.suffix)
    out=base.ROOT/'results/v3_p_20261103'/('p13_'+a.suffix)
    if a.child:
        if a.child.startswith('s'):
            return semantics(a.child=='s1', out/(a.child+'.json'))
        base.ImportProbe = ExportProbe
        return base.child(a.child.startswith('b'), out/(a.child+'.json'))
    lock=(base.ROOT/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(exist_ok=False)
    report=dict(done=False,success=False,diagnostic_only=True,children=[])
    for name in ('s0','s1','a0','b1','b2','a3'):
        env=os.environ.copy()
        env.update(TORCHINDUCTOR_CACHE_DIR=str(out/('compiler_cache_'+name)),
                   TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
        with (out/(name+'.log')).open('x') as log:
            try:
                run=subprocess.run([sys.executable,'-m','scripts.v3_lazy_metadata_cpu',
                    '--suffix',a.suffix,'--child',name],stdout=log,stderr=log,env=env,timeout=150)
                row=json.loads((out/(name+'.json')).read_text())
                report['children'].append(dict(name=name,exit=run.returncode,success=row['success'],error=row.get('error')))
            except BaseException as exc:
                report['children'].append(dict(name=name,success=False,error=repr(exc)))
        if not report['children'][-1]['success']:
            break
    report.update(done=True, success=len(report['children'])==6 and all(x['success'] for x in report['children']))
    (out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
