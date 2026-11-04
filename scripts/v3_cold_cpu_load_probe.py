"""CPU-only real checkpoint semantics; no training or fault injection."""
import fcntl
import gc
import hashlib
import json
import random
from pathlib import Path
import sys
import time


def main():
    root = Path('/workspace/SHIFT').resolve()
    lock = (root/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = root/'results/v3_p_20261103/p11_cpu_load02.json'
    if out.exists():
        raise FileExistsError(out)
    import torch
    from shift_full.cold_cpu_load import CPUModelLoad
    from shift_full.state_digest import digest_staged
    sys.path.insert(0, '/workspace/FLASH1022')
    from flash_v1.train import load_model
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    original_lazy = torch.cuda._lazy_init
    def forbidden(*args, **kwargs):
        raise RuntimeError('GPU/process-group operation forbidden in CPU load probe')
    torch.cuda._lazy_init = forbidden
    saved = {name:getattr(torch.distributed, name) for name in ('init_process_group','new_group','all_gather_object')}
    for name in saved:
        setattr(torch.distributed, name, forbidden)
    report = dict(done=False, success=False, performance_evidence=False, cuda_visible=False,
                  source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), root/'shift_full/cold_cpu_load.py')}, measurements=[])
    try:
        for background in (False, True):
            torch.manual_seed(2026)
            random.seed(2026)
            dtype = torch.get_default_dtype()
            started = time.monotonic()
            if background:
                task = CPUModelLoad(lambda:load_model(Path('/workspace/FLASH1022/models/qwen_moe')))
                model, timing = task.result()
                assert not task.thread.is_alive()
            else:
                model = load_model(Path('/workspace/FLASH1022/models/qwen_moe'))
                timing = dict(load_seconds=time.monotonic()-started)
            assert all(t.device.type == 'cpu' for t in (*model.parameters(), *model.buffers()))
            assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
            assert torch.get_default_dtype() == dtype
            hashes = dict(model=digest_staged(model.state_dict()),
                buffers=digest_staged(dict(model.named_buffers())), cpu_rng=digest_staged(torch.get_rng_state()),
                python_rng=digest_staged(random.getstate()))
            report['measurements'].append(dict(background=background, hashes=hashes,
                training=all(m.training for m in model.modules()), dtype=str(dtype), **timing))
            del model
            if background:
                del task
            gc.collect()
        a, b = report['measurements']
        assert a['hashes'] == b['hashes'] and a['training'] and b['training']
        # Prove background exceptions propagate and the worker is reclaimed.
        failed = CPUModelLoad(lambda:(_ for _ in ()).throw(ValueError('probe expected')))
        try:
            failed.result()
        except ValueError as error:
            assert str(error) == 'probe expected'
        else:
            raise AssertionError('load exception was swallowed')
        failed.thread.join(5)
        assert not failed.thread.is_alive()
        report['success'] = True
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        torch.cuda._lazy_init = original_lazy
        for name, method in saved.items():
            setattr(torch.distributed, name, method)
        report.update(done=True, cuda_initialized=torch.cuda.is_initialized(),
                      distributed_initialized=torch.distributed.is_initialized())
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(success=report['success'], error=report.get('error'))))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
