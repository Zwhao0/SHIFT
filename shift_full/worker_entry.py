"""Run a SHIFT training entrypoint from the fork-inherited fast-launch cache."""

import importlib
import os


def main():
    if os.environ.get('SHIFT_SINGLE_GPU_VISIBLE') == '1':
        devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        role=int(os.environ['LOCAL_RANK'])
        os.environ['CUDA_VISIBLE_DEVICES']=devices[role]
        os.environ['SHIFT_CUDA_DEVICE']='0'
    module_name = os.environ.get("SHIFT_ENTRY_MODULE")
    if not module_name:
        raise RuntimeError("SHIFT_ENTRY_MODULE must name the worker module to run")

    from shift_full.fast_launch import acquire_model

    cached_model = acquire_model()
    flash_train = importlib.import_module("flash_v1.train")

    def cached_load_model(_path):
        nonlocal cached_model
        if cached_model is None:
            raise RuntimeError('preloaded model already consumed')
        model, cached_model = cached_model, None
        return model

    flash_train.load_model = cached_load_model
    worker_module = importlib.import_module(module_name)
    worker_main = getattr(worker_module, "main", None)
    if not callable(worker_main):
        raise RuntimeError(f"worker module {module_name!r} has no callable main()")
    worker_main()


if __name__ == "__main__":
    main()
