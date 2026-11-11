"""Report the GPUs assigned to a standby Pod and optionally prime CUDA."""

import argparse
import json
import os
from pathlib import Path
import time


def prime_model(model_dir):
    """Read the staged weights into the node page cache without allocating GPU memory."""
    started = time.monotonic()
    buffer = bytearray(4 * 1024**2)
    size = 0
    with (model_dir / 'model.safetensors').open('rb', buffering=0) as stream:
        while count := stream.readinto(buffer):
            size += count
    if not size:
        raise RuntimeError('model weights are empty')
    return dict(bytes=size, seconds=time.monotonic() - started)


def probe(torch, expected_gpus, minimum_mib, profile, model_dir):
    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f'expected {expected_gpus} assigned GPUs, found {count}')
    if model_dir is not None:
        if not (model_dir / 'config.json').is_file():
            raise RuntimeError('model configuration is unavailable')
        if (model_dir / 'model.safetensors').stat().st_size == 0:
            raise RuntimeError('model weights are empty')
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        devices.append(dict(index=index, uuid=str(getattr(properties, 'uuid', '')),
                            name=properties.name, free_mib=free // (1024**2),
                            total_mib=total // (1024**2)))
    if not all(device['uuid'] for device in devices):
        raise RuntimeError('assigned GPU UUID is unavailable')
    if len({device['uuid'] for device in devices}) != count:
        raise RuntimeError('assigned GPU UUIDs are not unique')
    return dict(profile=profile, devices=devices, torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                ready=any(device['free_mib'] >= minimum_mib for device in devices))


def run(args):
    import torch

    identity = dict(node=os.environ['SHIFT_NODE_NAME'], instance=os.environ['SHIFT_POD_UID'])
    sequence = 0
    preflight_ms = warmup_ms = None
    while True:
        sequence += 1
        started = time.monotonic()
        try:
            report = probe(torch, args.gpus, args.minimum_mib, args.profile, args.model_dir)
        except (RuntimeError, OSError) as error:
            report = dict(profile=args.profile, devices=[], ready=False, error=str(error))
        probe_ms = round((time.monotonic() - started) * 1000)
        if preflight_ms is None:
            preflight_ms = probe_ms
        if report['ready'] and args.warm_cuda and warmup_ms is None:
            started = time.monotonic()
            try:
                for device in report['devices']:
                    if device['free_mib'] < args.minimum_mib:
                        continue
                    with torch.cuda.device(device['index']):
                        torch.ones(1, device='cuda').add_(1)
                        torch.cuda.synchronize()
            except RuntimeError as error:
                report.update(ready=False, error=str(error))
            else:
                warmup_ms = round((time.monotonic() - started) * 1000)
        row = dict(report, **identity, checked_at=time.monotonic(), sequence=sequence,
                   preflight_ms=preflight_ms, probe_ms=probe_ms, agent_cuda_warmup_ms=warmup_ms)
        temporary = args.snapshot.with_suffix('.tmp')
        temporary.write_text(json.dumps(row))
        temporary.replace(args.snapshot)
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('/tmp/shift-spare.json'))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--read', action='store_true')
    actions.add_argument('--check-ready', action='store_true')
    actions.add_argument('--prime-model', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--minimum-mib', type=int, default=512)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--profile', default='cuda-smoke')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--warm-cuda', action='store_true')
    args = parser.parse_args()
    if args.prime_model:
        if args.model_dir is None:
            parser.error('--prime-model requires --model-dir')
        print(json.dumps(prime_model(args.model_dir)))
    elif args.read or args.check_ready:
        try:
            report = json.loads(args.snapshot.read_text())
        except (OSError, ValueError):
            if args.check_ready:
                raise SystemExit(1)
            raise
        report['age_seconds'] = time.monotonic() - report.pop('checked_at')
        if args.check_ready:
            raise SystemExit(not (report['ready'] and 0 <= report['age_seconds'] <= 15))
        else:
            print(json.dumps(report))
    else:
        if args.gpus < 0 or min(args.minimum_mib, args.interval) <= 0:
            parser.error('GPU count must be nonnegative and memory and interval positive')
        run(args)


if __name__ == '__main__':
    main()
