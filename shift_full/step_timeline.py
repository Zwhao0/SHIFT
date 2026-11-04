"""One-step opt-in CPU/CUDA diagnosis; never performance acceptance evidence."""
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def stream_sha(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024), b''):
        value.update(chunk)
    return value.hexdigest()


def cpu_snapshot():
    tasks = {}
    for task in Path('/proc/self/task').glob('[0-9]*'):
        try:
            tasks[task.name] = dict(name=(task/'comm').read_text().strip(),
                                   schedstat=[int(x) for x in (task/'schedstat').read_text().split()])
        except FileNotFoundError:
            pass
    return dict(monotonic=time.monotonic(), affinity=sorted(os.sched_getaffinity(0)), tasks=tasks)


class StepTimeline:
    def __init__(self, torch, target_step, out, rank):
        self.torch, self.target_step, self.out, self.rank = torch, target_step, out, rank
        self.profiler = None
        self.captured = False

    def begin(self, step):
        if step != self.target_step or self.captured:
            return
        self.before = cpu_snapshot()
        self.profiler = self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        self.profiler.start()

    def scope(self, name):
        return self.torch.profiler.record_function('shift::'+name) if self.profiler else nullcontext()

    def end(self, committed):
        if self.profiler is None:
            return None
        profiler, self.profiler = self.profiler, None
        profiler.stop()
        after = cpu_snapshot()
        path = self.out/f'timeline_step{self.target_step}_r{self.rank}.json'
        profiler.export_chrome_trace(str(path))
        compressed = path.with_suffix('.json.gz')
        with path.open('rb') as source, gzip.open(compressed, 'wb') as target:
            shutil.copyfileobj(source, target)
        with path.open('rb') as source:
            raw_sha = stream_sha(source)
        with gzip.open(compressed, 'rb') as source:
            if stream_sha(source) != raw_sha:
                raise RuntimeError('timeline compression checksum mismatch')
        report = dict(performance_evidence=False, step=self.target_step, rank=self.rank,
                      committed=committed, before=self.before, after=after, uncompressed_bytes=path.stat().st_size,
                      raw_sha256=raw_sha)
        with compressed.open('rb') as source:
            report['compressed_sha256'] = stream_sha(source)
        path.with_suffix('.metadata.json').write_text(json.dumps(report, indent=2)+'\n')
        path.unlink()
        self.captured = True
        return dict(path=str(compressed), **{k:report[k] for k in
                    ('performance_evidence','step','committed','uncompressed_bytes','raw_sha256','compressed_sha256')})

    def close(self):
        if self.profiler is not None:
            profiler, self.profiler = self.profiler, None
            profiler.stop()
