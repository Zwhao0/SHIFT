"""Keep nested import attribution separate from real end-to-end performance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

MODULES = {'torch.utils.checkpoint', 'torch._dynamo', 'torch._dynamo.eval_frame',
           'torch._inductor', 'torch._inductor.lowering', 'sympy',
           'megatron.core.transformer.utils', 'megatron.core.tensor_parallel.random',
           'megatron.core', 'megatron.core.pipeline_parallel.schedules'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs, sources = [], {}
    for mode in ('sequential', 'threaded'):
        path = a.directory/(mode+'_importtime.log')
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        status_path = a.directory/(mode+'.json')
        sources[str(status_path)] = hashlib.sha256(status_path.read_bytes()).hexdigest()
        status = json.loads(status_path.read_text())
        assert status['success'] and status['state_before'] == status['state_after']
        assert not status['denied'] and not status['cuda_initialized'] and not status['distributed_initialized']
        rows = []
        for line in path.read_text().splitlines():
            match = re.match(r'import time:\s*(\d+)\s*\|\s*(\d+)\s*\| (.*)', line)
            if match:
                module = match[3]
                rows.append(dict(self_us=int(match[1]), cumulative_us=int(match[2]),
                                 depth=(len(module)-len(module.lstrip()))//2, module=module.strip()))
        assert rows
        runs.append(dict(mode=mode, total_import_rows=len(rows),
                         guarded_import_interval_seconds=status['import_seconds'],
                         selected=[r for r in rows if r['module'] in MODULES],
                         top_self=sorted(rows, key=lambda r:r['self_us'], reverse=True)[:12]))
    result = dict(runs=runs, sources_sha256=sources, performance_evidence=False, gpu_admitted=False,
                  policy='Complete importtime logs retained; profiler changes timing. Nested/repeated cumulative '
                         'values overlap and MUST NOT be summed. Compiler imports may be needed by later model/'
                         'optimizer/first-FB; no avoided full-path cost or production lazy mechanism proven.')
    with a.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(runs=len(runs), performance_evidence=False, gpu_admitted=False)))


if __name__ == '__main__':
    main()
