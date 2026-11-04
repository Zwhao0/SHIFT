"""Diagnostic CPU/CUDA timeline unions; overlaps are never summed as elapsed time."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([begin, end])
    return result


def occupancy(events):
    spans = union([(e['ts'], e['ts']+e['dur']) for e in events])
    if not spans:
        return dict(count=0)
    hull = spans[-1][1]-spans[0][0]
    busy = sum(end-begin for begin,end in spans)
    gaps = [b[0]-a[1] for a,b in zip(spans,spans[1:])]
    return dict(count=len(events), hull_ms=hull/1000, busy_union_ms=busy/1000,
                gaps_ms=(hull-busy)/1000, longest_gap_ms=max(gaps,default=0)/1000)


def analyze(path):
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    metadata = json.loads(path.with_suffix('').with_suffix('.metadata.json').read_text())
    if metadata.get('committed') is not True:
        raise ValueError('successful commit must be explicit in timeline metadata')
    if hashlib.sha256(compressed).hexdigest()!=metadata['compressed_sha256'] or hashlib.sha256(raw).hexdigest()!=metadata['raw_sha256']:
        raise ValueError('timeline SHA mismatch')
    events = [e for e in json.loads(raw)['traceEvents'] if e.get('ph')=='X' and 'dur' in e]
    phases = [e for e in events if e['name'].startswith('shift::') and e.get('cat')=='user_annotation']
    if sorted(e['name'] for e in phases) != ['shift::finite','shift::optimizer','shift::schedule','shift::should_commit']:
        raise ValueError('missing/duplicate timeline scopes')
    # Associate through CPU-op external id and phase time window, including
    # autograd worker threads. This is temporal association, not a call tree.
    ext_phase = {}
    for e in events:
        external = e.get('args',{}).get('External id')
        if external is None or e.get('cat')!='cpu_op':
            continue
        for phase in phases:
            if (e.get('pid')==phase.get('pid')
                    and phase['ts']<=e['ts']<phase['ts']+phase['dur']):
                ext_phase[external] = phase['name']
    gpu = [e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    if not gpu:
        raise ValueError('no CUDA timeline activity; do not infer GPU idle from CPU-only trace')
    devices = sorted({str(e.get('args',{}).get('device',e.get('pid'))) for e in gpu})
    if len(devices)!=1:
        raise ValueError('expected one selected rank device per trace')
    summaries = {}
    for phase in phases:
        name = phase['name']
        matched = [e for e in gpu if ext_phase.get(e.get('args',{}).get('External id'))==name]
        nccl = [e for e in matched if 'nccl' in e['name'].lower()]
        compute = [e for e in matched if 'nccl' not in e['name'].lower() and e.get('cat')=='kernel']
        names = defaultdict(lambda:dict(count=0,sum_kernel_ms=0.0))
        for e in matched:
            names[e['name']]['count'] += 1
            names[e['name']]['sum_kernel_ms'] += e['dur']/1000
        summaries[name] = dict(cpu_scope_ms=phase['dur']/1000, gpu=occupancy(matched),
            nccl=occupancy(nccl), non_nccl_kernels=occupancy(compute),
            top_kernel_duration_sums=[dict(name=k,**v) for k,v in sorted(names.items(),key=lambda x:-x[1]['sum_kernel_ms'])[:8]],
            note='Temporal CPU-window association across autograd threads, not direct call ancestry. NCCL/non-NCCL spans overlap; their hulls and busy times cannot be added.')
    runtime = [e['dur']/1000 for e in events if e.get('cat')=='cuda_runtime' and 'LaunchKernel' in e['name']]
    before,after = metadata['before'],metadata['after']
    deltas = []
    for tid,value in after['tasks'].items():
        old=before['tasks'].get(tid)
        if old:
            deltas.append(dict(tid=tid,name=value['name'],
                running_ms=(value['schedstat'][0]-old['schedstat'][0])/1e6,
                runqueue_wait_ms=(value['schedstat'][1]-old['schedstat'][1])/1e6))
    return dict(path=str(path),performance_evidence=False,compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        raw_sha256=hashlib.sha256(raw).hexdigest(), device=devices[0], phases=summaries,
        gpu_total=occupancy(gpu),unassigned_gpu_events=sum(e.get('args',{}).get('External id') not in ext_phase for e in gpu),
        launch_api=dict(count=len(runtime),mean_ms=statistics.fmean(runtime) if runtime else None,
                        maximum_ms=max(runtime,default=0)),
        affinity=before['affinity'],thread_count_before=len(before['tasks']),thread_count_after=len(after['tasks']),
        scheduler_snapshot_span_ms=(after['monotonic']-before['monotonic'])*1000,
        top_thread_runqueue_wait=sorted(deltas,key=lambda x:-x['runqueue_wait_ms'])[:10],
        limitations='Single intrusive selected-rank step; CUPTI/profiler init/stop perturb timing. CPU snapshots include profiler init/stop. Gaps are absent SHIFT activity, not whole-device idle: company GPU contexts are not traced. Kernel unions include stream waits/NCCL residency, not pure arithmetic; no causality/performance pass.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--step',type=int,default=11)
    a=p.parse_args()
    results={}
    for root in a.roots:
        paths=[]
        for rank in (0,4):
            matches=list(root.glob(f'physical_node*/rid0_g0/timeline_step{a.step}_r{rank}.json.gz'))
            if len(matches)!=1:raise ValueError('expected one physical replica0 timeline per rank')
            paths.append(matches[0])
        results[root.name]=[analyze(path) for path in paths]
    a.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({name:[dict(path=r['path'],phases={k:{q:v[q] for q in ('cpu_scope_ms','gpu','non_nccl_kernels')} for k,v in r['phases'].items()}) for r in rows] for name,rows in results.items()}))


if __name__=='__main__':main()
