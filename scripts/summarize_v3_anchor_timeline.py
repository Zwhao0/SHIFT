"""Check all-role anchor diagnostics and selected-rank operation histograms."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from scripts.v3_numerical_remote import STEPS, audit, compare_compact
from scripts.summarize_v3_timeline import analyze


def histograms(path):
    events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    events=[e for e in events if e.get('ph')=='X' and 'dur' in e]
    phases=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('shift::')]
    external={}
    cpu=defaultdict(Counter)
    for e in events:
        if e.get('cat')!='cpu_op':continue
        for phase in phases:
            if e.get('pid')==phase.get('pid') and phase['ts']<=e['ts']<phase['ts']+phase['dur']:
                cpu[phase['name']][e['name']]+=1
                key=e.get('args',{}).get('External id')
                if key is not None:external[key]=phase['name']
    kernels=defaultdict(Counter)
    for e in events:
        phase=external.get(e.get('args',{}).get('External id'))
        if phase and e.get('cat')=='kernel' and 'nccl' not in e['name'].lower():
            kernels[phase][e['name']]+=1
    optimizer=next(e for e in phases if e['name']=='shift::optimizer')
    end=optimizer['ts']+optimizer['dur']
    gpu=[e for e in events if e.get('cat')=='kernel']
    runtime=[e for e in events if e.get('cat')=='cuda_runtime']
    return dict(cpu_ops={k:dict(sorted(v.items())) for k,v in cpu.items()},
                non_nccl_kernels={k:dict(sorted(v.items())) for k,v in kernels.items()},
                coverage=dict(optimizer_cpu_lerp_ops=cpu['shift::optimizer'].get('aten::lerp_',0),
                    optimizer_observed_cuda_lerp_kernels=sum(v for k,v in kernels['shift::optimizer'].items() if 'lerp_scalar' in k),
                    last_kernel_relative_optimizer_end_ms=(max(e['ts']+e['dur'] for e in gpu)-end)/1000,
                    last_runtime_relative_optimizer_end_ms=(max(e['ts']+e['dur'] for e in runtime)-end)/1000,
                    note='CPU/GPU trace coverage must be checked before comparing optimizer kernel counts; missing tail is not less work.'))


def first_routing(base,candidate):
    """Absolute call counter differs after anchor load; use ordinal within layer."""
    _,ae=audit(base);_,be=audit(candidate)
    result=[]
    for role in ae:
        def rows(events):
            counters=Counter();mapped={}
            for e in events:
                if e['kind']!='routing' or e['step']+1 not in STEPS:continue
                group=(e['step'],e['layer']);key=(*group,counters[group]);counters[group]+=1
                mapped[key]=e
            return mapped
        ar,br=rows(ae[role]),rows(be[role])
        if ar.keys()!=br.keys():raise ValueError('routing event coverage differs')
        diffs=[key[0]+1 for key in ar if ar[key]['topk_hash']!=br[key]['topk_hash']]
        memberships=[key[0]+1 for key in ar if ar[key]['sorted_topk_hash']!=br[key]['sorted_topk_hash']]
        step2=[key for key in ar if key[0]==1]
        if not step2:raise ValueError('missing first forward routing')
        result.append(dict(role=role,first_ordered_difference_commit=min(diffs,default=None),
                           first_membership_difference_commit=min(memberships,default=None),
                           first_forward_calls=len(step2),first_forward_exact=all(
                               ar[k]['topk_hash']==br[k]['topk_hash'] and
                               ar[k]['sorted_topk_hash']==br[k]['sorted_topk_hash'] for k in step2)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]/'results'
    status=root/f'v3_b1_20261103_{a.suffix}'/'status.json'
    state=json.loads(status.read_text())
    if not state['done'] or not state.get('anchor_timeline_from'):raise ValueError('not completed anchor diagnostic')
    runs={r['mode']:root/r['session'] for r in state['rows']}
    if set(runs)!={'shift','nativeft','flat'}:raise ValueError('incomplete triad')
    data={};raw_sha={}
    for mode,directory in runs.items():
        signatures,events=audit(directory)
        for role,rows in events.items():
            loaded=[e for e in rows if e['kind']=='reference_anchor_loaded']
            if len(loaded)!=1 or loaded[0]['step']!=1 or loaded[0]['schema']!=2 or not loaded[0]['hashes_match']:
                raise ValueError('anchor restore not validated')
            path=directory/f'node{role[0]}'/f'events_g0_r{role[1]}.jsonl'
            raw_sha[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        data[mode]={}
        for rank in (0,4):
            paths=list(directory.glob(f'physical_node*/rid0_g0/timeline_step2_r{rank}.json.gz'))
            if len(paths)!=1:raise ValueError('missing/duplicate physical timeline')
            data[mode][rank]=dict(timeline=analyze(paths[0]),histograms=histograms(paths[0]))
    comparisons={}
    for candidate in ('nativeft','flat'):
        compact=compare_compact(runs['shift'],runs[candidate])
        first_hash=[]
        for rid in range(2):
            for rank in range(8):
                first_hash.append(dict(role=[rid,rank],categories={category:min(
                    (d[2] for d in compact['hash_differences'] if d[:2]==[rid,rank] and d[3]==category),default=None)
                    for category in ('gradients','model','optimizer')}))
        comparisons[candidate]=dict(compact=compact,first_hash_differences=first_hash,
                                    routing=first_routing(runs['shift'],runs[candidate]))
    output=dict(performance_evidence=False,rank_steps=480,source_status_sha256=hashlib.sha256(status.read_bytes()).hexdigest(),
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                raw_sha256=raw_sha,runs=data,comparisons=comparisons,
                limitations='Intrusive cold first F/B; schedule includes backward, local/outer finalization and routing audit. '
                'Equal histograms do not prove equal tensor shapes/values or operation order. No production timing inference.')
    with a.output.open('x') as stream:stream.write(json.dumps(output,indent=2)+'\n')
    print(json.dumps({m:{r:{k:sum(v.values()) for k,v in q['histograms']['non_nccl_kernels'].items()}
                         for r,q in ranks.items()} for m,ranks in data.items()}))


if __name__=='__main__':main()
