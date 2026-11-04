"""Reanalyze collected B1 evidence without replacing original server analyses."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.v3_benchmark_remote import analyze, stats, matched_comparison, SCHEMAS


def same_derived(a,b):
    # Host Python/libm stdev can differ by a few ulps; raw run metrics stay exact.
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_derived(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same_derived(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b


def independent_raw(directory):
    """Recompute from original events, without using the server analyzer."""
    sources, commits = {}, {}
    for replica in range(2):
        for rank in range(8):
            path = directory/f'node{replica}'/f'events_g0_r{rank}.jsonl'
            raw = path.read_bytes()
            sources[str(path.relative_to(directory))] = hashlib.sha256(raw).hexdigest()
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            rows = [e for e in events if e['kind']=='commit']
            if [e['step'] for e in rows] != list(range(1,81)):
                raise ValueError('raw missing/duplicate/extra commit')
            if any(e['kind']=='skipped' for e in events):raise ValueError('raw skipped attempt')
            for e in rows:
                if not (e['participants']==2 and e['contributed'] and e['finite_gradients'] and e['global_batch']==8):
                    raise ValueError('raw invalid joint commit')
                if e['step']%20==0 and set(e['hashes'])!=SCHEMAS:raise ValueError('raw incomplete audit')
            commits[replica,rank] = rows
    seconds = [max(rows[k-1]['seconds'] for rows in commits.values()) for k in range(11,81)]
    intervals = [max(rows[k-1]['commit_interval_seconds'] for rows in commits.values()) for k in range(11,81)]
    elapsed = max(rows[79]['monotonic']-rows[9]['monotonic'] for rows in commits.values())
    for rank in range(8):
        for left,right in zip(commits[0,rank],commits[1,rank]):
            if left['hashes']!=right['hashes']:raise ValueError('raw audit mismatch')
    return dict(rank_steps=sum(map(len,commits.values())),source_sha256=sources,
                step_mean=statistics.fmean(seconds),interval_mean=statistics.fmean(intervals),elapsed=elapsed)


def summarize(root, suffix):
    status_path = root/f'v3_b1_20261103_{suffix}'/'status.json'
    status = json.loads(status_path.read_text())
    if not status['done'] or not status['success']:
        raise ValueError('complete successful benchmark required')
    runs = []
    pools = {}
    for row in status['rows']:
        directory = root/row['session']
        result = analyze(directory, 80)
        raw = independent_raw(directory)
        if (raw['step_mean'] != result['windows']['fixed11_80']['mean'] or
            raw['interval_mean'] != result['fixed11_80_commit_intervals']['mean'] or
            raw['elapsed'] != result['fixed11_80_elapsed_seconds']):
            raise ValueError('independent raw metric mismatch')
        original = json.loads((directory/'analysis.json').read_text())
        if original != json.loads(json.dumps(result)):
            raise ValueError('server analysis differs from collected raw')
        values = [result['step_seconds'][k] for k in range(11, 81)]
        pools.setdefault(row['mode'], []).extend(values)
        runs.append(dict(session=row['session'], mode=row['mode'], **result,
                         independent_raw=raw, fixed_step_variance=statistics.variance(values),
                         original_analysis_sha256=hashlib.sha256((directory/'analysis.json').read_bytes()).hexdigest()))
    comparisons = {a+'->'+b:matched_comparison([
        dict(mode=r['mode'],windows=r['windows'],commit_intervals=r['fixed11_80_commit_intervals'],
             elapsed_seconds=r['fixed11_80_elapsed_seconds']) for r in runs],a,b)
        for a,b in ([('flat','nativeft'),('nativeft','shift'),('flat','shift')]
                    if len(set(status['schedule']))==3 else [tuple(status['schedule'][:2])])}
    if status.get('comparisons') is not None and not same_derived(comparisons,status['comparisons']):
        raise ValueError('server triad comparisons differ')
    baseline, candidate = status['schedule'][:2]
    means = {m:[r['windows']['fixed11_80']['mean'] for r in runs if r['mode']==m]
             for m in (baseline,candidate)}
    pairs = [100*(b/a-1) for a,b in zip(means[baseline],means[candidate])]
    if len(pairs)!=4:
        raise ValueError('four complete adjacent ABBA/BAAB pairs required')
    center = statistics.fmean(pairs)
    half = 3.182446*statistics.stdev(pairs)/2
    return dict(source_status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(),
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                analysis_function_sha256=hashlib.sha256(Path(__file__).with_name('v3_benchmark_remote.py').read_bytes()).hexdigest(),
                baseline=baseline, candidate=candidate, fixed_window=[11,80], runs=runs,
                rank_steps=sum(r['independent_raw']['rank_steps'] for r in runs),comparisons=comparisons,
                pooled={m:dict(**stats(v),step_variance=statistics.variance(v)) for m,v in pools.items()},
                pooled_overhead_percent=100*(statistics.fmean(means[candidate])/statistics.fmean(means[baseline])-1),
                pairs_percent=pairs, paired_percent_mean=center, paired_percent_t95=[center-half,center+half],
                run_means=means, run_variance={m:statistics.variance(v) for m,v in means.items()},
                note='All 70 steps/run retained including audits. t95 is exploratory over four run pairs; step samples are not independent repetitions. Original analyses preserved.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=p.parse_args()
    result=summarize(a.root,a.suffix)
    out=a.root/f'v3_b1_20261103_{a.suffix}'/'reanalyzed.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('rank_steps','comparisons')}))


if __name__=='__main__':main()
