"""Independently verify and summarize the one real-anchor P8 diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(directory):
    raw=directory/'status.json';x=json.loads(raw.read_text());manifest=json.loads((directory/'manifest.json').read_text())
    for path,item in manifest['files'].items():
        assert hashlib.sha256((directory/Path(path).name).read_bytes()).hexdigest()==item['sha256']
    assert x['done'] and x['success'] and not x['training'] and x['injections']==0
    names=('gpu_staged','cpu_staged','cpu_sha_only','gpu_breakdown')
    expected=x['rows'][0]['hashes'];assert len(expected)==9
    assert len(x['rows'])==12 and all(r['hashes']==expected for r in x['rows'])
    out=dict(source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),anchor_sha256=x['anchor_sha256'],
             nine_hashes_exact=True,measurements=12,inventory=x['inventory'],engines={})
    for name in names:
        rows=[r for r in x['rows'] if r['engine']==name]
        assert len(rows)==3 and {r['repeat'] for r in rows}=={0,1,2}
        mean=statistics.fmean(r['seconds'] for r in rows);assert mean==x['means'][name]
        out['engines'][name]=dict(mean_s=mean,min_s=min(r['seconds'] for r in rows),
            max_s=max(r['seconds'] for r in rows),sd_s=statistics.stdev(r['seconds'] for r in rows),
            fields={k:{m:statistics.fmean(r['fields'][k][m] for r in rows) for m in rows[0]['fields'][k]} for k in expected})
        if name=='gpu_breakdown':
            for r in rows:
                for m in r['fields'].values():
                    assert abs(m['total_s']-sum(m[k] for k in ('allocate_s','enqueue_s','wait_s','sha_s','traversal_residual_s')))<1e-9
    potential=statistics.fmean(min(r['fields']['model']['total_s'],r['fields']['optimizer']['total_s']) for r in x['rows'] if r['engine']=='cpu_sha_only')
    assert potential==x['hypothetical_sha_two_field_saving_s']
    out.update(optimistic_two_field_sha_hidden_s=potential,prototype_admission_s=.2,
               prototype_admitted=potential>=.2,performance_pass=False,
               cuda_extra_peak_bytes=max(r['cuda_peak_extra_bytes'] for r in x['rows']),
               process_maxrss_kib=max(r['process_maxrss_kib'] for r in x['rows']),
               pinned_live_slab_capacity_bytes=x['pinned_capacity_bytes'],
               limits='Single rank only. CPU total high-water is not incremental RSS; pinned allocator cache/reserved peak not measured. No parallel execution or end-to-end saving measured.')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=summarize(a.input)
    with a.output.open('x') as stream:stream.write(json.dumps(x,indent=2)+'\n')
    print(json.dumps({k:x[k] for k in ('nine_hashes_exact','optimistic_two_field_sha_hidden_s','prototype_admitted','performance_pass')}))


if __name__=='__main__':main()
