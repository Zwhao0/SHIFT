"""Reduce independently audited V4 runs to compact, credential-free evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('outer_manager'):
        from scripts.verify_v4_outer import summarize as summarize_outer
        return summarize_outer(root)
    audit=json.loads((root/'independent_audit.json').read_text())
    if not audit['pass_all'] or not audit['resources_clean']:
        raise ValueError('independent event and resource audits must pass')
    rows={r:[json.loads(x) for d in range(4) for x in (root/f'{r}{d}.jsonl').read_text().splitlines() if x.strip()] for r in 'ABC'}
    final=[x for r in 'AC' for x in rows[r] if x['kind']=='done']
    def state_signature(events):
        hashes={str(x['rank']):x['hashes'] for x in events}
        if set(hashes) != {str(rank) for rank in range(8)}:
            raise ValueError('state signature requires complete logical rank coverage')
        return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    tensor_totals={}
    for x in final:
        if not x['comparison']:continue
        for section,stats in x['comparison']['tensors'].items():
            total=tensor_totals.setdefault(section,dict(elements=0,unequal=0,max_abs=0.))
            total['elements']+=stats['elements'];total['unequal']+=stats['unequal']
            total['max_abs']=max(total['max_abs'],stats['max_abs'])
    state_matches={}
    if summary['oracle']:
        keys=final[0]['comparison']['exact']
        state_matches={key:all(x['comparison']['exact'][key] for x in final) for key in keys}
    result={key:summary[key] for key in ('name','topology','plan','oracle','functional_pass','oracle_exact','diagnostic_timing','state_channel_bytes','source_sha256','cleanup_errors')}
    result.update(independent_audit=audit, requested_cut=summary.get('requested_cut',summary['plan']['cut']),
        state_transport=summary.get('state_transport','cpu_socket'),
        transport_totals=summary.get('transport_totals'),
        gpu_prepare=summary.get('gpu_prepare',False),gpu_warmup=summary.get('gpu_warmup',False),
        reuse_local_groups=summary.get('reuse_local_groups',False),preconnect_state=summary.get('preconnect_state',False),
        defer_source_local_close=summary.get('defer_source_local_close',False),
        injected_slow_ms=summary.get('slow_ms',0),all_rank_final_state_exact=state_matches,
        boundary_state_signature=state_signature([x for r in 'AB' for x in rows[r] if x['kind']=='freeze']),
        final_state_signature=state_signature(final),
        final_tensor_comparison=tensor_totals,
        post_step_oracle_cases=sum(x['kind']=='oracle_step' for r in 'AC' for x in rows[r]),
        post_step_oracle_failures=sum(x['kind']=='oracle_step' and not x['exact'] for r in 'AC' for x in rows[r]),
        standby_warmup_seconds=[x['seconds'] for x in rows['C'] if x['kind']=='standby_gpu_warmed'],
        commits_during_standby_preparation=sum(x.get('standby_preparing',False) and not x.get('standby_ready',True) for x in rows['A'] if x['kind']=='commit' and x['phase']=='before'),
        full_parameters=next(x['full_parameters'] for x in rows['A'] if x['kind']=='model_ready'),
        model_id=next(x['model_id'] for x in rows['A'] if x['kind']=='model_ready'),
        model_revision=next(x['revision'] for x in rows['A'] if x['kind']=='model_ready'),
        evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix=='.jsonl' or p.name in ('summary.json','resources_before.json','resources_after.json','independent_audit.json')})
    result['local_group_counts'] = {r+str(d):[
        {k:x[k] for k in ('reused','discarded','identity_preserved')}
        for x in rows[r] if x['device']==d and x['kind']=='local_groups_reused']
        for r in 'AC' for d in range(4)} if summary.get('reuse_local_groups') else None
    result['source_local_group_closure'] = {str(d):[
        {k:x[k] for k in ('reused','discarded','seconds')}
        for x in rows['B'] if x['device']==d and x['kind']=='source_local_groups_closed']
        for d in range(4)} if summary.get('defer_source_local_close') else None
    if 'phase_seconds' in summary:
        # Preserve invocation order per kind without repeating event metadata.
        result['phase_seconds'] = {
            worker:{kind:[x['seconds'] for x in events if x['kind']==kind]
                    for kind in dict.fromkeys(x['kind'] for x in events)}
            for worker,events in summary['phase_seconds'].items()}
    if result['cleanup_errors']:
        result['cleanup_interpretation']=('worker training and independent resource audit passed; controller reported '+
            ('already-collected source unit' if 'boundary02' in summary['name'] else 'resource-report variable shadowing; corrected in later source'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--output',type=Path,default=Path('evidence/v4/summary.json'));a=p.parse_args()
    result=dict(schema=1,experiment='V4 intra-replica planned node replacement',
        runs=[summarize(root) for root in a.directories],
        limitations=['acceptance scope, topology and failure coverage are recorded per run; inspect each independent audit',
            'diagnostic single-run timings are not a controlled performance comparison',
            'regenerable raw evidence is removed after summarization; hashes cannot reconstruct it'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),runs=len(result['runs']),all_functional=all(x['functional_pass'] for x in result['runs']))))


if __name__=='__main__':main()
