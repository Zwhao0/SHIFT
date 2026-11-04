"""Independent raw-event/source checks for the one full P9 shadow diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root,provenance):
    manifest=json.loads((root/'physical_manifest.json').read_text())
    if {x['role'] for x in manifest}!=set('ABC') or len(manifest)!=3:raise ValueError('three roles required')
    raw={};sha={}
    for binding in manifest:
        role=binding['role'];folder=root/('physical_'+binding['physical_node'])/Path(binding['directory']).name
        raw[role]={}
        for rank in range(8):
            p=folder/f"events_g{binding['generation']}_r{rank}.jsonl"
            sha[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            if any(e['rank']!=rank for e in rows):raise ValueError('raw rank mismatch')
            raw[role][rank]=rows
    def one(role,rank,kind,step=None):
        rows=[e for e in raw[role][rank] if e['kind']==kind and (step is None or e.get('step')==step)]
        if len(rows)!=1:raise ValueError(f'{role}/{rank}/{kind}: expected one, got {len(rows)}')
        return rows[0]
    cut=one('A',0,'planned_cut_selected')['step'];checks=[];computed_times=[];relay_starts=[]
    for rank in range(8):
        b=one('B',rank,'planned_gradient_sent',cut);c=one('C',rank,'planned_shadow_computed',cut)
        restored=one('C',rank,'restored',cut);rollback=one('C',rank,'planned_shadow_rollback',cut)
        compared=one('C',rank,'planned_shadow_compared',cut);desc=one('B',rank,'planned_shadow_descriptor',cut)['descriptor']
        samples=[cut*8+micro*4+desc['replica_index']*2+desc['local_data_rank'] for micro in range(2)]
        if not (desc==c['descriptor'] and samples==b['sample_ids']==c['sample_ids'] and desc['participants']==2
                and desc['seq_len']==512 and desc['microbatches']==2 and desc['data']=='pattern'
                and desc['seed_base']==100000 and desc['rank']==rank and desc['step']==cut
                and desc['epoch']==1 and desc['schema']==1 and desc['replica_index'] in (0,1)):
            raise ValueError('sample/quorum/workload mismatch')
        for key in ('gradient_hash','parameter_gradient_hash','tail_hash'):
            if b[key]!=c[key] or compared['source'][key]!=b[key] or compared['shadow'][key]!=c[key]:
                raise ValueError(f'full pre-outer mismatch: rank{rank}/{key}')
        if not (compared['exact'] and rollback['hashes_match'] and rollback['ddp_clean']
                and len(rollback['hashes'])==9 and rollback['hashes']==restored['expected_hashes']==restored['actual_hashes']
                and c['owner_thread']==rollback['owner_thread']):raise ValueError('rollback/owner mismatch')
        def routes(role,shadow):
            result={}
            for e in raw[role][rank]:
                if e['kind']!='routing' or e['step']!=cut or bool(e.get('shadow'))!=shadow:continue
                result.setdefault(e['layer'],[]).append({k:e[k] for k in
                    ('topk_hash','sorted_topk_hash','min_boundary_margin','zero_margins','near_ties_1e6','tokens')})
            return result
        br=routes('B',False);cr=routes('C',True)
        if not br or br!=cr:raise ValueError('full routing sequence mismatch')
        receives=[e for e in raw['C'][rank] if e['kind']=='transport_recv' and e.get('step')==cut]
        if len(receives)!=2:raise ValueError('expected precopy and original relay only')
        computed_times.append(c['monotonic']);relay_starts.append(receives[1]['monotonic']-receives[1]['seconds'])
        preloaded=one('C',rank,'planned_precopy_loaded',cut)
        ready=one('C',rank,'planned_state_ready',cut)
        a_local=one('A',rank,'planned_local_gradient_ready',cut);a_freeze=one('A',rank,'planned_final_freeze',cut)
        checks.append(dict(rank=rank,samples=samples,route_calls=sum(map(len,br.values())),
            c_preloaded_to_computed_event_seconds=c['monotonic']-preloaded['monotonic'],
            c_callback_to_preouter_seconds=c['preouter_seconds'],capture_seconds=c['capture_seconds'],
            c_preloaded_to_rollback_seconds=rollback['monotonic']-preloaded['monotonic'],
            c_preloaded_to_state_ready_seconds=ready['monotonic']-preloaded['monotonic'],
            a_local_to_freeze_seconds=a_freeze['monotonic']-a_local['monotonic'],
            cuda_allocated_bytes=c['cuda_allocated_bytes'],cuda_peak_allocated_bytes=c['cuda_peak_allocated_bytes']))
    source=json.loads(provenance.read_text());source_checks={}
    for rel in ('shift_full/planned_train.py','shift_full/overlap_handoff.py','shift_full/planned_shadow.py',
                'scripts/v3_planned_remote.py','scripts/start_v3_planned.py'):
        actual=hashlib.sha256(Path(rel).read_bytes()).hexdigest();expected=source['runner_sha256'][rel]
        if actual!=expected:raise ValueError('local source changed since frozen prepare: '+rel)
        source_checks[rel]=actual
    return dict(all_roles_exact=True,cut=cut,roles=checks,source_sha256=source_checks,raw_sha256=sha,
        same_C_host_all_computed_to_earliest_original_relay_call_seconds=min(relay_starts)-max(computed_times),
        candidate_admission='Correctness pass. Plausible >=1s full-path saving only if added owner-fence/activation overhead <=.28s; separate protocol/card before prototype. Not measured end-to-end benefit.',
        production_contribution='original B relay',performance_evidence=False,
        timing_policy='Only same-rank monotonic differences. Shadow is intrusive; no cross-host subtraction, no published performance gate.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.root,a.provenance)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_sha256','source_sha256')}))

if __name__=='__main__':main()
