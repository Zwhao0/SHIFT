"""Collect already-audited compact results; never copies raw events or secrets."""
import hashlib
import json
from pathlib import Path
import statistics
from scripts.v41_control import command

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifests=['v41_manifest.json','v41_followup.json','v41_v4_recheck.json','v41_v1_performance.json']
    tasks=[t for m in manifests for t in json.loads((ROOT/'runtime/v3-reproduce'/m).read_text())['tasks']]
    names=[t['name'] for t in tasks]
    code='''from pathlib import Path
import json,hashlib
root=Path('/workspace/SHIFT/results');out={}
for name in NAMES:
 r=root/name
 s=json.loads((r/('summary.json' if (r/'summary.json').exists() else 'status.json')).read_text())
 a=json.loads((r/'v41_audit.json').read_text()) if (r/'v41_audit.json').exists() else None
 source=json.loads((r/'deployed_source.json').read_text())['expected'] if (r/'deployed_source.json').exists() else s.get('source_sha256',{})
 raw={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*.jsonl')} if name in ('v1_quick_20261104_u41_v1_audit1','v1_quick_20261104_u41_v1_audit2','v4_u41_v4_perf1') else {}
 out[name]=dict(summary={k:s[k] for k in ('seed','trigger','requested_trigger','initialization_seed','audit','audit_state','performance_no_audit','recovery_local_warmup','plan','reference_only','reference_run','post_layout','steady_mean_seconds','controller_seconds','decisions','role_hosts','source_sha256') if k in s},audit=a,sources=source,raw_sha256=raw)
a=json.loads((root/'unified_u41_v1_ref/selected_signatures.json').read_text())
b=json.loads((root/'unified_u41_v2_ref/selected_signatures.json').read_text())
c=json.loads((root/'v4_u41_v4_ref/reference_signatures.json').read_text())
assert a==b
assert all(x['sample_ids']==c['workers'][k][str(x['step'])]['sample_ids'] and all(x['hashes'][h]==v for h,v in c['workers'][k][str(x['step'])]['hashes'].items()) for k in a for x in a[k])
print(json.dumps(dict(cases=out,common_reference=dict(v1_v2_exact_nine_states_and_gradients=True,v1_v4_exact_nine_states_and_samples=True,logical_workers=16,common_steps=16,common_worker_steps=256,v4_reference_steps=32))))
'''.replace('NAMES',repr(names))
    packet=json.loads(command(code,True));cases=[]
    diagnostics={'v1_quick_20261104_u41_v1_audit1','v1_quick_20261104_u41_v1_audit2','v4_u41_v4_perf1'}
    for task in tasks:
        name=task['name'];row=packet['cases'][name];s=row['summary'];audit=row['audit']
        diagnostic=name in diagnostics
        assert diagnostic or audit and audit['pass_all'],name
        item=dict(name=name,accepted=not diagnostic,parameters=s,sources=row['sources'],independent_audit=audit)
        if 'launch' in task:
            args=task['launch'];item['command']=args
            for flag,key in (('--seed','seed'),('--trigger','trigger'),('--candidate-request-step','trigger')):
                if flag in args:item[key]=int(args[args.index(flag)+1])
        prepared=ROOT/'tmp'/f'{name}.prepare.json'
        if prepared.exists():
            item['verified_deployment_sha256']={k:v for k,v in json.loads(prepared.read_text())['runner_sha256'].items() if k.endswith('.py')}
        if diagnostic:
            item['raw_sha256']=row['raw_sha256']
            item['reason']=('cold join first kernels exceeded native 2-second gradient deadline; skipped retries; replaced by audited warmup-before-quorum and no-timed-hash trials'
                if name.startswith('v1_') else 'candidate deadline retained old owner; early controller teardown truncated candidate transport; strict audit failed; drain gate fixed and exact seed/trigger/deadline repeat plus forced-timeout audit passed')
        cases.append(item)
    by={x['name']:x for x in cases}
    mean=lambda n:by[n]['parameters']['steady_mean_seconds']
    pct=lambda x,y:100*(x/y-1)
    short_native=mean('unified_u41_native_final');old=json.loads((ROOT/'tmp/unified_base_before.summary.json').read_text())
    normal=dict(old_native_seconds=old['steady_mean_seconds'],final_short_native_seconds=short_native,
                common_core_change_percent=pct(short_native,old['steady_mean_seconds']),
                long_native_seconds=[mean('unified_u41_recheck_native1'),mean('unified_u41_recheck_native2')],
                long_v2_seconds=mean('unified_u41_recheck_v2'),
                v2_vs_faster_long_native_percent=pct(mean('unified_u41_recheck_v2'),min(mean('unified_u41_recheck_native1'),mean('unified_u41_recheck_native2'))),
                v2_post_layout_control_seconds=mean('unified_u41_recheck_v2_post'))
    perf={}
    for version,ns in dict(v1=['v1_quick_20261104_u41_v1_perf_final1','v1_quick_20261104_u41_v1_perf_final2'],
        v2=['v2_q2_u41_v2_perf1','v2_q2_u41_v2_perf2'],v4=['v4_u41_v4_perf_retry','v4_u41_v4_perf2']).items():
        perf[version]=[]
        for name in ns:
            a=by[name]['independent_audit'];w=a.get('windows') or a['timing']['fixed_windows']
            full=a.get('full_seconds') or a['timing']['request_to_first_common_commit_seconds']
            row=dict(name=name,complete_request_to_observed_common_seconds=full,
                     phase_max_rank_mean_seconds={k:v['max_rank_step']['mean'] for k,v in w.items()},
                     post_vs_own_pre_percent=pct(w['post']['max_rank_step']['mean'],w['pre']['max_rank_step']['mean']))
            if version in ('v1','v2'):
                row['post_vs_matched_AC_control_percent']=pct(w['post']['max_rank_step']['mean'],normal['v2_post_layout_control_seconds'])
            assert row['post_vs_own_pre_percent']<=10 or version=='v2' and row['post_vs_matched_AC_control_percent']<=10,name
            perf[version].append(row)
    assert abs(normal['common_core_change_percent'])<=10 and normal['v2_vs_faster_long_native_percent']<=10
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('shift_full','scripts') for p in (ROOT/d).glob('*.py')}
    report=dict(version='v4.1',title='代码重构统一',functional_pass=True,normal_and_post_timing_pass=True,
        global_batch=8,seq_len=512,data='random',initialization_seed=2026,
        rng_policy='replicated per logical shard; CPU/CUDA seed 70000+rank, Python seed 80000+rank; sample generator 100000+sample_id',
        initial_layout='complete 8-rank DP replicas on node0/node1; node3 standby; TP node-local; v4 crosses PP only after replacement',
        common_reference=packet['common_reference'],normal_timing=normal,performance=perf,cases=cases,source_sha256=sources,
        ancestry=dict(local_source_checkpoint='4fd55ec',original_remote_v4='7c8fe1e',v3_common_ancestor='8ee421c',publication_parent='origin/codex/v4-intra-replica; preserve local branch, no force'),
        limitations=['Shared company load is nonstationary: identical native code ranged 0.722 to 2.478 s; no isolated-speedup or statistical claim.',
          'Every fixed-window sample is retained, including slow steps. First common step is separate from the next four; teardown overshoot is separately retained.',
          'Native solo computes four microbatches to keep batch8, so it costs more than a joint step.',
          'Full times use one controller clock and include polling, preparation, native healing, transfer, necessary boundary validation and waiting; cross-host monotonic timestamps are never subtracted.',
          'Development audits perturb timing. V1/V2 performance runs do no timed hashes and require final nine-state development audit after five exact common commits. V4 retains boundary state validation but no per-step hashes.',
          'Recovery and migration timings characterize this new batch8 setup; they are not a controlled causal comparison against old batch16 v4 or a claim of arbitrary communication-fault tolerance.',
          'An initial native cold-join timing regression and one incomplete candidate-drain run are recorded as rejected diagnostics; neither is counted as a successful performance migration.'])
    dest=ROOT/'evidence/v4_1';dest.mkdir(exist_ok=True)
    # Minimal replay/comparison signatures; tensors and raw trajectories stay private and are deleted after audit.
    signatures={x['name']:{k:x['independent_audit'][k] for k in ('signature_sha256','final_signatures','last_validated_signatures') if k in (x['independent_audit'] or {})}
                for x in cases if x['independent_audit']}
    (dest/'selected_signatures.json').write_text(json.dumps(signatures,separators=(',',':'))+'\n')
    profiles={}
    for case in cases:
        case['parameters'].pop('source_sha256',None)
        for field in ('sources','verified_deployment_sha256'):
            if field not in case:continue
            value=case.pop(field);key=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
            profiles[key]=value;case[field+'_profile']=key
        if case['independent_audit']:
            for key in ('signature_sha256','final_signatures','last_validated_signatures'):
                case['independent_audit'].pop(key,None)
            case['independent_audit']['signatures_file']='selected_signatures.json#'+case['name']
    report['source_profiles']=profiles
    (dest/'summary.json').write_text(json.dumps(report,separators=(',',':'),ensure_ascii=False)+'\n')
    print(json.dumps(dict(cases=len(cases),normal=normal,performance=perf),ensure_ascii=False))


if __name__=='__main__':main()
