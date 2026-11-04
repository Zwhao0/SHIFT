"""Separate intrusive ten-step, full-anchor numerical diagnostic, no faults."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from scripts import v1_acceptance_remote as remote
from scripts.v3_benchmark_remote import NoFaultRotation, SCHEMAS, config
from shift_full.control import server_password

ROOT=Path(__file__).resolve().parents[1]
STEPS=list(range(2,12))


def load_events(directory):
    return {(rid,rank):[json.loads(s) for s in (directory/f'node{rid}'/f'events_g0_r{rank}.jsonl').read_text().splitlines() if s]
            for rid in range(2) for rank in range(8)}


def audit(directory):
    events=load_events(directory)
    signatures={}
    for role,rows in events.items():
        commits=[e for e in rows if e['kind']=='commit' and e['step'] in STEPS]
        if [e['step'] for e in commits]!=STEPS:raise RuntimeError('missing numerical joint steps')
        if any(e['kind']=='skipped' for e in rows):raise RuntimeError('numerical step skipped')
        if not all(e['participants']==2 and e['contributed'] and e['finite_gradients'] and e['global_batch']==8 for e in commits):
            raise RuntimeError('invalid numerical joint commit')
        sigs=[e for e in rows if e['kind']=='reference_signature' and e['step'] in STEPS]
        if [e['step'] for e in sigs]!=STEPS:raise RuntimeError('missing numerical signatures')
        if any(set(e['hashes'])!=SCHEMAS|{'gradients'} or e['manager_step']!=e['step'] for e in sigs):
            raise RuntimeError('invalid numerical signatures')
        signatures[role]={e['step']:e for e in sigs}
    for rank in range(8):
        for step in STEPS:
            if signatures[0,rank][step]['hashes']!=signatures[1,rank][step]['hashes']:
                raise RuntimeError(f'paired replica numerical mismatch {rank}:{step}')
    return signatures,events


def compare_compact(base,candidate):
    a,ae=audit(base);b,be=audit(candidate)
    diffs=[];routes=[];sample_errors=[]
    for role in a:
        for step in STEPS:
            if a[role][step]['sample_ids']!=b[role][step]['sample_ids']:sample_errors.append([*role,step])
            for key,value in a[role][step]['hashes'].items():
                if value!=b[role][step]['hashes'][key]:diffs.append([*role,step,key])
            ar=[e for e in ae[role] if e['kind']=='routing' and e['step']==step-1]
            br=[e for e in be[role] if e['kind']=='routing' and e['step']==step-1]
            if not ar or len(ar)!=len(br):raise RuntimeError('missing router events')
            for left,right in zip(ar,br):
                if left['layer']!=right['layer'] or left['topk_hash']!=right['topk_hash']:
                    routes.append(dict(role=role,step=step,layer=left['layer'],
                                       baseline_margin=left['min_boundary_margin'],candidate_margin=right['min_boundary_margin'],
                                       baseline_near_ties=left['near_ties_1e6'],candidate_near_ties=right['near_ties_1e6']))
    return dict(exact_pass=not(diffs or routes or sample_errors),hash_differences=diffs,
                router_differences=routes,sample_errors=sample_errors,rank_steps=16*len(STEPS),
                replica_pairs_exact=True)


def run_case(name,mode,anchor,strict=False,reuse_anchor=False,dump=True,sync_quorum=False,eager_groups=False,anchor_timeline=False,gradient_probe=False):
    c=config(mode,STEPS[-1],trace=anchor_timeline)
    if anchor_timeline:
        if strict or not reuse_anchor or dump:raise ValueError('anchor timeline requires formal reused anchor without dumps')
        c['train_args'] += ['--timeline-trace-step','2']
    if gradient_probe:
        if strict or not reuse_anchor or dump or STEPS != [2]:raise ValueError('probe must be one formal reused-anchor update')
        c['train_args'] += ['--gradient-probe-step','2']
    if sync_quorum:c['train_args'].append('--sync-quorum')
    if eager_groups:c['train_args'].append('--eager-groups')
    c['train_args']=[x for x in c['train_args'] if x!='--warmup-before-join']
    # Full CPU dumps are intentionally intrusive and may stall one host's disk.
    c['train_args'][c['train_args'].index('--communication-timeout')+1]='60'
    if strict:
        c['train_args'][c['train_args'].index('--seq-len')+1]='16'
        c['train_args'].append('--strict-fp32')
    c['train_args']+=['--reference-signatures','--audit-routing']
    if mode=='shift' and not reuse_anchor:c['train_args']+=['--reference-anchor-save',str(anchor)]
    else:c['train_args']+=['--reference-anchor',str(anchor),'--reference-anchor-step','1']
    if mode!='nativeft' and dump:
        c['train_args']+=['--reference-dump','--reference-dump-steps',','.join(map(str,STEPS))]
    s=NoFaultRotation(name,c,550,20261106,backup_mode='cold')
    try:
        # Refuse a diagnostic that would consume the reusable environment's space.
        for e in {s.endpoint_for(r) for r in range(2)}:
            code='import shutil; assert shutil.disk_usage("/workspace/SHIFT").free>180*1024**3,"insufficient numerical evidence space"'
            s.ssh(e,['python3','-c',code])
        s.start()
        s.await_('numerical_stopped',lambda d:all(s.rows(d,r,k,'stopped') for r in range(2) for k in range(8)),timeout=450)
        s.event('complete',faults=0)
    except BaseException as exc:
        s.event('failed',error=type(exc).__name__+': '+str(exc));raise
    finally:s.finish()
    audit(s.out)
    return s


def main():
    global STEPS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suffix',required=True)
    p.add_argument('--sync-quorum',action='store_true')
    p.add_argument('--eager-groups',action='store_true')
    p.add_argument('--strict-diagnostic',action='store_true',help='Separate FP32/TF32-off seq16 two-step diagnosis; never a performance result')
    p.add_argument('--compare-existing',help='CPU-only fixed-tolerance recompare of a completed suffix')
    p.add_argument('--validate-against',help='Formal-load SHIFT ten-step exact comparison against an existing anchor/run')
    p.add_argument('--gradient-probe-from',help='one formal original-anchor update, flat/SHIFT bounded slices on all ranks')
    p.add_argument('--anchor-timeline-from',help='Diagnostic only: reuse original formal anchor for all three modes; profile first step2 on ranks0/4, no tensor dumps')
    a=p.parse_args()
    if a.strict_diagnostic:STEPS=[2,3]
    if a.gradient_probe_from:STEPS=[2]
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    if a.compare_existing and not all(c.isalnum() or c in '_-' for c in a.compare_existing):p.error('invalid source suffix')
    if a.validate_against and not all(c.isalnum() or c in '_-' for c in a.validate_against):p.error('invalid validation suffix')
    if a.anchor_timeline_from and not all(c.isalnum() or c in '_-' for c in a.anchor_timeline_from):p.error('invalid anchor suffix')
    if a.gradient_probe_from and not all(c.isalnum() or c in '_-' for c in a.gradient_probe_from):p.error('invalid anchor suffix')
    if sum(bool(x) for x in (a.compare_existing,a.validate_against,a.strict_diagnostic,a.anchor_timeline_from,a.gradient_probe_from))>1:p.error('diagnostic modes are exclusive')
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/f'results/v3_b1_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    status=dict(done=False,success=False,rows=[],injections=0,diagnostic=True,strict_diagnostic=a.strict_diagnostic,steps=STEPS,sync_quorum=a.sync_quorum,eager_groups=a.eager_groups,started=time.time())
    path=out/'status.json'
    def save():path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    anchor=ROOT/'runtime'/f'v3-reference-{a.suffix}'
    sessions=[]
    try:
        if a.gradient_probe_from:
            source=json.loads((ROOT/f'results/v3_b1_20261103_{a.gradient_probe_from}/status.json').read_text())
            assert source['done'] and source['steps']==list(range(2,12)) and not source.get('strict_diagnostic')
            anchor=ROOT/'runtime'/f'v3-reference-{a.gradient_probe_from}'
            status.update(gradient_probe_from=a.gradient_probe_from,performance_evidence=False)
            for i,mode in enumerate(('flat','shift'),1):
                s=run_case(f'v3b1_{a.suffix}_{i:02d}_{mode}',mode,anchor,reuse_anchor=True,dump=False,gradient_probe=True)
                sessions.append(s)
                signatures,events=audit(s.out)
                for role,rows in events.items():
                    captures=[e for e in rows if e['kind']=='gradient_probe']
                    assert len(captures)==1 and captures[0]['step']==2
                    assert len(captures[0]['tensors'])==12
                status['rows'].append(dict(session=s.name,mode=mode,windows={}))
                save()
            result=compare_compact(sessions[0].out,sessions[1].out)
            (out/'shift_compact.json').write_text(json.dumps(result,indent=2)+'\n')
            assert not result['sample_errors'] and not result['router_differences']
            status.update(success=True,functional_pass=True,rank_steps=32)
            return
        if a.anchor_timeline_from:
            source=json.loads((ROOT/f'results/v3_b1_20261103_{a.anchor_timeline_from}/status.json').read_text())
            assert source['done'] and source['steps']==STEPS and not source.get('strict_diagnostic')
            anchor=ROOT/'runtime'/f'v3-reference-{a.anchor_timeline_from}'
            status.update(anchor_timeline_from=a.anchor_timeline_from,performance_evidence=False)
            for i,mode in enumerate(('shift','nativeft','flat'),1):
                s=run_case(f'v3b1_{a.suffix}_{i:02d}_{mode}',mode,anchor,reuse_anchor=True,dump=False,
                           sync_quorum=a.sync_quorum,eager_groups=a.eager_groups,anchor_timeline=True)
                sessions.append(s)
                status['rows'].append(dict(session=s.name,mode=mode,windows={}))
                if i>1:
                    result=compare_compact(sessions[0].out,s.out)
                    (out/f'{mode}_compact.json').write_text(json.dumps(result,indent=2)+'\n')
                    status[mode+'_exact']=result['exact_pass']
                save()
            # Expected flat divergence is retained, not a performance pass.
            status.update(success=True,functional_pass=True)
            return
        if a.compare_existing:
            from scripts.v3_compare_existing import compare_existing
            result=compare_existing(ROOT,a.compare_existing,a.suffix,out)
            status.update(flat_tolerance_pass=result['pass'],success=result['pass'],
                          compare_existing=a.compare_existing,training_started=False)
            return
        if a.validate_against:
            source=json.loads((ROOT/f'results/v3_b1_20261103_{a.validate_against}/status.json').read_text())
            assert source['done'] and source['steps']==STEPS and not source.get('strict_diagnostic')
            baseline=next(row['session'] for row in source['rows'] if row['mode']=='shift')
            anchor=ROOT/'runtime'/f'v3-reference-{a.validate_against}'
            s=run_case(f'v3b1_{a.suffix}_01_shift','shift',anchor,reuse_anchor=True,dump=False,sync_quorum=a.sync_quorum,eager_groups=a.eager_groups)
            status['rows'].append(dict(session=s.name,mode='shift',windows={}))
            result=compare_compact(ROOT/'results'/baseline,s.out)
            (out/'prior_shift_compact.json').write_text(json.dumps(result,indent=2)+'\n')
            status.update(prior_shift_exact=result['exact_pass'],success=result['exact_pass'],
                          validate_against=a.validate_against)
            return
        for i,mode in enumerate(('shift','nativeft','flat'),1):
            s=run_case(f'v3b1_{a.suffix}_{i:02d}_{mode}',mode,anchor,a.strict_diagnostic,sync_quorum=a.sync_quorum,eager_groups=a.eager_groups);sessions.append(s)
            status['rows'].append(dict(session=s.name,mode=mode,windows={}))
            if i>1:
                result=compare_compact(sessions[0].out,s.out)
                (out/f'{mode}_compact.json').write_text(json.dumps(result,indent=2)+'\n')
                status[mode+'_exact']=result['exact_pass']
            save()
        # Run fixed-tolerance CPU comparison on the physical host holding rid0.
        s=sessions[-1];e=s.endpoint_for(0);unit='shift-v3-numcompare-'+a.suffix
        report=ROOT/'results'/s.name/'flat_tolerance.json'
        base=ROOT/'results'/sessions[0].name/'rid0_g0';candidate=ROOT/'results'/s.name/'rid0_g0'
        cmd=['systemd-run','--quiet','--collect','--wait','--unit='+unit,'--property=RuntimeMaxSec=600',
             '--property=KillMode=control-group','--working-directory='+str(ROOT),
             '--setenv=PYTHONPATH='+str(ROOT),'--setenv=OMP_NUM_THREADS=1',
             str(ROOT/'runtime/torch214/bin/python'),'-m','shift_full.compare_reference',str(base),str(candidate),
             '--steps',','.join(map(str,STEPS)),'--output',str(report),'--threads','1']
        # A tolerance failure is evidence, rather than a controller failure.
        code='import subprocess; r=subprocess.run('+repr(cmd)+'); print(r.returncode)'
        try:
            status['tolerance_unit_exit']=s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True,timeout=650).strip()
            result=json.loads(s.ssh(e,['cat',str(report)],timeout=60))
            (out/'flat_tolerance.json').write_text(json.dumps(result,indent=2)+'\n')
            status['flat_tolerance_pass']=result['pass']
        finally:
            cleanup=('import subprocess; u='+repr(unit)+'; '
                     'subprocess.run(["systemctl","stop",u],capture_output=True,timeout=25); '
                     'r=subprocess.run(["systemctl","is-active",u],capture_output=True,text=True); '
                     'assert r.stdout.strip() not in ("active","activating","deactivating")')
            s.ssh(e,['sudo','-S','-p','','python3','-c',cleanup],sudo_input=True)
        status['success']=bool(status.get('nativeft_exact') and status.get('flat_tolerance_pass'))
    except BaseException as exc:status['error']=type(exc).__name__+': '+str(exc)
    finally:
        status.update(done=True,finished=time.time());save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
