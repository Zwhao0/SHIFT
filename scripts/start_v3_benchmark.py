"""Launch B1 in an independent bounded server systemd unit after prepare."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess as sp
import time
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True)
    p.add_argument('--digest-engine',choices=['reference','staged'],default='reference')
    p.add_argument('--overlap-cpu-model-load',action='store_true')
    p.add_argument('--overlap-cpu-imports',action='store_true')
    p.add_argument('--schedule',choices=['probe','common-probe','cold-probe','short-abba','abba','native-abba','triad','digest','digest-breakdown','numerical'],default='probe')
    p.add_argument('--trace-events',action='store_true')
    p.add_argument('--bucket-ready-trace',action='store_true')
    p.add_argument('--timeline-trace',action='store_true')
    p.add_argument('--sync-quorum',action='store_true')
    p.add_argument('--eager-groups',action='store_true')
    p.add_argument('--strict-diagnostic',action='store_true')
    p.add_argument('--compare-existing')
    p.add_argument('--validate-against')
    p.add_argument('--anchor-timeline-from')
    p.add_argument('--gradient-probe-from')
    p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args()
    if a.bucket_ready_trace and (a.schedule!='probe' or not a.trace_events or a.cpu_only):
        p.error('bucket readiness is a trace-enabled formal-load probe only')
    if a.overlap_cpu_model_load and (a.cpu_only or a.schedule not in ('cold-probe','numerical')):
        p.error('CPU model overlap admission requires cold-probe or numerical validation')
    if a.overlap_cpu_imports and (a.cpu_only or a.schedule not in ('cold-probe','numerical')):
        p.error('CPU import overlap admission requires cold-probe or numerical validation')
    if a.timeline_trace and (a.schedule not in ('probe','common-probe') or not a.trace_events or a.cpu_only or a.bucket_ready_trace):
        p.error('timeline is a separate trace-enabled formal-load probe only')
    if a.schedule in ('digest','digest-breakdown') and (a.digest_engine!='reference' or a.trace_events or a.sync_quorum or a.eager_groups or a.cpu_only):
        p.error('digest probe does not accept training/CPU-only flags')
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix): p.error('invalid suffix')
    if (a.strict_diagnostic or a.compare_existing or a.validate_against or a.anchor_timeline_from or a.gradient_probe_from) and a.schedule!='numerical':
        p.error('numerical options require --schedule numerical')
    if a.anchor_timeline_from and (a.cpu_only or a.timeline_trace or a.bucket_ready_trace or a.trace_events
                                   or a.strict_diagnostic or a.compare_existing or a.validate_against):
        p.error('anchor timeline is a separate numerical diagnostic')
    if a.gradient_probe_from and (a.cpu_only or a.trace_events or a.strict_diagnostic or a.anchor_timeline_from or a.compare_existing or a.validate_against or a.sync_quorum or a.eager_groups):
        p.error('gradient probe is a separate formal single-update diagnostic')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v3-start-'+str(__import__('os').getpid())+'.sock')
    unit='shift-v3-b1-'+a.suffix
    root='/workspace/SHIFT'
    out=root+'/results/v3_b1_20261103'
    run=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+('240' if a.schedule in ('digest','digest-breakdown') else '1800'),
         '--property=KillMode=control-group','--property=TimeoutStopSec=20',
         '--property=StandardOutput=append:'+out+'/'+a.suffix+'.log','--property=StandardError=inherit',
         '--working-directory='+root,'--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=SHIFT_DIGEST_ENGINE='+a.digest_engine]
    run.append('--setenv=SHIFT_OVERLAP_CPU_MODEL_LOAD='+('1' if a.overlap_cpu_model_load else '0'))
    run.append('--setenv=SHIFT_OVERLAP_CPU_IMPORTS='+('1' if a.overlap_cpu_imports else '0'))
    if a.cpu_only:
        run += [root+'/runtime/torch214/bin/python','-m','unittest','scripts.test_recovery_timeout',
                'scripts.test_v3_state','scripts.test_v3_benchmark','scripts.test_cold_launch',
                'shift_full.test_reference_audit','shift_full.test_rotation','shift_full.test_planned_protocol',
                'scripts.test_bucket_finalizer','scripts.test_v3_recovery','scripts.test_compare_progress']
    else:
        if a.schedule in ('digest','digest-breakdown'):
            run += ['--property=MemoryMax=16G','--setenv=CUDA_VISIBLE_DEVICES=0','--setenv=OMP_NUM_THREADS=1']
            module='scripts.v3_digest_remote' if a.schedule=='digest' else 'scripts.v3_digest_breakdown_remote'
            run += [root+'/runtime/torch214/bin/python','-m',module,'--suffix',a.suffix]
        elif a.schedule=='numerical':
            run += ['/usr/bin/python3','-m','scripts.v3_numerical_remote','--suffix',a.suffix]
            if a.strict_diagnostic:run.append('--strict-diagnostic')
            if a.compare_existing:run+=['--compare-existing',a.compare_existing]
            if a.validate_against:run+=['--validate-against',a.validate_against]
            if a.anchor_timeline_from:run+=['--anchor-timeline-from',a.anchor_timeline_from]
            if a.gradient_probe_from:run+=['--gradient-probe-from',a.gradient_probe_from]
        else:
            run += ['/usr/bin/python3','-m','scripts.v3_benchmark_remote','--suffix',a.suffix,'--schedule',a.schedule]
        if a.trace_events: run.append('--trace-events')
        if a.bucket_ready_trace: run.append('--bucket-ready-trace')
        if a.timeline_trace: run.append('--timeline-trace')
        if a.sync_quorum: run.append('--sync-quorum')
        if a.eager_groups: run.append('--eager-groups')
    credential_check='True' if a.cpu_only else 'Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file()'
    code=('from pathlib import Path; import subprocess,json; '
          'assert '+credential_check+', "Run prepare_v3_controller after the previous controller ends (single-use credential)"; '
          'Path('+repr(out)+').mkdir(parents=True,exist_ok=True); '
          'subprocess.run('+repr(run)+',check=True); '
          'print(json.dumps(dict(unit='+repr(unit)+',pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))')
    try:
        s.ensure_connection(e)
        result=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        result.update(cpu_only=a.cpu_only,launched=time.time())
        dest=s.root/'results/v3_b1_20261103';dest.mkdir(exist_ok=True)
        (dest/(a.suffix+'_owner.json')).write_text(json.dumps(result,indent=2)+'\n')
        provenance=s.root/'results/v3_a1_20261103/controller_prepare.json'
        if provenance.exists():
            (dest/(a.suffix+'_source_provenance.json')).write_bytes(provenance.read_bytes())
        print(json.dumps(result))
    finally:
        sp.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
