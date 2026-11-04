"""Independently audit raw events from the V4 native deadline probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _number(value):
    return (isinstance(value,(int,float)) and not isinstance(value,bool) and
            math.isfinite(value))


def verify(root):
    root=Path(root)
    errors=[]

    def check(ok,message):
        if not ok:errors.append(message)

    try:
        summary=json.loads((root/'summary.json').read_text())
    except (OSError,json.JSONDecodeError) as error:
        return dict(pass_all=False,errors=['summary.json: '+str(error)],case='invalid',
                    failure_layer='unresolved')
    if not isinstance(summary,dict):
        return dict(pass_all=False,errors=['summary must be a JSON object'],case='invalid',
                    failure_layer='unresolved')

    pg_timeout=summary.get('pg_timeout')
    delay=summary.get('delay')
    check(_number(pg_timeout) and pg_timeout>0,'positive native PG timeout')
    check(_number(pg_timeout) and any(math.isclose(pg_timeout,value,rel_tol=0,abs_tol=1e-9)
                                      for value in (2.0,10.0)),
          'PG timeout uses a supported 2s or 10s probe setting')
    check(_number(delay) and delay>=0,'nonnegative injected delay')
    if not _number(pg_timeout) or pg_timeout<=0:pg_timeout=None
    if not _number(delay) or delay<0:delay=None
    expected_timeout=(delay is not None and pg_timeout is not None and delay>pg_timeout)
    check(summary.get('expected_timeout') is expected_timeout,'summary timeout classification matches delay/deadline')
    check(summary.get('backend')=='nccl' and summary.get('gpu_used') is True,
          'probe used the native NCCL backend')
    check(summary.get('resources_clean') is True,'controller resource audit passed')
    check(type(summary.get('cancel_timer')) is bool,'cancel_timer flag is boolean')

    rows={}
    for replica in (0,1):
        path=root/f'R{replica}.jsonl'
        try:
            events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (OSError,json.JSONDecodeError) as error:
            errors.append(f'R{replica}.jsonl: {error}')
            events=[]
        rows[replica]=events
        check(bool(events),f'events present for replica {replica}')
        check(all(isinstance(x,dict) for x in events),f'event objects for replica {replica}')
        if not all(isinstance(x,dict) for x in events):
            rows[replica]=[]
            continue
        check(all(x.get('replica')==replica for x in events),f'replica binding {replica}')
        pid_values=[x.get('pid') for x in events]
        pids=set(pid_values) if all(isinstance(pid,(int,str,float,bool,tuple)) for pid in pid_values) else set()
        check(len(pids)==1 and all(type(pid) is int and pid>0 for pid in pids),
              f'PID continuity {replica}')
        check(all(type(x.get('step')) is int for x in events),f'step binding {replica}')

    pids=[next(iter({x.get('pid') for x in rows[r]}),None) for r in (0,1)]
    check(pids[0] is not None and pids[1] is not None and pids[0]!=pids[1],
          'worker PIDs are distinct')

    deadline_values=[]
    for replica,events in rows.items():
        ready=[x for x in events if x.get('kind')=='ready']
        check(len(ready)==1 and ready[0].get('step')==-1,
              f'one initial deadline report {replica}')
        if len(ready)==1:
            row=ready[0]
            valid=(_number(row.get('manager_timeout')) and row['manager_timeout']==90 and
                   _number(row.get('quorum_timeout')) and row['quorum_timeout']==180 and
                   pg_timeout is not None and _number(row.get('pg_timeout')) and
                   math.isclose(row['pg_timeout'],pg_timeout,rel_tol=0,abs_tol=1e-9))
            check(valid,f'effective Manager/quorum/PG deadlines {replica}')
            deadline_values.append(dict(replica=replica,manager_timeout=row.get('manager_timeout'),
                quorum_timeout=row.get('quorum_timeout'),pg_timeout=row.get('pg_timeout')))

    cancel_timer=summary.get('cancel_timer') is True
    case='native_pg_timeout' if expected_timeout else 'cancel_timer_control' if cancel_timer else 'normal'
    abort_events=[x for events in rows.values() for x in events if x.get('kind')=='timer_abort']
    error_events=[x for events in rows.values() for x in events if x.get('kind')=='error']
    timer_origin=None
    timer_elapsed=None

    for replica,events in rows.items():
        commits=[x for x in events if x.get('kind')=='commit']
        errors_for_worker=[x for x in events if x.get('kind')=='error']
        done=[x for x in events if x.get('kind')=='done']
        expected_steps=[0,1] if expected_timeout else [0,1,2]
        check([x.get('step') for x in events if x.get('kind')=='quorum_ready']==expected_steps,
              f'quorum-ready step coverage {replica}')
        check([x.get('step') for x in events if x.get('kind')=='allreduce_begin']==expected_steps,
              f'allreduce begin step coverage {replica}')
        check([x.get('step') for x in commits]==([0] if expected_timeout else [0,1,2]),
              f'commit coverage {replica}')
        check([x.get('committed_step') for x in commits]==([1] if expected_timeout else [1,2,3]),
              f'committed progress {replica}')

        successful_steps=[0] if expected_timeout else [0,1,2]
        ends=[x for x in events if x.get('kind')=='allreduce_end' and x.get('step') in successful_steps]
        decisions=[x for x in events if x.get('kind')=='commit_decision' and x.get('step') in successful_steps]
        check([x.get('step') for x in ends]==successful_steps and
              all(x.get('wait_return') is True and x.get('exact') is True for x in ends),
              f'completed exact AVG steps {replica}')
        check([x.get('step') for x in decisions]==successful_steps and
              all(x.get('decision') is True for x in decisions),f'commit decisions {replica}')

        if expected_timeout:
            check(not done,f'no done after native timeout {replica}')
            check(len(errors_for_worker)==1 and errors_for_worker[0].get('step')==1,
                  f'step 1 error {replica}')
            check(all(isinstance(errors_for_worker[0].get(key),str)
                      for key in ('error','manager_error','native_pg_error')) if errors_for_worker else False,
                  f'error layer fields {replica}')
        else:
            check(not errors_for_worker,f'no worker error {replica}')
            check(len(done)==1 and done[0].get('step')==2,f'worker done {replica}')

        guards=[x for x in events if x.get('kind')=='guard_exception_caught']
        expected_guards=[1] if cancel_timer else []
        check([x.get('step') for x in guards]==expected_guards,
              f'cancelled guard timer evidence {replica}')
        delays=[x for x in events if x.get('kind')=='injected_delay']
        if delay and delay>0 and replica==0:
            check(len(delays)==1 and delays[0].get('step')==1 and
                  _number(delays[0].get('seconds')) and math.isclose(delays[0]['seconds'],delay,rel_tol=0,abs_tol=1e-9),
                  'replica 0 injected delay')
        else:
            check(not delays,f'no unexpected injected delay {replica}')

    if expected_timeout:
        check(len(error_events)==2 and {x.get('replica') for x in error_events}=={0,1} and
              all(x.get('step')==1 for x in error_events),'two step 1 worker errors')
        aborted=any('aborted' in (str(x.get('native_pg_error',''))+' '+str(x.get('manager_error',''))).lower()
                    for x in error_events)
        check(aborted,'native PG or Manager error reports aborted')
        valid_origins={'cuda_stream','context/_run_context'}
        timers_valid=(bool(abort_events) and all(x.get('step')==1 and x.get('origin') in valid_origins and
            pg_timeout is not None and _number(x.get('timeout_seconds')) and
            math.isclose(x['timeout_seconds'],pg_timeout,rel_tol=0,abs_tol=1e-9) and
            _number(x.get('elapsed')) and pg_timeout<=x['elapsed']<pg_timeout+1 for x in abort_events))
        check(timers_valid,'native timer abort at the independent PG deadline')
        if abort_events:
            timer_origin=abort_events[0].get('origin')
            timer_elapsed=abort_events[0].get('elapsed')
        check(summary.get('success') is False,'timeout case reports worker failure')
        exit_codes=summary.get('exit_codes')
        check(isinstance(exit_codes,list) and len(exit_codes)==2 and
              all(type(code) is int and code!=0 for code in exit_codes),'both timeout workers exited nonzero')
        failure_layer='native_process_group_timeout' if timers_valid and aborted else 'unresolved'
    else:
        check(not error_events,'normal run has no error events')
        check(not abort_events,'normal run has no timer aborts')
        check(summary.get('success') is True,'normal case completed successfully')
        check(summary.get('exit_codes')==[0,0],'normal worker exit codes')
        failure_layer=None

    return dict(pass_all=not errors,errors=errors,case=case,expected_timeout=expected_timeout,
        manager_timeout_seconds=90,quorum_timeout_seconds=180,pg_timeout_seconds=pg_timeout,
        deadlines=deadline_values,failure_layer=failure_layer,timer_origin=timer_origin,
        timer_elapsed_seconds=timer_elapsed,error_count=len(error_events),timer_abort_count=len(abort_events),
        cancel_timer_guard_steps={str(replica):[x['step'] for x in rows[replica]
            if x.get('kind')=='guard_exception_caught'] for replica in (0,1)},
        resources_clean=summary.get('resources_clean') is True)


def summarize(root):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    audit=verify(root)
    raw_files=[root/'summary.json',root/'R0.jsonl',root/'R1.jsonl']
    raw_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in raw_files if path.is_file()}
    actual_auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(summary=summary,independent_audit=audit,pass_all=audit['pass_all'],
        actual_auditor_sha256=actual_auditor_sha256,raw_sha256=raw_sha256)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=verify(args.directory)
    (args.directory/'independent_deadline_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if not result['pass_all']:raise SystemExit(1)
