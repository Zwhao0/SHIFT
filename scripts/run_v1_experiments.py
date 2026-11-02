"""Serial post-endurance experiments; run only after the long run is collected."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session, endpoints_from_env
from shift_full.verify_rotation import verify
from shift_full.reporting import benchmark_summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v1'


def run(name, args, allowed=(0,)):
    OUT.mkdir(parents=True,exist_ok=True)
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as stream:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode not in allowed:
        raise RuntimeError(f'{name} exited {result.returncode}; see {OUT/(name+".log")}')
    print('DONE',name,flush=True)


def numerical():
    names={mode:f'v1_trajectory_{mode}_01' for mode in ('flat','shift')}
    from shift_full.move_snapshots import move
    from shift_full.remote_job import run_job
    probe=Session.__new__(Session);probe.root=ROOT;probe._sudo_password=None;probe.endpoints=endpoints_from_env()
    for mode,name in names.items():
        existing=ROOT/'results'/name/'controller.jsonl'
        if existing.exists():
            rows=[json.loads(line) for line in existing.read_text().splitlines()]
            if not all(any(e['kind']==kind for e in rows) for kind in ['complete','collected']):
                raise RuntimeError('incomplete existing trajectory session: '+name)
            print('REUSE',name,flush=True)
        else:
            run(name,['-m','shift_full.control','--session',name,'--config',f'configs/v1_trajectory_{mode}.json','--seconds','900'])
        probe.name=name
        transfer_file=OUT/(name+'_snapshot_transfer.json')
        if transfer_file.exists():
            manifest=json.loads(transfer_file.read_text())
            if manifest.get('file_count')!=24 or not manifest.get('server_cleanup_verified'):
                raise RuntimeError('incomplete snapshot transfer record')
        else:
            manifest=move(probe,name)
            transfer_file.write_text(json.dumps(manifest,indent=2)+'\n')
    endpoint=endpoints_from_env()['node1']
    remote='/workspace/SHIFT/results/'+names['shift']+'/trajectory_comparison.json'
    cmd=['/workspace/SHIFT/runtime/torch214/bin/python','-m','shift_full.compare_reference',
         '/workspace/SHIFT/results/'+names['flat'],'/workspace/SHIFT/results/'+names['shift'],
         '--steps','2,3,4','--output',remote]
    code=f'import subprocess; r=subprocess.run({cmd!r},cwd="/workspace/SHIFT"); print(r.returncode)'
    comparison_file=OUT/'trajectory_comparison.json'
    if not comparison_file.exists():
        rc=run_job(probe,endpoint,['python','-c',code],timeout=900).strip()
        if rc not in ('0','1'):
            raise RuntimeError('reference comparison failed: '+rc)
        comparison_file.write_text(probe.execute_endpoint(endpoint,['cat',remote],timeout=60))
    comparison=json.loads((OUT/'trajectory_comparison.json').read_text())
    if not all(comparison['checks'][key]['pass'] for key in ['files_present','loads_succeeded']):
        raise RuntimeError('trajectory snapshots missing or unreadable')
    run('trajectory_summary',['-m','shift_full.summarize_trajectory',str(ROOT/'results'/names['flat']),
                             str(ROOT/'results'/names['shift']),str(OUT/'trajectory_comparison.json'),
                             '--output',str(OUT/'trajectory_summary.json')],allowed=(0,1))
    summary=json.loads((OUT/'trajectory_summary.json').read_text())
    for key in ['input_read','same_initial_state','same_batch_plans','participants_and_global_batch']:
        if not summary['checks'][key]['pass']:
            raise RuntimeError('invalid trajectory comparison: '+key)


def backups():
    results=[]
    for mode in ('cold','cpu','gpu'):
        name=f'v1_backup_audit_{mode}_01'
        run(name,['-m','shift_full.rotation','--session',name,'--config','configs/shift_rotation_audit512.json',
                  '--seconds','900','--faults','1','--backup-mode',mode,'--seed','20261102'])
        report=verify(ROOT/'results'/name,1)
        (OUT/(name+'_verify.json')).write_text(json.dumps(report,indent=2)+'\n')
        if not report['pass']:
            raise RuntimeError(str(report['errors']))
    # Balanced mode order; identical workload, placement, seed, and failure target.
    for trial,modes in enumerate([('cold','cpu','gpu'),('gpu','cold','cpu'),('cpu','gpu','cold')],1):
        for mode in modes:
            name=f'v1_backup_timing_{mode}_{trial:02d}'
            run(name,['-m','shift_full.rotation','--session',name,'--config','configs/shift_rotation_timing.json',
                      '--seconds','900','--faults','1','--backup-mode',mode,'--seed','20261102'])
            report=verify(ROOT/'results'/name,1,require_audit=False)
            (OUT/(name+'_verify.json')).write_text(json.dumps(report,indent=2)+'\n')
            if not report['pass']:
                raise RuntimeError(str(report['errors']))
            fault=json.loads((ROOT/'results'/name/'fault_results.json').read_text())[0]
            results.append(dict(session=name,mode=mode,trial=trial,controller=fault,verified=report['faults'][0]))
            (OUT/'backup_timings.json').write_text(json.dumps(results,indent=2)+'\n')
    run('backup_summary',['scripts/summarize_v1_backups.py'])


def bench():
    for index,mode in enumerate(['flat','shift','shift','flat'],1):
        name=f'v1_bench_{mode}_{index:02d}'
        run(name,['-m','shift_full.rotation','--session',name,'--config',f'configs/v1_bench_{mode}.json',
                  '--seconds','600','--faults','0','--seed','20261102'])
    report,_=benchmark_summary([ROOT/'results/v1_bench_flat_01',ROOT/'results/v1_bench_flat_04'],
                               [ROOT/'results/v1_bench_shift_02',ROOT/'results/v1_bench_shift_03'])
    (OUT/'benchmark_abba.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['abba_complete']:
        raise RuntimeError('incomplete ABBA evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('numerical','backups','bench','all'),default='all')
    a=p.parse_args()
    long_dir=ROOT/'results/v1_rotation_long_20261102_01'
    rows=[json.loads(line) for line in (long_dir/'controller.jsonl').read_text().splitlines()]
    if not any(e['kind']=='collected' for e in rows) or not any(e['kind']=='complete' and e['faults']==60 for e in rows):
        raise RuntimeError('long experiment must finish and collect before other GPU experiments')
    report=verify(long_dir,60,3600)
    (OUT/'long_verify.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['pass']:
        raise RuntimeError(str(report['errors']))
    for name,fn in [('numerical',numerical),('backups',backups),('bench',bench)]:
        if a.phase in (name,'all'):
            fn()


if __name__=='__main__':
    main()
