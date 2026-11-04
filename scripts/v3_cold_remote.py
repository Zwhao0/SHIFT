"""Bounded modern-schema pure-cold fail-stop, with optional stdlib launcher."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from scripts import v1_acceptance_remote as remote
from scripts.v1_cold_remote import config, summarize
from scripts.v3_benchmark_remote import ServerOwnedRotation
from shift_full.control import server_password
from shift_full.verify_v3_recovery import verify_v3

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix',required=True)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--cold-launcher',action='store_true')
    p.add_argument('--sync-quorum',action='store_true')
    p.add_argument('--eager-groups',action='store_true')
    a=p.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    lock=(ROOT/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/f'results/v3_b2_20261103_{a.suffix}';out.mkdir(exist_ok=False)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    status=dict(done=False,success=False,seed=a.seed,cold_launcher=a.cold_launcher,
                formal_injections_before=remote.injections(),sync_quorum=a.sync_quorum,eager_groups=a.eager_groups,started=time.time())
    path=out/'status.json'
    def save():path.write_text(json.dumps(status,indent=2)+'\n')
    save();s=None
    try:
        c=config();c['cold_launcher']=a.cold_launcher
        if a.sync_quorum:c['train_args'].append('--sync-quorum')
        if a.eager_groups:c['train_args'].append('--eager-groups')
        name='v1_accept_20261103_v3cold_'+a.suffix
        remote.ensure_unused_session(name)
        s=ServerOwnedRotation(name,c,550,a.seed,backup_mode='cold')
        try:
            s.start()
            s.await_('prefault_training',lambda d:bool(s.common_commits(d,40)[0]))
            began=time.monotonic();s.event('fault_window_start',duration=0,requested_faults=1)
            s.fault(0)
            joint=s.fault_results[0]['first_joint_step']
            s.await_('post_recovery_progress',lambda d:bool(s.common_commits(d,joint+9)[0]))
            s.event('complete',faults=1,fault_window_seconds=time.monotonic()-began)
        except BaseException as exc:
            s.event('failed',error=type(exc).__name__+': '+str(exc));raise
        finally:s.finish()
        report=verify_v3(s.out)
        (out/'verify.json').write_text(json.dumps(report,indent=2)+'\n')
        status.update(summarize(s),functional_pass=report['pass'])
        status['success']=report['pass'] and status['time_pass']
    except BaseException as exc:status['error']=type(exc).__name__+': '+str(exc)
    finally:
        status.update(done=True,finished=time.time(),formal_injections_after=remote.injections(),session=s.name if s else None)
        save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
