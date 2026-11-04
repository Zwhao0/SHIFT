"""Server-owned V1 regression: common-only CPU warm pool, recorded injections.

Run on node0 with the temporary scoped SSH identity prepared by the operator.
No credentials are logged; the credential file is unlinked immediately.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import time

from scripts.recheck_v1_host import HostRotation
from shift_full.control import server_password
from shift_full.verify_rotation import verify
from shift_full.reporting import bench_session

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/v1_acceptance_20261103'
PRIVATE = ROOT/'runtime/v1-controller'
PASSWORD = None
POOL = '/workspace/SHIFT/runtime/generic-pool/preload.sock'


class ServerRotation(HostRotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sudo_password = PASSWORD

    def ensure_connection(self, endpoint, force=False):
        endpoint = self.coerce_endpoint(endpoint)
        if force:
            sp.run(['ssh', '-S', str(ROOT/'tmp'/endpoint.socket), '-O', 'exit',
                    'test@'+endpoint.ip], capture_output=True, timeout=5)
        if endpoint.physical_node == 'node0':
            # The command path is local, but inherited evidence collection uses
            # scp for every host, including node0. Recreate this per controller.
            sp.run(['ssh', '-i', str(PRIVATE/'id_ed25519'),
                    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'StrictHostKeyChecking=accept-new',
                    '-o', 'UserKnownHostsFile='+str(PRIVATE/'known_hosts'),
                    '-o', 'ControlMaster=auto', '-o', 'ControlPersist=3600',
                    '-o', 'ServerAliveInterval=5',
                    '-S', str(ROOT/'tmp'/endpoint.socket), '-fnNT',
                    'test@'+endpoint.ip], check=True, capture_output=True, timeout=20)
        else:
            self.ssh(endpoint, ['true'], timeout=20)

    def ssh(self, endpoint, cmd, timeout=45, sudo_input=False):
        endpoint = self.coerce_endpoint(endpoint)
        stdin = PASSWORD+'\n' if sudo_input else None
        if endpoint.physical_node == 'node0':
            args = cmd
        else:
            args = ['ssh', '-i', str(PRIVATE/'id_ed25519'), '-o', 'BatchMode=yes',
                    '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5',
                    '-o', 'ServerAliveCountMax=2', '-o', 'StrictHostKeyChecking=accept-new',
                    '-o', 'UserKnownHostsFile='+str(PRIVATE/'known_hosts'),
                    '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
                    '-o', 'ControlPath='+str(ROOT/'tmp'/endpoint.socket),
                    'test@'+endpoint.ip, shlex.join(cmd)]
        for attempt in range(6):
            try:
                result = sp.run(args, input=stdin, text=True, capture_output=True, timeout=timeout)
                if result.returncode != 255:
                    if result.returncode:
                        raise sp.CalledProcessError(result.returncode, ['remote-command'], output=result.stdout, stderr=result.stderr)
                    return result.stdout
            except sp.TimeoutExpired:
                if attempt == 5: raise
            if hasattr(self, 'log'):
                self.event('ssh_transport_retry', physical_node=endpoint.physical_node, attempt=attempt+1)
            time.sleep(min(1+attempt, 5))
        raise RuntimeError('SSH transport unavailable: '+endpoint.physical_node)


def config(mode, offset, steps=1000000):
    c = json.loads((ROOT/f'configs/v1_bench_{mode}.json').read_text())
    a = c['train_args']
    a += ['--audit-state', '--hash-every', '0']
    c.update(target_steps=steps, fast_launch=True, persistent_cpu_launcher=True,
             single_gpu_visibility=False, external_cpu_launcher_socket=POOL,
             cpu_launcher_generation_offset=offset, audit_initial=True)
    c['env']['SHIFT_GENERIC_PRELOAD'] = '1'
    c['env']['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    c['env']['SHIFT_COMMON_GPU_WARMUP'] = '1'
    return c


def injections():
    # Retain historical injection counts after raw evidence is archived locally.
    ledger_path = OUT/'formal_injection_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    archived = set(ledger.get('archived_sessions', []))
    active = sum(json.loads(line).get('kind') == 'injected'
                 for p in (ROOT/'results').glob('v1_accept_20261103_*/controller.jsonl')
                 if p.parent.name not in archived
                 for line in p.read_text().splitlines() if line.strip())
    return ledger.get('archived_formal_injections', 0) + active


def ensure_unused_session(name):
    ledger_path = OUT/'formal_injection_ledger.json'
    if ledger_path.exists() and name in json.loads(ledger_path.read_text()).get('archived_sessions', []):
        raise RuntimeError('archived session name cannot be reused; choose a new suffix')


def run_case(name, c, seed, fault):
    ensure_unused_session(name)
    s = ServerRotation(name, c, 600, seed, backup_mode='generic' if c['mode']=='shift' else 'cold')
    try:
        s.start()
        if fault:
            s.await_('warmup', lambda d: bool(s.common_commits(d, 40)[0]))
            began = time.monotonic()
            s.event('fault_window_start', duration=0, requested_faults=1)
            s.fault(0)
            joint = s.fault_results[0]['first_joint_step']
            s.await_('post_recovery_progress', lambda d: bool(s.common_commits(d, joint+19)[0]))
            s.event('complete', faults=1, fault_window_seconds=time.monotonic()-began)
        else:
            s.await_('all_stopped', lambda d: all(s.rows(d,r,k,'stopped') for r in range(2) for k in range(8)), timeout=450)
            s.event('complete', faults=0)
    except BaseException as error:
        s.event('failed', error=repr(error))
        raise
    finally:
        s.finish()
    return s


def joint_seconds(root):
    by_step = {}
    for p in root.glob('node[01]/events_g*_r*.jsonl'):
        for line in p.read_text().splitlines():
            e = json.loads(line)
            if e['kind'] == 'commit' and e['participants'] == 2 and e.get('contributed'):
                by_step.setdefault(e['step'], {})[(e['rid'],e['rank'])] = e
    valid = sorted(k for k,rows in by_step.items() if len(rows) == 16)
    # Warm-up and any state-hashing commits are excluded symmetrically.
    return [max(e['seconds'] for e in by_step[k].values()) for k in valid[10:]
            if not any(e.get('hashes') for e in by_step[k].values())]


def stop_pools():
    probe = ServerRotation.__new__(ServerRotation)
    probe.root = ROOT
    probe._sudo_password = PASSWORD
    from shift_full.control import endpoints_from_env
    code = '''from pathlib import Path
import os,signal,subprocess,json,time
p=Path('/workspace/SHIFT/runtime/generic-pool/owner.json')
if p.exists():
 pid=json.loads(p.read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and b'shift_full.launch_server' in cmd.read_bytes():
  from shift_min.supervisor import descendants
  for child in reversed(descendants(pid)):
   try:os.kill(child,signal.SIGKILL)
   except ProcessLookupError:pass
  os.kill(pid,signal.SIGTERM)
print('pool released')
'''
    errors = []
    for n,e in endpoints_from_env().items():
        try:probe.ssh(e,['sudo','-S','-p','','env','PYTHONPATH=/workspace/SHIFT','python3','-c',code],sudo_input=True)
        except Exception as error:errors.append(n+': '+str(error))
    return errors


def main():
    global PASSWORD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',nargs='+',type=int,default=[20261106,20261107,20261110])
    parser.add_argument('--suffix',default='r1')
    args=parser.parse_args()
    PASSWORD=server_password(PRIVATE/'credentials')
    (PRIVATE/'credentials').unlink()
    OUT.mkdir(parents=True,exist_ok=True)
    status=dict(done=False,success=False,rows=[],error=None)
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        for i,seed in enumerate(args.seeds,1):
            base=f'v1_accept_20261103_{args.suffix}_{i:02d}'
            flat=run_case(base+'_flat',config('flat',30000+100*i,40),seed,False)
            warm=run_case(base+'_warm',config('shift',40000+100*i),seed,True)
            report=verify(warm.out,1,require_audit=True)
            (OUT/(warm.name+'_verify.json')).write_text(json.dumps(report,indent=2)+'\n')
            f=warm.fault_results[0]
            manifest=json.loads((warm.out/'physical_manifest.json').read_text())
            binding=next(x for x in manifest if x['rid']==f['target'] and x['generation']==f['generation'])
            local_dir=warm.out/('physical_'+binding['physical_node'])/Path(binding['directory']).name
            launches=[json.loads(l) for l in (local_dir/'supervisor.jsonl').read_text().splitlines() if json.loads(l)['kind']=='launched']
            started=launches[0]['process_started_time']
            a,b=joint_seconds(flat.out),joint_seconds(warm.out)
            import statistics
            extra=statistics.fmean(b)/statistics.fmean(a)-1
            row=dict(seed=seed,flat=flat.name,warm=warm.name,backup=f['new_node'],killed=f['old_node'],
                     recovery_seconds=f['joint_time']-f['request_time'],
                     process_start_to_joint_seconds=f['joint_time']-started,preparation_seconds=f['preparation_seconds'],conservative_request_to_joint_seconds=f['joint_time']-f['request_time'],
                     fault_to_joint_seconds=f['fault_to_joint_seconds'],flat_mean_seconds=statistics.fmean(a),
                     shift_mean_seconds=statistics.fmean(b),flat_samples=len(a),shift_samples=len(b),
                     extra_step_percent=100*extra,functional_pass=report['pass'],
                     time_pass=f['joint_time']-f['request_time']<=10,step_pass=extra<=.05,
                     flat_seconds=a,shift_seconds=b)
            status['rows'].append(row)
            (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if not k.endswith('_seconds') or not isinstance(v,list)}),flush=True)
            if not report['pass']:
                raise RuntimeError('full-state fail-stop audit failed')
        status['success']=(len(status['rows'])==3 and {r['backup'] for r in status['rows']}=={'node0','node1','node3'}
                           and all(r['functional_pass'] and r['time_pass'] and r['step_pass'] for r in status['rows']))
    except BaseException as error:
        status['error']=repr(error)
    finally:
        status['pool_cleanup_errors']=stop_pools()
        status['formal_injections']=injections()
        status['done']=True
        (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return 0 if status['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
