"""Bounded P1f B-to-C random quick check on the three shared hosts.

Run one manifest row per invocation. This controller never prepares runtimes,
starts GPU work implicitly, or touches processes outside its exact systemd units.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess as sp
import time

from scripts import v1_acceptance_remote as remote
from scripts.v3_planned_remote import PlannedRotation, complete, SCHEMAS
from shift_full.control import CONTAINER_ROOT, server_password

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = remote.PRIVATE
SECONDS = 1200
GOAL_SECONDS = ('seconds', 'compute_seconds', 'schedule_seconds',
                'gradient_check_seconds', 'decision_seconds',
                'commit_interval_seconds', 'boundary_sync_seconds',
                'quorum_seconds', 'load_seconds', 'audit_seconds')
SOURCE_FILES = ['shift_full/' + name + '.py' for name in
                ('planned_train', 'planned_handoff', 'overlap_handoff',
                 'overlap_probe', 'prepared_pg', 'transport')]
SHARED_HASHES = ('model', 'optimizer', 'scheduler')

# This transient module runs in each torchrun worker before planned_train.
# The trigger is deliberately left to the original P1f coordinator; this
# wrapper only enforces the per-process GPU memory cap requested for v2.
WORKER_WRAPPER = '''import os, runpy, torch
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.cuda.set_per_process_memory_fraction(0.12, rank)
runpy.run_module("shift_full.planned_train", run_name="__main__", alter_sys=True)
'''


def _set_arg(args, flag, value):
    try:
        index = args.index(flag)
    except ValueError:
        args.extend([flag, str(value)])
    else:
        args[index + 1] = str(value)


def config(trigger, post_steps):
    """Build the audited, deterministic-shape p1f random-data workload."""
    c = json.loads((ROOT / 'configs/v1_bench_shift.json').read_text())
    args = c['train_args']
    _set_arg(args, '--communication-timeout', 30)
    _set_arg(args, '--max-seconds', SECONDS - 60)
    _set_arg(args, '--microbatches', 2)
    args += ['--audit-state', '--hash-every', '1', '--sync-quorum',
             '--fixed-batch', '--data', 'random', '--planned-variant', 'p1f',
             '--planned-source', 'B', '--planned-inject', 'healthy',
             '--planned-cycles', '1']
    c.update(entry_module='quick_v2_worker', target_steps=100000,
             lighthouse_port=31400, master_port=31500, min_replicas=2,
             heartbeat_timeout_ms=1000, join_timeout_ms=1000,
             quorum_tick_ms=100, recycle_after_worker_fault=False,
             fast_launch=False, persistent_cpu_launcher=False,
             single_gpu_visibility=False, audit_initial=True,
             external_lighthouse=True)
    c['env'].update(TORCH_NCCL_NONBLOCKING_TIMEOUT='60',
                    CUDA_DEVICE_MAX_CONNECTIONS='1')
    c['requested_trigger'] = trigger
    c['post_steps'] = post_steps
    return c


class QuickV2Rotation(PlannedRotation):
    def __init__(self, name, cfg, seconds, nodes):
        super().__init__(name, cfg, seconds, nodes, baseline={})
        self.wait_observations = {}

    def start_services(self, cycles):
        # Refuse any baseline other than the eight known company VLLM PIDs.
        baseline = {'nodes': {}}
        for name, endpoint in self.endpoints.items():
            rows = self.ssh(endpoint, ['nvidia-smi', '--query-compute-apps=pid,process_name',
                                       '--format=csv,noheader']).splitlines()
            company = [row for row in rows if 'VLLM::' in row]
            pids = {row.split(',', 1)[0].strip() for row in company}
            if len(rows) != 8 or len(company) != 8 or len(pids) != 8:
                raise RuntimeError(f'{name}: expected exactly eight company VLLM GPU PIDs and no other GPU compute')
            baseline['nodes'][name] = {'gpu': rows}
        self.baseline = baseline
        super().start_services(cycles)

    def install_worker_wrapper(self, role):
        endpoint = self.roles[role]
        path = CONTAINER_ROOT + '/results/' + self.name + '/quick_v2_worker.py'
        code = ('from pathlib import Path;p=Path(' + repr(path) + ');p.parent.mkdir(parents=True,exist_ok=True);'
                'p.write_text(' + repr(WORKER_WRAPPER) + ');p.chmod(0o644)')
        self.execute_endpoint(endpoint, ['python', '-c', code])

    def finish(self):
        # Nothing was launched if a read-only preflight failed.
        if not self.baseline.get('nodes'):
            if not self.log.closed:
                self.log.close()
            for endpoint in self.endpoints.values():
                sp.run(['ssh', '-S', str(self.root/'tmp'/endpoint.socket), '-O', 'exit',
                        'test@'+endpoint.ip], capture_output=True, timeout=5)
            return
        super().finish()

    def launch_role(self, role):
        rid, generation = self.role_bindings[role]
        old = self.config
        self.install_worker_wrapper(role)
        train_args = list(old['train_args']) + ['--planned-role', role]
        flash = old.get('env', {}).get('FLASH', '/workspace/FLASH1022')
        worker_path = CONTAINER_ROOT + '/results/' + self.name
        env = dict(old.get('env', {}), PYTHONPATH=':'.join((worker_path, CONTAINER_ROOT, flash)))
        self.config = dict(old, train_args=train_args, env=env)
        self.role_started.add(role)
        self.started_endpoints[self.roles[role].physical_node] = self.roles[role]
        try:
            self.start_supervisor(rid, self.roles[role], generation)
        finally:
            self.config = old

    def wait(self, label, predicate, timeout=180):
        deadline = min(self.deadline, time.monotonic() + timeout)
        previous_scan_end = None
        while time.monotonic() < deadline:
            scan_started = time.monotonic()
            data = self.data()
            scan_finished = time.monotonic()
            for role, files in data.items():
                bad = [e for e in files.get('supervisor.jsonl', [])
                       if e.get('kind') in ('unexpected_exit', 'supervisor_error')]
                errors = self.rows_role(data, role, 'planned_prepare_error')
                if bad or errors:
                    raise RuntimeError(f'{role}: {repr((bad or errors)[-1])}')
            observation = dict(scan_started_monotonic=scan_started,
                               scan_finished_monotonic=scan_finished,
                               scan_duration_seconds=scan_finished-scan_started,
                               previous_scan_end_monotonic=previous_scan_end,
                               since_previous_scan_end_seconds=(None if previous_scan_end is None
                                                                else scan_started-previous_scan_end),
                               requested_sleep_seconds=0.2)
            if predicate(data):
                self.wait_observations[label] = observation
                self.event(label, monotonic=scan_finished)
                return data
            previous_scan_end = scan_finished
            time.sleep(0.2)
        raise TimeoutError(label)


def _timed_fields(event):
    fields = {key: event[key] for key in ('time', 'monotonic', 'step', 'epoch',
              'participants', 'contributed', 'finite_gradients', 'global_batch',
              'fixed_batch', 'physical_members', 'planned_role') if key in event}
    for key in GOAL_SECONDS:
        if key in event:
            fields[key] = event[key]
    for key in ('local_forward_calls', 'cross_calls'):
        if key in event:
            fields[key] = event[key]
    return fields


def timing_samples(data):
    """Compact per-worker stage and per-step timings without duplicating hashes."""
    workers = []
    for role, files in data.items():
        for filename, rows in files.items():
            if not filename.startswith('events_g'):
                continue
            rank_text = filename.rsplit('_r', 1)[-1].split('.', 1)[0]
            rank = int(rank_text)
            phases, steps, handoff = [], [], []
            for event in rows:
                if event.get('kind') == 'phase':
                    phases.append(dict(name=event.get('name'), **_timed_fields(event)))
                elif event.get('kind') in ('begin', 'commit'):
                    steps.append(dict(kind=event['kind'], **_timed_fields(event)))
                elif (event.get('kind') in ('restored', 'reconfigure_begin', 'reconfigure_end')
                      or event.get('kind', '').startswith('planned_')):
                    handoff.append(dict(kind=event['kind'], **_timed_fields(event)))
            workers.append(dict(role=role, rank=rank, generation=int(filename.split('_')[1][1:]),
                                phases=phases, steps=steps, handoff=handoff))
    return workers


def verify_deployed_source(s, out):
    """Compare host-visible source on each endpoint without container exec."""
    import hashlib
    expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in SOURCE_FILES}
    code = ('import hashlib,json;from pathlib import Path;root=Path(' + repr(CONTAINER_ROOT) + ');'
            'names=' + repr(SOURCE_FILES) + ';'
            'print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}))')
    nodes = {}
    for name, endpoint in s.endpoints.items():
        actual = json.loads(s.execute_endpoint(endpoint, ['python', '-c', code]))
        nodes[name] = dict(passed=actual == expected, sha256=actual)
    result = dict(passed=all(row['passed'] for row in nodes.values()),
                  expected=expected, nodes=nodes)
    (Path(out)).write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('deployed planned source differs; prepare/deploy before run')
    return result


def audit(s, data, cut, goal):
    """Require complete B->C restoration and matching joint A/C commits."""
    for role in ('A', 'B'):
        commits = s.rows_role(data, role, 'commit')
        for step in range(1, cut + 1):
            rows = s.rows_role(data, role, 'commit', step=step)
            if not complete(rows):
                raise RuntimeError(f'{role}: incomplete pre-cut step {step}')
            for row in rows:
                if row.get('participants') != 2 or not row.get('contributed') or not row.get('finite_gradients'):
                    raise RuntimeError(f'{role}: invalid pre-cut commit {step}')
                if (set(row.get('hashes', {})) != SCHEMAS or row.get('fixed_batch') is not True
                        or row.get('global_batch') != 8):
                    raise RuntimeError(f'{role}: incomplete hashes at step {step}')
            if role == 'B':
                a_rows = s.rows_role(data, 'A', 'commit', step=step)
                for rank in range(8):
                    a = next(row for row in a_rows if row['rank'] == rank)
                    b = next(row for row in rows if row['rank'] == rank)
                    if any(a['hashes'][key] != b['hashes'][key] for key in SHARED_HASHES):
                        raise RuntimeError(f'A/B state hash mismatch at step {step} rank {rank}')
                samples = {sample for row in a_rows + rows for sample in row.get('sample_ids', [])}
                if samples != set(range((step-1)*8, step*8)):
                    raise RuntimeError(f'invalid pre-cut fixed-batch sample coverage at step {step}')
        if role == 'B' and any(e['step'] > cut for e in commits):
            raise RuntimeError('retired B committed after the selected cut')

    restored = s.rows_role(data, 'C', 'restored')
    state_ready = s.rows_role(data, 'C', 'planned_state_ready', step=cut)
    sent = s.rows_role(data, 'B', 'planned_state_sent', step=cut)
    if not complete(restored) or not complete(state_ready) or not complete(sent):
        raise RuntimeError('incomplete B-to-C state transfer')
    for row in restored:
        if (row.get('step') != cut or not row.get('hashes_match')
                or row.get('optimizer_tensors', 0) <= 0
                or set(row.get('actual_hashes', {})) != SCHEMAS
                or row.get('actual_hashes') != row.get('expected_hashes')):
            raise RuntimeError('C full-state restore audit failed')

    for step in range(cut + 1, goal + 1):
        a_rows = s.rows_role(data, 'A', 'commit', step=step)
        c_rows = s.rows_role(data, 'C', 'commit', step=step)
        if not complete(a_rows) or not complete(c_rows):
            raise RuntimeError(f'missing joint A/C commit at step {step}')
        for rank in range(8):
            a = next(row for row in a_rows if row['rank'] == rank)
            c = next(row for row in c_rows if row['rank'] == rank)
            for row in (a, c):
                if (row.get('epoch') != 1 or row.get('physical_members') != ['A', 'C']
                        or row.get('participants') != 2 or not row.get('contributed')
                        or not row.get('finite_gradients') or row.get('global_batch') != 8
                        or row.get('fixed_batch') is not True
                        or set(row.get('hashes', {})) != SCHEMAS):
                    raise RuntimeError(f'invalid A/C joint commit at step {step} rank {rank}')
            if any(a['hashes'][key] != c['hashes'][key] for key in SHARED_HASHES):
                raise RuntimeError(f'A/C shared state hash mismatch at step {step} rank {rank}')
        samples = {sample for row in a_rows + c_rows for sample in row.get('sample_ids', [])}
        expected_samples = set(range((step-1)*8, step*8))
        if samples != expected_samples:
            raise RuntimeError(f'invalid fixed-batch sample coverage at step {step}')
    if not complete(s.rows_role(data, 'B', 'planned_standby_verified')):
        raise RuntimeError('B standby did not finish retirement verification')
    return dict(functional_pass=True, cut=cut, goal=goal, post_steps=goal-cut,
                state_transfer_ranks=8, audited_joint_steps=list(range(cut+1, goal+1)))


def run_one(args):
    name = 'v2_q2_' + args.suffix
    cfg = config(args.trigger, args.post_steps)
    s = QuickV2Rotation(name, cfg, SECONDS, args.nodes.split(','))
    request_mono = None
    status = dict(done=False, success=False, name=name, seed=args.seed,
                  requested_trigger=args.trigger, requested_post_steps=args.post_steps,
                  source='B', target='C', nodes=args.nodes.split(','), variant='p1f',
                  seed_note='manifest provenance; planned_train fixes initialization and derives random batches from committed sample IDs')
    status_path = s.out / 'status.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        for endpoint in s.endpoints.values():
            s.ensure_connection(endpoint)
        verify_deployed_source(s, s.out / 'deployed_source.json')
        s.start_services(1)
        s.launch_role('A')
        s.launch_role('B')
        s.wait('ab_ready', lambda d: all(complete(s.rows_role(d, role, 'ready')) for role in 'AB'), timeout=600)
        for role in 'AB':
            s.mark_role(role, 'START')
        s.wait('ab_requested_trigger',
               lambda d: all(complete(s.rows_role(d, role, 'commit', step=args.trigger)) for role in 'AB'),
               timeout=600)

        # Same-controller monotonic request time intentionally precedes C launch,
        # so it includes all target preparation and is comparable to observation.
        request_mono = time.monotonic()
        s.event('REQUEST', monotonic=request_mono, seed=args.seed, intended_trigger=args.trigger)
        s.set_key('request', '1')
        s.launch_role('C')
        d = s.wait('c_local_ready', lambda data: complete(s.rows_role(data, 'C', 'planned_local_ready')), timeout=600)
        s.mark_role('C', 'START')
        s.set_key('arm', '1')
        d = s.wait('cut_selected', lambda data: bool(s.rows_role(data, 'A', 'planned_cut_selected')), timeout=60)
        cut = s.rows_role(d, 'A', 'planned_cut_selected')[-1]['step']
        goal = cut + args.post_steps
        for role in 'ABC':
            s.mark_role(role, 'TRAIN_UNTIL', str(goal))
        s.event('selected_cut', step=cut, monotonic=time.monotonic(), intended_trigger=args.trigger,
                deviation_steps=cut-args.trigger, goal=goal)

        d = s.wait('first_joint_commit_observed',
                   lambda data: all(complete(s.rows_role(data, role, 'commit', step=cut+1))
                                    for role in ('A', 'C')), timeout=600)
        first_observation = s.wait_observations['first_joint_commit_observed']
        first_observation.update(
            request_monotonic=request_mono,
            request_to_scan_start_seconds=first_observation['scan_started_monotonic']-request_mono,
            request_to_observed_seconds=first_observation['scan_finished_monotonic']-request_mono,
            polling_error_note=('same-controller monotonic interval; scan duration and preceding missed-scan gap are recorded, '
                                'but SSH/scan latency is not a hard upper bound; worker clocks across hosts are not subtracted'))
        d = s.wait('post_goal_observed',
                   lambda data: all(complete(s.rows_role(data, role, 'commit', step=goal))
                                    for role in ('A', 'C')), timeout=600)
        d = s.wait('all_workers_stopped',
                   lambda data: all(complete(s.rows_role(data, role, 'stopped')) for role in 'ABC'), timeout=180)
        full = s.data(full=True)
        functional = audit(s, full, cut, goal)
        result = dict(functional=functional, first_joint_observation=first_observation,
                      workers=timing_samples(full), requested_trigger=args.trigger,
                      observed_cut=cut, goal=goal, seed=args.seed,
                      worker_clock_note='per-worker monotonic values are retained but must not be subtracted across physical hosts')
        (s.out / 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(success=True, functional_pass=True, observed_cut=cut, goal=goal,
                      request_to_observed_joint_seconds=first_observation['request_to_observed_seconds'])
    except BaseException as exc:
        status['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        try:
            s.finish()
        finally:
            status.update(done=True, finished=time.time())
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            print(json.dumps(status), flush=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--trigger', type=int, choices=(9, 10), required=True)
    p.add_argument('--post-steps', type=int, default=5)
    p.add_argument('--nodes', default='node0,node1,node3')
    args = p.parse_args()
    if not args.suffix or not all(c.isalnum() or c in '_-' for c in args.suffix):
        p.error('invalid suffix')
    if args.post_steps != 5:
        p.error('quick manifest runs require exactly five post-steps')
    if args.nodes.split(',') != ['node0', 'node1', 'node3']:
        p.error('--nodes must be node0,node1,node3 in role order')
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    lock = (ROOT / 'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    credential = PRIVATE / 'credentials'
    if not credential.is_file():
        p.error('missing one-shot controller credential; run the approved controller preparation first')
    remote.PASSWORD = server_password(credential)
    credential.unlink()
    run_one(args)


if __name__ == '__main__':
    main()
