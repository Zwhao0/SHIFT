"""Two-plan bounded cold V1 fail-stop test with controller-clock timing."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import scripts.v1_acceptance_remote as remote
from scripts.v1_cold_remote import config as cold_config
from scripts.v3_benchmark_remote import ServerOwnedRotation
from shift_full.control import server_password, endpoints_from_env
from shift_full.verify_rotation import verify

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / 'runtime/v3-reproduce/company_gpu_baseline.json'
MEMORY_FRACTION = 0.12
POLL_SLEEP_SECONDS = 0.5
HASH_FIELDS = {'model', 'buffers', 'optimizer', 'scheduler', 'progress',
               'cpu_rng', 'cuda_rng', 'python_rng', 'tracker_rng'}


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_hashes():
    paths = [Path(__file__), ROOT/'scripts/v1_cold_remote.py',
             ROOT/'scripts/v3_benchmark_remote.py', ROOT/'shift_full/rotation.py',
             ROOT/'shift_full/supervisor.py', ROOT/'shift_full/train.py',
             ROOT/'shift_full/transport.py', ROOT/'shift_full/verify_rotation.py',
             ROOT/'shift_full/verify_v3_recovery.py', ROOT/'configs/v1_bench_shift.json']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    hashes['generated_worker_wrapper'] = hashlib.sha256(worker_module_source().encode()).hexdigest()
    return hashes


def worker_module_source():
    return '''import os\nimport runpy\nimport torch\n\nrank = int(os.environ["LOCAL_RANK"])\ntorch.cuda.set_device(rank)\ntorch.cuda.set_per_process_memory_fraction(0.12, rank)\nrunpy.run_module("shift_full.train", run_name="__main__", alter_sys=True)\n'''


class QuickV1Rotation(ServerOwnedRotation):
    """Keep Rotation's actual SIGKILL/recovery flow and instrument controller polls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_module = 'quick_v1_entry_' + self.name.replace('-', '_')
        self.worker_path = f'/workspace/SHIFT/shift_full/{self.worker_module}.py'
        self.kill_target = None
        self.kill_marker_started = None
        self.trigger_step_seen = None
        self.first_joint_observation = None
        self.first_joint_previous_scan = None
        self.await_observations = {}
        self._wrapper_nodes = set()

    def install_worker_module(self, endpoint):
        code = ('from pathlib import Path; p=Path(' + repr(self.worker_path) + '); '
                'p.write_text(' + repr(worker_module_source()) + ')')
        self.execute_endpoint(endpoint, ['python', '-c', code])
        self._wrapper_nodes.add(endpoint.physical_node)

    def start_supervisor(self, rid, endpoint, generation, standby=None):
        self.install_worker_module(endpoint)
        self.config['entry_module'] = 'shift_full.' + self.worker_module
        return super().start_supervisor(rid, endpoint, generation, standby)

    def event(self, kind, **fields):
        fields.pop('physical_ip', None)
        fields.pop('host_ip', None)
        fields.pop('ip', None)
        return super().event(kind, **fields)

    def mark(self, rid, name, text=''):
        if name.startswith('KILL_g'):
            self.kill_target = rid
            self.kill_marker_started = time.monotonic()
            self.event('fault_trigger_requested', rid=rid, marker=name,
                       monotonic=self.kill_marker_started,
                       physical_node=self.endpoint_for(rid).physical_node)
        return super().mark(rid, name, text)

    def _restored_step(self, data):
        if self.kill_target is None:
            return None
        target = self.kill_target
        rows = [self.rows(data, target, rank, 'restored') for rank in range(8)]
        if not all(rows):
            return None
        steps = {events[-1].get('step') for events in rows}
        return next(iter(steps)) if len(steps) == 1 else None

    def await_(self, label, predicate, timeout=120):
        phase_deadline = min(self.deadline, time.monotonic() + timeout)
        previous_scan = None
        joint_watch = label.startswith(('restored_', 'rejoined_'))
        while time.monotonic() < phase_deadline:
            scan_start = time.monotonic()
            data = [self.read(0), self.read(1)]
            scan_end = time.monotonic()
            for n in range(2):
                bad = [e for e in data[n].get('supervisor.jsonl', [])
                       if e['kind'] in ('unexpected_exit', 'supervisor_error')]
                if bad:
                    raise RuntimeError(f'{self.endpoint_for(n).physical_node}: {bad[-1]}')

            if joint_watch and self.first_joint_observation is None:
                restored_step = self._restored_step(data)
                if restored_step is not None:
                    steps, _ = self.common_commits(data, after=restored_step)
                    if steps:
                        self.first_joint_observation = dict(
                            step=steps[0], restored_step=restored_step,
                            scan_start_monotonic=scan_start,
                            scan_end_monotonic=scan_end,
                            scan_duration_seconds=scan_end-scan_start,
                            previous_incomplete_scan=(self.first_joint_previous_scan or previous_scan),
                            observation_window_seconds=(
                                scan_end-(self.first_joint_previous_scan or previous_scan)['scan_end_monotonic']
                                if (self.first_joint_previous_scan or previous_scan) else None),
                            configured_sleep_seconds=POLL_SLEEP_SECONDS,
                            observation_semantics='first controller scan containing all 16 rank commit rows; no cross-host clock subtraction')
                        self.event('first_joint_commit_observed',
                                   **self.first_joint_observation,
                                   trigger_monotonic=self.kill_marker_started,
                                   trigger_to_observation_seconds=(
                                       scan_end-self.kill_marker_started
                                       if self.kill_marker_started is not None else None))

            if predicate(data):
                self.await_observations[label] = dict(
                    scan_start_monotonic=scan_start,
                    scan_end_monotonic=scan_end,
                    scan_duration_seconds=scan_end-scan_start)
                if label == 'trigger_step_reached':
                    self.trigger_step_seen = scan_end
                self.event(label, generations=list(self.gen),
                           physical_nodes=[self.endpoint_for(n).physical_node for n in range(2)],
                           monotonic=scan_end, scan_start_monotonic=scan_start,
                           scan_duration_seconds=scan_end-scan_start)
                return data

            if joint_watch and self.first_joint_observation is None:
                previous_scan = dict(scan_start_monotonic=scan_start,
                                     scan_end_monotonic=scan_end,
                                     scan_duration_seconds=scan_end-scan_start)
                if self._restored_step(data) is not None:
                    self.first_joint_previous_scan = previous_scan
            time.sleep(POLL_SLEEP_SECONDS)
        raise TimeoutError(label)

    def finish(self):
        error = None
        try:
            super().finish()
        except BaseException as exc:
            error = exc
        cleanup_errors = []
        code = 'from pathlib import Path; Path(' + repr(self.worker_path) + ').unlink(missing_ok=True)'
        for endpoint in self.endpoints.values():
            try:
                self.execute_endpoint(endpoint, ['python', '-c', code], timeout=30)
            except Exception as exc:
                cleanup_errors.append(dict(node=endpoint.physical_node,
                                           error=type(exc).__name__))
        (self.out / 'worker_wrapper_cleanup.json').write_text(
            json.dumps(dict(path=self.worker_path, errors=cleanup_errors), indent=2)+'\n')
        if error is not None:
            raise error
        if cleanup_errors:
            raise RuntimeError('temporary worker wrapper cleanup failed')


def resource_snapshot(session, baseline, phase):
    result = {}
    for node, endpoint in session.endpoints.items():
        expected_rows = baseline.get('nodes', baseline)[node]['gpu']
        expected = {row.split(',', 1)[0].strip() for row in expected_rows
                    if 'VLLM::' in row}
        if len(expected) != 8:
            raise RuntimeError(f'{node}: baseline must contain eight company VLLM PIDs')
        raw = session.ssh(endpoint, ['nvidia-smi', '--query-compute-apps=pid,process_name',
                                     '--format=csv,noheader'])
        rows = [line.strip() for line in raw.splitlines() if line.strip()]
        entries = [line.split(',', 1) for line in rows]
        company = {pid.strip() for pid, name in entries if 'VLLM::' in name}
        current = {pid.strip() for pid, _ in entries}
        result[node] = dict(company_count=len(company), company_pids=sorted(company),
                            extra_pids=sorted(current-expected),
                            missing_company_pids=sorted(expected-company),
                            company_pids_unchanged=company == expected)
    (session.out / f'resources_{phase}.json').write_text(json.dumps(result, indent=2)+'\n')
    if any(not row['company_pids_unchanged'] or row['extra_pids'] for row in result.values()):
        raise RuntimeError(f'GPU PID baseline changed during {phase}')
    return result


def short_schema2_audit(result_dir, post_steps):
    errors = []
    try:
        manifest = json.loads((result_dir/'physical_manifest.json').read_text())
        faults = json.loads((result_dir/'fault_results.json').read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f'cannot load schema-2 evidence: {type(exc).__name__}']
    mapping = {(row['rid'], row['generation']): row for row in manifest}

    def event_rows(rid, generation, rank):
        binding = mapping.get((rid, generation))
        if not binding:
            return []
        path = (result_dir/('physical_'+binding['physical_node']) /
                Path(binding['directory']).name/f'events_g{generation}_r{rank}.jsonl')
        return read_rows(path)

    if len(faults) != 1:
        return ['exactly one fault result required']
    fault = faults[0]
    target, generation = fault.get('target'), fault.get('generation')
    restored_step, first_joint = fault.get('restored_step'), fault.get('first_joint_step')
    if not isinstance(target, int) or target not in (0, 1):
        return ['fault target is invalid']
    healthy = 1-target
    active_generations = {0: 0, 1: 0}
    active_generations[target] = generation
    rows_by_role = {}
    for rid in (0, 1):
        for rank in range(8):
            rows = event_rows(rid, active_generations[rid], rank)
            rows_by_role[rid, rank] = rows
            ready = [row for row in rows if row.get('kind') == 'ready']
            if len(ready) != 1 or ready[0].get('state_schema') != 2 or set(ready[0].get('hashes', {})) != HASH_FIELDS:
                errors.append(f'rid{rid} rank{rank}: complete schema-2 initial/replacement ready audit required')
            if rid == target:
                restored = [row for row in rows if row.get('kind') == 'restored' and row.get('step') == restored_step]
                if not restored or any(set(row.get('expected_hashes', {})) != HASH_FIELDS or
                                       set(row.get('actual_hashes', {})) != HASH_FIELDS or
                                       row.get('expected_hashes') != row.get('actual_hashes') or
                                       not row.get('hashes_match') for row in restored):
                    errors.append(f'rid{rid} rank{rank}: restored full schema-2 state mismatch')
            else:
                if not any(row.get('kind') == 'snapshot_readonly_verified' and
                           row.get('step') == restored_step and row.get('tensors', 0) > 0 for row in rows):
                    errors.append(f'rid{rid} rank{rank}: read-only source snapshot not verified')
    if fault.get('backup_mode') != 'cold' or fault.get('standby_resources'):
        errors.append('quick V1 must recover from a cold spare without prepared workers')
    if not isinstance(first_joint, int) or not isinstance(restored_step, int):
        errors.append('restored and first joint steps are required')
        return errors
    steps = list(range(first_joint, first_joint+post_steps))
    commit_rows = {}
    for rid in (0, 1):
        for rank in range(8):
            for step in steps:
                found = [row for row in rows_by_role[rid, rank]
                         if row.get('kind') == 'commit' and row.get('step') == step]
                if len(found) != 1 or not all(found[0].get(key) is True for key in
                                              ('contributed', 'finite_gradients')) or found[0].get('participants') != 2:
                    errors.append(f'rid{rid} rank{rank}: missing/invalid joint step {step}')
                else:
                    commit_rows[rid, rank, step] = found[0]
    for rank in range(8):
        for step in [first_joint]:
            a, b = commit_rows.get((0, rank, step)), commit_rows.get((1, rank, step))
            if not a or not b or set(a.get('hashes', {})) != HASH_FIELDS or a.get('hashes') != b.get('hashes'):
                errors.append(f'rank{rank}: step {step} lacks matching full schema-2 hashes')
    return errors


def timing_summary(session, trigger, post_steps):
    fault = session.fault_results[0]
    manifest = json.loads((session.out/'physical_manifest.json').read_text())
    workers, phases, lifecycle = {}, {}, {}
    stage_kinds = {'snapshot_begin', 'snapshot', 'snapshot_readonly_verified',
                   'transport_send', 'transport_recv', 'recovery_request',
                   'reconfigure_begin', 'reconfigure_end', 'state_loaded',
                   'restored', 'ready'}
    stage_fields = ('step', 'seconds', 'load_seconds', 'audit_seconds',
                    'gpu_bytes', 'control_bytes', 'tensors', 'cpu_tensor_bytes',
                    'optimizer_tensors', 'hashes_match', 'world_size', 'quorum_id',
                    'source_replica', 'source_rank')
    for item in manifest:
        directory = session.out/('physical_'+item['physical_node'])/Path(item['directory']).name
        role_key = f"rid{item['rid']}_g{item['generation']}"
        supervisor_rows = read_rows(directory/'supervisor.jsonl')
        launch = next((row for row in supervisor_rows if row.get('kind') == 'launched'), None)
        for rank in range(8):
            rows = read_rows(directory/f"events_g{item['generation']}_r{rank}.jsonl")
            commits = [dict(step=row.get('step'), participants=row.get('participants'),
                            contributed=row.get('contributed'), seconds=row.get('seconds'),
                            commit_interval_seconds=row.get('commit_interval_seconds'),
                            local_monotonic=row.get('monotonic'))
                       for row in rows if row.get('kind') == 'commit']
            key = f'{role_key}_rank{rank}'
            first_phase = next((row for row in rows if row.get('kind') == 'phase'), None)
            launch_wall = launch.get('process_started_time') if launch else None
            phase_start_wall = (first_phase.get('time')-first_phase.get('seconds')
                                if first_phase and isinstance(first_phase.get('time'), (int, float)) and
                                isinstance(first_phase.get('seconds'), (int, float)) else None)
            workers[key] = dict(
                commits=commits,
                supervisor_process_started_time=launch_wall,
                supervisor_launcher_pid=launch.get('launcher_pid') if launch else None,
                first_train_phase_start_wall_time=phase_start_wall,
                supervisor_to_first_train_phase_seconds=(phase_start_wall-launch_wall
                    if phase_start_wall is not None and isinstance(launch_wall, (int, float)) else None),
                phase_accounting_note=('same-host supervisor/worker wall timestamps; wrapper imports torch '
                    'before train phases, so this gap includes un-attributed wrapper/import startup'))
            phases[key] = [
                dict(name=row.get('name'), seconds=row.get('seconds'),
                     local_monotonic=row.get('monotonic'), wall_time=row.get('time'))
                for row in rows if row.get('kind') == 'phase']
            lifecycle[key] = [
                dict(kind=row.get('kind'), local_monotonic=row.get('monotonic'),
                     wall_time=row.get('time'), **{field: row.get(field) for field in stage_fields if field in row})
                for row in rows if row.get('kind') in stage_kinds]
    observation = session.first_joint_observation
    if observation is None:
        raise RuntimeError('controller did not record first joint commit observation')
    trigger_seen = session.trigger_step_seen
    return dict(trigger_step=trigger, post_steps=post_steps,
                worker_memory_fraction=MEMORY_FRACTION,
                worker_step_timings=workers, worker_phases=phases,
                worker_lifecycle_events=lifecycle,
                phase_accounting_limitation=('worker wrapper imports torch and sets the CUDA memory cap '
                    'before shift_full.train emits its python_imports phase; wrapper/import startup is included '
                    'in supervisor-to-first-phase timing but is not attributed to a train phase'),
                controller_observations=session.await_observations,
                trigger_step_seen_monotonic=trigger_seen,
                fault_marker_request_monotonic=session.kill_marker_started,
                first_joint_commit_observation=observation,
                trigger_step_to_first_joint_observation_seconds=(
                    observation['scan_end_monotonic']-trigger_seen if trigger_seen else None),
                fault_marker_to_first_joint_observation_seconds=(
                    observation['scan_end_monotonic']-session.kill_marker_started
                    if session.kill_marker_started is not None else None),
                original_rotation_wall_clock_fields=dict(
                    fault_time=fault.get('fault_time'), joint_time=fault.get('joint_time'),
                    fault_to_joint_seconds=fault.get('fault_to_joint_seconds'),
                    timing_status='legacy cross-host wall-clock delta; retained for audit only, not timing score'),
                trigger_to_recovery_stage_wall_clock_fields=dict(
                    request_time=fault.get('request_time'), ready_time=fault.get('ready_time'),
                    state_loaded_time=fault.get('state_loaded_time'),
                    limitation='worker wall clocks may differ; do not subtract across hosts'))


def run(args):
    name = f'v1_quick_20261104_{args.suffix}'
    out = ROOT/'results'/name
    if out.exists():
        raise FileExistsError(out)
    lock = (ROOT/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline = json.loads(BASELINE_PATH.read_text())
    remote.PASSWORD = server_password(remote.PRIVATE/'credentials')
    (remote.PRIVATE/'credentials').unlink()
    config = cold_config()
    config['lighthouse_ip'] = endpoints_from_env()['node0'].ip
    config['target_steps'] = 1000000
    config['entry_module'] = 'shift_full.train'
    config['env'].update(SHIFT_QUICK_V1_MEMORY_FRACTION=str(MEMORY_FRACTION),
                         SHIFT_GENERIC_PRELOAD='0', SHIFT_COMMON_GPU_WARMUP='0',
                         CUDA_DEVICE_MAX_CONNECTIONS='1')
    s = QuickV1Rotation(name, config, 600, args.seed, backup_mode='cold')
    status = dict(done=False, success=False, session=name, seed=args.seed,
                  trigger=args.trigger, post_steps=args.post_steps,
                  worker_memory_fraction=MEMORY_FRACTION, error=None)
    status_path = out/'status.json'
    status_path.write_text(json.dumps(status, indent=2)+'\n')
    try:
        for endpoint in s.endpoints.values():
            s.ensure_connection(endpoint)
        status['resources_before'] = resource_snapshot(s, baseline, 'before')
        s.start()
        data = s.await_('trigger_step_reached', lambda d: bool(
            s.common_commits(d, after=args.trigger-1)[0]))
        steps, _ = s.common_commits(data, after=args.trigger-1)
        if not steps or steps[0] != args.trigger:
            raise RuntimeError(f'expected observed common commit step {args.trigger}')
        s.event('fault_window_start', duration=0, requested_faults=1,
                trigger_step=args.trigger, monotonic=s.trigger_step_seen)
        s.fault(0)
        first_joint = s.first_joint_observation['step']
        goal_step = first_joint + args.post_steps - 1
        s.await_('post_recovery_progress', lambda d: bool(
            s.common_commits(d, after=goal_step-1)[0]))
        s.event('complete', faults=1, fault_window_seconds=(
            time.monotonic()-s.trigger_step_seen), monotonic=time.monotonic(),
            first_joint_step=first_joint, final_post_step=goal_step)
    except BaseException as exc:
        status['error'] = type(exc).__name__+': '+str(exc)
        if s.launched:
            s.event('failed', error=status['error'], monotonic=time.monotonic())
    finally:
        try:
            s.finish()
        except BaseException as exc:
            status['finish_error'] = type(exc).__name__+': '+str(exc)
        try:
            status['resources_after'] = resource_snapshot(s, baseline, 'after')
        except BaseException as exc:
            status['resource_error'] = type(exc).__name__+': '+str(exc)
    if s.fault_results:
        try:
            audit = verify(s.out, 1, require_audit=True)
            schema_errors = short_schema2_audit(s.out, args.post_steps)
            audit['schema2_short_plan_errors'] = schema_errors
            audit['schema2_short_plan_pass'] = not schema_errors
            audit['pass'] = audit['pass'] and not schema_errors
            (out/'verify.json').write_text(json.dumps(audit, indent=2)+'\n')
            timing = timing_summary(s, args.trigger, args.post_steps)
            (out/'timing_summary.json').write_text(json.dumps(timing, indent=2)+'\n')
            status.update(functional_pass=audit['pass'], verify_errors=audit['errors'],
                          schema2_short_plan_errors=schema_errors,
                          first_joint_step=s.fault_results[0].get('first_joint_step'),
                          final_post_step=timing['first_joint_commit_observation']['step']+args.post_steps-1,
                          timing_summary='timing_summary.json')
        except BaseException as exc:
            status['audit_error'] = type(exc).__name__+': '+str(exc)
    resource_ok = bool(status.get('resources_before') and status.get('resources_after'))
    status['resource_pass'] = resource_ok and not status.get('resource_error')
    status['success'] = bool(status.get('functional_pass') and status['resource_pass'] and
                             not status.get('error') and not status.get('finish_error'))
    status.update(done=True, finished=time.time())
    status_path.write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, separators=(',', ':')), flush=True)
    return 0 if status['success'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--trigger', type=int, required=True)
    parser.add_argument('--post-steps', type=int, required=True)
    args = parser.parse_args()
    if not args.suffix or not all(c.isalnum() or c in '_-' for c in args.suffix):
        parser.error('invalid suffix')
    if args.trigger < 1 or args.post_steps < 1:
        parser.error('trigger and post-steps must be positive')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
