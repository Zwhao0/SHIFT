"""Exercise the FLASH–SHIFT Lab path with replay, calibration or live profiling."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.export_v42_session import export
from shift_full.flash_bridge import alert, connect, keys


def run(args):
    root = args.root.resolve()
    results = root / 'results'
    run_dir = results / f'session_{args.session}'
    requests = root / 'runtime/v3-reproduce' / f'session_{args.session}' / 'requests'
    member_key, alert_key = keys(args.session)
    client = connect(args.redis_socket)
    if client.exists(member_key, alert_key):
        raise ValueError('use a unique replay session')
    flags = ['--normal-v1-exit'] if args.case == 'v1' else []
    if args.prepare_v1:
        flags.append('--prepare-v1')
    if args.replica == 0:
        flags.append('--lab-keep-dp-spread')
    topology = ['--nodes', 'node0', 'node1'] if args.two_runners else ['--lab-colocated-spare']
    rank = 4 * args.replica + 2 * args.stage
    if args.calibrate or args.benchmark == 'profile':
        flags.append('--flash-calibrate')
    if args.live_reference:
        flags.extend(('--flash-live-reference', str(args.live_reference.resolve()),
                      '--flash-inject-rank', str(rank), '--flash-inject-epochs', str(args.repeat),
                      '--flash-inject-step', str(args.inject_step)))
    policy = 'v4' if args.case == 'v1' else args.case
    bridge = [] if args.benchmark == 'off' else [
        '--flash-redis-socket', args.redis_socket, '--flash-slow-policy', policy]
    started = time.monotonic()
    with (results / f'{args.session}.controller.log').open('x') as log:
        worker = subprocess.Popen([
            sys.executable, '-m', 'scripts.session_remote', '--session', args.session,
            '--lab-experiment', args.experiment, '--root', str(root), '--flash-root', str(root.parent/'FLASH1022'),
            '--replica-world-size', '4', '--seq-len', '32', '--low-memory', '--seconds', str(420 + (args.repeat - 1) * 180),
            '--lab-node0', 'gpu-03', '--lab-node1', 'gpu-01', *topology, *bridge, *flags,
        ], cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
            stdout=log, stderr=subprocess.STDOUT)

    def summary():
        path = run_dir / 'summary.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def wait(predicate, seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            state = summary()
            if state.get('error') or state.get('rejected_commands'):
                raise RuntimeError(state.get('error') or state['rejected_commands'])
            for profile in state.get('flash_live_profiles', []):
                if profile.get('status') == 'error':
                    raise RuntimeError(profile['error'])
            if predicate(state):
                return state
            if worker.poll() is not None:
                raise RuntimeError(f'controller exited with status {worker.returncode}')
            time.sleep(0.5)
        raise TimeoutError('replay stage exceeded its deadline')

    def queue(identifier, epoch, action, **fields):
        request = dict(schema=1, session=args.session, id=identifier, expected_epoch=epoch,
                       action=action, replica_world_size=4, **fields)
        target = requests / f'{identifier}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(request))
        temporary.replace(target)

    source = ('flash_calibration' if args.calibrate else 'flash_live' if args.live_reference
              else 'shift_process_exit' if args.case == 'v1' else 'flash_replay')
    evidence = dict(case=args.case, signal=f'benchmark_{args.benchmark}' if args.benchmark else source,
                    runners=2 if args.two_runners else 3, prepared=args.prepare_v1, accepted=False)
    try:
        if args.benchmark:
            wait(lambda s: s.get('ready') and min(s.get('progress', {}).values(), default=0) >= 1, 155)
            queue('benchmark-stop', 0, 'stop', at_step=40)
            state = wait(lambda s: s.get('done') and s.get('resources_clean'), 175)
            worker.wait(timeout=10)
            measured = defaultdict(dict)
            gaps = defaultdict(dict)
            for path in (run_dir/'raw').glob('*/*.jsonl'):
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if (event['kind'] == 'session_commit' and event['participants'] == 2
                            and event['contributed'] and event['finite']):
                        identity = event['replica'], event['rank']
                        measured[event['step']][identity] = event['seconds']
                        if event.get('commit_gap') is not None:
                            gaps[event['step']][identity] = event['commit_gap']
            steps = {step: max(measured[step].values()) for step in range(4, 41)
                     if len(measured[step]) == 8}
            if worker.returncode or state['final_step'] != 40 or len(steps) != 37:
                raise RuntimeError('benchmark lacks exact common-commit coverage')
            profile = state.get('flash_live_profile', {})
            if args.benchmark == 'profile' and profile.get('status') != 'complete':
                raise RuntimeError('benchmark PROFILE was incomplete')
            commit_gaps = {step: max(gaps[step].values()) for step in steps if len(gaps[step]) == 8}
            if len(commit_gaps) != len(steps):
                raise RuntimeError('benchmark lacks exact commit-gap coverage')
            evidence.update(steps=steps, train_step_seconds=sum(steps.values()),
                            commit_gaps=commit_gaps, train_commit_gap_seconds=sum(commit_gaps.values()),
                            controller_wall_seconds=time.monotonic()-started)
            if args.benchmark != 'off':
                audit = json.loads((run_dir/'flash_epoch_0'/'bocd_audit.jsonl').read_text())
                evidence['monitor_audit'] = audit
                if audit['error'] or audit['finish_reason'] == 'request' or audit['count'] < 33:
                    raise RuntimeError('healthy benchmark did not complete MONITOR without a trigger')
            evidence['accepted'] = True
            return
        state = wait(lambda s: s.get('ready') and min(s.get('progress', {}).values(), default=0) >= args.warm_steps
                     and client.exists(member_key), 155)
        initial = json.loads(client.get(member_key))
        if args.calibrate:
            wait(lambda s: s.get('flash_live_profile', {}).get('status') == 'complete', 120)
            queue('calibration-stop', 0, 'stop')
            wait(lambda s: s.get('done') and s.get('resources_clean'), 65)
            worker.wait(timeout=10)
            evidence.update(summary=summary(), accepted=worker.returncode == 0,
                            reference=str(run_dir/'flash_reference.json'))
            return
        if args.case == 'v1':
            queue('v1-exit', 0, 'recover', policy='v1', source_replica=args.replica, target='auto')
        elif args.live_reference:
            evidence['predictions'] = []
            for epoch in range(args.repeat):
                def localized(state):
                    profiles = state.get('flash_live_profiles', [])
                    if len(profiles) <= epoch:
                        return False
                    profile = profiles[epoch]
                    audit = run_dir/f'flash_epoch_{epoch}'/'bocd_audit.jsonl'
                    if (profile['status'] == 'waiting_for_monitor' and audit.exists()
                            and json.loads(audit.read_text())['finish_reason'] == 'budget'):
                        raise TimeoutError(f'epoch {epoch}: BOCD finished without requesting PROFILE')
                    return 'prediction' in profile

                state = wait(localized, 180)
                prediction = state['flash_live_profiles'][epoch]['prediction']
                if prediction['alarm_rank'] != rank:
                    raise RuntimeError(f'epoch {epoch}: FLASH predicted {prediction["alarm_rank"]}, injected {rank}')
                evidence['predictions'].append(prediction)
            evidence['prediction'] = evidence['predictions'][0]
        else:
            # Synthetic prediction isolates the controller and transport from
            # the workload-specific detector and PROFILE calibration.
            last = min(state['progress'].values())
            prediction = dict(alarm_rank=rank, iteration_ids=list(range(last-7, last+1)),
                              manual_integration_only=False, abstention_reasons=[])
            event = alert(initial, prediction, {str(rank): rank for rank in range(8)})
            evidence['alert_published_monotonic'] = time.monotonic()
            evidence['alert_id'] = client.xadd(alert_key, {'event': json.dumps(event)})
            if args.duplicate_alert:
                evidence['duplicate_alert_id'] = client.xadd(alert_key, {'event': json.dumps(event)})
        state = wait(lambda s: s.get('epoch') == args.repeat and s['flash_monitor'][-1]['samples'] >= 36
                     and json.loads(client.get(member_key))['epoch'] == args.repeat, 190)
        memberships = state['flash_memberships']
        first = 4 * args.replica + (2 * args.stage if args.case == 'v4' else 0)
        expected = list(map(str, range(first, first + (2 if args.case == 'v4' else 4))))
        for previous, current in zip(memberships, memberships[1:]):
            changed = [rank for rank in previous['workers'] if previous['workers'][rank] != current['workers'][rank]]
            if changed != expected or any(previous['workers'][rank]['physical_node']
                                          == current['workers'][rank]['physical_node'] for rank in changed):
                raise RuntimeError('unexpected migrated rank scope or physical placement')
        evidence['migrated_logical_ranks'] = expected
        evidence['memberships'] = memberships
        queue('replay-stop', state['epoch'], 'stop')
        wait(lambda s: s.get('done') and s.get('resources_clean'), 65)
        worker.wait(timeout=10)
        if worker.returncode:
            raise RuntimeError(f'controller exited with status {worker.returncode}')
        evidence['session'] = export(run_dir)
        evidence['summary'] = summary()
        if args.live_reference:
            injection = {}
            handoff_injection = defaultdict(list)
            for path in (run_dir/'raw').glob('*/*.jsonl'):
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event['kind'] == 'flash_compute_delay':
                        injection[event['step']] = event['delay_launches']
                        if event['command_id'] and event['delay_launches']:
                            handoff_injection[event['command_id']].append(event['step'])
            if any(not injection.get(step) for prediction in evidence['predictions']
                   for step in prediction['iteration_ids']):
                raise RuntimeError('live PROFILE lacks confirmed delay in every observed step')
            evidence['delay_launches_by_step'] = injection
            evidence['delay_during_handoff'] = {cmd: sorted(steps) for cmd, steps in handoff_injection.items()}
        if args.duplicate_alert:
            expected_id = evidence['duplicate_alert_id']
            if not any(row['stream_id'] == expected_id and 'error' in row
                       for row in evidence['summary']['flash_alerts']):
                raise RuntimeError('duplicate alert was not fenced after the handoff')
        monitor_dir = run_dir/f'flash_epoch_{args.repeat}'
        audit = json.loads((monitor_dir/'bocd_audit.jsonl').read_text())
        evidence['replacement_monitor'] = audit
        if audit['error'] or audit['count'] < 33 or audit['finish_reason'] == 'request':
            raise RuntimeError('FLASH did not consume a complete warmup and detection window')
        evidence['accepted'] = True
    except BaseException as error:
        evidence.update(error=str(error), summary=summary())
        raise
    finally:
        if worker.poll() is None:
            worker.send_signal(signal.SIGINT)
            try:
                worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=10)
        evidence['summary'] = summary()
        args.evidence.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('v1', 'v2', 'v4'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--redis-socket', required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--replica', type=int, choices=(0, 1), default=1)
    parser.add_argument('--stage', type=int, choices=(0, 1), default=1)
    parser.add_argument('--warm-steps', type=int, default=36)
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--inject-step', type=int, default=36, help='first fault step relative to each epoch')
    parser.add_argument('--two-runners', action='store_true', help='use spare slots in the GPU03 runner')
    parser.add_argument('--prepare-v1', action='store_true', help='pre-stage a v1 target before controlled exit')
    parser.add_argument('--duplicate-alert', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate', action='store_true')
    mode.add_argument('--live-reference', type=Path)
    mode.add_argument('--benchmark', choices=('off', 'monitor', 'profile'))
    args = parser.parse_args()
    if args.case == 'v2' and args.replica != 1:
        parser.error('v2 currently supports DP1')
    if args.two_runners and args.replica != 1:
        parser.error('two-runner Lab scenarios reserve target slots for DP1')
    if args.prepare_v1 and args.case != 'v1':
        parser.error('prepared exit requires v1')
    if not 36 <= args.warm_steps <= 64 or (args.case == 'v1' and args.duplicate_alert):
        parser.error('invalid replay warmup or duplicate-alert setting')
    if (args.calibrate or args.live_reference or args.benchmark) and (args.case == 'v1' or args.duplicate_alert):
        parser.error('PROFILE is for slow-fault calibration and live migration')
    if args.repeat > 1 and (not args.live_reference or args.case != 'v4'):
        parser.error('repeated fault injection currently requires live v4')
    if not 36 <= args.inject_step <= 96 or (args.inject_step != 36 and not args.live_reference):
        parser.error('invalid live injection step')
    run(args)
