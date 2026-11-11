"""Drive one bounded acceptance session from the campus Kubernetes client."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.export_v42_session import export
from scripts.verify_session import verify
from shift_full.session_layout import SessionLayout

CASE_COUNTS = {'whole': 2, 'restore': 1, 'v4': 4}


def case_count(args):
    return 1 if args.v4_scope else CASE_COUNTS[args.mode]


def compare_pair(auto, fixed):
    """Compare two independently audited runs in the same Pods and on the same GPUs."""
    for key in ('configuration', 'physical_nodes', 'source_sha256', 'runner_gpu_uuids'):
        if auto[key] != fixed[key]:
            raise ValueError(f'paired runs differ in {key}')
    for result in (auto, fixed):
        if not result['accepted'] or not result['independent_audit']['pass_all'] or not result['resources_clean']:
            raise ValueError('paired runs must both pass the audit and release their actors')
    def inventory(result):
        return {row['node']: row['instance']
                for row in result['initial_spare_pool']['ranked']}

    if inventory(auto) != inventory(fixed):
        raise ValueError('paired runs require the same runner Pods and GPU inventory')
    if len(auto['cases']) != len(fixed['cases']):
        raise ValueError('paired runs have different operations')

    def measurement(auto_value, fixed_value, limit):
        if any(value is None or not math.isfinite(value) or value <= 0 for value in (auto_value, fixed_value)):
            raise ValueError('paired run is missing a positive timing measurement')
        ratio = auto_value / fixed_value
        return dict(auto_seconds=auto_value, fixed_seconds=fixed_value,
                    ratio=ratio, ratio_limit=limit, within_limit=ratio <= limit)

    def step_times(auto_window, fixed_window):
        return dict(mean=measurement(auto_window['mean_seconds'], fixed_window['mean_seconds'], 1.10),
                    p95=measurement(auto_window['step_seconds_distribution']['p95'],
                                    fixed_window['step_seconds_distribution']['p95'], 1.15))

    initial = step_times(auto['initial_native_5_to_16'], fixed['initial_native_5_to_16'])
    passes = all(metric['within_limit'] for metric in initial.values())
    operations = []
    for a_case, f_case in zip(auto['cases'], fixed['cases']):
        command = a_case['command']
        if (not a_case.get('placement') or f_case.get('placement')
                or a_case['target_gpu_uuids'] != f_case['target_gpu_uuids']
                or any(command[key] != f_case['command'][key]
                       for key in ('id', 'policy', 'source_replica', 'ranks', 'target'))):
            raise ValueError('paired runs must replay identical commands and target GPU UUIDs')
        metrics = step_times(a_case['steady_5_to_16'], f_case['steady_5_to_16'])
        metrics['request_to_first_joint'] = measurement(a_case['request_to_first_joint_seconds'],
                                                        f_case['request_to_first_joint_seconds'], 1.10)
        if command['policy'] == 'v1':
            metrics['exit_to_first_joint'] = measurement(a_case['fault_to_first_joint_seconds'],
                                                         f_case['fault_to_first_joint_seconds'], 1.10)
        passes &= all(metric['within_limit'] for metric in metrics.values())
        operations.append(dict(command=command['id'], metrics=metrics))
    return dict(auto_session=auto['session'], fixed_session=fixed['session'],
                within_limits=passes, initial_step_times=initial, operations=operations,
                different_candidate_preparation=any(
                    bool(a.get('candidate_prepared_before_exit')) != bool(f.get('candidate_prepared_before_exit'))
                    for a, f in zip(auto['cases'], fixed['cases'])))


def run(args):
    root = args.root.resolve()
    layout = SessionLayout(4)
    reference = json.loads(args.fixed_reference.read_text()) if args.fixed_reference else None
    if reference is not None and (not reference.get('accepted')
            or len(reference.get('cases', ())) != case_count(args)):
        raise ValueError('fixed reference requires a successful auto run with the same mode')
    run_dir = root / 'results' / f'session_{args.session}'
    requests = root / 'runtime/v3-reproduce' / f'session_{args.session}' / 'requests'
    logs = root / 'results'
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f'{root}:{root}/runtime/torch214/lib/python3.12/site-packages')
    with (logs / f'{args.session}.controller.log').open('xb') as log:
        command = [
            sys.executable, '-m', 'scripts.session_remote', '--session', args.session,
            '--lab-experiment', args.experiment, '--root', str(root), '--flash-root', str(root.parent/'FLASH1022'),
            '--replica-world-size', '4', '--seq-len', '32', '--low-memory', '--seconds', '1000',
        ]
        if args.normal_v1_exit:command.append('--normal-v1-exit')
        if args.prepare_v1:command.append('--prepare-v1')
        if args.fixed_reference:
            command.extend(['--lab-fixed-reference', str(args.fixed_reference.resolve())])
        if args.two_node:
            peer = 'gpu-01' if args.anchor_node == 'gpu-03' else 'gpu-03'
            command.extend(['--nodes','node0','node1','--lab-node0',args.anchor_node,'--lab-node1',peer])
        if args.colocated_spare:
            command.extend(['--lab-colocated-spare', '--lab-node0', 'gpu-03', '--lab-node1', 'gpu-01'])
        if args.keep_dp_spread:
            command.append('--lab-keep-dp-spread')
        controller = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)

    def read():
        summary = run_dir / 'summary.json'
        return json.loads(summary.read_text()) if summary.exists() else {}

    def wait(predicate, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            state = read()
            if state.get('error') or state.get('rejected_commands'):
                raise RuntimeError(state.get('error') or state['rejected_commands'])
            if predicate(state):
                return state
            if controller.poll() is not None:
                raise RuntimeError(f'controller exited with status {controller.returncode}')
            time.sleep(1)
        raise TimeoutError('Lab acceptance stage exceeded its deadline')

    def queue(identifier, epoch, action, **fields):
        request = dict(schema=1, session=args.session, id=identifier, expected_epoch=epoch,
                       action=action, replica_world_size=layout.world, **fields)
        path = requests / (identifier + '.json')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(request))
        temp.replace(path)

    def progressed(state, goal):
        return state.get('ready') and min(state.get('progress', {}).values(), default=0) >= goal

    order = ((1,0),(0,0),(1,1),(0,1)) if args.two_node else ((0,0),(1,0),(0,1),(1,1))
    if args.v4_scope:
        order = (tuple(int(index) for index in args.v4_scope.split('/')),)
    stages = [('v4', replica, layout.stage_ranks(stage)) for replica, stage in order]
    if args.mode == 'whole':
        stages = [(policy, 1, tuple(range(layout.world))) for policy in ('v2','v1')]
    if args.mode == 'restore':
        stages = [('v1', 1, tuple(range(layout.world)))]
    targets = ([case['command']['target'] for case in reference['cases']]
               if reference is not None else args.targets or ['auto'] * len(stages))

    try:
        wait(lambda state: progressed(state, 16), 300)
        for epoch, ((policy, replica, ranks), target) in enumerate(zip(stages, targets)):
            identifier = f'lab-{epoch + 1}-{policy}'
            queue(identifier, epoch, 'recover' if policy == 'v1' else 'migrate',
                  policy=policy, source_replica=replica, ranks=list(ranks), target=target)
            print(json.dumps(dict(command=identifier, stage='submitted', replica=replica, ranks=ranks)), flush=True)
            state = wait(lambda state: state.get('epoch') == epoch + 1 and not state.get('active_command'), 260)
            command = state['commands'][-1]
            print(json.dumps(dict(command=identifier, target=command['command']['target'],
                                  timing=command['first_common_observation'])), flush=True)
            wait(lambda state: progressed(state, command['completed_step'] + 16), 160)
        queue('lab-stop', len(stages), 'stop')
        wait(lambda state: state.get('done') and state.get('resources_clean'), 120)
        controller.wait(timeout=15)
        if controller.returncode:
            raise RuntimeError(f'controller exit status {controller.returncode}')
        evidence = export(run_dir)
        if reference is not None:
            try:
                evidence['fixed_pair'] = compare_pair(reference, evidence)
            except ValueError as error:
                evidence['fixed_pair'] = dict(within_limits=False, error=str(error))
                raise
    except (RuntimeError, TimeoutError, AssertionError, subprocess.TimeoutExpired) as error:
        if controller.poll() is None:
            controller.send_signal(2)
            try:
                controller.wait(timeout=80)
            except subprocess.TimeoutExpired:
                controller.terminate()
        evidence = dict(accepted=False, error=str(error), summary=read(), independent_audit=verify(run_dir))
        raise
    finally:
        if 'evidence' in locals():
            args.evidence.write_text(json.dumps(evidence, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=CASE_COUNTS, required=True)
    parser.add_argument('--v4-scope', choices=('0/0', '0/1', '1/0', '1/1'), help='profile one DP/PP migration')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--normal-v1-exit', action='store_true')
    parser.add_argument('--prepare-v1', action='store_true')
    parser.add_argument('--two-node', action='store_true')
    parser.add_argument('--colocated-spare', action='store_true')
    parser.add_argument('--keep-dp-spread', action='store_true')
    parser.add_argument('--anchor-node', choices=('gpu-01','gpu-03'), default='gpu-01')
    parser.add_argument('--targets', nargs='+', choices=('node0','node1','node3'),
                        help='replay the same placements with fixed targets for a timing comparison')
    parser.add_argument('--fixed-reference', type=Path,
                        help='replay the exact GPUs of an accepted auto run in the same runner Jobs')
    args = parser.parse_args()
    if args.evidence.exists():
        parser.error('evidence path already exists')
    if args.normal_v1_exit and args.mode == 'v4':
        parser.error('--normal-v1-exit requires --mode whole or restore')
    if args.prepare_v1 and not args.normal_v1_exit:
        parser.error('--prepare-v1 requires --normal-v1-exit')
    if args.fixed_reference and args.targets:
        parser.error('--fixed-reference selects the targets')
    if args.v4_scope and args.mode != 'v4':
        parser.error('--v4-scope requires --mode v4')
    if args.targets and len(args.targets) != case_count(args):
        parser.error('provide one fixed target per operation')
    if args.two_node and args.targets and 'node3' in args.targets:
        parser.error('two-node mode uses the node0 and node1 aliases')
    if args.two_node and args.colocated_spare:
        parser.error('--colocated-spare requires three runner Pods')
    if args.keep_dp_spread and (not args.colocated_spare or args.mode == 'v4'):
        parser.error('--keep-dp-spread requires the colocated whole-replica layout')
    run(args)


if __name__ == '__main__':
    main()
