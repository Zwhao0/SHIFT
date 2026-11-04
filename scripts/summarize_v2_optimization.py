"""Summarize local V2 optimization timing evidence without retaining worker rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_quick_validation import summarize as summarize_quick_validation


STAGE_FIELDS = {
    'schedule': 'schedule_seconds',
    'gradient_check': 'gradient_check_seconds',
    'decision': 'decision_seconds',
    'optimizer': 'optimizer_seconds',
    'step_sync': 'step_sync_seconds',
    'audit': 'audit_seconds',
}
def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def value_stats(values):
    values = [value for value in values if number(value)]
    if not values:
        return dict(n=0, mean_seconds=None, max_seconds=None)
    return dict(n=len(values), mean_seconds=statistics.fmean(values), max_seconds=max(values))


def commit_rows(timing):
    workers = timing.get('workers', [])
    if isinstance(workers, dict):
        workers = [dict(worker_key=key, **worker) for key, worker in workers.items()]
    rows = []
    for worker in workers:
        for row in worker.get('steps', []):
            if row.get('kind') == 'commit' and isinstance(row.get('step'), int):
                rows.append(row)
    return rows


def step_windows(timing, trigger, cut, goal):
    ranges = {
        'pre': (3, trigger),
        'during': (trigger + 1, cut),
        'first': (cut + 1, cut + 1),
        'post': (cut + 2, goal),
    }
    rows = commit_rows(timing)
    result = {}
    for name, (start, end) in ranges.items():
        selected = [row for row in rows if start <= row['step'] <= end]
        maxima = {}
        for row in selected:
            if number(row.get('seconds')):
                maxima.setdefault(row['step'], []).append(row['seconds'])
        per_step_max = {str(step): max(values) for step, values in sorted(maxima.items())}
        stage_stats = {
            stage: value_stats([row.get(field) for row in selected])
            for stage, field in STAGE_FIELDS.items()
        }
        result[name] = dict(
            step_range=[start, end],
            steps_with_data=len(per_step_max),
            mean_of_per_step_max_worker_seconds=value_stats(list(per_step_max.values()))['mean_seconds'],
            per_step_max_worker_seconds=per_step_max,
            stage_worker_seconds=stage_stats,
        )
    return result


def event_time(events, *kinds):
    for kind in kinds:
        row = next((item for item in events if item.get('kind') == kind), None)
        if row and number(row.get('monotonic')):
            return row['monotonic']
    return None


def controller_intervals(raw):
    timing = raw.get('timing', {})
    milestones = timing.get('controller_milestones', {})
    events = raw.get('controller_events', [])
    first_joint = timing.get('first_joint_observation', {})

    def milestone(name, *fallback_kinds):
        stamp = milestones.get(name)
        if number(stamp):
            return stamp
        return event_time(events, *fallback_kinds)

    points = [
        ('request', milestone('request', 'REQUEST', 'request')),
        ('target_prepared', milestone('target_prepared', 'target_prepared', 'c_local_ready')),
        ('frozen', milestone('frozen', 'frozen', 'freeze')),
        ('state_sent', milestone('state_sent', 'state_sent', 'planned_state_sent')),
        ('restored', milestone('restored', 'restored', 'planned_state_ready')),
        ('activated', milestone('activated', 'activated', 'planned_activated')),
        ('firstjoint', first_joint.get('scan_finished_monotonic') or
                       first_joint.get('scan_end_monotonic')),
    ]
    if raw.get('status', {}).get('variant') == 'p2':
        by_name=dict(points)
        points=[(key,by_name[key]) for key in ('request','target_prepared','state_sent','restored','frozen','activated','firstjoint')]
    intervals = []
    for (left, start), (right, end) in zip(points, points[1:]):
        seconds = end - start if number(start) and number(end) else None
        if seconds is not None and seconds < 0:
            raise ValueError(f'controller milestones are out of order: {left} -> {right}')
        intervals.append(dict(start=left, end=right, seconds=seconds))
    return intervals


def restore_signature_summary(signatures):
    ordered = sorted(signatures, key=lambda row: (
        str(row.get('role', '')), row.get('rank', -1), row.get('step', -1)))
    matches = [len(row.get('expected_hashes', {})) == 9 and row.get('expected_hashes') == row.get('actual_hashes')
               for row in ordered]
    payload = json.dumps(ordered, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return dict(
        n=len(ordered),
        full_hash_audits=sum(bool(len(row.get('expected_hashes', {})) == 9) for row in ordered),
        all_match=bool(ordered) and all(matches),
        sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest() if ordered else None,
    )


def summarize_run(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    timing = raw['timing']
    base = summarize_quick_validation(path)
    trigger = timing.get('requested_trigger', raw.get('status', {}).get('requested_trigger'))
    cut = timing.get('observed_cut', raw.get('status', {}).get('observed_cut'))
    goal = timing.get('goal', raw.get('status', {}).get('goal'))
    if not all(isinstance(value, int) for value in (trigger, cut, goal)):
        raise ValueError(f'{path}: requested trigger, observed cut, and goal must be integers')

    source_audit = raw.get('source_audit', {})
    restore = restore_signature_summary(raw.get('restore_signatures', []))
    cleanup_errors = raw.get('unit_cleanup', {}).get('errors', [])
    return dict(
        name=base['name'],
        evidence_file=path.name,
        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        functional_pass=bool(base['functional_pass']),
        resources_clean=bool(base['resources_clean']),
        cleanup_pass=not cleanup_errors,
        source_audit_pass=bool(source_audit.get('passed', False)),
        source_shas=source_audit.get('expected', {}),
        profile={key: raw.get('status', {}).get(key) for key in
                 ('audit_every_step', 'audit_state', 'variant', 'cold_launcher', 'digest_engine', 'overlap_cpu_model_load', 'overlap_cpu_imports', 'skip_restored_ready_audit')},
        independent_audit=raw.get('independent_audit'),
        final_audits_sha256=hashlib.sha256(json.dumps(raw.get('final_audits', []),sort_keys=True).encode()).hexdigest(),
        requested_trigger=trigger,
        observed_cut=cut,
        goal=goal,
        request_to_first_joint_observed_seconds=base['request_to_first_joint_observed_seconds'],
        committed_step_windows=step_windows(timing, trigger, cut, goal),
        controller_observation_intervals=controller_intervals(raw),
        restore_signatures=restore,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timing_files', nargs='+', type=Path,
                        help='one or more *_timing.json evidence files')
    parser.add_argument('--output', required=True, type=Path,
                        help='destination JSON file')
    args = parser.parse_args()

    runs = []
    for path in args.timing_files:
        if not path.is_file():
            parser.error(f'input does not exist: {path}')
        try:
            runs.append(summarize_run(path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f'{path}: {exc}')

    restore_payload = json.dumps(
        sorted(((run['name'], run['restore_signatures']['sha256']) for run in runs)),
        separators=(',', ':'), ensure_ascii=False)
    output = dict(
        schema=1,
        runs=runs,
        aggregate=dict(
            functional_pass=all(run['functional_pass'] for run in runs),
            resources_clean=all(run['resources_clean'] for run in runs),
            cleanup_pass=all(run['cleanup_pass'] for run in runs),
            source_audit_pass=all(run['source_audit_pass'] for run in runs),
            restore_signatures_sha256=hashlib.sha256(restore_payload.encode('utf-8')).hexdigest(),
        ),
        timing_semantics=(
            'Each window averages the per-step maximum seconds across committed workers. '
            'Stage means/maxima are independent worker samples; stage maxima are not additive. '
            'Controller intervals subtract same-controller monotonic observations and may be zero; '
            'they are not GPU phase durations.'
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'wrote {args.output} ({len(runs)} runs)')


if __name__ == '__main__':
    main()
