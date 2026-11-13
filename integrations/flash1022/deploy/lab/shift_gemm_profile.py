"""Calibrate or localize the bounded FLASH GEMM profile on the SHIFT layout."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from flash_v4.offline_localizer import _collect_health_samples, _gemm_family_and_features, _predict


def profiles(root, ranks):
    data = {}
    raw = {}
    for rank in range(ranks):
        path = root / f'rank_{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        summaries = [row for row in rows if row['kind'] == 'summary']
        if len(summaries) != 1:
            raise ValueError('one PROFILE summary per rank is required')
        summary = summaries[0]
        if (summary['rank'] != rank or not summary['window_complete']
                or summary['completed_profile_iterations'] != 8 or summary['dropped']):
            raise ValueError(f'rank {rank} has an incomplete PROFILE')
        raw[rank] = dict(summary=summary, event_rows=rows)
        iterations = defaultdict(list)
        for row in rows:
            stride = 8 if row['kind'] == 'lt_matmul' else summary['stride']
            key, sample, _ = _gemm_family_and_features(row, stride)
            if key:
                identity = json.dumps(key, separators=(',', ':'))
                iterations[row['iteration']].append((identity, sample))
        steps = sorted(iterations)
        if len(steps) != 8 or steps != list(range(steps[0], steps[0] + 8)):
            raise ValueError(f'rank {rank} has an incomplete GEMM window')
        if data and steps != sorted(data[0]):
            raise ValueError('PROFILE ranks disagree on the step window')
        data[rank] = iterations
    return data, raw


def calibrate(data):
    reference = {}
    for rank, steps in data.items():
        shape_times = defaultdict(list)
        for samples in steps.values():
            by_shape = defaultdict(list)
            for key, sample in samples:
                by_shape[key].append(sample['elapsed_us'])
            for key, values in by_shape.items():
                shape_times[key].append(median(values))
        reference[str(rank)] = {key: median(values) for key, values in shape_times.items() if len(values) >= 4}
        if not reference[str(rank)]:
            raise ValueError(f'rank {rank} lacks repeated healthy GEMM shapes')
    return dict(schema=1, healthy_steps=sorted(data[0]), reference=reference)


def shape_index(baseline):
    index = defaultdict(list)
    for key, elapsed in baseline.items():
        family, shape = json.loads(key)
        for axis in range(3):
            fixed = tuple(shape[:axis] + shape[axis + 1:])
            index[tuple(family), axis, fixed].append((shape[axis], elapsed))
    for points in index.values():
        points.sort()
    return index


def expected_time(key, baseline, index):
    if key in baseline:
        return baseline[key]
    family, shape = json.loads(key)
    neighbors = []
    for axis, size in enumerate(shape):
        fixed = tuple(shape[:axis] + shape[axis + 1:])
        points = index.get((tuple(family), axis, fixed), ())
        position = bisect_left(points, (size,))
        for other, elapsed in points[max(0, position - 1):position + 1]:
            ratio = max(size, other) / min(size, other)
            if ratio <= 1.5:
                neighbors.append((ratio, elapsed))
    if neighbors:
        return min(neighbors)[1]
    return None


def timing_ratios(data, healthy):
    ratios = {}
    coverage = {}
    steps = sorted(data[0])
    models = {tuple(row['family']): row['coefficients'] for row in healthy.get('gemm_models', ())}
    for rank, windows in data.items():
        baseline = healthy['reference'][str(rank)]
        index = shape_index(baseline)
        expected = {}
        for samples in windows.values():
            for key, sample in samples:
                if key in expected:
                    continue
                model = models.get(sample['family'])
                predicted = _predict(model, sample['features']) if model is not None else None
                expected[key] = predicted or expected_time(key, baseline, index)
        ratios[rank] = []
        coverage[str(rank)] = []
        for step in steps:
            samples = windows[step]
            matched = [(sample['elapsed_us'], expected[key], sample['sample_weight']) for key, sample in samples
                       if expected[key] is not None]
            coverage[str(rank)].append(len(matched) / len(samples))
            if coverage[str(rank)][-1] < 0.6:
                raise ValueError(f'rank {rank} has insufficient shape overlap')
            ratios[rank].append(sum(elapsed * weight for elapsed, _, weight in matched)
                                / sum(expected * weight for _, expected, weight in matched))
    return ratios, coverage


def regression_reference(data, raw):
    reference = calibrate(data)
    models, _, _ = _collect_health_samples(raw, 'gemm')
    reference.update(schema=2, gemm_models=[dict(family=list(family), coefficients=coefficients)
                                           for family, coefficients in sorted(models.items())])
    ratios, _ = timing_ratios(data, reference)
    reference['rank_offsets'] = {str(rank): median(values) for rank, values in ratios.items()}
    return reference


def localize(data, healthy):
    ratios, coverage = timing_ratios(data, healthy)
    steps = sorted(data[0])
    offsets = healthy.get('rank_offsets', {})
    for rank, values in ratios.items():
        offset = offsets.get(str(rank), 1)
        ratios[rank] = [value / offset for value in values]
    drift = [median(ratios[rank][i] for rank in data) for i in range(len(steps))]
    rank_scores = []
    for rank, values in ratios.items():
        normalized = [value / common for value, common in zip(values, drift)]
        rank_scores.append(dict(rank=rank, ratio=median(normalized),
                                persistence=sum(value >= 1.25 for value in normalized)))
    rank_scores.sort(key=lambda row: (-row['ratio'], row['rank']))
    best, next_best = rank_scores[:2]
    reasons = []
    if best['ratio'] < 1.5 or best['persistence'] < 6:
        reasons.append('insufficient_slowdown_or_persistence')
    if best['ratio'] / next_best['ratio'] < 1.25:
        reasons.append('ambiguous_rank_margin')
    return dict(alarm_rank=None if reasons else best['rank'], rank_order=rank_scores,
                abstention_reasons=reasons, iteration_ids=steps, coverage=coverage,
                manual_integration_only=False, fault_labels_read=False,
                evidence='flash_v4_gemm_regression' if offsets else 'shift_gemm_shape_match')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=(8,16), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--regression', action='store_true', help='fit the v4 GEMM model from healthy PROFILE')
    args = parser.parse_args()
    data, raw = profiles(args.input, args.ranks)
    if args.reference:
        result = localize(data, json.loads(args.reference.read_text()))
    elif args.regression:
        result = regression_reference(data, raw)
    else:
        result = calibrate(data)
    with args.output.open('x') as output:
        json.dump(result, output)


if __name__ == '__main__':
    main()
