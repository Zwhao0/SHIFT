"""Score a complete PROFILE using the frozen GEMM branch for Lab diagnosis."""

import argparse
import json
from pathlib import Path

from flash_v4.frozen_localizer import decide, decode_models, verify
from flash_v4.health_centered_fusion import write_new
from flash_v4.kernel_match_candidate import normalize
from flash_v4.offline_localizer import _read_observations, _score_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--topology', choices=('legacy', 'node_local_pp'), required=True)
    args = parser.parse_args()
    model_dir = Path('results/v4/frozen/r11_kernel_match_v3')
    receipt = json.loads((model_dir/'freeze_receipt.json').read_text())
    verify([receipt['model'], *receipt['source_files']])
    model = json.loads((model_dir/'model.json').read_text())
    reference = model['references'][args.topology]
    data = _read_observations(args.input_root)
    records, _ = _score_dataset(data, decode_models(reference['gemm_models']), 'gemm')
    times = {rank: {iteration: [value['known_weighted_actual_us'], value['known_weighted_predicted_us']]
                    for iteration, value in steps.items()} for rank, steps in records.items()}
    iterations = data[0]['iterations']
    gemm = normalize(times, {int(rank): offset for rank, offset in reference['gemm_offsets'].items()},
                     iterations)
    empty_kernel = [dict(rank=rank, score_by_iteration={}) for rank in range(16)]
    result = decide(dict(gemm={'rank_order': gemm}, kernel={'rank_order': empty_kernel}),
                    model['calibration']['by_topology'][args.topology])
    result.update(iteration_ids=list(iterations), manual_integration_only=True, fault_labels_read=False,
                  evidence='gemm_only_diagnostic')
    write_new(args.input_root/'gemm_diagnostic.json', result)
    print(json.dumps({key: result[key] for key in ('alarm_rank', 'iteration_ids', 'abstention_reasons')}))


if __name__ == '__main__':
    main()
