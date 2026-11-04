"""Audit the one-update probe and reconstruct only the observed small slices."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct


def bf16(value):
    bits = struct.unpack('I', struct.pack('f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('f', struct.pack('I', bits))[0]


def possible_sums(values):
    """All BF16 binary addition trees, plus one final rounding of the real sum.

    Diagnostic envelopes only: matching does not identify NCCL's execution tree.
    """
    @lru_cache(None)
    def sums(mask):
        if mask & (mask-1) == 0:
            return {values[mask.bit_length()-1]}
        answer = set()
        subset = (mask-1) & mask
        while subset:
            other = mask ^ subset
            if other and subset < other:
                answer.update(bf16(a+b) for a in sums(subset) for b in sums(other))
            subset = (subset-1) & mask
        return answer
    return sums((1 << len(values))-1) | {bf16(sum(values))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from scripts import v3_numerical_remote as n
    n.STEPS = [2]
    root = Path(__file__).resolve().parents[1]/'results'
    status = json.loads((root/f'v3_b1_20261103_{a.suffix}/status.json').read_text())
    assert status['done'] and status['success'] and status['steps'] == [2] and status['gradient_probe_from'] == 'num01'
    runs = {r['mode']: root/r['session'] for r in status['rows']}
    assert set(runs) == {'flat', 'shift'}
    data, raw_sha, same_mode = {}, {}, {}
    for mode, directory in runs.items():
        signatures, events = n.audit(directory)
        probes = {}
        for (rid, rank), rows in events.items():
            raw = directory/f'node{rid}'/f'events_g0_r{rank}.jsonl'
            raw_sha[str(raw.relative_to(root))] = hashlib.sha256(raw.read_bytes()).hexdigest()
            loaded = [r for r in rows if r['kind'] == 'reference_anchor_loaded']
            assert len(loaded) == 1 and loaded[0]['schema'] == 2 and loaded[0]['hashes_match'] and loaded[0]['step'] == 1
            captures = [r for r in rows if r['kind'] == 'gradient_probe']
            assert len(captures) == 1 and captures[0]['step'] == 2
            capture = captures[0]
            assert capture['metadata']['physical_rank'] == rid*8+rank
            probes[rid*8+rank] = capture
        data[mode] = probes
        prior = root/f'v3b1_panchor01_{"01_shift" if mode == "shift" else "03_flat"}'
        same_mode[mode] = n.compare_compact(prior, directory)
        assert same_mode[mode]['exact_pass'], 'probe changed same-mode committed state/gradients/routing'
    cross = n.compare_compact(runs['flat'], runs['shift'])
    assert not cross['sample_errors'] and not cross['router_differences']
    def tensor(mode, role, phase, name):
        rows = [r for r in data[mode][role]['tensors'] if r['phase'] == phase and r['name'] == name]
        assert len(rows) == 1
        return rows[0]
    comparisons, reconstruction = [], []
    for role in range(16):
        left = data['flat'][role]
        right = data['shift'][role]
        assert len(left['tensors']) == len(right['tensors']) == 12
        names = [r['name'] for r in left['tensors'] if r['phase'] == 'pre_local']
        for name in names:
            for phase in ('pre_local', 'post_local', 'post_outer', 'post_tail'):
                x, y = tensor('flat', role, phase, name), tensor('shift', role, phase, name)
                assert (x['shape'], x['indices'], x['dtype']) == (y['shape'], y['indices'], y['dtype'])
                delta = [abs(a-b) for a, b in zip(x['values'], y['values'])]
                comparisons.append(dict(role=role, name=name, phase=phase, dense=x['dense'], elements=len(delta),
                                        different=sum(d != 0 for d in delta), max_abs=max(delta)))
            for mode in ('flat', 'shift'):
                pre = tensor(mode, role, 'pre_local', name)
                local = tensor(mode, role, 'post_local', name)
                outer = tensor(mode, role, 'post_outer', name)['values']
                tail = tensor(mode, role, 'post_tail', name)['values']
                members = pre['members'] if mode == 'flat' else [(role//8)*8+r for r in pre['members']]
                expected_members = ([role] if not pre['dense'] else [role//8*8+role%8//4*4+role%2, role//8*8+role%8//4*4+2+role%2])
                if mode == 'flat':
                    expected_members = ([role%8, role%8+8] if not pre['dense'] else
                                        [role%8//4*4+role%2, role%8//4*4+2+role%2, role%8//4*4+8+role%2, role%8//4*4+10+role%2])
                assert members == expected_members
                assert pre['scaling'] == (.25 if mode == 'flat' else .5)
                values = [tensor(mode, member, 'pre_local', name)['values'] for member in members]
                outside = 0
                real_errors = []
                for i, observed in enumerate(local['values']):
                    scaled = tuple(bf16(v[i]*pre['scaling']) for v in values)
                    outside += observed not in possible_sums(scaled)
                    real_errors.append(abs(observed-sum(v[i]*pre['scaling'] for v in values)))
                if mode == 'shift':
                    peers = [tensor(mode, role%8+r*8, 'post_local', name)['values'] for r in range(2)]
                    outer_mismatch = sum(z != bf16(bf16(x+y)/2) for x,y,z in zip(*peers,outer))
                else:
                    outer_mismatch = sum(x != y for x,y in zip(local['values'], outer))
                reconstruction.append(dict(mode=mode, role=role, name=name, members=members, scaling=pre['scaling'],
                    local_outside_bf16_tree_envelope=outside, local_max_abs_real_error=max(real_errors),
                    outer_reconstruction_mismatches=outer_mismatch, tail_changes=sum(x != y for x,y in zip(outer,tail))))
    result = dict(diagnostic=True, performance_evidence=False, rank_steps=32, raw_sha256=raw_sha,
                  same_mode_prior=same_mode, cross_mode=cross, comparisons=comparisons, reconstruction=reconstruction,
                  limitation='Only selected slices; BF16 tree envelope does not prove actual NCCL summation tree or unselected gradient equivalence. No performance savings measured.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(rank_steps=32, same_mode_exact={k:v['exact_pass'] for k,v in same_mode.items()},
        differences={phase:sum(r['different'] for r in comparisons if r['phase']==phase) for phase in ('pre_local','post_local','post_outer','post_tail')},
        local_outside_envelope=sum(r['local_outside_bf16_tree_envelope'] for r in reconstruction),
        outer_mismatches=sum(r['outer_reconstruction_mismatches'] for r in reconstruction),
        tail_changes=sum(r['tail_changes'] for r in reconstruction))))


if __name__ == '__main__':
    main()
