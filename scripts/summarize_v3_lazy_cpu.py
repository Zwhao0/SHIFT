"""Independently verify P13 full CPU startup evidence; never grants GPU admission."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    rows = {n:json.loads((a.directory/(n+'.json')).read_text()) for n in ('a0', 'b1', 'b2', 'a3')}
    status = json.loads((a.directory/'status.json').read_text())
    assert status['done'] and status['success'] and len(status['children']) == 4
    reference = rows['a0']
    for name, x in rows.items():
        assert x['success'] and x['done'] and not x['denied'] and not x['optional_calls']
        assert not x['cuda_initialized'] and not x['distributed_initialized']
        assert x['parameter_elements'] == 494712832 and x['training']
        assert x['optimizer_state_entries'] == 0  # Initial construction; no optimizer update claimed.
        assert not x['preexisting_compiler_modules']
        assert x['hashes'] == reference['hashes'] and x['state_before'] == reference['state_before']
        assert x['state_after'] == reference['state_after']
        assert x['compiler_modules_after'] == ['torch._dynamo', 'torch._inductor']
        assert x['utils_sha256'] == 'fe89a3458286a036d8d36e9df9d191eaf0d048eb39a2172af96c4bb414b5deb9'
        for module, source in x['sources'].items():
            for other in rows.values():
                if module in other['sources']:
                    assert source == other['sources'][module], module
        assert x['sources']['megatron.core.jit']['sha256'] == hashlib.sha256(
            json.loads((a.directory.parent/'p13_jit_source.json').read_text())['text'].encode()).hexdigest()
    savings = [rows['a0']['total_seconds']-rows['b1']['total_seconds'],
               rows['a3']['total_seconds']-rows['b2']['total_seconds']]
    baseline_only = sorted(set(reference['sources'])-set(rows['b1']['sources']))
    metadata = {name:dict(original=reference['wrapper_metadata'][name]['attributes'],
                         candidate=rows['b1']['wrapper_metadata'][name]['attributes'])
                for name in ('gelu_impl', 'erf_gelu')}
    report = dict(functional_cpu=True, full_cpu_budget_admitted=min(savings) >= .5,
        semantics_admitted=False, gpu_admitted=False, performance_gates_passed=False,
        release_admitted=False, diagnostic_only=True, savings_seconds=savings,
        mean_saving_seconds=sum(savings)/len(savings),
        measurements={n:{k:x[k] for k in ('total_seconds', 'imports_seconds', 'model_seconds',
                      'optimizer_seconds', 'first_import')} for n,x in rows.items()},
        hashes=reference['hashes'], state_before=reference['state_before'], state_after=reference['state_after'],
        metadata_gap=metadata, baseline_only_modules=baseline_only,
        raw_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.directory.glob('*') if f.is_file()},
        limits=['ABBA is two chronological comparisons, not rank repeats or final performance verification',
                'full CPU optimizer covers dependency imports, not exact partitioned DDP execution',
                'compiled-wrapper API/metadata and actual GPU first-FB costs remain unproved',
                'all initial model tensors/buffers and full optimizer/scheduler/RNG compared; optimizer has no update state'])
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('functional_cpu', 'full_cpu_budget_admitted',
        'savings_seconds', 'semantics_admitted', 'gpu_admitted')}))


if __name__ == '__main__':
    main()
