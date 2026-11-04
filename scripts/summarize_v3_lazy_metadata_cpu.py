"""Independent bounded P13 genuine-export CPU check; never grants GPU admission."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    status=json.loads((a.directory/'status.json').read_text())
    assert status['done'] and status['success'] and len(status['children'])==6
    sem={n:json.loads((a.directory/(n+'.json')).read_text()) for n in ('s0','s1')}
    for row in sem.values():
        assert row['success'] and row['done'] and not row['denied']
        assert row['concurrent_identity'] and row['rng_exact']
        assert not row['cuda_initialized'] and not row['distributed_initialized']
        assert len(row['expected_rejected_cuda_probe'])==1
        assert row['cache_source_sha256']=='1edadc1e1229adbd7c6f84cef43a04c031add6e1793be0f12b73f22798966156'
        assert row['compiler_metrics']['generated_kernel_count']>0
    for name in ('gelu_impl','erf_gelu'):
        original=sem['s0']['functions'][name]
        candidate=sem['s1']['functions'][name]
        for key in ('attributes','signature','config_sha256','invalid_input','recovery_output','isolate_recompiles_id'):
            assert original[key]==candidate[key], (name,key)
        assert len(original['samples'])==len(candidate['samples'])==3
        for left,right in zip(original['samples'],candidate['samples']):
            for key in ('size','output','gradient'):
                assert left[key]==right[key],(name,key)
    rows={n:json.loads((a.directory/(n+'.json')).read_text()) for n in ('a0','b1','b2','a3')}
    reference=rows['a0']
    jit_sha=hashlib.sha256(json.loads((a.directory.parent/'p13_jit_source.json').read_text())['text'].encode()).hexdigest()
    for row in rows.values():
        assert row['success'] and row['done'] and not row['denied'] and not row['optional_calls']
        assert not row['cuda_initialized'] and not row['distributed_initialized']
        assert row['parameter_elements']==494712832 and row['training'] and row['optimizer_state_entries']==0
        assert not row['preexisting_compiler_modules']
        for key in ('hashes','state_before','state_after','wrapper_metadata'):
            assert row[key]==reference[key],key
        assert row['utils_sha256']=='fe89a3458286a036d8d36e9df9d191eaf0d048eb39a2172af96c4bb414b5deb9'
        assert row['sources']['megatron.core.jit']['sha256']==jit_sha
        for module,source in row['sources'].items():
            for other in rows.values():
                if module in other['sources']:
                    assert source==other['sources'][module],module
    savings=[rows['a0']['total_seconds']-rows['b1']['total_seconds'],
             rows['a3']['total_seconds']-rows['b2']['total_seconds']]
    report=dict(diagnostic_only=True,bounded_cpu_semantics_pass=True,
        genuine_metadata_api=True,real_default_inductor_kernels=True,
        complete_cpu_initial_state_exact=True,full_cpu_budget_admitted=min(savings)>=.5,
        savings_seconds=savings,gpu_admitted=False,production_admitted=False,
        performance_gates_passed=False,release_admitted=False,
        materialization={n:row.get('materialization',[]) for n,row in rows.items()},
        measurements={n:{k:row[k] for k in ('total_seconds','imports_seconds','model_seconds','optimizer_seconds')} for n,row in rows.items()},
        raw_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.directory.glob('*') if f.is_file()},
        limits=['Bounded CPU ordinary export/API equivalence, not every possible Torch mode or GPU fallback',
                'Genuine aot_compile callable returned; AOT packaging execution not tested',
                'No nonempty optimizer update in full-model budget probe',
                'Two chronological CPU budget comparisons, not independent training performance validation',
                'Expected cache CUDA query rejected before native initialization, retained separately from forbidden calls'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('bounded_cpu_semantics_pass','full_cpu_budget_admitted','savings_seconds','gpu_admitted')}))


if __name__=='__main__':
    main()
