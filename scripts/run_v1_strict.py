"""Two-update strict-FP32 control, preserving the original tolerances."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_v1_experiments import ROOT, OUT, run
from shift_full.control import Session, endpoints_from_env
from shift_full.remote_job import run_job


def main():
    for mode in ('flat','shift'):
        name=f'v1_strict_{mode}_01'
        run(name,['-m','shift_full.control','--session',name,'--config',f'configs/v1_strict_{mode}.json','--seconds','900'])
    for mode in ('flat','shift'):
        for p in (ROOT/f'results/v1_strict_{mode}_01').glob('node[01]/events_g0_r*.jsonl'):
            ready=next(json.loads(l) for l in p.read_text().splitlines() if json.loads(l)['kind']=='ready')
            assert ready['matmul_allow_tf32'] is False and ready['float32_matmul_precision']=='highest'
    probe=Session.__new__(Session);probe.root=ROOT;probe._sudo_password=None
    e=endpoints_from_env()['node0'];remote='/workspace/SHIFT/results/v1_strict_shift_01/strict_comparison.json'
    cmd=['/workspace/SHIFT/runtime/torch214/bin/python','-m','shift_full.compare_reference',
         '/workspace/SHIFT/results/v1_strict_flat_01','/workspace/SHIFT/results/v1_strict_shift_01',
         '--steps','2,3','--threads','4','--output',remote]
    code=f'import subprocess; r=subprocess.run({cmd!r},cwd="/workspace/SHIFT"); print(r.returncode)'
    assert run_job(probe,e,['python','-c',code],timeout=1200).strip() in ('0','1')
    comparison=run_job(probe,e,['cat',remote],timeout=60)
    (OUT/'strict_comparison.json').write_text(comparison)
    run('strict_summary',['-m','shift_full.summarize_trajectory',
        str(ROOT/'results/v1_strict_flat_01'),str(ROOT/'results/v1_strict_shift_01'),
        str(OUT/'strict_comparison.json'),'--output',str(OUT/'strict_summary.json')],allowed=(0,1))
    print('STRICT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
