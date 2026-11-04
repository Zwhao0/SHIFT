"""The predeclared benchmark window must retain slow/audited steps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.v3_benchmark_remote import analyze, SCHEMAS, NoFaultRotation, matched_comparison
from scripts.reanalyze_v3_benchmark import independent_raw

class BenchmarkTest(unittest.TestCase):
    def test_anchor_timeline_preserves_formal_workload_and_all_checks(self):
        from scripts import v3_numerical_remote as numerical
        from unittest.mock import MagicMock
        session=MagicMock()
        session.endpoint_for.return_value='same-host'
        with patch.object(numerical,'NoFaultRotation',return_value=session) as rotation, patch.object(numerical,'audit') as audit:
            numerical.run_case('diagnostic','flat',Path('/original-anchor'),reuse_anchor=True,dump=False,anchor_timeline=True)
            config=rotation.call_args.args[1];args=config['train_args']
            self.assertEqual(args[args.index('--seq-len')+1],'512')
            self.assertEqual(args[args.index('--timeline-trace-step')+1],'2')
            self.assertEqual(args[args.index('--reference-anchor')+1],'/original-anchor')
            self.assertTrue({'--audit-state','--reference-signatures','--audit-routing','--measure-step','--trace-events'}<=set(args))
            self.assertNotIn('--reference-anchor-save',args)
            self.assertNotIn('--reference-dump',args)
            self.assertNotIn('--warmup-before-join',args)
            audit.assert_called_once_with(session.out)
            session.finish.assert_called_once()
        with self.assertRaises(ValueError):
            numerical.run_case('invalid','shift',Path('/anchor'),anchor_timeline=True)

    def test_digest_engine_is_explicitly_propagated_to_all_workers(self):
        from scripts import v1_acceptance_remote as remote
        with patch.object(remote.ServerRotation,'__init__',return_value=None):
            with patch.dict('os.environ',{'SHIFT_DIGEST_ENGINE':'staged'}):
                config={'env':{}}
                NoFaultRotation('test',config,1,2)
                self.assertEqual(config['env']['SHIFT_DIGEST_ENGINE'],'staged')
            with patch.dict('os.environ',{'SHIFT_DIGEST_ENGINE':'invalid'}):
                with self.assertRaises(ValueError):NoFaultRotation('test',{'env':{}},1,2)

    def test_all_joint_steps_including_slow_audit_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rid in range(2):
                d=root/f'node{rid}';d.mkdir()
                for rank in range(8):
                    hashes={k:'same' for k in SCHEMAS}
                    rows=[dict(kind='ready',hashes=hashes,state_inventory={})]
                    rows += [dict(kind='commit',step=k,participants=2,contributed=True,finite_gradients=True,
                                  global_batch=8,hashes=hashes if k%20==0 else {},sample_ids=[rid],
                                  monotonic=k+70*(k>=20),seconds=71 if k==20 else 1,commit_interval_seconds=71 if k==20 else 1)
                             for k in range(1,81)]
                    (d/f'events_g0_r{rank}.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            result=analyze(root,80)
            self.assertEqual(result['windows']['fixed11_80']['count'],70)
            self.assertEqual(result['windows']['fixed11_80']['mean'],2)
            self.assertEqual(result['windows']['fixed11_80']['maximum'],71)
            self.assertEqual(result['fixed11_80_elapsed_seconds'],140)
            raw=independent_raw(root)
            self.assertEqual(raw['rank_steps'],1280)
            self.assertEqual(raw['step_mean'],2)
            self.assertEqual(raw['interval_mean'],2)
            self.assertEqual(raw['elapsed'],140)
            p=root/'node1/events_g0_r7.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
            with self.assertRaisesRegex(RuntimeError,'missing/duplicate'):analyze(root,80)
            with self.assertRaisesRegex(ValueError,'missing/duplicate'):independent_raw(root)

    def test_three_cost_denominators_and_complete_intervals(self):
        rows=[]
        for _ in range(4):
            for mode,v in [('flat',1),('nativeft',1.06),('shift',1.07)]:
                rows.append(dict(mode=mode,windows={'fixed11_80':{'mean':v}},
                    commit_intervals={'mean':v*2},elapsed_seconds=v*140))
        result=matched_comparison(rows,'flat','shift')
        for item in result['metrics'].values():self.assertAlmostEqual(item['paired_percent_mean'],7)
        result=matched_comparison(rows,'nativeft','shift')
        self.assertAlmostEqual(result['metrics']['elapsed_seconds']['pooled_overhead_percent'],100*(1.07/1.06-1))
        with self.assertRaisesRegex(ValueError,'four complete'):matched_comparison(rows[:-1],'flat','shift')

    def test_no_fault_entry_refuses_injection(self):
        with self.assertRaisesRegex(RuntimeError,'forbidden'):NoFaultRotation.__new__(NoFaultRotation).fault(0)

if __name__=='__main__':unittest.main()
