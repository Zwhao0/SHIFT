"""CPU checks for the actual descriptor/finalizer/rollback diagnostic helpers."""
import copy
import argparse
import ast
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from shift_full.planned_shadow import validate_descriptor, LocalFinalizer, shadow_transaction


class ShadowTest(unittest.TestCase):
    def test_actual_trainer_cli_guard_before_any_cuda(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'shift_full/planned_train.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        body=[]
        for n in main.body:
            if isinstance(n,ast.ImportFrom) and n.module=='shift_full.planned_state':break
            body.append(n)
        body.append(ast.parse('return args').body[0])
        main.body=body;ns={'argparse':argparse}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'<actual-trainer-cli>','exec'),ns)
        base=['trainer','--session','test','--replica','1','--generation','1','--host-ip','c',
            '--planned-role','C','--planned-variant','p2','--planned-hosts','{}',
            '--planned-shadow-recompute','--planned-inject','slow-compute','--reference-signatures',
            '--audit-state','--audit-routing','--sync-quorum','--fixed-batch','--seq-len','512',
            '--data','pattern','--backend','nccl','--bucket-size','67108864']
        with patch.object(sys,'argv',base):self.assertTrue(ns['main']().planned_shadow_recompute)
        # Only B/A read the existing anchor; C must receive B's actual k state.
        wrong=list(base);wrong[wrong.index('C')]='B'
        import io
        from contextlib import redirect_stderr
        with patch.object(sys,'argv',wrong),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):ns['main']()
        with patch.object(sys,'argv',wrong+['--reference-anchor','/existing']):self.assertTrue(ns['main']().reference_anchor)
        ownership=[s.replace('--planned-shadow-recompute','--planned-ownership-recompute') for s in base]
        with patch.object(sys,'argv',ownership):self.assertTrue(ns['main']().planned_ownership_recompute)
        with patch.object(sys,'argv',ownership+['--planned-shadow-recompute']),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):ns['main']()

    def descriptor(self):
        return dict(schema=1,rank=6,step=10,epoch=1,seq_len=512,microbatches=2,
            data='random',vocab_size=32000,seed_base=100000,participants=2,
            replica_index=0,local_data_rank=1,sample_ids=[81,85])

    def validate(self,d):
        return validate_descriptor(d,rank=6,step=10,epoch=1,seq_len=512,
            microbatches=2,data='random',vocab_size=32000)

    def test_actual_quorum_index_not_assumed(self):
        d=self.descriptor();self.assertEqual(self.validate(d),[81,85])
        d.update(replica_index=1,sample_ids=[83,87]);self.assertEqual(self.validate(d),[83,87])

    def test_reject_missing_stale_and_wrong_order(self):
        for key in self.descriptor():
            d=self.descriptor();del d[key]
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(d)
        for updates in [dict(step=9),dict(epoch=2),dict(rank=5),dict(seq_len=16),dict(data='pattern'),
                        dict(seed_base=9),dict(participants=1),dict(replica_index=None),dict(sample_ids=[85,81])]:
            d=self.descriptor();d.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.validate(d)

    def test_finalizer_only_native_local_and_once_on_owner(self):
        calls=[];ddp=SimpleNamespace(finish_grad_sync=lambda:calls.append('local'))
        local=LocalFinalizer(ddp);errors=[]
        def other():
            try:local([ddp])
            except RuntimeError as exc:errors.append(str(exc))
        t=threading.Thread(target=other);t.start();t.join()
        self.assertEqual(len(errors),1);self.assertEqual(calls,[])
        with self.assertRaises(RuntimeError):local([object()])
        local([ddp]);self.assertEqual(calls,['local'])
        with self.assertRaises(RuntimeError):local([ddp])

    def test_restore_clear_verify_and_native_finalizer_on_each_failure(self):
        for failure in (None,'execute','capture','restore','clear_after'):
            calls=[];native=object();config=SimpleNamespace(finalize_model_grads_func=native)
            ddp=SimpleNamespace(finish_grad_sync=lambda:calls.append('local'))
            def execute():
                config.finalize_model_grads_func([ddp]);calls.append('execute')
                if failure=='execute':raise ValueError(failure)
            def capture():
                calls.append('capture')
                if failure=='capture':raise ValueError(failure)
                return 'hash'
            def restore():
                self.assertIs(config.finalize_model_grads_func,native);calls.append('restore')
                if failure=='restore':raise ValueError(failure)
            def clear():
                calls.append('clear')
                if failure=='clear_after' and calls.count('clear')==2:raise ValueError(failure)
            def verify():calls.append('verify')
            kwargs=dict(config=config,ddp=ddp,execute=execute,capture=capture,restore=restore,clear=clear,verify=verify)
            if failure:
                with self.assertRaises(ValueError):shadow_transaction(**kwargs)
            else:self.assertEqual(shadow_transaction(**kwargs),'hash')
            self.assertIs(config.finalize_model_grads_func,native)
            self.assertEqual(calls[-1],'verify');self.assertEqual(calls.count('restore'),1)
            self.assertEqual(calls.count('clear'),2)

    def test_missing_schedule_finalizer_rejected_and_rolled_back(self):
        calls=[];config=SimpleNamespace(finalize_model_grads_func=None)
        with self.assertRaisesRegex(RuntimeError,'missed'):
            shadow_transaction(config=config,ddp=SimpleNamespace(finish_grad_sync=lambda:None),
                execute=lambda:None,capture=lambda:None,restore=lambda:calls.append('restore'),
                clear=lambda:calls.append('clear'),verify=lambda:calls.append('verify'))
        self.assertEqual(calls,['clear','restore','clear','verify'])
        self.assertIsNone(config.finalize_model_grads_func)

if __name__=='__main__':unittest.main()
