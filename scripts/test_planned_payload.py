"""CPU execution of the trainer's actual nested checkpoint/relay functions.

Extract closures to avoid starting Megatron/distributed main. CUDA RNG/stream
APIs are replaced by a separate CPU byte-state; model, Adam and hashes are real.
This does not replace full-payload GPU/relay/cycle end-to-end validation.
"""
import ast
from contextlib import ExitStack
import copy
from pathlib import Path
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from shift_full import planned_train
from shift_full.planned_state import validate_state
from shift_full.state_inventory import tracker_snapshot,restore_tracker,restore_buffers


class PlannedPayloadTest(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.cuda_rng=torch.tensor([1,2,3],dtype=torch.uint8)
        def set_cuda(value):self.cuda_rng=value.clone()
        fake=dict(get_rng_state=lambda:self.cuda_rng.clone(),set_rng_state=set_cuda,
                  current_stream=lambda:SimpleNamespace(synchronize=lambda:None),
                  memory_allocated=lambda *_:0,max_memory_allocated=lambda *_:0)
        for key,value in fake.items():self.stack.enter_context(patch.object(torch.cuda,key,value))
        torch.manual_seed(2026);random.seed(2026)
        stage=torch.nn.Linear(2,2)
        stage.register_buffer('persistent',torch.tensor([4.]))
        stage.register_buffer('scratch',torch.tensor([5.]),persistent=False)
        optimizer=torch.optim.AdamW(stage.parameters(),lr=1e-4,foreach=False)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.98)
        stage(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step()
        tracker=SimpleNamespace(states={'tp':torch.tensor([11,12],dtype=torch.uint8)})
        tracker.get_states=lambda:tracker.states
        tracker.set_states=lambda states:setattr(tracker,'states',states)
        self.ns=dict(torch=torch,random=random,time=time,digest=planned_train.digest,
                     tensor_count=planned_train.tensor_count,cpu_tree=planned_train.cpu_tree,
                     tracker_snapshot=tracker_snapshot,restore_tracker=restore_tracker,
                     restore_buffers=restore_buffers,validate_state=validate_state,
                     stage=stage,optimizer=optimizer,scheduler=scheduler,rng_tracker=tracker,
                     progress={'updates':7},rank=3,device_rank=0,manager=None,boundary_rng=None,
                     handoff=SimpleNamespace(new_epoch=1),
                     args=SimpleNamespace(audit_state=True,transport='gpu',replica=1),event=lambda *a,**k:None)
        tree=ast.parse(Path(planned_train.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        names={'rng_snapshot','current_hashes','export_state','load_state','apply_tail'}
        funcs=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in names]
        wrapper=ast.parse('def build():\n audit_joint_pending=False\n return locals()').body[0]
        wrapper.body[1:1]=funcs
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'<planned-closures>','exec'),self.ns)
        self.fn=self.ns['build']()

    def test_snapshot_restores_adam_buffers_and_all_rng(self):
        payload=planned_train.cpu_tree(self.fn['export_state']())
        expected=dict(payload['hashes'])
        with torch.no_grad():
            self.ns['stage'].weight.add_(10)
            self.ns['stage'].scratch.add_(20)
        self.ns['optimizer'].param_groups[0]['lr']=9
        self.ns['scheduler'].last_epoch+=5
        self.ns['progress']['updates']=99
        self.ns['rng_tracker'].states['tp'].add_(4)
        torch.rand(4);random.random();torch.cuda.set_rng_state(torch.tensor([9],dtype=torch.uint8))
        self.fn['load_state'](payload)
        self.assertEqual(self.fn['current_hashes'](),expected)
        self.assertEqual(self.ns['stage'].scratch.item(),5)
        self.assertGreater(planned_train.tensor_count(self.ns['optimizer'].state_dict()['state']),0)
        self.ns['rng_tracker'].states['tp'].add_(1)
        self.assertEqual(payload['tracker_rng']['tp'].tolist(),[11,12])

    def test_boundary_buffers_and_tracker_frozen_and_tail_roundtrips_each_epoch(self):
        rng=self.fn['rng_snapshot']()
        self.ns['boundary_rng']=dict(step=7,rng=rng,
            buffers={n:b.clone() for n,b in self.ns['stage'].named_buffers()},copy_ready=SimpleNamespace(synchronize=lambda:None))
        self.ns['stage'].scratch.add_(100);self.ns['rng_tracker'].states['tp'].add_(1)
        payload=self.fn['export_state']()
        self.assertEqual(payload['all_buffers']['scratch'].item(),5)
        self.assertEqual(payload['tracker_rng']['tp'].tolist(),[11,12])
        for epoch in range(1,5):
            step=7+epoch;self.ns['progress']['updates']=step;self.ns['handoff'].new_epoch=epoch
            tail=dict(schema=2,planned_epoch=epoch,planned_role=1,source_rank=3,
                step=step,progress=dict(updates=step),all_buffers={n:b.clone() for n,b in self.ns['stage'].named_buffers()},**self.fn['rng_snapshot']())
            expected=planned_train.digest(tail)
            self.ns['stage'].scratch.add_(10);self.ns['rng_tracker'].states['tp'].add_(3)
            torch.rand(2);random.random();torch.cuda.set_rng_state(torch.tensor([99],dtype=torch.uint8))
            applied=self.fn['apply_tail'](tail)
            self.assertEqual(planned_train.digest(applied),expected)

if __name__=='__main__':unittest.main()


class ShadowPayloadTest(unittest.TestCase):
    """Execute the actual pre-receiver callback with CPU tensors/fake local schedule."""
    def setUp(self):
        PlannedPayloadTest.setUp(self)
        import json,threading
        ns=self.ns;stage=ns['stage']
        stage.hf_config=SimpleNamespace(vocab_size=32000);stage.extra_positions=lambda _:0
        params=list(stage.parameters())
        for p in params:p.main_grad=torch.zeros_like(p)
        bucket=SimpleNamespace(grad_data=torch.zeros(8),params_with_grad=set(),communication_handle=None,communication_issued=False)
        buf=SimpleNamespace(buckets=[bucket],grad_data=bucket.grad_data,is_last_microbatch=True)
        def clear():
            buf.grad_data.zero_();bucket.params_with_grad=set();bucket.communication_handle=None;bucket.communication_issued=False
            buf.is_last_microbatch=True
            for p in params:p.grad_added_to_main_grad=False;p.main_grad.zero_()
        ddp=SimpleNamespace(buffers=[buf],expert_parallel_buffers=[],zero_grad_buffer=clear,
            finish_grad_sync=lambda:setattr(bucket,'communication_issued',True))
        stage.config=SimpleNamespace(finalize_model_grads_func=object())
        descriptor=dict(schema=1,rank=3,step=7,epoch=1,seq_len=16,microbatches=2,data='pattern',vocab_size=32000,
            seed_base=100000,participants=2,replica_index=0,local_data_rank=0,sample_ids=[56,60])
        ns.update(json=json,threading=threading,ddp=ddp,route_calls={'prior':1},shadow_active=[False],
            state=SimpleNamespace(get_data_parallel_rank=lambda:0),forward_step=object())
        ns['args'].seq_len=16;ns['args'].microbatches=2;ns['args'].data='pattern'
        ns['handoff'].cut=7;ns['handoff'].coord=SimpleNamespace(get=lambda _:json.dumps(descriptor).encode())
        ns['export_state']=SimpleNamespace(role_metadata=lambda:dict(schema=2,source_rank=3,step=7,progress=dict(updates=7),
            all_buffers=dict(stage.named_buffers()),**self.fn['rng_snapshot']()))
        ns['current_hashes']=self.fn['current_hashes'];ns['rng_snapshot']=self.fn['rng_snapshot']
        self.fail_schedule=False
        def schedule(**kwargs):
            ns['route_calls']['shadow']=1
            self.assertEqual(len(list(kwargs['data_iterator'])),2)
            for p in params:p.main_grad.add_(1);p.grad_added_to_main_grad=True
            bucket.grad_data.add_(1);stage.scratch.add_(3);torch.rand(1);ns['rng_tracker'].states['tp'].add_(2)
            stage.config.finalize_model_grads_func([ddp])
            if self.fail_schedule:raise ValueError('fake schedule failure')
        ns['get_forward_backward_func']=lambda:schedule
        native_arange=torch.arange
        self.stack.enter_context(patch.object(torch,'arange',lambda *a,**kw:native_arange(*a,**{k:v for k,v in kw.items() if k!='device'})))
        tree=ast.parse(Path(planned_train.__file__).read_text())
        callback=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='shadow_recompute')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),'<actual-shadow-callback>','exec'),ns)
        self.expected=self.fn['current_hashes']();self.native_finalizer=stage.config.finalize_model_grads_func

    def check_clean(self):
        self.assertEqual(self.fn['current_hashes'](),self.expected)
        self.assertEqual(self.ns['route_calls'],{'prior':1});self.assertEqual(self.ns['shadow_active'],[False])
        self.assertIs(self.ns['stage'].config.finalize_model_grads_func,self.native_finalizer)

    def test_actual_callback_roundtrip(self):
        result=self.ns['shadow_recompute'](self.expected)
        self.assertEqual(result['sample_ids'],[56,60]);self.check_clean()

    def test_actual_callback_schedule_exception_roundtrip(self):
        self.fail_schedule=True
        with self.assertRaisesRegex(ValueError,'fake schedule failure'):self.ns['shadow_recompute'](self.expected)
        self.check_clean()


class OwnershipPayloadTest(unittest.TestCase):
    """Same actual trainer closure; success retains real CPU gradients/tail."""
    setUp=ShadowPayloadTest.setUp
    check_clean=ShadowPayloadTest.check_clean

    def test_actual_callback_fresh_audit_then_retains_and_cancels(self):
        calls=[];original=self.ns['current_hashes']
        self.ns['current_hashes']=lambda:(calls.append('fresh'),original())[1]
        c=self.ns['shadow_recompute'](self.expected,retain=True)
        self.assertEqual(calls,['fresh'])
        c.validate()
        self.assertEqual(c.result['sample_ids'],[56,60])
        self.assertEqual(self.ns['stage'].scratch.item(),8)
        self.assertTrue(all(bool(p.main_grad.count_nonzero()) for p in self.ns['stage'].parameters()))
        self.assertIs(self.ns['stage'].config.finalize_model_grads_func,self.native_finalizer)
        c.cancel();self.check_clean();self.assertEqual(calls,['fresh','fresh'])

    def test_actual_callback_exception_rolls_back_all_nine_hashes(self):
        self.fail_schedule=True
        with self.assertRaisesRegex(ValueError,'fake schedule failure'):
            self.ns['shadow_recompute'](self.expected,retain=True)
        self.check_clean()

    def test_actual_callback_rejects_gradient_identity_and_rng_mutation(self):
        for mutation in ('version','identity','rng','scheduler'):
            with self.subTest(mutation=mutation):
                c=self.ns['shadow_recompute'](self.expected,retain=True)
                p=next(self.ns['stage'].parameters())
                if mutation=='version':p.main_grad.add_(2)
                elif mutation=='identity':p.main_grad=p.main_grad.clone()
                elif mutation=='rng':torch.rand(1)
                else:self.ns['scheduler'].last_epoch+=1
                with self.assertRaises(RuntimeError):c.validate()
                if mutation=='scheduler':self.ns['scheduler'].last_epoch-=1
                c.cancel();self.check_clean()

    def test_actual_callback_continuation_does_not_finish_local_twice(self):
        c=self.ns['shadow_recompute'](self.expected,retain=True)
        calls=[];ddp=self.ns['ddp']
        ddp.finish_grad_sync=lambda:calls.append('illegal_local')
        def native(models):
            models[0].finish_grad_sync();calls.append('native_tppp')
        c.finish(native_finalizer=native,outer=lambda:calls.append('new_outer'))
        self.assertEqual(calls,['new_outer','native_tppp'])
        with self.assertRaises(RuntimeError):c.cancel()

    def test_pinned_native_finalizer_and_actual_outer_wrapper_continue_once(self):
        # Execute the installed vendor's real finalizer and TP/PP functions.
        # Only distributed transport is instrumented; CPU tensors are real.
        import hashlib
        from typing import List
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        path=Path('/workspace/FLASH1022/Greyhound-master/Megatron-LM/megatron/core/distributed/finalize_model_grads.py')
        source=path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
            '051790f45f896159d6c98fe6e911fbb8bb969119addd854ae14434a6aacad820')
        c=self.ns['shadow_recompute'](self.expected,retain=True)
        calls=[];ddp=self.ns['ddp'];p=next(self.ns['stage'].parameters())
        p.sequence_parallel=True
        embedding=SimpleNamespace(main_grad=torch.ones(2))
        ddp.parameters=lambda:[p];ddp.pre_process=True
        ddp.share_embeddings_and_output_weights=True
        ddp.shared_embedding_or_output_weight=lambda:embedding
        parallel=SimpleNamespace(get_tensor_model_parallel_world_size=lambda:2,
            get_tensor_model_parallel_group=lambda:'TP',is_rank_in_embedding_group=lambda **kw:True,
            get_pipeline_model_parallel_world_size=lambda:2,is_pipeline_first_stage=lambda **kw:True,
            is_rank_in_position_embedding_group=lambda:False,get_embedding_group=lambda:'PP')
        ns=dict(torch=torch,List=List,TransformerConfig=object,parallel_state=parallel,
            _flatten_dense_tensors=_flatten_dense_tensors,_unflatten_dense_tensors=_unflatten_dense_tensors,
            get_model_config=lambda m:SimpleNamespace(timers=None,sequence_parallel=True),
            get_attr_wrapped_model=lambda m,a,**kw:m if kw.get('return_model_obj') else getattr(m,a))
        tree=ast.parse(source)
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(path),'exec'),ns)
        # Extract the exact trainer wrapper, with its real local_finished branch.
        trainer=ast.parse(Path(planned_train.__file__).read_text())
        finish=next(n for n in ast.walk(trainer) if isinstance(n,ast.FunctionDef) and n.name=='finish_grad_sync')
        wrapper=ast.parse('def build():\n manager=supplied_manager\n return finish_grad_sync').body[0]
        wrapper.body.insert(1,finish)
        manager=SimpleNamespace(allreduce=lambda t:(calls.append('outer'),SimpleNamespace(wait=lambda:None))[1])
        env=dict(self.ns,args=SimpleNamespace(profile_timings=False,backend='nccl',planned_variant='p2'),
            handoff=SimpleNamespace(is_relay_step=lambda:False,authorize=lambda m:None,emit=lambda *a,**kw:None),
            supplied_manager=manager,buffers=ddp.buffers,cross_timings={},cross_calls=[0],
            original_finish=lambda: calls.append('ILLEGAL_LOCAL'))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),'<actual-outer-wrapper>','exec'),env)
        outer=env['build']()
        with patch.object(torch.distributed,'all_reduce',side_effect=lambda t,group:calls.append(group)):
            c.seal();c.activate()
            c.finish(native_finalizer=ns['finalize_model_grads'],outer=lambda:outer(ddp,local_finished=True))
        self.assertEqual(calls,['outer','TP','PP'])
        self.assertEqual(env['cross_calls'],[1])
        self.assertEqual(c.state,'finished')
