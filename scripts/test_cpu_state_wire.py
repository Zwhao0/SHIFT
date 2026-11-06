"""Exercise real packing/decoding of CPU Adam state without a CUDA context."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'requires retained Torch runtime')
class CPUStateWireTests(unittest.TestCase):
    def transport(self,pg,staged):
        from shift_full.transport import PackedGPUTransport
        from datetime import timedelta
        t=PackedGPUTransport.__new__(PackedGPUTransport)
        t._pg=pg;t._device=torch.device('cpu');t._timeout=timedelta(seconds=5)
        t._event=None;t._stage_cpu=staged
        return t

    def test_large_cpu_state_roundtrips_as_data_not_metadata(self):
        from datetime import timedelta
        queue=[]
        done=SimpleNamespace(wait=lambda timeout:None)
        def send(values,rank,tag):queue.append((tag,values[0].clone()));return done
        def recv(values,rank,tag):
            actual,tensor=queue.pop(0);self.assertEqual(tag,actual)
            values[0].copy_(tensor);return done
        pg=SimpleNamespace(send=send,recv=recv)
        payload=dict(optimizer={'state':{0:dict(exp_avg=torch.arange(800_000,dtype=torch.float32),
            exp_avg_sq=torch.ones(800_000),step=torch.tensor(7.))}},progress={'updates':7},rng=torch.arange(32,dtype=torch.uint8))
        source=self.transport(pg,True);target=self.transport(pg,True)
        with patch('shift_full.transport._CHUNK_BYTES',64*1024),patch.object(torch.cuda,'current_stream',return_value=SimpleNamespace(synchronize=lambda:None)):
            source.send_checkpoint([1],7,payload,timedelta(seconds=5))
            self.assertLess(queue[1][1].numel(),64*1024)
            self.assertGreater(len(queue),90)
            actual=target.recv_checkpoint(0,'',7,timedelta(seconds=5))
        self.assertEqual(actual['progress'],payload['progress'])
        for name,tensor in payload['optimizer']['state'][0].items():
            self.assertTrue(torch.equal(tensor,actual['optimizer']['state'][0][name]))
            self.assertEqual(actual['optimizer']['state'][0][name].device.type,'cpu')
        self.assertTrue(torch.equal(actual['rng'],payload['rng']))
        self.assertFalse(queue)

    def test_ordinary_profile_retains_cpu_control_data_limit(self):
        t=self.transport(None,False)
        with self.assertRaisesRegex(ValueError,'4 MiB'):
            t._pack(dict(adam=torch.zeros(1_100_000)),1)

    def test_frozen_source_checks_actual_tensors_and_rejects_pre_send_mutation(self):
        from shift_full.overlap_handoff import FrozenTransport
        from datetime import timedelta
        events=[];done=SimpleNamespace(wait=lambda timeout:None)
        pg=SimpleNamespace(send=lambda *args,**kwargs:done)
        t=FrozenTransport.__new__(FrozenTransport)
        t._pg=pg;t._device=torch.device('cpu');t._timeout=timedelta(seconds=5)
        t._event=lambda kind,**fields:events.append((kind,fields));t._stage_cpu=True
        value=dict(moment=torch.ones(4096),counter=torch.tensor(1.))
        event=SimpleNamespace(record=lambda:None,wait=lambda:None)
        with patch.object(torch.cuda,'current_stream',return_value=SimpleNamespace(synchronize=lambda:None)),patch.object(torch.cuda,'Event',return_value=event),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0):
            t.freeze(value,1);t.send_checkpoint([1],1,None,timedelta(seconds=5))
            audit=[fields for kind,fields in events if kind=='snapshot_readonly_verified']
            self.assertEqual(audit[-1]['tensors'],2)
            value['moment'].add_(1)
            with self.assertRaisesRegex(RuntimeError,'changed before'):
                t.send_checkpoint([1],1,None,timedelta(seconds=5))


if __name__=='__main__':unittest.main()
