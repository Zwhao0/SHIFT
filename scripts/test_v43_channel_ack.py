"""Actual channel closure ordering: a streamed receiver must acknowledge first."""
import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ChannelAckTests(unittest.TestCase):
    def run_transfer(self,source,fail_ack=False):
        calls=[]
        class Store:
            def get(self,key):return b'12'
            def set(self,key,value):calls.append('ack')
            def wait(self,keys,timeout):
                self_test.assertEqual(keys,['source_delivered/2']);calls.append('wait-ack')
                if fail_ack:raise TimeoutError('receiver not ready')
        class Transport:
            _stage_cpu=True
            def __init__(self,pg,device,timeout,event):self.event=event
            def send_checkpoint(self,*args):calls.append('send');self.emit('transport_send')
            def recv_checkpoint(self,*args):calls.append('receive');self.emit('transport_recv');return {}
            def emit(self,kind):self.event(kind,gpu_bytes=12,control_bytes=8,cpu_tensor_bytes=4)
            def metadata(self):return ''
        self_test=self
        path=Path(__file__).resolve().parents[1]/'shift_full/v4_transaction_train.py'
        channel=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.ClassDef) and n.name=='Channel')
        function=next(n for n in channel.body if isinstance(n,ast.FunctionDef) and n.name=='transfer')
        ns=dict(__package__='shift_full',time=__import__('time'),device=0,rank=2,
            torch=SimpleNamespace(device=lambda *args:None),boundary_hashes=lambda packet:None,
            s=SimpleNamespace(timeout=timedelta(seconds=2),event=lambda kind,**kw:calls.append(kind)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'<actual-channel>','exec'),ns)
        obj=SimpleNamespace(thread=None,error=None,pg=SimpleNamespace(shutdown=lambda:calls.append('shutdown')),
            store=Store(),round_id=1,is_source=source,kind='source')
        with patch.dict('sys.modules',{'shift_full.transport':SimpleNamespace(PackedGPUTransport=Transport)}):
            if fail_ack:
                with self.assertRaises(TimeoutError):ns['transfer'](obj,{})
            else:ns['transfer'](obj,{})
        return calls

    def test_source_keeps_channel_until_ack(self):
        calls=self.run_transfer(True)
        self.assertLess(calls.index('wait-ack'),calls.index('state_sent'))
        self.assertLess(calls.index('state_sent'),calls.index('shutdown'))

    def test_receiver_acknowledges_only_after_receive(self):
        calls=self.run_transfer(False)
        self.assertLess(calls.index('receive'),calls.index('ack'))
        self.assertLess(calls.index('ack'),calls.index('shutdown'))

    def test_missing_ack_cannot_report_state_sent(self):
        calls=self.run_transfer(True,True)
        self.assertNotIn('state_sent',calls)
        self.assertIn('shutdown',calls)


if __name__=='__main__':unittest.main()
