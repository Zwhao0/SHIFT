"""Protocol guards remain effective while retired communicators still exist."""
import unittest,threading
from unittest.mock import patch
from types import SimpleNamespace
from shift_full.planned_handoff import EpochGate, PlannedHandoff

class PlannedProtocolTest(unittest.TestCase):
 def handoff(self,role):
  h=PlannedHandoff.__new__(PlannedHandoff);h.role=role;h.gate=EpochGate();h.old_epoch_revoked=False;return h
 def test_fence_denies_old_manager_before_new_activation(self):
  for role in ('A','B'):
   h=self.handoff(role);old=SimpleNamespace(_shift_epoch=0);h.authorize(old)
   h.old_epoch_revoked=True
   with self.assertRaises(ValueError):h.authorize(old)
 def test_only_new_active_owners_authorized(self):
  for role in ('A','C'):
   h=self.handoff(role);h.old_epoch_revoked=True;h.gate.activate(1,7,7,True)
   h.authorize(SimpleNamespace(_shift_epoch=1))
   with self.assertRaises(ValueError):h.authorize(SimpleNamespace(_shift_epoch=0))
  h=self.handoff('B');h.gate.activate(1,7,7,True)
  with self.assertRaises(ValueError):h.authorize(SimpleNamespace(_shift_epoch=1))
 def test_stale_state_or_unready_c_cannot_activate(self):
  for epoch,step,loaded,ready in [(0,7,7,True),(2,7,7,True),(1,7,6,True),(1,7,7,False)]:
   with self.assertRaises(ValueError):EpochGate().activate(epoch,step,loaded,ready)
 def test_cleanup_attempts_all_groups_and_propagates_error(self):
  class PG:
   def __init__(self,fail=False):self.fail=fail;self._pg=object();self.aborted=False;self.closed=False
   def abort(self,errored=False):
    self.aborted=True
    if self.fail:raise RuntimeError('test cleanup error')
   def shutdown(self):self.closed=True;self._pg=None
  first,second=PG(True),PG();h=self.handoff('A');h.deferred=True;h.cleanup_thread=None;h.cleanup_cancel=threading.Event();h.device=0;h.rank=0;h.cut=7;h.errors=[];h.retired_pgs=[first,second];h.emit=lambda *a,**k:None;h.coord=SimpleNamespace(set=lambda *a:None)
  control=SimpleNamespace(check=lambda keys:True)
  with patch('shift_full.planned_handoff.torch.cuda.set_device'),patch('shift_full.planned_handoff.dist.TCPStore',return_value=control):
   h.start_retired_cleanup()
   with self.assertRaises(RuntimeError):h.finish_retired_cleanup()
  self.assertTrue(first.closed and second.closed and second.aborted)
 def test_cancel_cleanup_wait_without_hanging(self):
  h=self.handoff('A');h.deferred=True;h.cleanup_thread=None;h.cleanup_cancel=threading.Event();h.cleanup_cancel.set();h.device=0;h.rank=0;h.cut=7;h.errors=[];h.retired_pgs=[];h.emit=lambda *a,**k:None
  with patch('shift_full.planned_handoff.torch.cuda.set_device'),patch('shift_full.planned_handoff.dist.TCPStore',return_value=SimpleNamespace(check=lambda keys:False)):
   h.start_retired_cleanup();h.cleanup_thread.join(timeout=1)
  self.assertFalse(h.cleanup_thread.is_alive())
if __name__=='__main__':unittest.main()
