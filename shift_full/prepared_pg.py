"""One-epoch application adapter for reusing a physically prepared torchFT PG.

No Manager handles are exchanged. A new Manager owns this PG from construction.
This intentionally rejects later reconfiguration; it is not a fault-recovery PG.
"""
from datetime import timedelta
import torch
from shift_full.compat import RecoveryNCCL

class PreparedNCCL(RecoveryNCCL):
 def __init__(self,rank,global_ranks,event,timeout=timedelta(seconds=30)):
  super().__init__(timeout=timeout);self.expected_rank=rank;self.expected_global=list(global_ranks);self.emit=event;self.prepared=False;self.bound_quorum=None;self.native_creations=0
 def prepare(self,store_addr,replica_id,group_rank,group_world_size):
  if self.prepared:raise RuntimeError('candidate already prepared')
  super().configure(store_addr,replica_id,self.expected_rank,2,0,group_rank,group_world_size,self.expected_global)
  self.native_creations+=1
  value=torch.ones(1,device='cuda');self.allreduce([value],torch.distributed.AllreduceOptions()).wait();torch.cuda.current_stream().synchronize()
  if value.item()!=2:raise RuntimeError('candidate warmup failed')
  self.parent_identity=id(self.parent);self.prepared=True;self.emit('candidate_pg_ready',parent_identity=self.parent_identity,native_creations=self.native_creations)
 def configure(self,store_addr,replica_id,rank,world_size,quorum_id=None,group_rank=None,group_world_size=None,global_ranks=None):
  if not self.prepared or rank!=self.expected_rank or world_size!=2 or list(global_ranks or [])!=self.expected_global:
   raise RuntimeError('candidate membership mismatch/unprepared')
  if self.bound_quorum is not None and self.bound_quorum!=quorum_id:raise RuntimeError('prepared PG is restricted to one epoch/quorum')
  if id(self.parent)!=self.parent_identity:raise RuntimeError('candidate parent changed')
  self.bound_quorum=quorum_id
  self._replica_id=replica_id;self._quorum_id=quorum_id;self._group_rank=group_rank;self._group_world_size=group_world_size;self._global_ranks=global_ranks;self._rank=rank
  self.emit('candidate_pg_reused',quorum_id=quorum_id,parent_identity=id(self.parent),native_creations=self.native_creations,global_ranks=global_ranks)
