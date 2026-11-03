"""CPU-only application protocol negatives; invoke with installed runtime."""
import json
from shift_full.planned_handoff import EpochGate

def main():
 rows=[]
 def reject(name,fn):
  try:fn()
  except ValueError:rows.append(dict(name=name,rejected=True))
  else:raise AssertionError(name)
 g=EpochGate();assert g.cancel()==('A','B') and g.epoch==0;rows.append(dict(name='cancel_preserves_AB',passed=True))
 reject('stale_epoch',lambda:g.activate(0,17,17,True));reject('skipped_epoch',lambda:g.activate(2,17,17,True));reject('stale_state',lambda:g.activate(1,17,16,True));reject('unready_C',lambda:g.activate(1,17,17,False))
 g.activate(1,17,17,True)
 reject('old_B_old_epoch',lambda:g.authorize('B',0));reject('old_B_new_epoch',lambda:g.authorize('B',1));reject('old_A_epoch',lambda:g.authorize('A',0));reject('cancel_after_activation',g.cancel)
 g.authorize('A',1);g.authorize('C',1)
 print(json.dumps(dict(rows=rows,limitation='Application EpochGate; no forced stale Rust RPC')))
if __name__=='__main__':main()
