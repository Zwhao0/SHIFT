"""Read-only deployed source equality; no GPU work and no credential output."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,endpoints_from_env,CONTAINER_ROOT

def verify(out):
 root=Path(__file__).resolve().parents[1];s=Session.__new__(Session);s.root=root;s._sudo_password=None
 names=['shift_full/'+n+'.py' for n in ('planned_train','planned_handoff','overlap_handoff','overlap_probe','prepared_pg','transport')]
 expected={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names};nodes={}
 code='import hashlib,json;from pathlib import Path;print(json.dumps({n:hashlib.sha256(Path('+repr(CONTAINER_ROOT)+',n).read_bytes()).hexdigest() for n in '+repr(names)+'}))'
 for name,e in endpoints_from_env().items():
  s.ensure_connection(e);actual=json.loads(s.execute_endpoint(e,['python','-c',code]));nodes[name]=dict(passed=actual==expected,sha256=actual)
 result=dict(passed=all(n['passed'] for n in nodes.values()),expected=expected,nodes=nodes)
 Path(out).write_text(json.dumps(result,indent=2)+'\n')
 if not result['passed']:raise RuntimeError('deployed source differs; deploy before experiment')
 return result
if __name__=='__main__':
 r=verify(sys.argv[1]);print(json.dumps(dict(passed=r['passed'],nodes=len(r['nodes']),files=len(r['expected']))))
