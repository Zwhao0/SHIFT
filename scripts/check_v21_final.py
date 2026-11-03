"""Capture installed-runtime protocol tests, preservation and read-only cleanup."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shift_full.control import Session,endpoints_from_env,CONTAINER_ROOT

def main():
 root=Path(__file__).resolve().parents[1];s=Session.__new__(Session);s.root=root;s._sudo_password=None;e=endpoints_from_env()['node0'];s.ensure_connection(e)
 code='''import json,subprocess,sys
out={}
for name,args in [('unit',['-m','unittest','shift_full.test_planned_protocol','-v']),('protocol',['-m','scripts.probe_planned_protocol'])]:
 p=subprocess.run([sys.executable]+args,cwd='/workspace/SHIFT',capture_output=True,text=True);out[name]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(out))
'''
 result=json.loads(s.execute_endpoint(e,[CONTAINER_ROOT+'/runtime/torch214/bin/python','-c',code]))
 assert all(r['exit_code']==0 for r in result.values()),'protocol tests failed'
 subprocess.run([sys.executable,'scripts/check_planned_cleanup.py'],cwd=root,check=True,capture_output=True)
 result['cleanup']=json.loads((root/'results/planned_handoff/cleanup.json').read_text())
 code=subprocess.run(['git','diff','--quiet','f834664e5a7ce539f0249825c55d3b8d9877772f','--','results/v1','results/v2','shift_full/train.py'],cwd=root).returncode
 result['preservation']=dict(passed=code==0,baseline='f834664e5a7ce539f0249825c55d3b8d9877772f',scope=['results/v1','results/v2','shift_full/train.py'])
 assert code==0,'published evidence or v1 entry changed'
 result['passed']=True
 out=root/'results/overlap_v21/final_checks.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,unit=result['unit']['exit_code'],protocol=result['protocol']['exit_code'],cleanup=result['cleanup']['passed'],preserved=True)))
if __name__=='__main__':main()
