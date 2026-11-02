"""Trace one worst step-3 AdamW model coordinate from existing CPU snapshots."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shift_full.control import Session, endpoints_from_env
from shift_full.remote_job import run_job

CODE=r'''import json,math,sys,torch
paths=sys.argv[1:5]; name=sys.argv[5]; index=int(sys.argv[6])
def load(path): return torch.load(path,map_location="cpu",weights_only=False,mmap=True)
def val(x): return float(x.detach().reshape(-1)[0].item()) if isinstance(x,torch.Tensor) else float(x)
def one(p2,p3):
 a,b=load(p2),load(p3); names=list(a["gradients"]); assert name in names and names==list(b["gradients"])
 ids=[i for g in a["optimizer"]["param_groups"] for i in g["params"]]; assert len(ids)==len(names)
 pos=names.index(name); pid=ids[pos]; group=next(g for g in a["optimizer"]["param_groups"] if pid in g["params"])
 sa=a["optimizer"]["state"][pid]; sb=b["optimizer"]["state"][pid]; p_a=a["model"][name]; p_b=b["model"][name]
 assert p_a.shape==p_b.shape==a["gradients"][name].shape==b["gradients"][name].shape and index<p_a.numel()
 def at(t): return val(t.detach().reshape(-1)[index])
 p2v,p3v,g2,g3=at(p_a),at(p_b),at(a["gradients"][name]),at(b["gradients"][name])
 m2,m3=at(sa["exp_avg"]),at(sb["exp_avg"]); v2,v3=at(sa["exp_avg_sq"]),at(sb["exp_avg_sq"])
 beta1,beta2=group["betas"]; lr=float(group["lr"]); eps=float(group["eps"]); wd=float(group["weight_decay"]); step=int(val(sb["step"]))
 mp=beta1*m2+(1-beta1)*g3; vp=beta2*v2+(1-beta2)*g3*g3
 def predict(m,v): return p2v*(1-lr*wd)-lr*(m/(1-beta1**step))/(math.sqrt(v/(1-beta2**step))+eps)
 return dict(step=step,p2=p2v,p3=p3v,g2=g2,g3=g3,m2=m2,m3=m3,m3_pred=mp,v2=v2,v3=v3,v3_pred=vp,
  m3_recurrence_residual=m3-mp,v3_recurrence_residual=v3-vp,p3_pred_observed_moments=predict(m3,v3),p3_pred_recurrent_moments=predict(mp,vp),
  actual_update=p3v-p2v,lr=lr,betas=[beta1,beta2],eps=eps,weight_decay=wd,adam_step=step)
out={"flat":one(paths[0],paths[1]),"shift":one(paths[2],paths[3])}
print(json.dumps(out,separators=(",",":")))
'''

def main():
    report=json.loads((ROOT/'results/v1/strict_comparison.json').read_text()); candidates=[]
    for rank,row in report['byrank']['3'].items():
        for d in row['by_category']['model']['top_differences']:
            if d.get('path','').startswith("model['") and isinstance(d.get('first_largest_abs_flat_index'),int):
                candidates.append((float(d.get('max_abs') or 0),int(rank),d))
    if not candidates: raise SystemExit('No step-3 model coordinate in comparison report')
    _,rank,best=max(candidates,key=lambda x:x[0]); name=best['path'][7:-2]; idx=best['first_largest_abs_flat_index']
    paths=[f'/workspace/SHIFT/results/v1_strict_{m}_01/reference_step{s}_r{rank}.pt' for m in ('flat','shift') for s in (2,3)]
    probe=Session.__new__(Session); probe.root=ROOT; probe._sudo_password=None; ep=endpoints_from_env()['node0']
    probe.ensure_connection(ep)
    raw=run_job(probe,ep,['/workspace/SHIFT/runtime/torch214/bin/python','-c',CODE,*paths,name,str(idx)],timeout=900)
    values=json.loads(raw.strip().splitlines()[-1]); f,s=values['flat'],values['shift']
    out={'source':'results/v1/strict_comparison.json','selected':{'step':3,'rank':rank,'parameter':name,'flat_index':idx,'max_abs':best['max_abs']},'coordinate':values,
         'cross_mode_differences':{k:f[k]-s[k] for k in ('p2','p3','g2','g3','m2','m3','v2','v3')},
         'interpretation':'Scalar FP64 recurrence and AdamW formula diagnostics from saved FP32 values; recurrence residuals include FP32 update rounding. This isolates one coordinate and does not establish a global cause.'}
    dest=ROOT/'results/v1/adam_coordinate.json'; dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'rank':rank,'parameter':name,'index':idx,'output':str(dest)}))

if __name__=='__main__': main()
