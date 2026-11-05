"""Scoped CPU-only status/audit export for the v4 recovery optimization round."""
import argparse
import json
from pathlib import Path
import subprocess
from scripts.v41_control import command

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/v4_recovery_20261105'


def inspect(name,audit=False):
    if not name.startswith('v4_') or not all(c.isalnum() or c in '_-' for c in name):raise ValueError('invalid run name')
    code='''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/workspace/SHIFT')
p=Path('/workspace/SHIFT/results')/NAME
s=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else {}
if not AUDIT:
 print(json.dumps({k:s.get(k) for k in ('name','functional_pass','resources_clean','error','final_step','decisions')}))
else:
 from scripts.audit_v41_remote import v4,clean,stats
 a=v4(p,s);a['resources']=clean(p)
 (p/'v41_audit.json').write_text(json.dumps(a))
 from scripts.quick_v4_timing import _load_worker_events
 es=_load_worker_events(p) if s.get('transactional_switch') and not s.get('reference_only') else {}
 a['stage_durations']={kind:stats([e['seconds'] for rows in es.values() for e in rows if e['kind']==kind and 'seconds' in e]) for kind in sorted({e['kind'] for rows in es.values() for e in rows if 'seconds' in e})}
 a['state_hash_calls']={key:dict(total=sum(e['kind']=='state_hash' for e in rows),timed=sum(e['kind']=='state_hash' and not e.get('final_audit') for e in rows),final_audit=sum(e['kind']=='state_hash' and e.get('final_audit') is True for e in rows)) for key,rows in es.items()}
 groups={}
 for rows in es.values():
  final=False
  for e in rows:
   if e['kind']=='final_audit_begin':final=True
   if 'seconds' not in e or e['kind']=='commit':continue
   key='/'.join((e['role'],e['kind'],str(e.get('epoch','')), 'final' if final else 'training'))
   groups.setdefault(key,[]).append(e['seconds'])
 a['role_stage_durations']={key:stats(values) for key,values in groups.items()}
 a0=es.get('g0_A0',[])
 cuts=[e['step'] for e in a0 if e['kind']=='boundary_selected']
 a['a0_boundary_commit_gaps_seconds']={str(cut):next(e['monotonic'] for e in a0 if e['kind']=='commit' and e['step']==cut+1)-next(e['monotonic'] for e in a0 if e['kind']=='commit' and e['step']==cut) for cut in cuts}
 a['a0_max_commit_gap_seconds']=max((right['monotonic']-left['monotonic'] for left,right in zip([e for e in a0 if e['kind']=='commit'],[e for e in a0 if e['kind']=='commit'][1:])),default=None)
 signatures=a.pop('final_signatures',{})
 reference=json.loads((p/'reference_signatures.json').read_text()) if s.get('reference_only') and (p/'reference_signatures.json').exists() else None
 keys=('seed','initialization_seed','performance_no_audit','boundary_validation','digest_engine','plan','reference_only','reference_run','final_reference_run','post_layout','candidate_request_step','slow_ms','restore_deadline','stall_restore_rank','fail_restore_rank','source_stall_round','cancel_round','source_deadline','decisions','role_hosts')
 print(json.dumps(dict(name=NAME,parameters={k:s[k] for k in keys if k in s},independent_audit=a,source_sha256=s.get('source_sha256'),selected_signatures=signatures,reference_signatures=reference)))
'''.replace('NAME',repr(name)).replace('AUDIT',repr(audit))
    try:x=json.loads(command(code,True))
    except subprocess.CalledProcessError as error:
        # Do not print argv/credential endpoints from CalledProcessError.
        raise RuntimeError((error.stderr or '')[-5000:]) from None
    if audit:
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        path=EVIDENCE/'summary.json';summary=json.loads(path.read_text()) if path.exists() else dict(base_commit='e4d4f51eef0ab7343ccc7af750b17a42644bfa25',runs=[])
        signatures=x.pop('selected_signatures');reference=x.pop('reference_signatures')
        sp=EVIDENCE/'selected_signatures.json';selected=json.loads(sp.read_text()) if sp.exists() else {}
        selected[name]=dict(final=signatures,**({'reference':reference} if reference else {}));sp.write_text(json.dumps(selected,separators=(',',':'))+'\n')
        summary['runs']=[r for r in summary['runs'] if r['name']!=name]+[x]
        path.write_text(json.dumps(summary,indent=2)+'\n')
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name');p.add_argument('--audit',action='store_true');args=p.parse_args()
    x=inspect(args.name,args.audit)
    if args.audit:
        a=x['independent_audit'];print(json.dumps(dict(name=args.name,pass_all=a['pass_all'],timing=a.get('timing'),stages=a.get('stage_durations'))))
    else:print(json.dumps(x))


if __name__=='__main__':main()
