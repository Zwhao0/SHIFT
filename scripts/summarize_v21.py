"""Validate bounded v2.1 matrices and summarize same-host timing."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_planned_handoff import analyze
from scripts.summarize_v2_final import dist,boot_ci,pct
from scripts.analyze_overlap_cycles import analyze as analyze_cycles
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/overlap_v21';OUT=ROOT/'results/v2_1'
ORDER=['p1f','p2','p2','p1f','p1f','p2','p2','p1f','p1f','p2']
def read(p):return json.loads(p.read_text())
def rows(n):return [json.loads(l) for l in (RAW/(n+'.jsonl')).read_text().splitlines() if l.strip()]

def main():
 errors=[];audit=rows('audit');timing=rows('timing');scenarios=rows('scenarios');metrics=[]
 if len(audit)!=3 or not all(r['passed'] and r['exit_code']==0 for r in audit):errors.append('need 3 independent full audits')
 if len(timing)!=10 or [r['variant'] for r in timing]!=ORDER:errors.append('need 5+5 interleaved timings')
 source=read(RAW/'deployed_protocol.json')['expected']
 for r in audit+timing:
  p=ROOT/'results'/r['name']
  if not r['passed'] or r['exit_code']:errors.append(r['name']+': execution failed');continue
  if read(p/'deployed_source.json')['expected']!=source:errors.append(r['name']+': worker source differs')
  a=analyze(p)
  if not a['all_checks_pass']:errors.append(r['name']+': analysis failed')
  if r in timing:
   if a['post_steps']!=50 or len(a['ac_intervals_seconds'])!=49:errors.append(r['name']+': not post50')
   metrics.append(dict(name=r['name'],variant=r['variant'],gap=a['last_ab_to_first_ac_seconds'],extra=a['extra_gap_seconds'],request=a['request_to_first_ac_observed_controller_seconds'],post50=a['post_ac_total_seconds'],relay_pause=a.get('relay_pause_seconds'),precopy_hidden_ranks=a.get('precopy_sender_finished_before_local_gradient_ready_ranks'),intervals=a['ac_intervals_seconds']))
 success={r['scenario']:r for r in scenarios if r['passed'] and not r['exit_code']}
 expected={'prepare-fail','ready-timeout','stale-state','relay-reject','slow-compute'}
 if set(success)!=expected:errors.append('scenario matrix incomplete')
 for name,r in success.items():
  p=ROOT/'results'/r['name'];s=read(p/('planned_status.json' if name=='slow-compute' else 'scenario_status.json'))
  if not s['passed']:errors.append(name+': did not pass')
  if read(p/'deployed_source.json')['expected']!=source:errors.append(name+': worker source differs')
 # Cycle analyzers write their own full check details.
 cycles={}
 for name in ('v2_1_cycles3_audit02','v2_1_cycles10_retry01','v2_1_lr_cycles3_audit01'):
  code=analyze_cycles(ROOT/'results'/name);cycles[name]=read(ROOT/'results'/name/'cycle_analysis.json')
  if not cycles[name]['all_checks_pass']:errors.append(name+': cycle analysis failed')
 summary={}
 for v in ('p1f','p2'):
  q=[r for r in metrics if r['variant']==v];intervals=[x for r in q for x in r['intervals']]
  summary[v]=dict(n=len(q),metrics={k:dist([r[k] for r in q]) for k in ('gap','extra','request','post50')},post49=dict(n=len(intervals),median=pct(intervals,.5),p95=pct(intervals,.95),max=max(intervals)),sessions=q)
 delta={k:boot_ci([r[k] for r in metrics if r['variant']=='p1f'],[r[k] for r in metrics if r['variant']=='p2']) for k in ('gap','extra','request','post50')} if len(metrics)==10 else {}
 for d in delta.values():d['difference']='median(p1f) - median(p2)'
 lr=read(RAW/'lr_boundary_checks.json');final_checks=read(RAW/'final_checks.json')
 if not lr['passed'] or not final_checks['passed']:errors.append('final checks failed')
 p1f_lr=read(RAW/'lr_p1f_checks.json')
 if not p1f_lr['passed']:errors.append('p1f LR compatibility failed')
 result=dict(p1f_lr_compatibility=p1f_lr,lr_compatibility=lr,final_checks=final_checks,final_worker_sha256=read(RAW/'deployed_final.json')['expected'],status='complete' if not errors else 'failed',errors=errors,audit=audit,timing=summary,p1f_minus_p2_bootstrap=delta,scenarios=success,failed_attempts=[r for r in scenarios if not r['passed']],cycles=cycles,worker_sha256=source,
  scope='Before-first-new-update rejection recovery; no arbitrary post-update rollback. Slow-compute is injected 250ms wall delay, no detector or physical GPU throttling.',method='Independent timing sessions, 5+5 interleaved, post50. Commit-gap timing uses A-host monotonic; do not compare p2 begin with v2 activation wait. Negative extra is retained and is not zero downtime.')
 OUT.mkdir(exist_ok=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],errors=errors)))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
