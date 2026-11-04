"""Deploy separately, then start one bounded server-owned planned controller."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess as sp
from shift_full.control import Session,endpoints_from_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',required=True)
    p.add_argument('--digest-engine',choices=['reference','staged'],default='reference')
    p.add_argument('--nodes',default='node0,node1,node3');p.add_argument('--variant',choices=['steady','p2'],default='p2')
    p.add_argument('--cycles',type=int,default=1);p.add_argument('--scenario',choices=['healthy','cancel','stale-state','relay-reject','slow-compute'],default='healthy')
    p.add_argument('--slow-ms',type=float,default=0);p.add_argument('--post-steps',type=int,default=10);p.add_argument('--anchor',action='store_true')
    p.add_argument('--steady-steps',type=int,default=12)
    p.add_argument('--shadow-recompute',action='store_true')
    p.add_argument('--ownership-recompute',action='store_true')
    p.add_argument('--baseline',type=Path,default=Path('results/v3_c1_20261103/resources_c1a_final.json'))
    a=p.parse_args()
    if a.shadow_recompute and not (a.anchor and a.variant=='p2' and a.cycles==1 and a.scenario=='slow-compute'):
        p.error('shadow requires one p2 slow-compute anchor diagnostic')
    if a.ownership_recompute and (a.shadow_recompute or not (a.anchor and a.variant=='p2' and a.cycles==1 and a.scenario=='slow-compute')):
        p.error('ownership first admission requires one original-anchor p2 slow-compute cycle')
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v3-planned-'+str(os.getpid())+'.sock')
    root='/workspace/SHIFT';out=root+'/results/v3_c1_20261103';unit='shift-v3-c1-'+a.suffix
    baseline=json.loads(a.baseline.read_text());baseline_path=out+'/'+a.suffix+'_baseline.json'
    args=['--suffix',a.suffix,'--nodes',a.nodes,'--variant',a.variant,'--cycles',str(a.cycles),'--scenario',a.scenario,'--slow-ms',str(a.slow_ms),'--post-steps',str(a.post_steps),'--baseline',baseline_path]
    args+=['--steady-steps',str(a.steady_steps)]
    if a.anchor:args+=['--anchor']
    if a.shadow_recompute:args+=['--shadow-recompute']
    if a.ownership_recompute:args+=['--ownership-recompute']
    cmd=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=2100','--property=KillMode=control-group',
         '--property=TimeoutStopSec=20','--property=StandardOutput=append:'+out+'/'+a.suffix+'.log','--property=StandardError=inherit',
         '--working-directory='+root,'--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=SHIFT_DIGEST_ENGINE='+a.digest_engine,'/usr/bin/python3','-m','scripts.v3_planned_remote',*args]
    code='from pathlib import Path;import subprocess,json;assert Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file(),"prepare required";assert not Path('+repr(root+'/results/v3_c1_20261103_'+a.suffix)+').exists(),"session name already used";Path('+repr(out)+').mkdir(parents=True,exist_ok=True);Path('+repr(baseline_path)+').open("x").write('+repr(json.dumps(baseline))+');subprocess.run('+repr(cmd)+',check=True);print(json.dumps(dict(unit='+repr(unit)+',pid=int(subprocess.check_output(["systemctl","show",'+repr(unit)+',"--property=MainPID","--value"],text=True)))))'
    try:
        s.ensure_connection(e);result=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        dest=s.root/'results/v3_c1_20261103';dest.mkdir(exist_ok=True)
        (dest/(a.suffix+'_owner.json')).write_text(json.dumps(result,indent=2)+'\n')
        (dest/(a.suffix+'_source_provenance.json')).write_bytes((s.root/'results/v3_a1_20261103/controller_prepare.json').read_bytes())
        print(json.dumps(result))
    finally:sp.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
