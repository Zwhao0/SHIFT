"""Launch one prepared, bounded V1/V2 quick controller on node0."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from shift_full.control import Session, endpoints_from_env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('version',choices=['v1','v2'])
    p.add_argument('--suffix',required=True)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--trigger',required=True,type=int)
    p.add_argument('--post-steps',default=5,type=int)
    p.add_argument('--cold-launcher',action='store_true')
    p.add_argument('--communication-timeout',type=float)
    p.add_argument('--digest-engine',choices=['reference','staged'],default='reference')
    p.add_argument('--overlap-cpu-model-load',action='store_true')
    p.add_argument('--audit-every-step',action='store_true')
    p.add_argument('--performance-no-audit',action='store_true')
    p.add_argument('--slow-ms',type=float,default=0)
    p.add_argument('--variant',choices=['p1f','p2'],default='p1f')
    a=p.parse_args()
    if a.slow_ms < 0 or (a.slow_ms and a.version!='v2'):p.error('slow-ms requires v2 and nonnegative delay')
    if a.audit_every_step and a.version != 'v2':p.error('audit-every-step applies to v2 only')
    if a.variant!='p1f' and a.version != 'v2':p.error('planned variant applies to v2 only')
    if a.performance_no_audit and a.audit_every_step:p.error('choose migration auditing or performance timing')
    if a.version != 'v1' and a.communication_timeout is not None:
        p.error('communication-timeout override applies to v1 only')
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):p.error('invalid suffix')
    if min(a.trigger,a.post_steps)<1:p.error('positive step counts required')
    s=Session.__new__(Session);s.root=Path(__file__).resolve().parents[1];s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket=f'ssh-quick-start-{os.getpid()}.sock')
    root='/workspace/SHIFT';unit='shift-quick-'+a.suffix+'-controller'
    args=['--suffix',a.suffix,'--seed',str(a.seed),'--trigger',str(a.trigger),'--post-steps',str(a.post_steps)]
    if a.cold_launcher:args.append('--cold-launcher')
    if a.audit_every_step:args.append('--audit-every-step')
    if a.performance_no_audit:args.append('--performance-no-audit')
    if a.version == 'v2':args+=['--variant',a.variant,'--slow-ms',str(a.slow_ms)]
    if a.communication_timeout is not None:
        if not 0 < a.communication_timeout <= 90:p.error('communication-timeout must be in (0, 90]')
        args+=['--communication-timeout',str(a.communication_timeout)]
    command=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec=1020',
        '--property=KillMode=control-group','--property=TimeoutStopSec=20','--working-directory='+root,
        '--property=StandardOutput=append:'+root+'/runtime/v3-reproduce/'+a.suffix+'.controller.log',
        '--property=StandardError=inherit','--setenv=PYTHONPATH='+root,'--setenv=PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/python3','-m','scripts.quick_'+a.version+'_remote',*args]
    if a.version == 'v1':
        command[command.index('/usr/bin/python3'):command.index('/usr/bin/python3')]=[
            '--setenv=SHIFT_DIGEST_ENGINE='+a.digest_engine,
            '--setenv=SHIFT_OVERLAP_CPU_MODEL_LOAD='+('1' if a.overlap_cpu_model_load else '0')]
    else:
        command.insert(command.index('/usr/bin/python3'), '--setenv=SHIFT_DIGEST_ENGINE='+a.digest_engine)
        command.insert(command.index('/usr/bin/python3'), '--setenv=SHIFT_OVERLAP_CPU_MODEL_LOAD='+('1' if a.overlap_cpu_model_load else '0'))
    code='from pathlib import Path;import subprocess,json;assert Path('+repr(root+'/runtime/v1-controller/credentials')+').is_file(),"prepare required";subprocess.run('+repr(command)+',check=True);print(json.dumps(dict(unit='+repr(unit)+')))'
    try:
        s.ensure_connection(e)
        print(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
    finally:
        subprocess.run(['ssh','-S',str(s.root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
