"""Bounded live acceptance driver; no second training start between commands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.sessionctl import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--start',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--two-each',action='store_true');p.add_argument('--steady-steps',type=int,choices=(0,80),default=0);args=p.parse_args()
    def call(*options):
        cmd=[sys.executable,'-m','scripts.sessionctl',*options,'--session',args.session]
        x=subprocess.run(cmd,capture_output=True,text=True,timeout=160)
        if x.returncode:raise RuntimeError(x.stderr[-2000:])
        print(x.stdout.strip(),flush=True)
    def wait(predicate,seconds=160):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            s=read(args.session)
            if s.get('error'):raise RuntimeError(s['error'])
            if predicate(s):return s
            time.sleep(2)
        raise TimeoutError('live acceptance progress deadline')
    if args.start:call('start','--seconds','900',*(['--audit'] if args.audit else []))
    wait(lambda s:s.get('ready') and min(s.get('progress',{'0':0,'1':0}).values())>=max(6,args.steady_steps))
    cases=[('v2','node3','whole'),('v4','node1','partial'),('v1','node3','recover')]
    if args.two_each:cases += [('v4','node0','partial-return'),('v2','node1','whole-return'),('v1','node3','recover-second')]
    for policy,target,identifier in cases:
        state=read(args.session)
        if any(c['command']['id']==identifier for c in state.get('commands',[])):continue
        call('command','--policy',policy,'--target',target,'--id',identifier)
        result=wait(lambda s:any(c['command']['id']==identifier for c in s.get('commands',[])))
        print(json.dumps(dict(completed=policy,epoch=result['epoch'],progress=result['progress'],seconds=result['commands'][-1]['controller_seconds'])),flush=True)
        if policy=='v2' and args.steady_steps:
            goal=result['commands'][-1]['completed_step']+args.steady_steps
            wait(lambda s:min(s['progress'].values())>=goal,seconds=200)
            print(json.dumps(dict(steady_after=identifier,steps=args.steady_steps)),flush=True)
    call('stop','--id','finish')
    result=wait(lambda s:s.get('done'))
    assert result['resources_clean'] and not result['cleanup_errors']
    print(json.dumps(dict(done=True,epoch=result['epoch'],final_step=result['final_step'],resources_clean=True)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        from scripts.v41_control import command
        session=sys.argv[sys.argv.index('--session')+1]
        unit='shift-session-'+session+'-controller'
        try:command('import subprocess;subprocess.run(["systemctl","kill","--kill-who=main","--signal=SIGINT",'+repr(unit)+'],capture_output=True)',True)
        except BaseException:pass
        print(type(error).__name__+': '+str(error),flush=True)
        raise SystemExit(1)
