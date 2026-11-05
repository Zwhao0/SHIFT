"""Serial same-session native / p2 out / p2 back 80-step timing control."""
import argparse
import json
import subprocess
import sys
import time
from scripts.sessionctl import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);a=p.parse_args()
    def call(action,*options):
        x=subprocess.run([sys.executable,'-m','scripts.sessionctl',action,'--session',a.session,*options],capture_output=True,text=True,timeout=160)
        if x.returncode:raise RuntimeError(x.stderr[-1000:])
        print(x.stdout.strip(),flush=True)
    def wait(test,seconds=200):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            s=read(a.session)
            if s.get('error'):raise RuntimeError(s['error'])
            if test(s):return s
            time.sleep(2)
        raise TimeoutError('steady control deadline')
    call('start','--seconds','900')
    wait(lambda s:s.get('ready') and min(s.get('progress',{'0':0,'1':0}).values())>=80)
    print(json.dumps(dict(native_80=True)),flush=True)
    for target,cid in [('node3','steady-out'),('node1','steady-back')]:
        call('command','--policy','v2','--target',target,'--id',cid)
        s=wait(lambda s:any(c['command']['id']==cid for c in s.get('commands',[])))
        done=s['commands'][-1]['completed_step'];wait(lambda s:min(s['progress'].values())>=done+80)
        print(json.dumps(dict(completed=cid,stable_steps=80,after=done)),flush=True)
    call('stop','--id','finish')
    s=wait(lambda s:s.get('done'));assert s['resources_clean'] and not s['cleanup_errors']
    print(json.dumps(dict(done=True,final_step=s['final_step'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        from scripts.v41_control import command
        name=sys.argv[sys.argv.index('--session')+1]
        try:command('import subprocess;subprocess.run(["systemctl","kill","--kill-who=main","--signal=SIGINT",'+repr('shift-session-'+name+'-controller')+'],capture_output=True)',True)
        except BaseException:pass
        print(type(error).__name__+': '+str(error),flush=True);raise SystemExit(1)
