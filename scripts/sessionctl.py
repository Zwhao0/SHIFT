"""Start once, then send recovery/migration policies to the running SHIFT session."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.v41_control import command
from shift_full.session_control import Command

ROOT=Path(__file__).resolve().parents[1]


def transport(code):
    for attempt in range(3):
        try:return command(code,True)
        except subprocess.CalledProcessError:
            if attempt==2:raise RuntimeError('scoped SSH transport failed after three attempts') from None
            time.sleep(attempt+1)


def read(name):
    path='/workspace/SHIFT/results/session_'+name+'/summary.json'
    return json.loads(transport('from pathlib import Path;import json;p=Path('+repr(path)+');print(p.read_text() if p.exists() else json.dumps({"pending":True}))'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('start','status','command','stop'))
    p.add_argument('--session',required=True);p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--audit',action='store_true');p.add_argument('--initial-steps',type=int,default=0)
    p.add_argument('--replica-world-size',type=int,choices=(4,8),default=8)
    p.add_argument('--seq-len',type=int,default=512);p.add_argument('--low-memory',action='store_true')
    p.add_argument('--nodes',nargs='+',choices=('node0','node1','node3'),default=['node0','node1','node3'])
    p.add_argument('--policy',choices=('v1','v2','v4'));p.add_argument('--source-replica',type=int)
    p.add_argument('--target');p.add_argument('--id');p.add_argument('--expected-epoch',type=int)
    p.add_argument('--at-step',type=int);p.add_argument('--ranks')
    a=p.parse_args()
    if not a.session or not all(c.isalnum() or c in '_-' for c in a.session):p.error('invalid session')
    if a.action=='start':
        prepared=subprocess.run([sys.executable,'-m','scripts.prepare_v3_controller','--nodes',*a.nodes],cwd=ROOT,capture_output=True,text=True,timeout=150)
        if prepared.returncode:raise RuntimeError('scoped controller prepare failed')
        manifest=ROOT/'results/v3_a1_20261103/controller_prepare.json'
        (ROOT/'runtime/v3-reproduce'/('session_'+a.session+'_deployment.json')).write_bytes(manifest.read_bytes())
        unit='shift-session-'+a.session+'-controller'
        args=['systemd-run','--quiet','--collect','--unit='+unit,'--property=RuntimeMaxSec='+str(a.seconds+60),
            '--property=KillMode=control-group','--property=TimeoutStopSec=30','--working-directory=/workspace/SHIFT',
            '--property=StandardOutput=append:/workspace/SHIFT/runtime/v3-reproduce/session_'+a.session+'.controller.log',
            '--property=StandardError=inherit','--setenv=PYTHONPATH=/workspace/SHIFT','--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--setenv=CUDA_VISIBLE_DEVICES=','/workspace/SHIFT/runtime/torch214/bin/python','-m','scripts.session_remote',
            '--session',a.session,'--seconds',str(a.seconds),'--initial-steps',str(a.initial_steps)]
        args+=['--nodes',*a.nodes,'--replica-world-size',str(a.replica_world_size),'--seq-len',str(a.seq_len)]
        if a.low_memory:args+=['--low-memory']
        if a.audit:args+=['--audit']
        print(command('import subprocess,json;subprocess.run('+repr(args)+',check=True);print(json.dumps(dict(unit='+repr(unit)+')))',True));return
    state=read(a.session)
    if a.action=='status':
        print(json.dumps({k:state.get(k) for k in ('session','ready','epoch','progress','active_command','commands','rejected_commands','done','error','resources_clean')}));return
    if state.get('done') or state.get('error'):p.error('session is finished')
    epoch=a.expected_epoch if a.expected_epoch is not None else state.get('epoch',0)+bool(state.get('active_command'))
    identifier=a.id or (a.action+'-'+str(state.get('command_count',0)+1))
    payload=dict(schema=1,session=a.session,id=identifier,expected_epoch=int(epoch),at_step=a.at_step,
                 action='stop' if a.action=='stop' else 'recover' if a.policy=='v1' else 'migrate',
                 replica_world_size=state.get('configuration',{}).get('replica_world_size',8))
    if a.action=='command':
        if not a.policy or not a.target:p.error('command requires --policy and --target')
        payload.update(policy=a.policy,source_replica=a.source_replica if a.source_replica is not None else 0 if a.policy=='v4' else 1,target=a.target)
        if a.ranks:payload['ranks']=[int(x) for x in a.ranks.split(',')]
        elif a.policy=='v4':
            world=state.get('configuration',{}).get('replica_world_size',8)
            payload['ranks']=list(range(world//2,world))
    Command.from_dict(payload)
    path='/workspace/SHIFT/runtime/v3-reproduce/session_'+a.session+'/requests/'+identifier+'.json'
    code='from pathlib import Path;import json;p=Path('+repr(path)+');data='+repr(json.dumps(payload))+';p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() or json.loads(p.read_text())==json.loads(data),"command id payload conflict";tmp=p.with_suffix(".tmp");tmp.write_text(data);tmp.replace(p);print(json.dumps(dict(id='+repr(identifier)+',status="queued")))'
    print(transport(code))


if __name__=='__main__':main()
