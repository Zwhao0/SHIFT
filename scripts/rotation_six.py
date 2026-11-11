"""Bounded serial recovery rotations for three-node or split two-node layouts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.sessionctl import read
from scripts.v41_control import command

CASES=[('v2',1,'node3'),('v4',0,'node1'),('v4',0,'node0'),
       ('v2',1,'node1'),('v1',0,'node3'),('v1',1,'node0')]
V4_STAGES = [(0, 0, 'node3'), (1, 0, 'node0'), (0, 1, 'node1'), (1, 1, 'node3')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True)
    p.add_argument('--start',action='store_true');p.add_argument('--steady-steps',type=int,default=80)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--two-node-native',action='store_true',help='two real native recoveries on sixteen training GPUs')
    mode.add_argument('--two-node-migration',action='store_true',help='two v2 and two v4 operations on eight training GPUs')
    mode.add_argument('--v4-all-stages', action='store_true', help='dedicated three-node v4 test covering all four DP/PP pairs')
    p.add_argument('--seq-len',type=int,default=32);p.add_argument('--low-memory',action='store_true')
    a=p.parse_args();assert a.steady_steps>=10
    cases=([('v1',0,'node0'),('v1',1,'node1')] if a.two_node_native else
           [('v2',1,'node0'),('v4',0,'node1'),('v4',0,'node0'),('v2',1,'node1')] if a.two_node_migration else CASES)
    stages = [None] * len(cases)
    if a.v4_all_stages:
        cases = [('v4', replica, target) for replica, _, target in V4_STAGES]
        stages = [stage for _, stage, _ in V4_STAGES]
    two_nodes=a.two_node_native or a.two_node_migration
    def ctl(*args):
        r=subprocess.run([sys.executable,'-m','scripts.sessionctl',*args,'--session',a.session],capture_output=True,text=True,timeout=170)
        if r.returncode:raise RuntimeError('session command failed; inspect private controller log')
    def wait(predicate,seconds=240):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            state=read(a.session)
            if state.get('error'):raise RuntimeError(state['error'])
            if predicate(state):return state
            time.sleep(2)
        raise TimeoutError('session rotation progress deadline')
    evidence=Path(__file__).resolve().parents[1]/'runtime/v3-reproduce'/('rotation_'+a.session+'.json')
    result=dict(session=a.session,cases=cases,stages=stages,standby_checks=[],two_node_native=a.two_node_native,two_node_migration=a.two_node_migration)
    def standby(node):
        # PID sets stay private; only counts and equality are exported here.
        code='''import json,subprocess
from pathlib import Path
root=Path('/workspace/SHIFT/results/session_'''+a.session+''')
baseline=json.loads((root/'resources_before.json').read_text())[NODE]['company']
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines()
company={line.split(',',1)[0].strip():line.split(',',1)[1].strip() for line in rows if 'VLLM::' in line}
extra=[line for line in rows if 'VLLM::' not in line]
print(json.dumps(dict(node=NODE,company_unchanged=company==baseline,extra_gpu_count=len(extra))))
'''.replace('NODE',repr(node))
        # resources_before lives on the controller; pass its exact private row
        # to the selected host rather than assuming a shared filesystem.
        basecode='from pathlib import Path;import json;print(json.dumps(json.loads(Path('+repr('/workspace/SHIFT/results/session_'+a.session+'/resources_before.json')+').read_text())['+repr(node)+']["company"]))'
        baseline=json.loads(command(basecode,True))
        first=code.index("root=Path(");last=code.index('rows=subprocess')
        code=code[:first]+'baseline='+repr(baseline)+'\n'+code[last:]
        until=time.monotonic()+30
        while True:
            packet=json.loads(command(code,True,node))
            if not packet['company_unchanged']:raise RuntimeError('company GPU baseline changed')
            if packet['extra_gpu_count']==0:break
            if time.monotonic()>until:raise RuntimeError('standby host still owns test GPU workers')
            time.sleep(1)
        result['standby_checks'].append(packet)
        evidence.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(whole_host_standby=node,extra_gpu=0)),flush=True)
    try:
        if a.start:ctl('start','--seconds','1600','--seq-len',str(a.seq_len),
            '--replica-world-size','4' if a.two_node_migration or a.v4_all_stages else '8',
            *(['--nodes','node0','node1'] if two_nodes else []),*(['--low-memory'] if a.low_memory else []))
        wait(lambda s:s.get('ready') and min(s.get('progress',{'0':0,'1':0}).values())>=a.steady_steps)
        if not two_nodes:standby('node3')
        for index,(policy,source,target) in enumerate(cases):
            identifier=f'rotate-{index+1}-{policy}'
            stage_args = [] if stages[index] is None else ['--pp-stage', str(stages[index])]
            ctl('command','--policy',policy,'--source-replica',str(source),'--target',target,
                '--expected-epoch',str(index),'--id',identifier,*stage_args)
            state=wait(lambda s:s.get('epoch')==index+1 and not s.get('active_command'))
            record=state['commands'][-1]
            print(json.dumps(dict(policy=policy,source=source,target=target,epoch=state['epoch'],
                request_seconds=record['first_common_observation']['request_seconds'],
                fault_seconds=record['first_common_observation']['fault_seconds'])),flush=True)
            if not two_nodes and not a.v4_all_stages:
                if index==0:standby('node1')
                if index==4:standby('node0')
            goal=record['completed_step']+a.steady_steps
            wait(lambda s:min(s['progress'].values())>=goal)
        ctl('stop','--expected-epoch',str(len(cases)),'--id','rotation-finish')
        state=wait(lambda s:s.get('done') and 'resources_clean' in s)
        assert state['resources_clean'] and not state['cleanup_errors']
        if not two_nodes and not a.v4_all_stages:
            assert {x['node'] for x in result['standby_checks']}=={'node0','node1','node3'}
        result.update(done=True,epoch=state['epoch'],final_step=state['final_step'])
        evidence.write_text(json.dumps(result,indent=2)+'\n')
    except BaseException:
        unit='shift-session-'+a.session+'-controller'
        try:command('import subprocess;subprocess.run(["systemctl","kill","--kill-who=main","--signal=SIGINT",'+repr(unit)+'],capture_output=True)',True)
        except BaseException:pass
        raise


if __name__=='__main__':main()
