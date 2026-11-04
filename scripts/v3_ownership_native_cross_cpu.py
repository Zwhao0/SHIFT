"""Server-owned bounded three-host CPU FT lifecycle probe, no GPU/training."""
import argparse
from dataclasses import replace
from datetime import timedelta
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suffix',required=True)
    parser.add_argument('--early',action='store_true');a=parser.parse_args()
    if not a.suffix or not all(c.isalnum() or c in '_-' for c in a.suffix):parser.error('invalid suffix')
    import torch
    import torch.distributed as dist
    from torchft._torchft import LighthouseServer
    from scripts import v1_acceptance_remote as remote
    from scripts.v3_ownership_native_cpu import worker
    from shift_full.control import endpoints_from_env,server_password
    root=Path(__file__).resolve().parents[1]
    lock=(root/'runtime/v3-training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote.PASSWORD=server_password(remote.PRIVATE/'credentials');(remote.PRIVATE/'credentials').unlink()
    session=remote.ServerRotation.__new__(remote.ServerRotation);session.root=root;session._sudo_password=remote.PASSWORD
    stage='p10b4' if a.early else 'p10b3'
    eps={k:replace(e,socket=f'ssh-{stage}-{a.suffix}-{k}.sock') for k,e in endpoints_from_env().items()}
    roles=dict(A=eps['node0'],B=eps['node3'],C=eps['node1'])
    base=root/f'results/v3_{stage}_{a.suffix}';base.mkdir(exist_ok=False)
    status=dict(done=False,success=False,training_started=False,gpu_used=False,injections=0,started=time.time(),cases={})
    def save():(base/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    units=[];processes=[];lights=[];stores=[]
    save()
    def sudo(e,code):return session.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True)
    try:
        manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for d in ('shift_full','scripts') for p in (root/d).glob('*.py')}
        (base/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for e in eps.values():session.ensure_connection(e)
        # Only C lacks an original anchor. Transfer CPU guard shape/metadata
        # fixtures, with genuine small buffers/RNG/param_groups/scheduler.
        fixture=base/'guard_fixture';fixture.mkdir();inventory={}
        def proxy(v):
            if isinstance(v,torch.Tensor):return torch.zeros((),dtype=v.dtype).expand(v.shape)
            if isinstance(v,dict):return {k:proxy(x) for k,x in v.items()}
            if isinstance(v,list):return [proxy(x) for x in v]
            if isinstance(v,tuple):return tuple(proxy(x) for x in v)
            return v
        for rank in range(8):
            source=root/'runtime/v3-reference-num01'/f'anchor_r{rank}.pt'
            payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
            payload=dict(payload,model=proxy(payload['model']),optimizer=proxy(payload['optimizer']))
            target=fixture/source.name;torch.save(payload,target)
            inventory[rank]=dict(source=str(source),source_bytes=source.stat().st_size,
                source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
                fixture_bytes=target.stat().st_size,fixture_sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest())
        (fixture/'provenance.json').write_text(json.dumps(inventory,indent=2)+'\n')
        bundle=base/'guard_fixture.tgz'
        with tarfile.open(bundle,'w:gz') as t:t.add(fixture,arcname='guard_fixture')
        cpath=roles['C'].host_root+'/results/'+base.name
        transfer='/tmp/shift-p10b3-'+a.suffix+'.tgz'
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ControlPath='+str(root/'tmp'/roles['C'].socket),
            str(bundle),'test@'+roles['C'].ip+':'+transfer],check=True,capture_output=True,timeout=30)
        sudo(roles['C'],'from pathlib import Path;import tarfile;d=Path('+repr(cpath)+');d.mkdir(exist_ok=False);'
             'p=Path('+repr(transfer)+');t=tarfile.open(p);t.extractall(d,filter="data");t.close();p.unlink()')
        cases=([('success',False),('future_reject',False),('stale_k',False),('guard_fallback',True)]
               if a.early else [('success',False),('guard_fallback',True)])
        for name,failure in cases:
            out=base/name;out.mkdir();processes=[];lights=[];stores=[]
            coord=dist.TCPStore(roles['A'].ip,0,None,True,wait_for_workers=False)
            cfg=dict(out=str(out),world=8,failure=failure,coord=coord.port,control_host=roles['A'].ip,
                hosts={r:e.ip for r,e in roles.items()},offset=500,nonce=a.suffix+'-'+name,
                lh={},stores={},anchor_guards=True,early=a.early,early_case=name)
            for epoch,members in [('old','AB'),('AC','AC'),('AB','AB')]:
                lh=LighthouseServer(bind=roles['A'].ip+':0',min_replicas=2,
                    join_timeout_ms=1000,quorum_tick_ms=100,heartbeat_timeout_ms=1000)
                lights.append(lh);cfg['lh'][epoch]='http://'+roles['A'].ip+':'+lh.address().rsplit(':',1)[1]
                cfg['stores'][epoch]={}
                for r in members:
                    cfg['stores'][epoch][r]={'old':32020,'AC':32021,'AB':32022}[epoch]
                    if r=='A':stores.append(dist.TCPStore(roles[r].ip,cfg['stores'][epoch][r],None,True,wait_for_workers=False))
            coord.set('cut','6');coord.set('ownership_nonce',cfg['nonce'])
            (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
            for r in 'BC':
                e=roles[r];d=e.host_root+'/results/'+base.name+'/'+name;unit=f'shift-v3-{stage}-{a.suffix}-{name}-{r}'
                cmd=['systemd-run','--quiet','--unit='+unit,'--property=RuntimeMaxSec=125','--property=KillMode=control-group',
                    '--property=TimeoutStopSec=10','--property=StandardOutput=append:'+d+'.log','--property=StandardError=inherit',
                    '--working-directory='+e.host_root,'--setenv=PYTHONPATH='+e.host_root,'--setenv=CUDA_VISIBLE_DEVICES=',
                    '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
                    '--setenv=TOKIO_WORKER_THREADS=2','--setenv=SHIFT_DIGEST_ENGINE=staged']
                if r=='C':cmd+=['--setenv=SHIFT_OWNERSHIP_GUARD_FIXTURE='+cpath+'/guard_fixture']
                cmd += [e.host_root+'/runtime/torch214/bin/python','-m','scripts.v3_ownership_native_cpu','--role',r,'--config',d+'/config.json']
                code='from pathlib import Path;import subprocess;d=Path('+repr(d)+');d.mkdir(parents=True,exist_ok=False);(d/"config.json").write_text('+repr(json.dumps(cfg))+');subprocess.run('+repr(cmd)+',check=True)'
                sudo(e,code);units.append((e,unit,d))
            ctx=mp.get_context('spawn');processes=[ctx.Process(target=worker,args=('A',r,cfg)) for r in range(8)]
            for p in processes:p.start()
            deadline=time.monotonic()+105
            for p in processes:p.join(max(0,deadline-time.monotonic()))
            result=dict(A=dict(exit_codes=[p.exitcode for p in processes],success=all(p.exitcode==0 for p in processes)),remote={})
            # Wait for each role supervisor's terminal evidence, without keeping
            # a desktop SSH command or controller-owned shell as its lifetime.
            for r in 'BC':
                e=roles[r];d=e.host_root+'/results/'+base.name+'/'+name
                end=time.monotonic()+15
                while True:
                    code='from pathlib import Path;import json;p=Path('+repr(d+'/role_'+r+'.json')+');print(p.read_text() if p.exists() else "null")'
                    row=json.loads(sudo(e,code))
                    if row is not None:break
                    if time.monotonic()>end:raise TimeoutError('remote role terminal evidence')
                    time.sleep(.3)
                result['remote'][r]=row
            result['success']=result['A']['success'] and all(r['success'] for r in result['remote'].values())
            status['cases'][name]=result;save()
            for lh in lights:lh.shutdown()
            lights=[];stores=[];del coord
            if not result['success']:break
        status['success']=len(status['cases'])==len(cases) and all(c['success'] for c in status['cases'].values())
    except BaseException as exc:
        status.update(error=type(exc).__name__,traceback=traceback.format_exc())
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:p.join(5)
        for lh in lights:lh.shutdown()
        status['remote_units']=[]
        for e,unit,d in units:
            try:
                code='import subprocess,json;from pathlib import Path;stop=subprocess.run(["systemctl","stop",'+repr(unit)+'],capture_output=True,text=True);show=subprocess.run(["systemctl","show",'+repr(unit)+',"--property=ActiveState,SubState,Result,MainPID,LoadState","--no-pager"],capture_output=True,text=True);p=Path('+repr(d)+');print(json.dumps(dict(files={str(f.relative_to(p)):f.read_text() for f in p.glob("*") if f.is_file()},log=p.with_suffix(".log").read_text(),unit=show.stdout,unit_stderr=show.stderr,unit_exit=show.returncode,stop_exit=stop.returncode,stop_stderr=stop.stderr)))'
                raw=json.loads(sudo(e,code));(base/(unit+'.json')).write_text(json.dumps(raw,indent=2)+'\n')
                status['remote_units'].append(dict(node=e.physical_node,unit=unit,status=raw['unit']))
            except BaseException as exc:status.setdefault('cleanup_errors',[]).append(type(exc).__name__)
        for e in eps.values():
            subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)
        status.update(done=True,finished=time.time());save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
