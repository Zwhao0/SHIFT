"""Bounded CPU-only cross-host fence timing. No training/Manager/GPU.

Server uses its own monotonic clock for broadcast -> all eight matching acks.
Client owner reserves before publishing ready; workers only parse/fence/ack.
This measures control transport, NOT total .20s activation admission.
"""
import argparse
import fcntl
import json
from pathlib import Path
import queue
import socket
import threading
import time
from shift_full.ownership_control import FenceWire
from shift_full.ownership_protocol import Cut,LocalFence


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['server','client'])
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=31780)
    p.add_argument('--nonce',required=True);p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not 1<=a.rounds<=30:raise ValueError('invalid/reused probe')
    lock=(Path(__file__).resolve().parents[1]/'runtime/v3-training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=dict(mode=a.mode,training_started=False,gpu_used=False,success=False,rounds=[],errors=[])
    wires=[];listener=None;threads=[]
    try:
        if a.mode=='server':
            listener=socket.socket();listener.settimeout(30);listener.bind((a.host,a.port));listener.listen(8)
            wires=[FenceWire(listener.accept()[0],timeout=10) for _ in range(8)]
            # Connections can arrive in any order; fixed first greeting binds rank.
            ranked={}
            for w in wires:
                hello=json.loads(w.reader.readline(4097))
                rank=hello['rank']
                if (type(rank) is not int or not 0<=rank<8 or rank in ranked or
                        hello!=dict(kind='hello',rank=rank,cut=dict(epoch=1,step=0,nonce=a.nonce))):
                    raise ValueError('bad or duplicate hello')
                ranked[rank]=w
            for i in range(a.rounds):
                cut=Cut(i+1,i,a.nonce)
                for rank,w in ranked.items():w.receive('ready',cut,rank)
                start=time.monotonic()
                for rank,w in ranked.items():w.send('fence',cut,rank)
                for rank,w in ranked.items():w.receive('ack',cut,rank)
                result['rounds'].append(dict(round=i,all8_ack_seconds=time.monotonic()-start))
                for rank,w in ranked.items():w.send('done',cut,rank)
        else:
            jobs=[queue.Queue() for _ in range(8)];errors=queue.Queue()
            def control(rank,w):
                try:
                    for _ in range(a.rounds):
                        f=jobs[rank].get(timeout=30)
                        w.send('ready',f.cut,rank)
                        w.fence_and_ack(f,rank)
                        w.receive('done',f.cut,rank)
                        jobs[rank].task_done()
                except BaseException as exc:errors.put(type(exc).__name__)
            for rank in range(8):
                w=FenceWire(socket.create_connection((a.host,a.port),timeout=10),timeout=10);wires.append(w)
                w.send('hello',Cut(1,0,a.nonce),rank)
                t=threading.Thread(target=control,args=(rank,w),daemon=True);t.start();threads.append(t)
            for i in range(a.rounds):
                start=time.monotonic();fences=[LocalFence(Cut(i+1,i,a.nonce)) for _ in range(8)]
                for f in fences:f.reserve(quorum_complete=True,state_readers_done=True)
                reserved=time.monotonic()-start
                for job,f in zip(jobs,fences):job.put(f)
                end=time.monotonic()+15
                while any(job.unfinished_tasks for job in jobs):
                    if not errors.empty():raise RuntimeError(errors.get())
                    if time.monotonic()>end:raise TimeoutError('control workers')
                    time.sleep(.0002)
                if not all(f.fenced and not f.drained and not f.closed for f in fences):
                    raise RuntimeError('ack did training drain/closure')
                result['rounds'].append(dict(round=i,reserve8_seconds=reserved))
            for t in threads:t.join(timeout=1)
            if any(t.is_alive() for t in threads):raise RuntimeError('unjoined worker')
        result['success']=True
    except BaseException as exc:
        result['errors'].append(type(exc).__name__)
        raise
    finally:
        for w in wires:w.close()
        if listener:listener.close()
        result['thread_count_after']=threading.active_count()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rounds'}),flush=True)

if __name__=='__main__':main()
