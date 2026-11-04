"""Bounded epoch Lighthouses; owned by one exact systemd cgroup on role A."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--cycles',type=int,required=True);p.add_argument('--seconds',type=int,required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.directory);out.mkdir(parents=True,exist_ok=False)
    ports=[cfg['lighthouse_port']]+[31410+20*r+i for r in range(a.cycles) for i in (0,1)]
    children=[];streams=[]
    try:
        for port in ports:
            stream=(out/f'lighthouse_{port}.log').open('w');streams.append(stream)
            cmd=[cfg['python'],'/workspace/SHIFT/vendor/bin/torchft_lighthouse','--bind',f'[::]:{port}','--min_replicas','2',
                 '--heartbeat_timeout_ms','1000','--join_timeout_ms','1000','--quorum_tick_ms','100']
            children.append(subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT))
        time.sleep(.5)
        if any(c.poll() is not None for c in children):raise RuntimeError('epoch lighthouse startup failed')
        (out/'owner.json').write_text(json.dumps(dict(ports=ports,pids=[c.pid for c in children])))
        end=time.monotonic()+a.seconds
        while time.monotonic()<end and not (out/'STOP').exists():
            if any(c.poll() is not None for c in children):raise RuntimeError('epoch lighthouse exited')
            time.sleep(.1)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for stream in streams:stream.close()
        (out/'cleanup.json').write_text(json.dumps(dict(returncodes=[c.returncode for c in children])))


if __name__=='__main__':main()
