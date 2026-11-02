"""Run bounded container commands detached and retrieve their captured output."""
import json, subprocess, time, uuid

from .control import CONTAINER_ROOT

WRAPPER = r'''
import json,os,signal,subprocess,sys,time
from pathlib import Path
from shift_full.process_tree import enable_subreaper,reap_orphans
enable_subreaper()
d=Path(sys.argv[1]); timeout=float(sys.argv[2]); cmd=json.loads(sys.argv[3]); d.mkdir(parents=True,exist_ok=False)
def atomic(path,value):
 t=path.with_suffix(".tmp"); t.write_text(json.dumps(value)); os.replace(t,path)
atomic(d/"pid.json",{"wrapper_pid":os.getpid()}); started=time.monotonic(); code=127; timed=False; error=None
p=None
try:
 with (d/"stdout").open("wb") as out,(d/"stderr").open("wb") as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True); atomic(d/"pid.json",{"wrapper_pid":os.getpid(),"child_pid":p.pid})
  try: code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   timed=True; os.killpg(p.pid,signal.SIGKILL); p.wait(); code=124
except Exception as e: error=type(e).__name__; code=127
if p is not None:
 if p.poll() is None: os.killpg(p.pid,signal.SIGKILL)
 p.wait()
reap_orphans()
atomic(d/"status.json",{"exit_code":code,"timed_out":timed,"error":error,"seconds":time.monotonic()-started})
'''

POLL = r'''
import json,sys
from pathlib import Path
d=Path(sys.argv[1]); status=d/"status.json"
if not status.is_file():
 p=d/"pid.json"; print(json.dumps({"done":False,"pid":json.loads(p.read_text()) if p.exists() else None})); sys.exit(0)
s=json.loads(status.read_text()); out=(d/"stdout").read_text(errors="replace"); err=(d/"stderr").read_text(errors="replace")[-2048:]
print(json.dumps({"done":True,"status":s,"stdout":out,"stderr":err},separators=(",",":")))
'''


def run_job(session, endpoint, command, timeout=900):
    """Return captured stdout; retry only short status reads after SSH 255."""
    job = "/tmp/SHIFT-jobs/" + uuid.uuid4().hex
    launch = ["exec", "-d", "-w", CONTAINER_ROOT, endpoint.container, "python", "-c",
              WRAPPER, job, str(timeout), json.dumps(list(command), separators=(",", ":"))]
    try:
        session.docker(endpoint, launch, timeout=30)
    except subprocess.CalledProcessError as error:
        if error.returncode != 255: raise
        session.ensure_connection(endpoint, force=True)
    deadline = time.monotonic() + timeout + 60
    start_deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            raw = session.execute_endpoint(endpoint, ["python", "-c", POLL, job], timeout=45)
        except subprocess.TimeoutExpired:
            session.ensure_connection(endpoint, force=True); continue
        except subprocess.CalledProcessError as error:
            if error.returncode != 255: raise
            session.ensure_connection(endpoint, force=True); continue
        result = json.loads(next(line for line in reversed(raw.splitlines()) if line.strip()))
        if result.get("done"):
            status = result["status"]
            if status.get("exit_code") != 0:
                raise RuntimeError("remote job failed: exit=%s timeout=%s error=%s stderr=%s" %
                                   (status.get("exit_code"), status.get("timed_out"), status.get("error"), result.get("stderr", "")))
            return result.get("stdout", "")
        if result.get("pid") is None and time.monotonic() >= start_deadline:
            raise RuntimeError("remote job launch could not be confirmed")
        time.sleep(3)
    raise TimeoutError("remote job status timed out: " + job)
