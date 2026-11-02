"""Move the 24 allowlisted trajectory snapshots from node0 to node1."""

import json
import time

from .control import CONTAINER_ROOT
from .remote_job import run_job

PORT = 31587
SESSIONS = {"v1_trajectory_flat_01", "v1_trajectory_shift_01"}
FILES = tuple(f"reference_step{s}_r{r}.pt" for s in (2, 3, 4) for r in range(8))

SOURCE_CODE = r'''
import hashlib,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve(); names=json.loads(sys.argv[2]); pid=Path(sys.argv[3]); out=[]; total=time.monotonic()
if pid.exists(): raise SystemExit("transfer pid file already exists")
for n in names:
 p=r/n
 if p.parent.resolve()!=r or p.is_symlink() or not p.is_file(): raise SystemExit("missing or unsafe source: "+n)
 t=time.monotonic(); h=hashlib.sha256(); size=0
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b); size+=len(b)
 if size!=p.stat().st_size or not size: raise SystemExit("source changed or empty: "+n)
 out.append({"name":n,"size_bytes":size,"sha256":h.hexdigest(),"hash_seconds":time.monotonic()-t})
 print(".",flush=True)
print(json.dumps({"files":out,"total_hash_seconds":time.monotonic()-total},separators=(",",":")))
'''
DEST_STATE_CODE = r'''
import hashlib,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]); names=json.loads(sys.argv[2]); finals=[r/n for n in names]; parts=[r/(n+".partial") for n in names]
if not r.is_dir(): raise SystemExit("destination session directory missing")
exists=lambda p:p.exists() or p.is_symlink()
if not any(map(exists,finals+parts)): print(json.dumps({"state":"empty"},separators=(",",":"))); sys.exit(0)
if not all(p.is_file() and not p.is_symlink() for p in finals) or any(map(exists,parts)):
 print(json.dumps({"state":"mixed"},separators=(",",":"))); sys.exit(0)
out=[]
for n,p in zip(names,finals):
 t=time.monotonic(); h=hashlib.sha256(); size=0
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b); size+=len(b)
 out.append({"name":n,"size_bytes":size,"sha256":h.hexdigest(),"existing_verify_seconds":time.monotonic()-t}); print(".",flush=True)
print(json.dumps({"state":"complete","files":out},separators=(",",":")))
'''
SERVER_CODE = r'''
import http.server,json,os,signal,sys,threading
from pathlib import Path
session,bind,port,root_arg,pidfile,names_json=sys.argv[1:]
root=Path(root_arg).resolve(); allowed=set(json.loads(names_json))
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  n=self.path.partition("?")[0].lstrip("/"); p=root/n
  if "/" in n or n not in allowed or p.parent.resolve()!=root or p.is_symlink() or not p.is_file():
   self.send_error(404); return
  self.send_response(200); self.send_header("Content-Length",str(p.stat().st_size)); self.end_headers()
  try:
   with p.open("rb") as f:
    for b in iter(lambda:f.read(1024*1024),b""): self.wfile.write(b)
  except (BrokenPipeError,ConnectionResetError,FileNotFoundError): pass
 def log_message(self,*_): pass
class Server(http.server.ThreadingHTTPServer): daemon_threads=True
s=Server((bind,int(port)),Handler); pid=os.getpid()
fd=os.open(pidfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,"w") as f: f.write(str(pid)); f.flush(); os.fsync(f.fileno())
signal.signal(signal.SIGTERM,lambda *_:threading.Thread(target=s.shutdown,daemon=True).start())
try: s.serve_forever(poll_interval=.2)
finally:
 s.server_close()
 try:
  if Path(pidfile).read_text().strip()==str(pid): Path(pidfile).unlink()
 except FileNotFoundError: pass
'''
CONTROL_CODE = r'''
import json,os,signal,subprocess,sys,time
action,pidfile,path=sys.argv[1:]
def proc(pid,field):
 try:
  value=open("/proc/%d/%s"%(pid,field),"rb").read()
  return value.decode() if isinstance(value,bytes) else value
 except OSError: return subprocess.check_output(["ps","-p",str(pid),"-o",("command=" if field=="cmdline" else "stat=")],text=True)
if action=="wait":
 end=time.monotonic()+20
 while time.monotonic()<end:
  try:
   pid=int(open(pidfile).read()); cmd=proc(pid,"cmdline")
   if path in cmd: print(json.dumps({"pid":pid},separators=(",",":"))); sys.exit(0)
  except (OSError,ValueError): pass
  time.sleep(.1)
 raise SystemExit("HTTP server did not become ready")
if not os.path.isfile(pidfile): raise SystemExit("HTTP server pid file missing")
pid=int(open(pidfile).read()); cmd=proc(pid,"cmdline")
if path not in cmd: raise SystemExit("saved pid command line does not match session path")
os.kill(pid,signal.SIGTERM); end=time.monotonic()+15
while time.monotonic()<end:
 try:
  state=proc(pid,"stat").strip(); state=state.rsplit(")",1)[1].split()[0] if ")" in state else state.split()[0]
  if state=="Z": break
  os.kill(pid,0)
 except (ProcessLookupError,FileNotFoundError): break
 time.sleep(.1)
else: raise SystemExit("HTTP server did not stop")
try:
 if open(pidfile).read().strip()==str(pid): os.unlink(pidfile)
except FileNotFoundError: pass
print(json.dumps({"stopped":True,"pid":pid},separators=(",",":")))
'''
DOWNLOAD_CODE = r'''
import hashlib,json,os,sys,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
r=Path(sys.argv[1]); base=sys.argv[2].rstrip("/"); manifest=json.loads(sys.argv[3]); opener=build_opener(ProxyHandler({})); out=[]
for x in manifest:
 n=x["name"]; dst=r/n; part=r/(n+".partial")
 if dst.exists() or dst.is_symlink() or part.exists() or part.is_symlink(): raise SystemExit("refusing existing target: "+n)
 t=time.monotonic(); h=hashlib.sha256(); size=0
 with opener.open(base+"/"+n,timeout=900) as src, part.open("xb") as f:
  for b in iter(lambda:src.read(4*1024*1024),b""): f.write(b); h.update(b); size+=len(b)
  f.flush(); os.fsync(f.fileno())
 transfer=time.monotonic()-t; digest=h.hexdigest()
 if size!=x["size_bytes"] or digest!=x["sha256"]: raise SystemExit("download hash/size mismatch: "+n)
 os.link(str(part),str(dst)); os.unlink(str(part))
 fd=os.open(str(r),os.O_RDONLY); os.fsync(fd); os.close(fd)
 out.append({"name":n,"size_bytes":size,"sha256":digest,"transfer_hash_seconds":transfer})
 print(".",flush=True)
print(json.dumps({"files":out},separators=(",",":")))
'''
DELETE_CODE = r'''
import hashlib,json,os,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve(); manifest=json.loads(sys.argv[2])
for x in manifest:
 p=r/x["name"]
 if p.parent.resolve()!=r or p.is_symlink() or not p.is_file(): raise SystemExit("source changed: "+x["name"])
 h=hashlib.sha256(); size=0
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b); size+=len(b)
 if size!=x["size_bytes"] or h.hexdigest()!=x["sha256"]: raise SystemExit("source hash changed: "+x["name"])
 print(".",flush=True)
for x in manifest: (r/x["name"]).unlink(); print(".",flush=True)
fd=os.open(str(r),os.O_RDONLY); os.fsync(fd); os.close(fd)
print(json.dumps({"deleted":[x["name"] for x in manifest]},separators=(",",":")))
'''
def _run(probe, endpoint, code, args, timeout):
    raw = run_job(probe, endpoint, ["python", "-c", code, *args], timeout)
    result = json.loads(next(line for line in reversed(raw.splitlines()) if line.strip()))
    if not isinstance(result, dict):
        raise RuntimeError("remote transfer step returned invalid JSON")
    return result


def move(probe, name):
    """Move only the allowlisted session snapshots; retain sources on failure."""
    if (not isinstance(name, str) or not name.startswith("v1_trajectory_")
            or name not in SESSIONS or getattr(probe, "name", None) != name):
        raise ValueError("session is not an allowlisted v1 trajectory")
    started = time.monotonic()
    node0, node1 = probe.endpoints["node0"], probe.endpoints["node1"]
    src = dst = f"{CONTAINER_ROOT}/results/{name}"
    pidfile = f"/tmp/SHIFT-{name}-transfer.pid"
    names = json.dumps(FILES, separators=(",", ":"))
    source = _run(probe, node0, SOURCE_CODE, [src, names, pidfile], 300)
    records = source.get("files", [])
    if len(records) != 24 or {x.get("name") for x in records} != set(FILES):
        raise RuntimeError("source manifest must contain exactly 24 snapshots")
    records = [next(x for x in records if x["name"] == n) for n in FILES]
    manifest = json.dumps(records, separators=(",", ":"), sort_keys=True)
    state = _run(probe, node1, DEST_STATE_CODE, [dst, names], 300)
    if state.get("state") == "mixed": raise RuntimeError("mixed destination files; refusing resume")
    resumed = state.get("state") == "complete"
    copied = state.get("files", []) if resumed else []
    stopped = {"pid": None}
    if resumed:
        copied_by_name = {x["name"]: x for x in copied}
        for x in records:
            y = copied_by_name[x["name"]]
            if (x["sha256"], x["size_bytes"]) != (y.get("sha256"), y.get("size_bytes")):
                raise RuntimeError("existing destination differs: " + x["name"])
    else:
        server_path = src; running = False
        try:
            running = True
            probe.docker(node0, ["exec", "-d", "-w", CONTAINER_ROOT, node0.container,
                                "python", "-c", SERVER_CODE, name, node0.ip, str(PORT),
                                src, pidfile, names], timeout=30)
            server = _run(probe, node0, CONTROL_CODE, ["wait", pidfile, server_path], 30)
            url = f"http://{node0.ip}:{PORT}"
            copied = _run(probe, node1, DOWNLOAD_CODE, [dst, url, manifest], 900).get("files", [])
            if len(copied) != 24 or {x.get("name") for x in copied} != set(FILES):
                raise RuntimeError("destination verification is incomplete")
            copied_by_name = {x["name"]: x for x in copied}
            for x in records:
                y = copied_by_name[x["name"]]
                if (x["sha256"], x["size_bytes"]) != (y.get("sha256"), y.get("size_bytes")):
                    raise RuntimeError("destination differs: " + x["name"])
            stopped = _run(probe, node0, CONTROL_CODE, ["stop", pidfile, server_path], 30)
            if stopped.get("stopped") is not True or stopped.get("pid") != server.get("pid"):
                raise RuntimeError("HTTP server cleanup was not verified")
            running = False
        except BaseException:
            if running:
                try: _run(probe, node0, CONTROL_CODE, ["stop", pidfile, server_path], 30)
                except Exception: pass
            raise
    deleted = _run(probe, node0, DELETE_CODE, [src, manifest], 300).get("deleted")
    if deleted != list(FILES): raise RuntimeError("source cleanup did not confirm exact allowlist")

    destination = {x["name"]: x for x in copied}
    files, by_step = [], {}
    for x in records:
        y = destination[x["name"]]
        step, rank = x["name"].split("_r")
        step, rank = int(step.rsplit("step", 1)[1]), int(rank.split(".", 1)[0])
        files.append({"step": step, "rank": rank, "name": x["name"],
                      "source": {"node": node0.physical_node, "sha256": x["sha256"],
                                 "size_bytes": x["size_bytes"], "hash_seconds": x["hash_seconds"],
                                 "deleted_after_verification": True},
                      "destination": {"node": node1.physical_node, "sha256": y["sha256"],
                                      "size_bytes": y["size_bytes"],
                                      "transfer_hash_seconds": y.get("transfer_hash_seconds"),
                                      "existing_verify_seconds": y.get("existing_verify_seconds")}})
        row = by_step.setdefault(str(step), {"file_count": 0, "source_bytes": 0,
                                             "destination_bytes": 0,
                                             "source_hash_seconds": 0.0,
                                             "destination_transfer_hash_seconds": 0.0})
        row["file_count"] += 1
        row["source_bytes"] += x["size_bytes"]
        row["destination_bytes"] += y["size_bytes"]
        row["source_hash_seconds"] += x["hash_seconds"]
        row["destination_transfer_hash_seconds"] += y.get("transfer_hash_seconds") or 0
        row.setdefault("destination_existing_verify_seconds", 0.0)
        row["destination_existing_verify_seconds"] += y.get("existing_verify_seconds") or 0
    return {"session": name, "source_node": node0.physical_node,
            "destination_node": node1.physical_node, "server_port": PORT,
            "server_pid": stopped["pid"], "server_cleanup_verified": True,
            "server_started": not resumed, "resumed_existing": resumed,
            "source_manifest_seconds": source.get("total_hash_seconds"),
            "total_seconds": time.monotonic() - started,
            "file_count": len(files), "bytes_total": sum(x["source"]["size_bytes"] for x in files),
            "by_step": by_step, "files": files}
