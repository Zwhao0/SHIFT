"""Remove only remote /tmp evidence copies proven identical to local SHA archives.

Never touches raw results, anchors, environments, models or dependency caches.
Run after a controller completes. Local evidence archives remain authoritative.
"""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import time

from shift_full.control import Session, endpoints_from_env


def main():
    root=Path(__file__).resolve().parents[1]
    candidates=[]
    for group in ('v3_b1_20261103','v3_b2_20261103','v3_c1_20261103'):
        for p in sorted((root/'results'/group).glob('*-evidence.tgz')):
            check=p.with_suffix(p.suffix+'.sha256')
            if not check.is_file():continue
            expected=check.read_text().split()[0]
            if len(expected)!=64 or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('local evidence checksum mismatch: '+p.name)
            suffix=p.name.removesuffix('-evidence.tgz')
            if not suffix or not all(c.isalnum() or c in '_-' for c in suffix):
                raise ValueError('invalid suffix')
            candidates.append(dict(path='/tmp/shift-v3-b1-'+suffix+'-evidence.tgz',sha256=expected))
    s=Session.__new__(Session);s.root=root;s._sudo_password=None
    e=replace(endpoints_from_env()['node0'],socket='ssh-v3-clean-'+str(os.getpid())+'.sock')
    code='''from pathlib import Path
import fcntl,hashlib,json
lock=Path('/workspace/SHIFT/runtime/v3-training.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[]
for item in '''+repr(candidates)+''':
 p=Path(item['path'])
 if not p.exists():continue
 if not p.is_file() or p.is_symlink():raise RuntimeError('unexpected file type')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('remote evidence mismatch')
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=item['sha256']))
 p.unlink()
print(json.dumps(dict(removed=rows,reclaimed_bytes=sum(r['bytes'] for r in rows))))
'''
    try:
        s.ensure_connection(e)
        report=json.loads(s.ssh(e,['sudo','-S','-p','','python3','-c',code],sudo_input=True))
        report.update(observed_at=time.time(),policy='remote temporary duplicates only; local SHA archives and all raw results retained')
        out=root/'results/v3_b2_20261103'/('cleanup-'+str(time.time_ns())+'.json')
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(removed_count=len(report['removed']),reclaimed_bytes=report['reclaimed_bytes'],report=str(out))))
    finally:
        subprocess.run(['ssh','-S',str(root/'tmp'/e.socket),'-O','exit','test@'+e.ip],capture_output=True,timeout=5)


if __name__=='__main__':main()
