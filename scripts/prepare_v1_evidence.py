"""Collect small metadata and compressed raw evidence; exclude model snapshots."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v1'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sessions=[]
    for source in sorted((ROOT/'results').glob('v1_*')):
        if not (source/'controller.jsonl').is_file():
            continue
        name=source.name;sessions.append(name)
        metadata=OUT/'sessions'/name;metadata.mkdir(parents=True,exist_ok=True)
        raw=OUT/'raw'/name;raw.mkdir(parents=True,exist_ok=True)
        for filename in ['config.json','controller.jsonl','physical_manifest.json','fault_results.json']:
            if (source/filename).is_file():
                shutil.copy2(source/filename,metadata/filename)
        for node in ['node0','node1','node3']:
            if (source/(node+'.tgz')).is_file():
                shutil.copy2(source/(node+'.tgz'),raw/(node+'.tgz'))
    code=[]
    for folder in ['shift_full','shift_min','configs','scripts','vendor','runtime/torch214-manifest']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                code.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),size_bytes=p.stat().st_size))
    (OUT/'source_manifest.json').write_text(json.dumps(dict(files=code),indent=2)+'\n')
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='manifest.json':
            files.append(dict(path=str(p.relative_to(OUT)),sha256=digest(p),size_bytes=p.stat().st_size))
    (OUT/'manifest.json').write_text(json.dumps(dict(sessions=sessions,files=files),indent=2)+'\n')
    print(json.dumps(dict(sessions=len(sessions),files=len(files),total_bytes=sum(p['size_bytes'] for p in files))))


if __name__=='__main__':
    main()
