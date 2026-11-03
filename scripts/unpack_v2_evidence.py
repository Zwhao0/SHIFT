"""Verify the public manifest, then unpack bounded regular-file evidence."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('results/v2'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.evidence/'manifest.json').read_text())['files']
 for name,row in manifest.items():
  f=a.evidence/name
  if len(f.read_bytes())!=row['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'checksum mismatch: {name}')
 a.output.mkdir(parents=True,exist_ok=True)
 for f in sorted((a.evidence/'bundles').rglob('*.tar.gz'))+[p for n in ('metadata.tar.gz','components.tar.gz','micro.tar.gz') if (p:=a.evidence/n).is_file()]:
  dest=(a.output/'components' if f.name=='components.tar.gz' else a.output).resolve();dest.mkdir(parents=True,exist_ok=True)
  with tarfile.open(f) as t:
   members=t.getmembers()
   if any(not (m.isfile() or m.isdir()) or not (dest/m.name).resolve().is_relative_to(dest) for m in members):raise ValueError(f'unsafe archive: {f.name}')
   t.extractall(dest,members=members)
 print(f'verified {len(manifest)} files; unpacked evidence into {a.output}')
if __name__=='__main__':main()
