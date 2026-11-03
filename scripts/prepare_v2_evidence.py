#!/usr/bin/env python3
"""Build scrubbed, size-bounded upload evidence under results/v2."""
import fnmatch,hashlib,io,json,re,shutil,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; OUT=RESULTS/'v2'
MAX=25_000_000; PART=16*1024*1024
META={'config.json','controller.jsonl','physical_manifest.json','provenance.json'}
STATUS=re.compile(r'^(?:planned_|cancel_|steady_)?(?:status|analysis)\.json$')
SECRET_KEY=re.compile(r'(?:^|[_-])(?:password|passwd|secret|token|credential|api[_-]?key|access[_-]?key|authorization|cookie|private[_-]?key|client[_-]?secret)(?:$|[_-])',re.I)
SECRET_TEXT=re.compile(r'(?i)(\b(?:password|passwd|secret|token|credential|api[_-]?key|access[_-]?key|authorization|cookie|client[_-]?secret)\b\s*[:=]\s*)[^\s,;]+')
PRIVATE_IP=re.compile(r'(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])')
URL_AUTH=re.compile(r'(?i)(https?://)[^/@\s]+@'); BEARER=re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+')
PATHS=re.compile(r'(?i)(?:/home|/Users)/[^/\s"\']+(?:/[^\s"\']*)?|/workspace/[^\s"\']*')
EXTRA_PATH=re.compile(r'(?i)/[^\s"\']*/(?:runtime|models)(?:/[^\s"\']*)?')

def safe_name(n): return not re.search(r'auth|credential|secret|token|key',n,re.I)
def scrub_text(s):
 s=URL_AUTH.sub(r'\1<redacted>@',s); s=BEARER.sub('Bearer <redacted>',s); s=SECRET_TEXT.sub(r'\1<redacted>',s)
 s=PRIVATE_IP.sub('<private-ip>',s); s=EXTRA_PATH.sub('<excluded-path>',s); return PATHS.sub('<redacted-path>',s)
def scrub(x,k=''):
 if SECRET_KEY.search(str(k)): return '<redacted>'
 if str(k).lower() in {'root','host_root','model_path','runtime_path','cache_dir'}: return '<redacted-path>'
 if isinstance(x,dict): return {a:scrub(b,a) for a,b in x.items()}
 if isinstance(x,list): return [scrub(a) for a in x]
 return scrub_text(x) if isinstance(x,str) else x
def clean(p):
 raw=p.read_bytes()
 if p.suffix=='.json':
  try: return (json.dumps(scrub(json.loads(raw)),indent=2,ensure_ascii=False)+'\n').encode()
  except (ValueError,UnicodeDecodeError): pass
 if p.suffix=='.jsonl':
  lines=[]
  for line in raw.decode('utf-8','replace').splitlines():
   try: lines.append(json.dumps(scrub(json.loads(line)),sort_keys=True,ensure_ascii=False))
   except ValueError: lines.append(scrub_text(line))
  return ('\n'.join(lines)+'\n').encode()
 return scrub_text(raw.decode('utf-8','replace')).encode()
def ismeta(p): return p.name in META or bool(STATUS.fullmatch(p.name))
def inputs(s):
 out=[]
 for p in s.iterdir():
  if p.is_file() and not p.is_symlink() and ismeta(p): out.append((f'{s.name}/{p.name}',p))
 for d in sorted(s.glob('physical_node*')):
  if not d.is_dir() or d.is_symlink(): continue
  for p in d.iterdir():
   n=p.name; ok=ismeta(p) or n in {'supervisor.jsonl','lighthouse.log'} or (n.startswith('events') and n.endswith('.jsonl')) or (n.startswith('torchrun') and n.endswith('.log'))
   if p.is_file() and not p.is_symlink() and ok: out.append((f'{s.name}/{d.name}/{n}',p))
 return out
def tar(path,items):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(path,'w:gz') as t:
  for name,p in items:
   b=clean(p); info=tarfile.TarInfo(name); info.size=len(b); info.mtime=0; info.mode=0o644; t.addfile(info,io.BytesIO(b))
 if path.stat().st_size<MAX: return True
 path.unlink(); return False
def chunks(b):
 out=[]; buf=bytearray()
 for line in b.splitlines(keepends=True):
  while len(line)>PART:
   if buf: out.append(bytes(buf)); buf.clear()
   out.append(line[:PART]); line=line[PART:]
  if len(buf)+len(line)>PART: out.append(bytes(buf)); buf.clear()
  buf.extend(line)
 return out+[bytes(buf)] if buf else out or [b'']
def bundle(group,s,items):
 d=OUT/'bundles'/group
 if tar(d/f'{s.name}.tar.gz',items): return
 parts={'metadata':[]}
 for node in s.glob('physical_node*'):
  if node.is_dir() and not node.is_symlink(): parts[node.name]=[]
 for arc,p in items:
  node=next((n for n in parts if n!='metadata' and f'/{n}/' in arc),'metadata'); parts[node].append((arc,p))
 for node,subset in parts.items():
  if not subset: continue
  dst=d/f'{s.name}_{node}.tar.gz'
  if tar(dst,subset): continue
  for arc,p in subset:
   for i,b in enumerate(chunks(clean(p)),1):
    name=f'{s.name}_{node}_{p.name}.part{i:03}.tar.gz'; info=tarfile.TarInfo(f'{arc}.part{i:03}'); info.size=len(b); info.mtime=0; info.mode=0o644
    with tarfile.open(d/name,'w:gz') as t: t.addfile(info,io.BytesIO(b))
    if (d/name).stat().st_size>=MAX: raise RuntimeError(f'bundle part exceeds 25 MB: {name}')
def copy_session(group,s):
 d=OUT/'sessions'/group/s.name; d.mkdir(parents=True,exist_ok=True); copied=[]
 for p in s.iterdir():
  if p.is_file() and not p.is_symlink() and ismeta(p): (d/p.name).write_bytes(clean(p)); copied.append(p.name)
 pd=d/'provenance'
 for p in sorted(s.glob('physical_node*/provenance.json')):
  if p.is_file() and not p.is_symlink(): pd.mkdir(exist_ok=True); (pd/f'{p.parent.name}.json').write_bytes(clean(p)); copied.append(f'provenance/{p.parent.name}.json')
 have_status=any(STATUS.fullmatch(Path(n).name) and 'status' in n for n in copied)
 have_analysis=any(STATUS.fullmatch(Path(n).name) and 'analysis' in n for n in copied)
 have_prov=any(n=='provenance.json' or n.startswith('provenance/') for n in copied)
 expected=('config.json','controller.jsonl','physical_manifest.json')
 missing=[n for n in expected if n not in copied]+[n for n,ok in (('status',have_status),('analysis',have_analysis),('provenance',have_prov)) if not ok]
 rec={'session':s.name,'group':group,'source':str(s.relative_to(ROOT)),'included':copied,'missing_expected':missing}
 (d/'session.json').write_text(json.dumps(rec,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def source_manifest():
 files=set((ROOT/'shift_full').glob('*.py')); scripts=ROOT/'scripts'
 files.update(p for pat in ('*planned*.py','*v2*.py') for p in scripts.glob(pat))
 rows=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(files) if p.is_file() and not p.is_symlink() and safe_name(p.name)]
 configs=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((ROOT/'configs').glob('*.json')) if p.is_file() and not p.is_symlink() and safe_name(p.name)]
 (OUT/'source_manifest.json').write_text(json.dumps({'sources':rows,'config_hashes':configs},indent=2)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for n in ('sessions','bundles','legacy'):
  p=OUT/n
  if p.exists(): shutil.rmtree(p)
 ss=[('legacy',p) for p in sorted(RESULTS.glob('ph_*')) if p.is_dir() and not p.is_symlink()]
 ss += [('final',p) for p in sorted(RESULTS.glob('v2_*')) if p.is_dir() and not p.is_symlink() and p!=OUT and (p/'controller.jsonl').is_file()]
 for g,s in ss: copy_session(g,s); bundle(g,s,inputs(s))
 docs=RESULTS/'planned_handoff'; pats=('SUMMARY.md','LIMITS.md','native_capabilities.md','aggregate.json','final_stage_baseline.json','final_stage_baseline.md','protocol_negative.json','protocol_negative.md','versions.json','installed_vs_wheel.json','cleanup.json','v2_final_*.jsonl','v1_preservation*.json','v1_preservation*.md')
 legacy=OUT/'legacy'; legacy.mkdir(exist_ok=True)
 for p in sorted(docs.iterdir()):
  if p.is_file() and not p.is_symlink() and any(fnmatch.fnmatch(p.name,x) for x in pats): (legacy/p.name).write_bytes(clean(p))
 # Keep upload writes bounded: session metadata is one archive, with a small index.
 records=[json.loads(p.read_text()) for p in sorted((OUT/'sessions').glob('*/*/session.json'))]
 (OUT/'session_index.json').write_text(json.dumps(records,indent=2)+'\n')
 with tarfile.open(OUT/'metadata.tar.gz','w:gz') as t: t.add(OUT/'sessions',arcname='sessions')
 shutil.rmtree(OUT/'sessions')
 component_items=[]
 for pattern in ('ph_shadow_*','ph_candidate_*'):
  for d in sorted(docs.glob(pattern)):
   component_items.extend((str(p.relative_to(docs)),p) for p in sorted(d.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.log'))
 component_items.extend((p.name,p) for p in sorted(docs.glob('overlap_result*.json')))
 if not tar(OUT/'components.tar.gz',component_items):raise RuntimeError('components archive exceeds 25 MB')
 old_source=docs/'source_delta_v1.tgz'
 if old_source.is_file():shutil.copy2(old_source,legacy/'source_delta_v1.tgz')
 source_manifest()
 files={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and not p.is_symlink() and p.name!='manifest.json'}
 note='manifest.json omits its own hash; existing summary.json and SUMMARY.md are preserved and included.'
 (OUT/'manifest.json').write_text(json.dumps({'files':files,'note':note},indent=2)+'\n')
 if any(p.stat().st_size>=MAX for p in (OUT/'bundles').rglob('*.tar.gz')): raise RuntimeError('bundle exceeds 25 MB')
 print(f'prepared {len(ss)} sessions; {len(files)} manifest entries')
if __name__=='__main__': main()
