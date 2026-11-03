"""Package v2.1 logs using the v2 scrubber; never modify v1/v2 evidence."""
import hashlib,json,shutil,tarfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import prepare_v2_evidence as pack

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/v2_1';RAW=ROOT/'results/overlap_v21'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=True);pack.OUT=OUT
 pack.META.update({'cycle_status.json','cycle_analysis.json','scenario_status.json','deployed_source.json','preflight.json'})
 pack.STATUS=__import__('re').compile(r'^(?:(?:planned|cancel|steady|cycle|scenario)_)?(?:status|analysis)\.json$')
 for name in ('bundles','sessions','records'):
  p=OUT/name
  if p.exists():shutil.rmtree(p)
 sessions=[p for p in sorted((ROOT/'results').glob('v2_1_*')) if p.is_dir() and (p/'controller.jsonl').exists()]
 for s in sessions:
  group='final' if s.name.startswith('v2_1_final_') or s.name in ('v2_1_cycles3_audit02','v2_1_cycles10_retry01','v2_1_lr_cycles3_audit01','v2_1_lr_p1f_audit01') else 'calibration'
  pack.copy_session(group,s)
  items=pack.inputs(s)
  for p in sorted(s.glob('physical_node*/lighthouse_round*.log')):items.append((str(p.relative_to(s.parent)),p))
  pack.bundle(group,s,items)
 records=[json.loads(p.read_text()) for p in sorted((OUT/'sessions').glob('*/*/session.json'))]
 (OUT/'session_index.json').write_text(json.dumps(records,indent=2)+'\n')
 with tarfile.open(OUT/'metadata.tar.gz','w:gz') as t:t.add(OUT/'sessions',arcname='sessions')
 shutil.rmtree(OUT/'sessions')
 d=OUT/'records';d.mkdir()
 for p in sorted(RAW.iterdir()):
  if p.is_file() and p.suffix in ('.json','.jsonl','.md') and not p.name.startswith('upload_') and p.name not in ('REPORT.template.md','lr_compatibility_note.md','prepublish_validation.json'):(d/p.name).write_bytes(pack.clean(p))
 for p in sorted(RAW.glob('*source.tgz')):shutil.copy2(p,d/p.name)
 micro=ROOT/'results/planned_handoff/v2_1_relay_micro01'
 if micro.exists():pack.tar(OUT/'micro.tar.gz',[(str(p.relative_to(micro.parent)),p) for p in sorted(micro.iterdir()) if p.suffix in ('.json','.jsonl','.log')])
 source=[p for folder in ('shift_full','scripts') for p in (ROOT/folder).glob('*.py')]
 (OUT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(source)},indent=2)+'\n')
 files={str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
 assert all(r['bytes']<25_000_000 for r in files.values())
 (OUT/'manifest.json').write_text(json.dumps(dict(files=files,note='Public logs scrubbed; source snapshots preserve executable source; manifest omits itself.'),indent=2)+'\n')
 print(json.dumps(dict(sessions=len(sessions),files=len(files),bytes=sum(r['bytes'] for r in files.values()))))
if __name__=='__main__':main()
