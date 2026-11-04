"""Fresh launcher topology and failure teardown, without importing torch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ColdLauncherTest(unittest.TestCase):
    def launch(self, source, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'probe.py').write_text(source)
            env=dict(os.environ,PYTHONPATH=tmp+os.pathsep+str(Path(__file__).resolve().parents[1]),PROBE_OUT=tmp)
            result=subprocess.run([sys.executable,'-m','shift_full.cold_launch','--nnodes=2',
                                   '--nproc_per_node=2','--node_rank=1','--master_addr=127.0.0.1',
                                   '--master_port=29860','--module','probe',*extra],env=env,
                                  text=True,capture_output=True,timeout=12)
            return result,{p.stem:json.loads(p.read_text()) for p in root.glob('rank*.json')}

    def test_fresh_topology(self):
        result,rows=self.launch('''import json,os,sys
from pathlib import Path
assert 'torch' not in sys.modules
r=os.environ['LOCAL_RANK']
Path(os.environ['PROBE_OUT'], 'rank'+r+'.json').write_text(json.dumps({k:os.environ[k] for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_PORT')}))
''')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([rows['rank'+str(i)]['RANK'] for i in range(2)],['2','3'])
        self.assertTrue(all(r['WORLD_SIZE']=='4' and r['LOCAL_WORLD_SIZE']=='2' for r in rows.values()))

    def test_worker_failure_reaps_peer(self):
        result,rows=self.launch('''import json,os,time
from pathlib import Path
r=os.environ['LOCAL_RANK']
Path(os.environ['PROBE_OUT'],'rank'+r+'.json').write_text(json.dumps({'pid':os.getpid()}))
if r=='0':
 time.sleep(.2)
 raise SystemExit(7)
time.sleep(60)
''')
        self.assertEqual(result.returncode,7,result.stderr)
        for row in rows.values():
            with self.assertRaises(ProcessLookupError):os.kill(row['pid'],0)


if __name__=='__main__':unittest.main()
