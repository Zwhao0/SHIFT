"""A partial comparison never passes even when every completed pair matches."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from shift_full.compare_reference import main


class ComparisonProgressTest(unittest.TestCase):
    def test_partial_cannot_be_accepted_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            payload={key:{'x':torch.tensor([1.,2.])} for key in ('gradients','model','optimizer','scheduler')}
            for directory in (a,b):
                torch.save(payload,directory/'reference_step1_r0.pt')
            output=root/'report.json'
            with patch.object(sys,'argv',['compare',str(a),str(b),'--steps','1','--ranks','0','--output',str(output)]):
                self.assertEqual(main(),0)
            final=json.loads(output.read_text()); partial=json.loads(output.with_suffix('.partial.json').read_text())
            self.assertTrue(final['pass']); self.assertTrue(final['complete'])
            self.assertFalse(partial['pass']); self.assertFalse(partial['complete'])
            self.assertTrue(partial['byrank']['1']['0']['pass'])


if __name__=='__main__':unittest.main()
