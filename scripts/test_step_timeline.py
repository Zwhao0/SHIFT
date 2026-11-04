"""One-shot capture, bounded streaming archive and event-field compatibility."""
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shift_full.step_timeline import StepTimeline


class TimelineTest(unittest.TestCase):
    def test_capture_once_and_event_fields_do_not_shadow_rank(self):
        calls=[]
        profiler=SimpleNamespace(start=lambda:calls.append('start'),stop=lambda:calls.append('stop'),
            export_chrome_trace=lambda path:Path(path).write_text('{"traceEvents": []}'))
        torch=SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kw:profiler,
            ProfilerActivity=SimpleNamespace(CPU=1,CUDA=2),record_function=lambda name:nullcontext()))
        with TemporaryDirectory() as tmp, patch('shift_full.step_timeline.cpu_snapshot',return_value={}):
            trace=StepTimeline(torch,11,Path(tmp),0)
            trace.begin(10)
            self.assertIsNone(trace.end(True))
            trace.begin(11)
            report=trace.end(False)
            event=dict(kind='timeline_diagnostic',rank=0,**report)
            self.assertFalse(event['committed'])
            with gzip.open(report['path'],'rt') as source:
                self.assertEqual(json.load(source),{'traceEvents':[]})
            self.assertFalse((Path(tmp)/'timeline_step11_r0.json').exists())
            trace.begin(11)
            trace.close()
            self.assertEqual(calls,['start','stop'])


if __name__=='__main__':unittest.main()
