"""CPU fixtures for the independent V4 native deadline-probe auditor."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_v4_deadline import summarize, verify


class DeadlineAuditTest(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root=Path(directory.name)
        self.make_case()

    def make_case(self, *, timeout=False, cancel_timer=False, pg_timeout=2.0, delay=None):
        if delay is None:delay=3.0 if timeout else 0.0
        expected_timeout=delay>pg_timeout
        summary=dict(success=not expected_timeout,expected_timeout=expected_timeout,
            pass_expected=not expected_timeout,exit_codes=[1,1] if expected_timeout else [0,0],
            backend='nccl',pg_timeout=pg_timeout,delay=delay,cancel_timer=cancel_timer,
            gpu_used=True,resources_clean=True,effective_deadlines=[],timers=[],errors=[])
        (self.root/'summary.json').write_text(json.dumps(summary))

        for replica in (0,1):
            events=[]
            def add(kind,step,**fields):
                events.append(dict(kind=kind,replica=replica,step=step,pid=5000+replica,
                                   monotonic=len(events)+1,**fields))
            add('ready',-1,manager_timeout=90.0,quorum_timeout=180.0,pg_timeout=pg_timeout)
            steps=(0,1) if expected_timeout else (0,1,2)
            for step in steps:
                add('quorum_ready',step)
                add('allreduce_begin',step)
                if step==1 and replica==0 and delay:
                    add('injected_delay',step,seconds=delay)
                if not expected_timeout or step==0:
                    add('allreduce_end',step,seconds=0.01,wait_return=True,error='None',
                        native_pg_error='None',exact=True)
                    add('commit_decision',step,decision=True)
                    add('commit',step,committed_step=step+1)
                if cancel_timer and step==1:
                    add('guard_exception_caught',step)
                if expected_timeout and step==1:
                    if replica==0:
                        add('timer_abort',step,origin='cuda_stream',timeout_seconds=pg_timeout,
                            elapsed=pg_timeout+0.25)
                    add('error',step,error='RuntimeError: operation aborted',
                        manager_error='RuntimeError: ProcessGroupNCCL aborted',
                        native_pg_error='RuntimeError: native process group aborted')
            if not expected_timeout:add('done',2)
            (self.root/f'R{replica}.jsonl').write_text(
                ''.join(json.dumps(event)+'\n' for event in events))

    def update_summary(self,**fields):
        path=self.root/'summary.json';summary=json.loads(path.read_text());summary.update(fields)
        path.write_text(json.dumps(summary))

    def read(self,replica):
        return [json.loads(line) for line in (self.root/f'R{replica}.jsonl').read_text().splitlines()]

    def save(self,replica,events):
        (self.root/f'R{replica}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))

    def test_normal_case_is_audited_from_raw_not_pass_expected(self):
        self.update_summary(pass_expected=False)
        report=verify(self.root)
        self.assertTrue(report['pass_all'],report['errors'])
        self.assertEqual(report['case'],'normal')
        self.assertIsNone(report['failure_layer'])

    def test_expected_timeout_is_attributed_from_raw(self):
        self.make_case(timeout=True)
        self.update_summary(pass_expected=False)
        report=verify(self.root)
        self.assertTrue(report['pass_all'],report['errors'])
        self.assertEqual(report['failure_layer'],'native_process_group_timeout')
        self.assertEqual(report['timer_origin'],'cuda_stream')
        self.assertEqual(report['error_count'],2)
        self.assertEqual(report['timer_abort_count'],1)

    def test_cancelled_guard_timer_is_a_success_case(self):
        self.make_case(cancel_timer=True)
        report=verify(self.root)
        self.assertTrue(report['pass_all'],report['errors'])
        self.assertEqual(report['case'],'cancel_timer_control')
        self.assertEqual(report['cancel_timer_guard_steps'],{'0':[1],'1':[1]})

    def test_ten_second_native_deadline_with_three_second_delay_is_normal(self):
        self.make_case(pg_timeout=10.0,delay=3.0)
        report=verify(self.root)
        self.assertTrue(report['pass_all'],report['errors'])
        self.assertFalse(report['expected_timeout'])
        self.assertIsNone(report['failure_layer'])

    def test_effective_deadline_layer_must_match_configuration(self):
        events=self.read(0)
        events[0]['manager_timeout']=2.0
        self.save(0,events)
        self.assertFalse(verify(self.root)['pass_all'])

        self.make_case(timeout=True)
        events=self.read(1)
        events[0]['pg_timeout']=10.0
        self.save(1,events)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_timeout_without_timer_abort_is_rejected_even_if_summary_passes(self):
        self.make_case(timeout=True)
        events=self.read(0);events=[x for x in events if x['kind']!='timer_abort'];self.save(0,events)
        self.update_summary(pass_expected=True)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_step_one_commit_after_timeout_is_rejected(self):
        self.make_case(timeout=True)
        events=self.read(1)
        events.append(dict(kind='commit',replica=1,step=1,pid=5001,monotonic=100,committed_step=2))
        self.save(1,events)
        self.assertFalse(verify(self.root)['pass_all'])

    def test_uncancelled_timer_abort_is_rejected(self):
        self.make_case(cancel_timer=True)
        events=self.read(0)
        events.append(dict(kind='timer_abort',replica=0,step=1,pid=5000,monotonic=100,
                           origin='context/_run_context',timeout_seconds=2.0,elapsed=2.1))
        self.save(0,events)
        report=verify(self.root)
        self.assertFalse(report['pass_all'])
        self.assertFalse(report['errors']==[])

    def test_summary_includes_actual_auditor_and_raw_hashes(self):
        result=summarize(self.root)
        self.assertTrue(result['pass_all'])
        self.assertEqual(len(result['actual_auditor_sha256']),64)
        self.assertEqual(set(result['raw_sha256']),{'summary.json','R0.jsonl','R1.jsonl'})
        for name in ('summary.json','R0.jsonl','R1.jsonl'):
            expected=hashlib.sha256((self.root/name).read_bytes()).hexdigest()
            self.assertEqual(result['raw_sha256'][name],expected)
        self.assertNotIn('company_pids',json.dumps(result))


if __name__=='__main__':
    unittest.main()
