"""Small CPU-only synthetic tests for session verification."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_session import verify
from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.training_core import sample_plan


def _hash(rank, salt=0):
    return {key: f'{rank + salt:064x}' for key in REFERENCE_HASH_KEYS}


class SessionVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'session_smoke'
        self.root.mkdir()
        self._make_good_run()

    def tearDown(self):
        self.temp.cleanup()

    def _make_good_run(self):
        self.actors = {
            'g0_A': dict(node='node0', role='A', replica=0, devices=[0,1,2,3], generation=0, unit='u-g0-A'),
            'g0_B': dict(node='node0', role='B', replica=0, devices=[4,5,6,7], generation=0, unit='u-g0-B'),
            'g0_D': dict(node='node1', role='D', replica=1, devices=[0,1,2,3], generation=0, unit='u-g0-D'),
            'g0_E': dict(node='node1', role='E', replica=1, devices=[4,5,6,7], generation=0, unit='u-g0-E'),
        }
        active = {}
        for rank in range(8):
            if rank < 4:
                active[f'0/{rank}'] = ['g0_A', rank]
                active[f'1/{rank}'] = ['g0_D', rank]
            else:
                active[f'0/{rank}'] = ['g0_B', rank-4]
                active[f'1/{rank}'] = ['g0_E', rank-4]
        self.summary = dict(name='session_smoke', session='smoke', done=True, error=None,
            resources_clean=True, cleanup_errors=[], audit=False, boundary_validation='runtime',
            epoch=0, actors=self.actors, active=active, commands=[])
        (self.root / 'summary.json').write_text(json.dumps(self.summary))
        (self.root / 'raw').mkdir()
        pid = 1000
        for label, desc in self.actors.items():
            dest = self.root / 'raw' / desc['node']
            dest.mkdir(exist_ok=True)
            for device in range(4):
                replica = desc['replica']
                rank = device if desc['role'] in ('A','D') else device+4
                local = rank % 4 // 2
                commits = sample_plan(0, participants=2, replica=replica, local_data_rank=local)
                common = dict(role=desc['role'], rank=rank, replica=replica, device=device,
                              actor_round=0, round=0, pid=pid, monotonic=10.0,
                              time=100.0, command_id=None)
                commit = dict(common, kind='session_commit', step=1, hashes=None,
                    sample_ids=commits, participants=2, contributed=True,
                    participating_rank=replica, seconds=.25, finite=True)
                done = dict(common, kind='session_done', step=1, hashes=_hash(rank),
                            engine_identity={'training': rank})
                (dest / f'{label}{device}.jsonl').write_text(
                    json.dumps(commit)+'\n'+json.dumps(dict(done, monotonic=10.1))+'\n')
                pid += 1
        resources = {node: dict(company={str(i): 'VLLM::model' for i in range(node_i*100, node_i*100+8)}, extra_gpu=[])
                     for node_i, node in enumerate(('node0','node1','node2'))}
        (self.root / 'resources_before.json').write_text(json.dumps(resources))
        (self.root / 'resources_after.json').write_text(json.dumps(resources))

    def _save_summary(self):
        (self.root / 'summary.json').write_text(json.dumps(self.summary))

    def _events(self, label, device):
        d = self.actors[label]
        path = self.root / 'raw' / d['node'] / f'{label}{device}.jsonl'
        return path, [json.loads(x) for x in path.read_text().splitlines()]

    def test_good_one_loop_without_commands(self):
        result = verify(self.root)
        self.assertTrue(result['pass_all'], result['errors'])
        self.assertEqual(result['statistics']['final_workers'], 16)
        self.assertEqual(result['statistics']['commands'], 0)

    def test_good_one_loop_full_hash_audit(self):
        self.summary['audit'] = True
        self.summary['boundary_validation'] = 'full'
        self._save_summary()
        for label, desc in self.actors.items():
            for device in range(4):
                path, events = self._events(label, device)
                rank = events[0]['rank']
                for event in events:
                    if event['kind'] == 'session_commit':
                        event['hashes'] = _hash(rank)
                path.write_text('\n'.join(json.dumps(x) for x in events)+'\n')
        result = verify(self.root)
        self.assertTrue(result['pass_all'], result['errors'])

    def test_rejects_incomplete_final_hash(self):
        path, events = self._events('g0_A', 0)
        events[-1]['hashes'].pop('scheduler')
        path.write_text('\n'.join(json.dumps(x) for x in events)+'\n')
        result = verify(self.root)
        self.assertFalse(result['pass_all'])
        self.assertTrue(any('final nine-state hash missing 0/0' in e for e in result['errors']))

    def test_rejects_pid_discontinuity(self):
        path, events = self._events('g0_B', 2)
        events[1]['pid'] += 99
        path.write_text('\n'.join(json.dumps(x) for x in events)+'\n')
        result = verify(self.root)
        self.assertTrue(any('PID continuity g0_B2.jsonl' in e for e in result['errors']))

    def test_rejects_wrong_sample_plan(self):
        path, events = self._events('g0_D', 1)
        events[0]['sample_ids'] = [999]
        path.write_text('\n'.join(json.dumps(x) for x in events)+'\n')
        result = verify(self.root)
        self.assertTrue(any('sample plan mismatch g0_D1.jsonl/step1' in e for e in result['errors']))
        self.assertTrue(any('global sample coverage mismatch step 1' in e for e in result['errors']))

    def test_rejects_nonsequential_command_epoch(self):
        policies = ('v2','v4','v1')
        self.summary['epoch'] = 3
        self.summary['commands'] = [dict(sequence=i+1, status='completed', completed_step=10+i*10,
            command=dict(id=f'c{i}', policy=policy, expected_epoch=i+1 if i == 1 else i, source_replica=0 if policy=='v4' else 1,
                         at_step=2+i*10)) for i, policy in enumerate(policies)]
        self._save_summary()
        result = verify(self.root)
        self.assertTrue(any('command sequence/epoch mismatch' in e for e in result['errors']))

    def test_expected_counts_option(self):
        result = verify(self.root, expected_counts={'v1':2,'v2':2,'v4':2})
        self.assertFalse(result['pass_all'])
        self.assertEqual(result['policy_counts'], {'v1':0,'v2':0,'v4':0})
        self.assertTrue(any('policy counts mismatch' in e for e in result['errors']))

    def test_rejects_missing_v2_source_handoff(self):
        target = 'g1_D'
        self.actors[target] = dict(node='node2', role='D', replica=1, devices=list(range(8)),
                                   generation=1, unit='u-g1-D')
        self.summary['actors'] = self.actors
        self.summary['epoch'] = 1
        self.summary['commands'] = [dict(sequence=1, status='completed', completed_step=10,
            target_actor=target, command=dict(id='v2', policy='v2', expected_epoch=0,
                source_replica=1, target='node2', at_step=2),
            first_common_observation=dict(request_seconds=1.0, fault_seconds=None))]
        dest = self.root / 'raw' / 'node2'
        dest.mkdir()
        for device in range(8):
            event = dict(kind='worker_imported', role='D', rank=device, replica=1, device=device,
                         actor_round=1, round=0, pid=3000+device, monotonic=20., time=200.)
            (dest / f'{target}{device}.jsonl').write_text(json.dumps(event)+'\n')
        self._save_summary()
        result = verify(self.root)
        self.assertTrue(any('v2 source handoff: planned cut selection missing' in e or
                            'v2 source handoff: planned_precopy_window_open missing' in e
                            for e in result['errors']), result['errors'])


class PartialScopeEvidenceTests(unittest.TestCase):
    def test_dp1_pp0_requires_decision_from_retained_pp1(self):
        from collections import defaultdict
        from scripts.verify_session import _flatten, _verify_v4

        rows = defaultdict(lambda: defaultdict(list))
        actors = {
            'old': dict(role='D', replica=1, devices=[0,1], generation=0),
            'retained': dict(role='E', replica=1, devices=[2,3], generation=0),
            'peer': dict(role='A', replica=0, devices=[0,1,2,3], generation=0),
            'new': dict(role='C', replica=1, ranks=[0,1], devices=[0,1], generation=1),
        }
        before = {(replica, rank): ('peer', rank) if replica == 0 else
                  ('old', rank) if rank < 2 else ('retained', rank - 2)
                  for replica in (0, 1) for rank in range(4)}
        after = before | {(1,0): ('new',0), (1,1): ('new',1)}
        command = dict(sequence=1, source_role='B', target_role='C', target_actor='new', outer_store_service=1,
                       command=dict(expected_epoch=0, source_replica=1, ranks=[0,1]))

        def event(placement, replica, rank, kind, **fields):
            actor, device = placement[replica,rank]
            row = dict(kind=kind, rank=rank, replica=replica, round=0, step=10, **fields)
            rows[actor][device].append(row)
            return row

        switched = dict(decision='replace', owner='C', epoch=1)
        for replica, rank in before:
            event(before,replica,rank,'boundary_selected')
            event(before,replica,rank,'freeze')
            if replica == 0:
                for kind in ('peer_compute_ready', 'peer_compute_begin', 'peer_native_gradient_ready'):
                    event(before,replica,rank,kind)
            else:
                event(after,replica,rank,'activated',**switched)
                if rank >= 2:event(after,replica,rank,'outer_rebuilt',epoch=1)
        decision = event(before,1,2,'switch_decision',validation='runtime',
                         candidate_results={'0':True, '1':True},**switched)
        transport = dict(transport='packed_gpu', packet_source='source', bytes=3, gpu_bytes=2, control_bytes=1)
        for rank in (0,1):
            event(before,1,rank,'retired')
            event(before,1,rank,'state_sent',**transport)
            event(after,1,rank,'state_received',**transport)
            readiness = dict(schema=1,epoch=1,rank=rank,cut=10,target='C',restored=True)
            event(after,1,rank,'candidate_validated',readiness=readiness)
            event(after,1,rank,'candidate_activated',readiness=readiness)
        errors = []
        _verify_v4(command,before,after,rows,actors,_flatten(rows),errors,'runtime',world=4)
        self.assertEqual(errors,[])
        decision['rank'] = 0
        _verify_v4(command,before,after,rows,actors,_flatten(rows),errors,'runtime',world=4)
        self.assertIn('v4 switch decision missing',errors)


if __name__ == '__main__':
    unittest.main()
