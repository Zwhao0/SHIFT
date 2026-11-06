"""CPU-only verifier coverage for whole-replica v1 rotation."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_v42_session import export as export_session
from scripts.verify_session import _resource_check, verify
from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.training_core import sample_plan


def _hash(rank):
    return {key: f'{rank:064x}' for key in REFERENCE_HASH_KEYS}


def _build_v1_run(root, source_replica, world=8):
    width = world // 2
    tp = world // 4
    actors = {
        'g0_A': dict(node='node0', role='A', replica=0, devices=list(range(width)), generation=0, unit='u-g0-A'),
        'g0_B': dict(node='node0', role='B', replica=0, devices=list(range(width,world)), generation=0, unit='u-g0-B'),
        'g0_D': dict(node='node1', role='D', replica=1, devices=list(range(width)), generation=0, unit='u-g0-D'),
        'g0_E': dict(node='node1', role='E', replica=1, devices=list(range(width,world)), generation=0, unit='u-g0-E'),
    }
    target = 'g1_D'
    actors[target] = dict(node='node2', role='D', replica=source_replica,
                          devices=list(range(world)), generation=1, unit='u-g1-D')
    source_labels = {'g0_A', 'g0_B'} if source_replica == 0 else {'g0_D', 'g0_E'}
    healthy_replica = 1 - source_replica
    active = {}
    for rank in range(world):
        source_pair = (target, rank)
        healthy_pair = (('g0_D', rank) if rank < width else ('g0_E', rank-width)) if healthy_replica == 1 else (
            ('g0_A', rank) if rank < width else ('g0_B', rank-width))
        active[f'{source_replica}/{rank}'] = list(source_pair)
        active[f'{healthy_replica}/{rank}'] = list(healthy_pair)

    raw = root / 'raw'
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    pid_by_label = {}
    next_pid = 1000
    for label, desc in actors.items():
        dest = raw / desc['node']
        dest.mkdir(exist_ok=True)
        pid_by_label[label] = []
        for device in range(len(desc['devices'])):
            rank = device if desc['role'] in ('A', 'D') else device + width
            pid = next_pid
            next_pid += 1
            pid_by_label[label].append(pid)
            common = dict(role=desc['role'], rank=rank, replica=desc['replica'], device=device,
                          actor_round=desc['generation'], round=0, pid=pid, monotonic=20.0,
                          time=100.0, command_id=None)
            events = []
            steps = [1] if label in source_labels else (list(range(2,7)) if label == target else list(range(1,7)))
            for step in steps:
                samples = sample_plan(step-1, participants=2, replica=desc['replica'],
                                      local_data_rank=(rank % width) // tp)
                events.append(dict(common, kind='session_commit', step=step,
                    hashes=None, sample_ids=samples, participants=2, contributed=True,
                    participating_rank=desc['replica'], seconds=.25, finite=True,
                    command_id='v1' if step >= 2 else None))
            if label == target:
                events.append(dict(common, kind='session_native_restored', step=1,
                                   source_replica=healthy_replica))
            if label not in source_labels:
                events.append(dict(common, kind='session_command_done', step=2,
                                   command_id='v1', policy='v1'))
                events.append(dict(common, kind='session_done', step=6, hashes=_hash(rank),
                                   engine_identity={'training': rank}))
            (dest / f'{label}{device}.jsonl').write_text(
                ''.join(json.dumps(e)+'\n' for e in events))

    proof = []
    for index, label in enumerate(sorted(source_labels)):
        proof.append(dict(unit=actors[label]['unit'], signal='SIGKILL',
                          worker_pids=pid_by_label[label], main_pid=8000+index,
                          all_workers_dead=True, systemctl_returncode=0))
    command = dict(id='v1', policy='v1', expected_epoch=0, action='recover',
                   source_replica=source_replica, target='node2', at_step=1,
                   ranks=list(range(world)))
    rec = dict(sequence=1, status='completed', completed_step=2, controller_seconds=1.5, target_actor=target,
               devices=list(range(world)), command=command, requested_monotonic=10.0,
               fault_monotonic=10.0, fault_kind='SIGKILL', fault_proof=proof,
               first_common_observation=dict(step=2, request_seconds=1.0,
                   fault_seconds=1.0, observed_monotonic=11.0))
    summary = dict(name='v1_rotation', session='test', done=True, error=None,
        resources_clean=True, cleanup_errors=[], audit=False, boundary_validation='runtime',
        epoch=1, final_step=6, actors=actors, active=active, commands=[rec])
    if world == 4:
        summary['configuration'] = dict(replica_world_size=4,tp=1,pp=2,ep=2,batch=8,seq_len=32)
    (root / 'summary.json').write_text(json.dumps(summary))
    resources = {node: dict(company={str(i): 'company' for i in range(8)}, extra_gpu=[])
                 for node in ('node0', 'node1', 'node2')}
    (root / 'resources_before.json').write_text(json.dumps(resources))
    (root / 'resources_after.json').write_text(json.dumps(resources))


class V1RotationVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'session'
        self.root.mkdir()
        _build_v1_run(self.root, 0)

    def tearDown(self):
        self.temp.cleanup()

    def _summary(self):
        path = self.root / 'summary.json'
        return path, json.loads(path.read_text())

    def test_source_zero_two_pp_units_to_d8_passes(self):
        result = verify(self.root)
        self.assertTrue(result['pass_all'], result['errors'])
        self.assertEqual(result['source_manifest'][0]['source_replica'], 0)

    def test_existing_source_one_path_still_passes(self):
        _build_v1_run(self.root, 1)
        result = verify(self.root)
        self.assertTrue(result['pass_all'], result['errors'])
        self.assertEqual(result['source_manifest'][0]['source_replica'], 1)

    def test_four_rank_candidate_can_use_upper_four_physical_slots(self):
        _build_v1_run(self.root,1,world=4)
        path,summary=self._summary()
        summary['actors']['g1_D']['devices']=[4,5,6,7]
        summary['commands'][0]['devices']=[4,5,6,7]
        path.write_text(json.dumps(summary))
        result=verify(self.root)
        self.assertTrue(result['pass_all'],result['errors'])
        summary['actors']['g1_D']['devices']=[4,5,6,8]
        summary['commands'][0]['devices']=[4,5,6,8]
        path.write_text(json.dumps(summary))
        self.assertFalse(verify(self.root)['pass_all'])

    def test_rejects_candidate_on_wrong_replica(self):
        path, summary = self._summary()
        summary['commands'][0]['command']['source_replica'] = 1
        path.write_text(json.dumps(summary))
        result = verify(self.root)
        self.assertFalse(result['pass_all'])
        self.assertTrue(any('replacement must be a 8-rank D actor on the source replica'
                            in error for error in result['errors']), result['errors'])

    def test_rejects_changed_healthy_mapping(self):
        path, summary = self._summary()
        summary['active']['1/0'] = ['g1_D', 0]
        path.write_text(json.dumps(summary))
        result = verify(self.root)
        self.assertFalse(result['pass_all'])
        self.assertTrue(any('active logical rank mismatch 1/0' in error or
                            'summary final active map disagrees with command sequence' in error
                            for error in result['errors']), result['errors'])

    def test_rejects_missing_pid_from_a_source_unit_proof(self):
        path, summary = self._summary()
        summary['commands'][0]['fault_proof'][0]['worker_pids'].pop()
        path.write_text(json.dumps(summary))
        result = verify(self.root)
        self.assertFalse(result['pass_all'])
        self.assertTrue(any('source death proof PID set mismatch' in error
                            for error in result['errors']), result['errors'])

    def test_world4_tp1_v1_rotation_passes(self):
        _build_v1_run(self.root, 0, world=4)
        result = verify(self.root)
        self.assertTrue(result['pass_all'], result['errors'])
        self.assertEqual(result['configuration']['replica_world_size'], 4)
        self.assertEqual(result['statistics']['final_workers'], 8)

    def test_world4_rejects_inconsistent_parallel_configuration(self):
        _build_v1_run(self.root, 0, world=4)
        path, summary = self._summary()
        for key, wrong in (('tp', 2), ('pp', 1), ('ep', 1)):
            with self.subTest(key=key):
                summary['configuration'][key] = wrong
                path.write_text(json.dumps(summary))
                result = verify(self.root)
                self.assertFalse(result['pass_all'])
                self.assertTrue(any(f'configuration {key} must be' in error
                                    for error in result['errors']), result['errors'])
                summary['configuration'][key] = {'tp':1,'pp':2,'ep':2}[key]

    def test_export_world4_uses_dynamic_cohorts_and_configuration(self):
        _build_v1_run(self.root, 0, world=4)
        original = Path.read_bytes
        verifier_path = Path(__file__).with_name('verify_session.py')
        def read_bytes(path):
            if str(path) == '/workspace/SHIFT/scripts/verify_session.py':
                return original(verifier_path)
            return original(path)
        with patch.object(Path, 'read_bytes', read_bytes):
            packet = export_session(self.root)
        self.assertTrue(packet['accepted'])
        self.assertEqual(packet['configuration']['replica_world_size'], 4)
        case = packet['cases'][0]
        self.assertEqual(len(case['windows']['first_joint']['steps']), 1)
        self.assertEqual(case['windows']['post4']['steps'], [3,4,5,6])

    def test_resource_evidence_accepts_two_matching_nodes(self):
        resources = {node: dict(company={str(i): 'pid' for i in range(8)}, extra_gpu=[])
                     for node in ('node0', 'node1')}
        (self.root / 'resources_before.json').write_text(json.dumps(resources))
        (self.root / 'resources_after.json').write_text(json.dumps(resources))
        errors = []
        _resource_check(self.root, errors)
        self.assertEqual(errors, [])

    def test_resource_evidence_rejects_different_two_node_sets(self):
        before = {node: dict(company={str(i): 'pid' for i in range(8)}, extra_gpu=[])
                  for node in ('node0', 'node1')}
        after = dict(before)
        after['node2'] = after.pop('node1')
        (self.root / 'resources_before.json').write_text(json.dumps(before))
        (self.root / 'resources_after.json').write_text(json.dumps(after))
        errors = []
        _resource_check(self.root, errors)
        self.assertTrue(any('same two or three nodes' in error for error in errors), errors)


if __name__ == '__main__':
    unittest.main()
