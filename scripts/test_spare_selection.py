from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.lab_acceptance import compare_pair
from scripts.lab_session_backend import LabSessionBackend
from shift_full.spare_agent import prime_model, probe
from shift_full.spare_pool import SparePool
from shift_full.spare_selection import Selection, SpareReport, plan_replacement, select_spare


def spare(node, *, slots=4, mib=4096, setup_ms=100):
    return SpareReport(node, f'{node}-pod', 99, frozenset({'lab-w4'}),
                       tuple((f'gpu{i}', mib) for i in range(slots)), True, setup_ms)


def select(*reports, **options):
    return select_spare(reports, **(dict(profile='lab-w4', ranks=4, min_gpu_mib=3072,
                                        now=100, max_age=5) | options))


class SpareSelectionTest(unittest.TestCase):
    def test_one_spare_and_source_exclusion(self):
        decision = select(spare('source', setup_ms=0), spare('spare'), excluded_nodes={'source'})
        self.assertEqual(decision.target.node, 'spare')
        self.assertEqual(decision.target.instance, 'spare-pod')
        self.assertEqual(len(decision.target.slots), 4)
        self.assertEqual(decision.rejected, (('source', 'excluded'),))

    def test_unreachable_runner_does_not_disqualify_other_candidates(self):
        backend = LabSessionBackend.__new__(LabSessionBackend)
        backend.pods = {'offline': ('runner', 'offline-pod')}
        backend.inventory = Mock(side_effect=subprocess.TimeoutExpired('kubectl', 4))
        offline = backend.candidate_inventory('offline')
        healthy = replace(spare('healthy'), observed_at=offline.observed_at)
        decision = select(offline, healthy, now=offline.observed_at)
        self.assertEqual(decision.target.node, 'healthy')
        self.assertEqual(decision.rejected, (('offline', 'not_ready'),))

    def test_agent_needs_staged_weights_but_can_report_partially_free_devices(self):
        mib = 1024**2
        cuda = Mock()
        cuda.device_count.return_value = 2
        cuda.get_device_properties.side_effect = lambda index: SimpleNamespace(uuid=str(index), name='test')
        cuda.mem_get_info.side_effect = lambda index: ((128, 8192)[index] * mib, 16384 * mib)
        torch = SimpleNamespace(cuda=cuda, __version__='test', version=SimpleNamespace(cuda='test'))
        with TemporaryDirectory() as directory:
            model = Path(directory)
            (model/'config.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):
                probe(torch, 2, 512, 'shift-session', model)
            (model/'model.safetensors').write_bytes(b'weights')
            report = probe(torch, 2, 512, 'shift-session', model)
            primed = prime_model(model)
        self.assertTrue(report['ready'])
        self.assertEqual([device['free_mib'] for device in report['devices']], [128, 8192])
        self.assertEqual(primed['bytes'], len(b'weights'))

    def test_filters_are_hard_constraints(self):
        decision = select(
            replace(spare('old'), observed_at=90), replace(spare('future'), observed_at=101),
            replace(spare('warming'), ready=False), replace(spare('wrong'), profiles=frozenset({'lab-w8'})),
            spare('small', slots=3), spare('low-memory', mib=2048),
        )
        self.assertIsNone(decision.target)
        self.assertEqual(dict(decision.rejected), {
            'old': 'stale_report', 'future': 'stale_report', 'warming': 'not_ready',
            'wrong': 'incompatible_profile', 'small': 'insufficient_slots', 'low-memory': 'insufficient_slots',
        })

    def test_deterministic_ranking(self):
        reports = (spare('slow', mib=8000, setup_ms=300), spare('b', mib=5000), spare('a', mib=5000),
                   spare('less-room'), spare('unknown', mib=8000, setup_ms=None))
        decision = select(*reports)
        self.assertEqual([row.node for row in decision.ranked], ['a', 'b', 'less-room', 'slow', 'unknown'])
        self.assertEqual(decision, select(*reversed(reports)))

    def test_stage_and_replica_use_different_capacity(self):
        report = replace(spare('partial'), free_gpu_mib=(('a', 1024), ('b', 4100), ('c', 3500), ('d', 2048)))
        self.assertIsNone(select(report).target)
        self.assertEqual(select(report, ranks=2).target.slots, ('b', 'c'))

    def test_duplicate_instance_report_cannot_overwrite(self):
        report = spare('spare')
        with self.assertRaisesRegex(ValueError, 'one identified report'):
            select(report, replace(report, instance='new-pod'))

    def test_gpu_slot_cannot_count_twice(self):
        report = replace(spare('spare'), free_gpu_mib=(('a', 4096),) * 4)
        with self.assertRaisesRegex(ValueError, 'GPU slot inventory'):
            select(report)

    def test_manual_fault_type_changes_rank_scope_and_target(self):
        reports = (spare('source', setup_ms=0), spare('whole'), spare('stage', slots=2, setup_ms=50))
        for fault, replica, rank, policy, action, target, ranks in (
            ('process_exit', 0, None, 'v1', 'recover', 'whole', (0, 1, 2, 3)),
            ('replica_slow', 1, None, 'v2', 'migrate', 'whole', (0, 1, 2, 3)),
            ('rank_slow', 0, 3, 'v4', 'migrate', 'stage', (2, 3)),
        ):
            with self.subTest(fault=fault):
                plan = plan_replacement(reports, fault=fault, source_replica=replica,
                                        affected_rank=rank, source_node='source', world=4, profile='lab-w4',
                                        min_gpu_mib=3072, now=100, max_age=5)
                self.assertEqual((plan.policy, plan.action, plan.decision.target.node, plan.ranks),
                                 (policy, action, target, ranks))
                if action == 'migrate':
                    command = plan.migration_command(session='run', command_id='incident-1', expected_epoch=2)
                    self.assertEqual((command.target, command.ranks, command.expected_epoch), (target, ranks, 2))
                else:
                    with self.assertRaisesRegex(ValueError, 'separate execution path'):
                        plan.migration_command(session='run', command_id='incident-1', expected_epoch=2)

    def test_partial_placement_maps_all_replica_stage_pairs(self):
        for replica in (0, 1):
            for stage, ranks in enumerate(((0, 1), (2, 3))):
                with self.subTest(replica=replica, stage=stage):
                    plan = plan_replacement((spare('target'),), fault='rank_slow', source_replica=replica,
                                            affected_rank=ranks[-1], source_node='source', world=4,
                                            profile='lab-w4', min_gpu_mib=3072, now=100, max_age=5)
                    command = plan.migration_command(session='run', command_id='slow', expected_epoch=2)
                    self.assertEqual((command.source_replica, command.ranks), (replica, ranks))
        with self.assertRaisesRegex(ValueError, 'scope is unsupported'):
            plan_replacement((spare('target'),), fault='rank_slow', source_replica=0,
                             affected_rank=4, source_node='source', world=4, profile='lab-w4',
                             min_gpu_mib=3072, now=100, max_age=5)

    def test_pool_claim_leaves_other_slots_available_and_release_needs_fresh_report(self):
        pool = SparePool({'spare'})
        report = spare('spare')
        pool.record(report)
        options = dict(profile='lab-w4', ranks=2, min_gpu_mib=3072, now=100, max_age=5)
        first = pool.claim('first', **options).target
        second = pool.claim('second', **options).target
        self.assertTrue(set(first.slots).isdisjoint(second.slots))
        self.assertIsNone(pool.decide(**options).target)
        pool.release_after_exit('first', now=100)
        pool.record(report)
        self.assertIsNone(pool.decide(**options).target)
        pool.record(replace(report, observed_at=101))
        self.assertEqual(pool.decide(**(options | {'now': 101})).target.slots, first.slots)

    def test_new_pod_cannot_overwrite_an_unreleased_claim(self):
        pool = SparePool({'spare'})
        report = spare('spare')
        pool.record(report)
        options = dict(profile='lab-w4', ranks=2, min_gpu_mib=3072, now=100, max_age=5)
        self.assertIsNotNone(pool.claim('first', **options).target)
        pool.record(replace(report, instance='replacement-pod', observed_at=100))
        self.assertEqual(pool.decide(**options).rejected, (('spare', 'not_ready'),))

    def test_session_rotation_reserves_each_target_and_preserves_pp0(self):
        pool = SparePool({'node0', 'node1', 'node3'})
        a, b, c = spare('node0', slots=6), spare('node1'), spare('node3')
        for report in (a, b, c):
            pool.record(report)
        pool.track_running('dp0', Selection(a.node, a.instance, ('gpu0', 'gpu1', 'gpu2', 'gpu3')))
        pool.track_running('dp1', Selection(b.node, b.instance, ('gpu0', 'gpu1', 'gpu2', 'gpu3')))
        options = dict(world=4, profile='lab-w4', min_gpu_mib=3072, max_age=10)

        v2 = pool.claim_replacement('v2', fault='replica_slow', source_replica=1,
                                    source_node=b.node, now=100, **options)
        self.assertEqual(v2.decision.target.node, c.node)
        pool.release_after_exit('dp1', now=100)
        pool.record(replace(b, observed_at=101))

        v4 = pool.claim_replacement('v4', fault='rank_slow', source_replica=0,
                                    source_node=a.node, affected_rank=2, now=102, **options)
        self.assertEqual(v4.decision.target.node, b.node)
        pool.release_after_exit('dp0', slots=('gpu2', 'gpu3'), now=102)
        pool.record(replace(a, observed_at=103))

        v1 = pool.claim_replacement('v1', fault='process_exit', source_replica=1,
                                    source_node=c.node, now=104, **options)
        self.assertEqual(v1.decision.target.node, a.node)
        self.assertEqual(v1.decision.target.slots, ('gpu2', 'gpu3', 'gpu4', 'gpu5'))
        pool.release_after_exit('v2', now=104)
        pool.record(replace(c, observed_at=105))
        self.assertEqual(pool.decide(profile='lab-w4', ranks=4, min_gpu_mib=3072,
                                     now=105, max_age=10).target.node, c.node)


class LabPerformancePairTest(unittest.TestCase):
    def test_pair_checks_devices_and_reports_slow_recovery(self):
        window = dict(mean_seconds=2.0, step_seconds_distribution={'p95': 2.5})
        command = dict(id='lab-1-v1', policy='v1', source_replica=1, ranks=[0, 1, 2, 3], target='node3')
        case = dict(command=command, target_gpu_uuids=['d', 'b', 'a', 'c'],
                    request_to_first_joint_seconds=30.0, fault_to_first_joint_seconds=20.0,
                    steady_5_to_16=window, placement={'ranked': [command['target']]})
        auto = dict(session='auto', accepted=True, independent_audit={'pass_all': True},
                    resources_clean=True, configuration={}, physical_nodes={'node3': 'gpu-03'},
                    runner_gpu_uuids={'node3': ['a', 'b', 'c', 'd']}, source_sha256={},
                    initial_spare_pool={'ranked': [{'node': 'node3', 'instance': 'pod'}]},
                    initial_native_5_to_16=window, cases=[case])
        fixed = deepcopy(auto)
        fixed['session'] = 'fixed'
        fixed['cases'][0].pop('placement')
        self.assertTrue(compare_pair(auto, fixed)['within_limits'])

        fixed['cases'][0]['fault_to_first_joint_seconds'] = 16.0
        comparison = compare_pair(auto, fixed)
        self.assertFalse(comparison['within_limits'])
        self.assertEqual(comparison['operations'][0]['metrics']['exit_to_first_joint']['ratio'], 1.25)

        fixed['cases'][0]['target_gpu_uuids'].reverse()
        with self.assertRaisesRegex(ValueError, 'target GPU UUIDs'):
            compare_pair(auto, fixed)


if __name__ == '__main__':
    unittest.main()
