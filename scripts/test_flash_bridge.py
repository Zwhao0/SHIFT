import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.flash_live_controller import LiveProfile
from shift_full.flash_bridge import Bridge, alert, command, connect, membership
from shift_full.flash_live import RankProfile
from shift_full.flash_monitor import EpochSamples
from shift_full.session_control import Command
from shift_full.session_layout import SessionLayout


class FlashBridgeTest(unittest.TestCase):
    def setUp(self):
        self.layout = SessionLayout(4)
        self.active = {}
        self.actors = {}
        for replica in range(2):
            for stage in range(2):
                label = f'g0-{replica}-{stage}'
                ranks = self.layout.stage_ranks(stage)
                self.actors[label] = dict(generation=0, node=f'node{replica}', physical_node=f'gpu-0{replica + 1}',
                                          pod_uid=f'pod-{replica}',
                                          gpu_uuids=[f'gpu-{replica}-{rank}' for rank in ranks])
                for index, rank in enumerate(ranks):
                    self.active[replica, rank] = (label, index)
        self.initial = membership('flash-bridge-test', 0, 1, self.layout, self.active, self.actors)
        self.prediction = dict(alarm_rank=6, manual_integration_only=False,
                               abstention_reasons=[], iteration_ids=list(range(41, 49)))
        self.rank_map = {str(rank): rank for rank in range(8)}

    def test_localized_rank_maps_to_the_configured_scope(self):
        event = alert(self.initial, self.prediction, self.rank_map)
        full = Command.from_dict(command(self.initial, event, 'v2', 48, 32))
        stage = Command.from_dict(command(self.initial, event, 'v4', 48, 32))
        self.assertEqual((full.source_replica, full.ranks, full.target), (1, (0, 1, 2, 3), 'auto'))
        self.assertEqual((stage.source_replica, stage.ranks, stage.target), (1, (2, 3), 'auto'))
        self.assertEqual(command(self.initial, event, 'v4', 50, 32)['id'], stage.id)
        permutation = {**self.rank_map, '6': 3, '3': 6}
        self.assertEqual(alert(self.initial, self.prediction, permutation)['rank'], 3)

    def test_stale_and_replaced_worker_alerts_are_fenced(self):
        old_event = alert(self.initial, self.prediction, self.rank_map)
        self.actors['replacement'] = dict(generation=1, node='node3', physical_node='gpu-03',
                                          pod_uid='new-pod', gpu_uuids=['new-gpu-0', 'new-gpu-1'])
        self.active.update({(1, 2): ('replacement', 0), (1, 3): ('replacement', 1)})
        new = membership(self.initial['session'], 1, 66, self.layout,
                         self.active, self.actors, 'completed-command')
        with self.assertRaisesRegex(ValueError, 'epoch'):
            command(new, old_event, 'v4', 74, 32)
        self.prediction['iteration_ids'] = list(range(67, 75))
        current = alert(new, self.prediction, self.rank_map)
        self.assertEqual(current['worker']['gpu_uuid'], 'new-gpu-0')
        self.assertEqual(new['workers']['5'], self.initial['workers']['5'])
        self.assertEqual(command(new, current, 'v4', 74, 32)['expected_epoch'], 1)
        with self.assertRaisesRegex(ValueError, 'active worker'):
            command(new, dict(current, worker=old_event['worker']), 'v4', 74, 32)

    def test_only_committed_recent_automatic_windows_trigger(self):
        for prediction in (dict(self.prediction, alarm_rank=None),
                           dict(self.prediction, manual_integration_only=True),
                           dict(self.prediction, abstention_reasons=['ambiguous'])):
            self.assertIsNone(alert(self.initial, prediction, self.rank_map))
        with self.assertRaises(ValueError):
            alert(self.initial, dict(self.prediction, iteration_ids=[41] * 8), self.rank_map)
        event = alert(self.initial, self.prediction, self.rank_map)
        for progress in (47, 81):
            with self.assertRaisesRegex(ValueError, 'observation window'):
                command(self.initial, event, 'v4', progress, 32)
        cross_session = copy.deepcopy(event)
        cross_session['session'] = 'other'
        with self.assertRaisesRegex(ValueError, 'session'):
            command(self.initial, cross_session, 'v4', 48, 32)

    def test_monitor_waits_for_the_new_worker_at_the_committed_step(self):
        self.actors['replacement'] = dict(generation=1, node='node3', physical_node='gpu-03',
                                          pod_uid='new-pod', gpu_uuids=['new-gpu-0'])
        self.active[1, 2] = ('replacement', 0)
        snapshot = membership(self.initial['session'], 1, 66, self.layout, self.active, self.actors)
        rows = {name: [] for name in self.actors}
        for (replica, rank), (actor, _) in self.active.items():
            rows[actor].append(dict(kind='session_commit', session_epoch=1, step=66,
                                    participants=2, contributed=True, finite=True,
                                    replica=replica, rank=rank, seconds=0.25))
        replacement = rows['replacement'].pop()
        rows['g0-1-1'].append(replacement)
        rows['replacement'].append(dict(replacement, session_epoch=0))
        feed = EpochSamples(snapshot)
        self.assertEqual(list(feed.collect(rows)), [])
        rows['replacement'].append(replacement)
        self.assertEqual(list(feed.collect(rows)),
                         [dict(iteration=1, step=66, time_ms=250.0, active_workers=8)])
        self.assertEqual(list(feed.collect(rows)), [])

    def test_reprofile_uses_new_epoch_commits_and_current_store_progress(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'profile_request.json').write_text('{}')
            store = Mock()
            store.multi_get.return_value = [b'103', b'102']
            profiler = LiveProfile(directory, None, directory / 'reference.json', None, Mock())
            profiler._score = Mock()
            snapshot = dict(self.initial, epoch=1, first_step=67)
            profiler.event = {'epoch': 0}
            profiler.begin_epoch(snapshot, store)
            self.assertIsNone(profiler.event)
            log = directory / 'train.jsonl'
            profiler.poll(snapshot, {0: 100, 1: 100}, log, {}, store)
            store.set.assert_called_with('session/flash_window', '1 67 106')
            rows = {actor: [] for actor in self.actors}
            for worker in snapshot['workers'].values():
                rows[worker['actor']].append(dict(kind='flash_profile_ready', rank=worker['rank'],
                    session_epoch=0, first_step=106, last_step=113))
            profiler.poll(snapshot, {0: 113, 1: 113}, log, rows, store)
            profiler._score.assert_not_called()
            for events in rows.values():
                events.extend(dict(event, session_epoch=1) for event in tuple(events))
            profiler.poll(snapshot, {0: 113, 1: 113}, log, rows, store)
            profiler._score.assert_called_once_with(snapshot, 4)

    def test_interrupted_profile_drains_before_the_same_process_rearms(self):
        runtime = SimpleNamespace(replica=0, rank=0, layout=self.layout, epoch=0,
                                  command=None, in_protocol=False, c={}, event=Mock())
        native = Mock()
        native.flash_v4_prepare_window.return_value = 0
        with TemporaryDirectory() as directory, patch.dict(os.environ, FLASH_V4_OUTPUT_DIR=directory), \
                patch('shift_full.flash_live.ctypes.CDLL', return_value=native):
            profile = RankProfile(runtime)
            profile.note_window(b'0 1 10')
            profile.schedule(10, lambda: None)
            profile.interrupted(10)
            native.flash_v4_flush.assert_called_once()
            runtime.event.assert_called_once_with('flash_profile_failed', first_step=10, step=10,
                                                  reason='training command interrupted PROFILE')
            runtime.epoch = 1
            profile.note_window(b'1 24 62')
            for step in range(62, 70):
                profile.schedule(step, lambda: None)
            self.assertEqual(native.flash_v4_flush.call_count, 1)
            profile.committed(69)
            self.assertEqual(native.flash_v4_flush.call_count, 2)
            native.flash_v4_prepare_window.assert_called_with(os.fsencode(Path(directory)/'epoch_1'))
            runtime.event.assert_called_with('flash_profile_ready', first_step=62, last_step=69)

    def test_lab_delay_persists_until_the_retiring_epoch_changes(self):
        runtime = SimpleNamespace(replica=0, rank=0, layout=self.layout, epoch=0,
                                  command=object(), in_protocol=True,
                                  c={'flash_inject_rank': 0, 'flash_inject_epochs': 2}, event=Mock())
        native = Mock()
        native.flash_lab_gemm_delay.return_value = 64
        with TemporaryDirectory() as directory, patch.dict(os.environ, FLASH_V4_OUTPUT_DIR=directory), \
                patch('shift_full.flash_live.ctypes.CDLL', return_value=native):
            profile = RankProfile(runtime)
            profile.note_window(b'0 1 0')
            profile.schedule(36, lambda: None)
            runtime.event.assert_called_once_with('flash_compute_delay', step=36, delay_launches=64)
            runtime.epoch = 1
            profile.schedule(37, lambda: None)
            self.assertEqual(native.flash_lab_gemm_delay.call_count, 2)

    @unittest.skipUnless(os.getenv('SHIFT_TEST_REDIS_SOCKET'), 'requires a private Redis server')
    def test_real_redis_membership_alert_and_resume(self):
        client = connect(os.environ['SHIFT_TEST_REDIS_SOCKET'])
        bridge = Bridge(client, self.initial['session'])
        self.addCleanup(client.delete, bridge.membership_key, bridge.alert_key)
        bridge.publish(self.initial)
        snapshot = json.loads(client.get(bridge.membership_key))
        event = alert(snapshot, self.prediction, self.rank_map)
        stream_id = client.xadd(bridge.alert_key, {'event': json.dumps(event)})
        cursor, raw = bridge.read()
        self.assertEqual(cursor, stream_id)
        self.assertEqual(command(snapshot, json.loads(raw), 'v4', 48, 32)['ranks'], [2, 3])
        bridge.cursor = cursor
        self.assertIsNone(bridge.read())
        self.assertEqual(Bridge(client, self.initial['session']).read()[0], stream_id)


if __name__ == '__main__':
    unittest.main()
