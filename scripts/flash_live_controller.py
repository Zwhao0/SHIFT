"""Collect FLASH's bounded SHIFT PROFILE and run its CPU inference entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shift_full.flash_bridge import alert


class LiveProfile:
    def __init__(self, directory, flash_root, reference, transport, bridge):
        self.directory = directory
        self.flash_root = flash_root
        self.reference = reference
        self.transport = transport
        self.bridge = bridge
        self.state = dict(status='waiting_for_monitor' if reference else 'waiting_for_calibration')
        self.history = [self.state]
        self.event = None

    def begin_epoch(self, snapshot, store):
        epoch, first = snapshot['epoch'], snapshot['first_step']
        if epoch:
            if self.state['status'] == 'collecting':
                self.state['status'] = 'superseded'
            self.state = dict(status='waiting_for_monitor' if self.reference else 'disabled')
            self.history.append(self.state)
        self.event = None
        self.state.update(epoch=epoch, epoch_first_step=first)
        store.set('session/flash_window', f'{epoch} {first} 0')

    def poll(self, snapshot, progress, monitor_log, rows, store):
        if self.event is not None:
            self._publish()
            return
        if self.state['status'] in ('complete', 'error', 'disabled'):
            return
        trigger = monitor_log.parent / 'profile_request.json'
        if 'first_step' not in self.state:
            if self.reference and not trigger.exists():
                return
            epoch, epoch_first = self.state['epoch'], self.state['epoch_first_step']
            if min(progress.values()) < epoch_first + 7:
                return
            committed = store.multi_get(['session/progress/0', 'session/progress/1'])
            first = max(max(progress.values()), *map(int, committed)) + 3
            store.set('session/flash_window', f'{epoch} {epoch_first} {first}')
            self.state.update(status='collecting', first_step=first, last_step=first + 7,
                              scheduled_wall_time=time.time())
            if self.reference:
                self.state['monitor_request'] = json.loads(trigger.read_text())
                self.state['monitor_request_wall_time'] = trigger.stat().st_mtime
            return
        first, last = self.state['first_step'], self.state['last_step']
        missing = []
        for rank, worker in snapshot['workers'].items():
            events = [row for row in rows[worker['actor']]
                      if row['kind'] in ('flash_profile_ready', 'flash_profile_failed')
                      and row['rank'] == worker['rank'] and row.get('session_epoch') == snapshot['epoch']
                      and row['first_step'] == first]
            if any(row['kind'] == 'flash_profile_failed' for row in events):
                self.state.update(status='error', error=f'rank {rank} failed PROFILE')
                return
            if not any(row.get('last_step') == last for row in events):
                missing.append(rank)
        if missing:
            if min(progress.values()) >= last + 4:
                self.state.update(status='error', error=f'PROFILE records missing for ranks {missing}')
            return
        if min(progress.values()) >= last:
            self._score(snapshot, snapshot['replica_world_size'])
            if self.event is not None:
                self._publish()

    def _publish(self):
        self.state['stream_id'] = self.bridge.client.xadd(
            self.bridge.alert_key, {'event': json.dumps(self.event)})
        self.state['published_wall_time'] = time.time()
        self.state['published_monotonic'] = time.monotonic()
        self.event = None

    def _score(self, snapshot, world):
        started = time.monotonic()
        root = self.directory / 'workers' / 'flash_native'
        epoch = snapshot['epoch']
        scope = self.directory / f'flash_epoch_{epoch}' if epoch else self.directory
        observed = scope / 'flash_observations'
        paths = {}
        for rank, worker in snapshot['workers'].items():
            node = worker['runner']
            directory = root / worker['actor']
            if epoch:
                directory /= f'epoch_{epoch}'
            paths.setdefault(node, {})[rank] = str(directory / f'rank_{rank}.jsonl')
        try:
            observed.mkdir()
            for node, files in paths.items():
                code = ('import json; from pathlib import Path; files=' + repr(files) +
                        '; print(json.dumps({rank: Path(path).read_text() for rank, path in files.items()}))')
                for rank, raw in json.loads(self.transport(node, code)).items():
                    (observed / f'rank_{rank}.jsonl').write_text(raw)
            output = scope / ('flash_prediction.json' if self.reference else 'flash_reference.json')
            command = [sys.executable, str(self.flash_root/'deploy/lab/shift_gemm_profile.py'),
                       '--input', str(observed), '--output', str(output), '--ranks', str(2 * world)]
            if self.reference:
                command.extend(('--reference', str(self.reference)))
            else:
                command.append('--regression')
            subprocess.run(command, check=True, timeout=30, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(self.flash_root)))
            self.state.update(status='complete', output=str(output.relative_to(self.directory)),
                              collection_and_score_seconds=time.monotonic()-started)
            if self.reference:
                prediction = json.loads(output.read_text())
                if prediction['iteration_ids'] != list(range(self.state['first_step'], self.state['last_step'] + 1)):
                    raise ValueError('scored steps differ from the requested window')
                self.state['prediction'] = prediction
                rank_map = {str(rank): rank for rank in range(2 * world)}
                self.event = alert(snapshot, prediction, rank_map)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            message = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
            self.state.update(status='error', error=message[-1200:])
