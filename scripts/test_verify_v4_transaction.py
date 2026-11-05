"""Synthetic CPU fixtures for the independent transaction-run auditor."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_v4_transaction import summarize, verify
from shift_full.reference_audit import REFERENCE_HASH_KEYS
from shift_full.v4_protocol import Plan


def _hashes(rank, step):
    return {key: hashlib.sha256(f'{key}/{rank}/{step}'.encode()).hexdigest()
            for key in REFERENCE_HASH_KEYS}


def _sample_ids(step, replica, rank):
    return [(step - 1) * 16 + micro * 4 + replica * 2 + (rank % 4) // 2
            for micro in range(4)] if step else []


class Fixture:
    def __init__(self, root, outcomes, *, reference=False, cross_node_tp=False,
                 timeouts=(), invalid_rounds=(), cancel_round=None, recover_rounds=(),
                 boundary_validation='full', final_reference=False):
        self.root = Path(root)
        self.cycles = len(outcomes)
        self.outcomes = outcomes
        self.cuts = [2 + i for i in range(self.cycles)]
        self.final_step = self.cuts[-1] + 2
        self.reference = reference
        self.cross_node_tp = cross_node_tp
        self.timeouts = set(timeouts)
        self.invalid_rounds = set(invalid_rounds)
        self.recover_rounds = set(recover_rounds)
        self.boundary_validation = boundary_validation
        self.final_reference_enabled = final_reference
        source = ('B', 0)
        self.rounds = []
        self.actor_specs = {(0, role) for role in 'ABCDE'}
        for round_id, decision in enumerate(outcomes):
            role, generation = source
            target = 'C' if role == 'B' else 'B'
            self.actor_specs.add((round_id, target))
            active_after = (role, generation) if decision == 'retain' else (target, round_id)
            self.rounds.append(dict(round=round_id, cut=self.cuts[round_id], source=source,
                target=target, decision=decision, candidate=(target, round_id),
                active_after=active_after))
            if decision == 'replace':
                source = (target, round_id)
        self.labels = {spec: f'g{spec[0]}_{spec[1]}' for spec in self.actor_specs}
        self.events = {(label, device): [] for label in self.labels.values() for device in range(4)}
        self.ticks = {key: 0 for key in self.events}
        self.pid = {key: 20000 + index for index, key in enumerate(self.events)}
        self.summary = dict(name='synthetic-transaction', transactional_switch=True,
            overlap_peer=True, outer_manager=True, functional_pass=True, error=None,
            resources_clean=True, cleanup_errors=[], cycles=self.cycles,
            between_steps=1, plan=dict(cut=2, post_steps=2, microbatches=4, seq_len=8),
            final_step=self.final_step, cancel_round=(next((r['round'] for r in self.rounds
                if r['decision'] == 'retain'), -1) if cancel_round is None else cancel_round),
            cross_node_tp=cross_node_tp,
            topology=dict(global_batch=16),
            outer_timeout=90, reference_run='fixture-reference' if reference else None,
            reference_sha256=None, oracle_exact=True if reference else None,
            decisions=[], source_recovery_stops=[],
            source_recovery_scope=dict(failure_window='after_freeze_before_ready',
                transport_failure='fail_stop'),
            actors=[dict(generation=g, role=role) for g, role in sorted(self.actor_specs)])
        if boundary_validation != 'full':
            self.summary['boundary_validation'] = boundary_validation
        self.reference = self._make_reference() if reference else None
        self.final_reference = self._make_reference() if final_reference else None
        if self.final_reference:
            self.final_reference['run'] = 'v4_fixture-final-reference'

    def _rank(self, label, device):
        role = label.split('_', 1)[1]
        return Plan.logical_rank({'D': 'A', 'E': 'B'}.get(role, role), device, self.cross_node_tp)

    def _replica(self, label):
        return int(label.endswith('_D') or label.endswith('_E'))

    def _manager(self, step):
        return dict(step=step, batches_committed=2 * step)

    def _manager_identity(self, label, device):
        marker = 30000 + sum(map(ord, label)) * 10 + device
        return dict(manager=marker, server=marker + 1, client=marker + 2,
                    executor=marker + 3, pg=marker + 4, service_epoch=0)

    def _readiness(self, label, device, cut, epoch, target):
        return dict(schema=1, epoch=epoch, rank=self._rank(label, device),
                    cut=cut, target=target, restored=True)

    def _training_identity(self, label, device):
        marker = 40000 + sum(map(ord, label)) * 10 + device
        return dict(model=marker, optimizer=marker + 1, ddp=marker + 2,
                    parameters=[marker + 3], adam=[marker + 4], buckets=[marker + 5])

    def _make_reference(self):
        workers = {}
        for replica in range(2):
            for rank in range(8):
                workers[f'{replica}/{rank}'] = {
                    str(step): dict(hashes=_hashes(rank, step), sample_ids=_sample_ids(step, replica, rank),
                                    outer_manager=self._manager(step))
                    for step in range(self.final_step + 1)}
        return dict(schema=1, run='fixture-reference', steps=self.final_step,
                    seq_len=8, microbatches=4, seed=20261106, source_sha256={}, workers=workers)

    def add(self, label, device, kind, round_id, **fields):
        role = label.split('_', 1)[1]
        rank = self._rank(label, device)
        key = (label, device)
        self.ticks[key] += 1
        self.events[key].append(dict(kind=kind, role=role, rank=rank, replica=self._replica(label),
            device=device, pid=self.pid[key], time=100000.0 + self.ticks[key],
            monotonic=float(self.ticks[key]), round=round_id,
            actor_round=int(label[1:label.index('_')]), **fields))

    def _initial_actor_events(self):
        active = {'g0_A', 'g0_B', 'g0_D', 'g0_E'}
        for label in self.labels.values():
            generation = int(label[1:label.index('_')])
            role = label.split('_', 1)[1]
            for device in range(4):
                if label in active:
                    self.add(label, device, 'outer_ready', 0, outer_epoch=0,
                        timeout_seconds=90, quorum_timeout_seconds=90, connect_timeout_seconds=90)
                if self.reference and generation == 0 and role in ('A', 'B', 'D', 'E'):
                    rank = self._rank(label, device); replica = self._replica(label)
                    self.add(label, device, 'reference_initial', 0, step=0, hashes=_hashes(rank, 0),
                        sample_ids=[], outer_manager=self._manager(0))

    def _commit(self, label, device, step, phase, round_id, epoch, *, peer=False):
        rank = self._rank(label, device); replica = self._replica(label)
        manager_step = step - 1
        self.add(label, device, 'outer_quorum_requested', round_id,
                 outer_epoch=epoch, step=manager_step)
        if peer:
            self.add(label, device, 'peer_compute_begin', round_id, step=manager_step)
            self.add(label, device, 'peer_native_gradient_ready', round_id, step=manager_step)
        self.add(label, device, 'outer_quorum_wait', round_id,
                 outer_epoch=epoch, step=manager_step, seconds=0.2)
        self.add(label, device, 'outer_begin', round_id, outer_epoch=epoch,
            step=manager_step, participants=2, participating_rank=replica)
        if peer:
            self.add(label, device, 'peer_outer_gradient_ready', round_id,
                     step=manager_step, seconds=0.3)
        self.add(label, device, 'outer_commit', round_id, outer_epoch=epoch, step=manager_step,
            committed_step=step, batches_committed=2 * step, global_fenced=True)
        self.add(label, device, 'commit', round_id, epoch=epoch, phase=phase, step=step,
            sample_ids=_sample_ids(step, replica, rank), finite=True,
            global_activation_observed=(phase == 'after'))
        if self.reference:
            self.add(label, device, 'reference_comparison', round_id, step=step, phase=phase,
                exact=True, hashes=_hashes(rank, step), sample_ids=_sample_ids(step, replica, rank),
                outer_manager=self._manager(step))

    def _round_events(self):
        # Initial training reaches cut zero before the first transaction opens.
        for label in ('g0_A', 'g0_B', 'g0_D', 'g0_E'):
            role = label.split('_', 1)[1]
            for device in range(4):
                rank = self._rank(label, device); replica = self._replica(label)
                for step in range(1, self.cuts[0] + 1):
                    self._commit(label, device, step, 'before', 0, 0)

        for state in self.rounds:
            r, cut = state['round'], state['cut']
            source_label = self.labels[(state['source'][1], state['source'][0])]
            candidate_label = self.labels[(state['candidate'][1], state['candidate'][0])]
            active_labels = ('g0_A', 'g0_D', 'g0_E', source_label)
            for label in active_labels:
                for device in range(4):
                    rank = self._rank(label, device)
                    state_hashes = None if self.boundary_validation == 'runtime' else _hashes(rank, cut)
                    self.add(label, device, 'boundary_selected', r, epoch=r, step=cut,
                             requested_cut=cut)
                    self.add(label, device, 'freeze', r, step=cut, hashes=state_hashes,
                             identity_preserved=True)
                    self.add(label, device, 'outer_retention_boundary', r, step=cut,
                        identity=self._manager_identity(label, device),
                        training_identity=self._training_identity(label, device),
                        manager_state=self._manager(cut))
            for device in range(4):
                rank = self._rank(source_label, device)
                state_hashes = None if self.boundary_validation == 'runtime' else _hashes(rank, cut)
                channel = dict(transport='packed_gpu', bytes=104, gpu_bytes=64,
                               control_bytes=40, cpu_tensor_bytes=10,
                               hashes=state_hashes, oracle_hashes=None)
                self.add(candidate_label, device, 'candidate_prepared', r)
                recovery = r in self.recover_rounds
                if recovery:
                    self.add(source_label, device, 'old_owner_rejected', r, step=cut)
                    self.add(source_label, device, 'source_stalled', r, step=cut)
                    peer_hashes = _hashes(self._rank('g0_E', device), cut)
                    if self.boundary_validation == 'runtime':
                        peer_hashes = None
                    peer_channel = dict(channel, hashes=peer_hashes, packet_source='peer')
                    self.add('g0_E', device, 'peer_state_authorized', r, step=cut,
                        hashes=peer_hashes, expected_hashes=state_hashes,
                        **({'version_checked': True} if self.boundary_validation == 'runtime' else {}))
                    self.add('g0_E', device, 'state_sent', r, step=cut, **peer_channel)
                    self.add(candidate_label, device, 'state_received', r, step=cut, **peer_channel)
                else:
                    source_channel = dict(channel, packet_source='source')
                    self.add(source_label, device, 'state_sent', r, step=cut, **source_channel)
                    self.add(candidate_label, device, 'state_received', r, step=cut, **source_channel)
                if r in self.invalid_rounds and device == 3:
                    self.add(candidate_label, device, 'candidate_rejected', r, step=cut,
                        hashes=(None if self.boundary_validation == 'runtime' else _hashes(rank, cut + 100)),
                        expected_hashes=state_hashes,
                        reason='complete state signature mismatch')
                elif r not in self.timeouts or device < 2:
                    self.add(candidate_label, device, 'candidate_validated', r, step=cut,
                             hashes=state_hashes, expected_hashes=state_hashes, reason=None,
                             **({'readiness': self._readiness(candidate_label, device, cut, r + 1, state['target'])}
                                if self.boundary_validation == 'runtime' else {}))

            results = {str(self._rank(candidate_label, device)): True for device in range(4)}
            if r in self.invalid_rounds:
                results[str(self._rank(candidate_label, 3))] = False
            decision = state['decision']
            owner = state['target'] if decision == 'replace' else state['source'][0]
            timed_out = r in self.timeouts
            if decision == 'replace':
                decision_results = results
            else:
                decision_results = {} if timed_out else results
            recovering = r in self.recover_rounds
            if recovering:
                self.add('g0_A', 0, 'source_recovery_selected', r, step=cut,
                    source=state['source'][0], target=state['target'])
            self.add('g0_A', 0, 'switch_decision', r, step=cut, epoch=r + 1,
                decision=decision, owner=owner, timeout=timed_out, candidate_results=decision_results,
                source_recovery=recovering,
                **({'validation': 'runtime'} if self.boundary_validation == 'runtime' else {}))

            # Per-worker retention/activation events preserve production causal order.
            for label in ('g0_A', 'g0_D', 'g0_E'):
                for device in range(4):
                    rank = self._rank(label, device)
                    self.add(label, device, 'outer_retained', r, step=cut,
                        identity=self._manager_identity(label, device),
                        training_identity=self._training_identity(label, device),
                        manager_state=self._manager(cut), channel_reconnected=rank in
                            ({1, 3, 5, 7} if self.cross_node_tp else {4, 5, 6, 7}))
            if decision == 'retain':
                for device in range(4):
                    self.add(source_label, device, 'outer_retained', r, step=cut,
                        identity=self._manager_identity(source_label, device),
                        training_identity=self._training_identity(source_label, device),
                        manager_state=self._manager(cut), channel_reconnected=self._rank(source_label, device) in
                            ({1, 3, 5, 7} if self.cross_node_tp else {4, 5, 6, 7}))
                    self.add(source_label, device, 'inner_groups_retained_after_cancel', r, step=cut)
                    self.add('g0_A', device, 'inner_groups_retained_after_cancel', r, step=cut)
                for device in range(4):
                    self.add(candidate_label, device, 'candidate_cancelled', r, step=cut)
                    self.add(candidate_label, device, 'standby_done', r, step=cut)
            else:
                for device in range(4):
                    if not recovering:
                        self.add(source_label, device, 'old_owner_rejected', r, step=cut)
                        self.add(source_label, device, 'retired', r, step=cut)
                    self.add(candidate_label, device, 'outer_ready', r, outer_epoch=r + 1,
                        timeout_seconds=90, quorum_timeout_seconds=90, connect_timeout_seconds=90)
                    self.add(candidate_label, device, 'candidate_activated', r, step=cut,
                        identity=self._manager_identity(candidate_label, device),
                        training_identity=self._training_identity(candidate_label, device),
                        manager_state=self._manager(cut), channel_reconnected=self._rank(candidate_label, device) in
                            ({1, 3, 5, 7} if self.cross_node_tp else {4, 5, 6, 7}),
                        owner=state['target'], decision='replace',
                        hashes=(None if self.boundary_validation == 'runtime' else
                                _hashes(self._rank(candidate_label, device), cut)),
                        identity_preserved=True, peer_compute_started=True, peer_gradients_ready=True,
                        **({'readiness': self._readiness(candidate_label, device, cut, r + 1, state['target'])}
                           if self.boundary_validation == 'runtime' else {}))
            # D/E publish retained state, then start their first post-decision step.
            for role in ('D', 'E'):
                label = 'g0_' + role
                for device in range(4):
                    rank = self._rank(label, device)
                    hashes = None if self.boundary_validation == 'runtime' else _hashes(rank, cut)
                    self.add(label, device, 'peer_inner_groups_retained', r, identity_preserved=True)
                    self.add(label, device, 'peer_compute_ready', r, step=cut,
                             hashes=hashes, identity_preserved=True)
            for device in range(4):
                a_fields = dict(epoch=r + 1, step=cut,
                    hashes=(None if self.boundary_validation == 'runtime' else _hashes(device, cut)),
                    identity_preserved=True, owner=owner, decision=decision,
                    peer_compute_started=True, peer_gradients_ready=True)
                self.add('g0_A', device, 'activated', r, **a_fields)
            if decision == 'retain':
                for device in range(4):
                    self.add(source_label, device, 'activated', r,
                        epoch=r + 1, step=cut,
                        hashes=(None if self.boundary_validation == 'runtime' else _hashes(self._rank(source_label, device), cut)),
                        identity_preserved=True, owner=owner, decision=decision,
                        peer_compute_started=True, peer_gradients_ready=True)
            else:
                for device in range(4):
                    self.add(candidate_label, device, 'activated', r,
                        epoch=r + 1, step=cut,
                        hashes=(None if self.boundary_validation == 'runtime' else _hashes(self._rank(candidate_label, device), cut)),
                        identity_preserved=True, owner=owner, decision=decision,
                        peer_compute_started=True, peer_gradients_ready=True)

            end = self.cuts[r + 1] if r + 1 < self.cycles else self.final_step
            for device in range(4):
                self._commit('g0_A', device, cut + 1, 'after', r, r + 1,
                             peer=False) if end > cut else None
            # Emit each post segment for all active participants, adding peer wait proof on its first step.
            active_role, active_generation = state['active_after']
            active_label = self.labels[(active_generation, active_role)]
            for label in ('g0_A', 'g0_D', 'g0_E', active_label):
                for device in range(4):
                    role = label.split('_', 1)[1]
                    replica = self._replica(label); rank = self._rank(label, device)
                    for step in range(cut + 1, end + 1):
                        peer = role in ('D', 'E') and step == cut + 1
                        if label == 'g0_A' and step == cut + 1:
                            # Already added below with the common loop.
                            pass
                        if label == 'g0_A' and step == cut + 1:
                            continue
                        self._commit(label, device, step, 'after', r, r + 1, peer=peer)

    def _final_events(self):
        final_role, final_generation = self.rounds[-1]['active_after']
        final_label = self.labels[(final_generation, final_role)]
        for label in ('g0_A', 'g0_D', 'g0_E', final_label):
            role = label.split('_', 1)[1]
            for device in range(4):
                rank = self._rank(label, device); replica = self._replica(label)
                if self.boundary_validation == 'runtime':
                    self.add(label, device, 'final_audit_ready', self.cycles - 1, step=self.final_step)
                    self.add(label, device, 'final_audit_begin', self.cycles - 1, step=self.final_step)
                    self.add(label, device, 'state_hash', self.cycles - 1, step=self.final_step,
                             final_audit=True, calls=1)
                hashes = _hashes(rank, self.final_step)
                if self.final_reference_enabled:
                    self.add(label, device, 'final_reference_comparison', self.cycles - 1,
                             step=self.final_step, exact=True, hashes=hashes)
                comparison = None
                if self.reference:
                    comparison = dict(exact={key: True for key in REFERENCE_HASH_KEYS},
                        trajectory_exact=True, manager_exact=True, tensors={},
                        method='independent_nine_state_sha256')
                self.add(label, device, 'done', self.cycles - 1, step=self.final_step,
                    comparison=comparison, identity_preserved=True,
                    hashes=hashes,
                    **({'state_hash_calls': 1} if self.boundary_validation == 'runtime' else {}))

    def write(self):
        self._initial_actor_events()
        self._round_events()
        self._final_events()
        self.summary['decisions'] = [
            {key: event.get(key) for key in ('round', 'step', 'decision', 'owner', 'epoch',
                'timeout', 'candidate_results', 'source_recovery', 'validation')}
            for event in self.events[('g0_A', 0)] if event['kind'] == 'switch_decision']
        self.summary['source_recovery_stops'] = [dict(round=r,
            generation=self.rounds[r]['source'][1], role=self.rounds[r]['source'][0],
            unit_stopped=True, stop_returncode=0,
            pre=dict(LoadState='loaded', ActiveState='active', SubState='running', MainPID=25000+r),
            post=dict(LoadState='not-found', ActiveState='inactive', SubState='dead', MainPID=0))
            for r in sorted(self.recover_rounds)]
        self.root.mkdir(parents=True, exist_ok=True)
        for (label, device), events in self.events.items():
            (self.root / f'{label}{device}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
        if self.reference:
            raw_reference = json.dumps(self.reference, separators=(',', ':')).encode()
            (self.root / 'reference_signatures.json').write_bytes(raw_reference)
            self.summary['reference_sha256'] = hashlib.sha256(raw_reference).hexdigest()
        if self.final_reference:
            raw_final_reference = json.dumps(self.final_reference, separators=(',', ':')).encode()
            (self.root / 'final_reference_signatures.json').write_bytes(raw_final_reference)
            self.summary['final_reference_run'] = self.final_reference['run'].removeprefix('v4_')
            self.summary['final_reference_sha256'] = hashlib.sha256(raw_final_reference).hexdigest()
        self.summary['actors'] = [dict(generation=g, role=role) for g, role in sorted(self.actor_specs)]
        pids = {'node0': range(1000, 1008), 'node1': range(2000, 2008), 'node3': range(3000, 3008)}
        resources = {node: dict(company={str(pid): 'VLLM::fixture' for pid in ids}, extra_gpu=[])
                     for node, ids in pids.items()}
        for filename in ('resources_before.json', 'resources_after.json'):
            (self.root / filename).write_text(json.dumps(resources))
        (self.root / 'summary.json').write_text(json.dumps(self.summary))
        return self.root


class VerifyV4TransactionTest(unittest.TestCase):
    def fixture(self, outcomes, *, reference=False, cross_node_tp=False, timeouts=(), invalid_rounds=(), cancel_round=None,
                recover_rounds=(), boundary_validation='full', final_reference=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        return Fixture(root, outcomes, reference=reference, cross_node_tp=cross_node_tp,
                       timeouts=timeouts, invalid_rounds=invalid_rounds,
                       cancel_round=cancel_round, recover_rounds=recover_rounds,
                       boundary_validation=boundary_validation,
                       final_reference=final_reference).write()

    def test_single_round_replace_and_retain(self):
        for outcome in ('replace', 'retain'):
            with self.subTest(outcome=outcome):
                audit = verify(self.fixture([outcome]))
                self.assertTrue(audit['pass_all'], audit['errors'])
                self.assertEqual(audit['decisions'], [outcome])
                self.assertEqual(audit['commit_count'], 16 * 4)

    def test_multi_round_replace_then_retain(self):
        audit = verify(self.fixture(['replace', 'retain']))
        self.assertTrue(audit['pass_all'], audit['errors'])
        self.assertEqual(audit['cuts'], [2, 3])
        self.assertEqual(len(audit['peer_outer_gradient_ready_seconds']), 16)

    def test_timeout_may_have_partial_candidate_validation_evidence(self):
        audit = verify(self.fixture(['retain'], timeouts={0}, cancel_round=-1))
        self.assertTrue(audit['pass_all'], audit['errors'])

    def test_invalid_candidate_hash_retains_existing_source(self):
        audit = verify(self.fixture(['retain'], invalid_rounds={0}, cancel_round=-1))
        self.assertTrue(audit['pass_all'], audit['errors'])

    def test_multi_round_retain_then_replace(self):
        audit = verify(self.fixture(['retain', 'replace']))
        self.assertTrue(audit['pass_all'], audit['errors'])

    def test_cross_node_tp_uses_even_odd_logical_rank_binding(self):
        audit = verify(self.fixture(['replace'], cross_node_tp=True))
        self.assertTrue(audit['pass_all'], audit['errors'])

    def test_source_recovery_uses_authorized_peer_snapshot_and_stops_source(self):
        root = self.fixture(['replace'], recover_rounds={0})
        audit = verify(root)
        self.assertTrue(audit['pass_all'], audit['errors'])
        summary = json.loads((root / 'summary.json').read_text())
        self.assertEqual(summary['decisions'][0]['source_recovery'], True)
        self.assertEqual(summary['source_recovery_stops'], [
            dict(round=0, generation=0, role='B', unit_stopped=True,
                 stop_returncode=0,
                 pre=dict(LoadState='loaded', ActiveState='active', SubState='running', MainPID=25000),
                 post=dict(LoadState='not-found', ActiveState='inactive', SubState='dead', MainPID=0))])

    def test_source_recovery_rejects_missing_peer_authorization(self):
        root = self.fixture(['replace'], recover_rounds={0})
        path = root / 'g0_E0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [x for x in events if x['kind'] != 'peer_state_authorized']
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('peer recovery state' in e for e in audit['errors']))

    def test_source_recovery_rejects_mismatched_peer_hashes_and_missing_stop(self):
        root = self.fixture(['replace'], recover_rounds={0})
        path = root / 'g0_E2.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        auth = next(x for x in events if x['kind'] == 'peer_state_authorized')
        auth['hashes'] = _hashes(auth['rank'], auth['step'] + 7)
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        summary_path = root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['source_recovery_stops'] = []
        summary_path.write_text(json.dumps(summary))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('peer recovery state' in e for e in audit['errors']))
        self.assertTrue(any('stopped old source units' in e for e in audit['errors']))

    def test_source_recovery_rejects_recovery_marker_changed_in_summary(self):
        root = self.fixture(['replace'], recover_rounds={0})
        summary_path = root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['decisions'][0]['source_recovery'] = False
        summary_path.write_text(json.dumps(summary))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('immutable round 0' in e for e in audit['errors']))

    def test_source_recovery_stop_must_bind_exact_source_generation(self):
        root = self.fixture(['replace'], recover_rounds={0})
        summary_path = root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['source_recovery_stops'][0]['generation'] = 4
        summary_path.write_text(json.dumps(summary))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('stopped old source units' in e for e in audit['errors']))

    def test_source_recovery_stop_rejects_missing_or_failed_stop_proof(self):
        for field, value in (('pre', None), ('stop_returncode', 1)):
            with self.subTest(field=field):
                root = self.fixture(['replace'], recover_rounds={0})
                summary_path = root / 'summary.json'
                summary = json.loads(summary_path.read_text())
                if value is None:
                    summary['source_recovery_stops'][0][field].pop('MainPID')
                else:
                    summary['source_recovery_stops'][0][field] = value
                summary_path.write_text(json.dumps(summary))
                audit = verify(root)
                self.assertFalse(audit['pass_all'])
                self.assertTrue(any('stopped cleanly' in e for e in audit['errors']))

    def test_source_recovery_requires_scope_limit(self):
        root = self.fixture(['replace'], recover_rounds={0})
        summary_path = root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['source_recovery_scope']['failure_window'] = 'during_transport'
        summary_path.write_text(json.dumps(summary))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('scope states' in e for e in audit['errors']))

    def test_reference_manifest_and_comparisons_are_recomputed(self):
        audit = verify(self.fixture(['replace'], reference=True))
        self.assertTrue(audit['pass_all'], audit['errors'])

    def test_runtime_validation_and_final_reference_pass(self):
        root = self.fixture(['replace'], boundary_validation='runtime', final_reference=True)
        audit = verify(root)
        self.assertTrue(audit['pass_all'], audit['errors'])
        self.assertEqual(audit['boundary_validation'], 'runtime')

    def _edit_event(self, root, filename, predicate, edit):
        path = Path(root) / filename
        events = [json.loads(line) for line in path.read_text().splitlines()]
        event = next((x for x in events if predicate(x)), None)
        self.assertIsNotNone(event, f'fixture event not found: {filename}')
        edit(event)
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))

    def test_runtime_rejects_missing_candidate_readiness(self):
        root = self.fixture(['replace'], boundary_validation='runtime')
        self._edit_event(root, 'g0_C0.jsonl', lambda x: x.get('kind') == 'candidate_validated',
                         lambda x: x.pop('readiness'))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('runtime readiness' in e for e in audit['errors']))

    def test_runtime_rejects_wrong_candidate_readiness_rank_and_cut(self):
        for field, delta in (('rank', 1), ('cut', 1)):
            with self.subTest(field=field):
                root = self.fixture(['replace'], boundary_validation='runtime')
                self._edit_event(root, 'g0_C0.jsonl', lambda x: x.get('kind') == 'candidate_validated',
                                 lambda x: x['readiness'].__setitem__(field, x['readiness'][field] + delta))
                audit = verify(root)
                self.assertFalse(audit['pass_all'])
                self.assertTrue(any('runtime readiness' in e for e in audit['errors']))

    def test_runtime_requires_final_done_hash(self):
        root = self.fixture(['replace'], boundary_validation='runtime')
        self._edit_event(root, 'g0_A0.jsonl', lambda x: x.get('kind') == 'done',
                         lambda x: x.pop('hashes'))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('final done/identity' in e for e in audit['errors']))

    def test_runtime_rejects_hash_smuggled_into_boundary(self):
        root = self.fixture(['replace'], boundary_validation='runtime')
        self._edit_event(root, 'g0_B0.jsonl', lambda x: x.get('kind') == 'freeze',
                         lambda x: x.__setitem__('hashes', _hashes(x['rank'], x['step'])))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('runtime boundary carries a state hash' in e for e in audit['errors']))

    def test_runtime_rejects_final_reference_hash_mismatch(self):
        root = self.fixture(['replace'], boundary_validation='runtime', final_reference=True)
        reference_path = root / 'final_reference_signatures.json'
        reference = json.loads(reference_path.read_text())
        reference['workers']['0/0'][str(self.fixture_final_step(root))]['hashes']['model'] = 'f' * 64
        raw = json.dumps(reference, separators=(',', ':')).encode()
        reference_path.write_bytes(raw)
        summary_path = root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['final_reference_sha256'] = hashlib.sha256(raw).hexdigest()
        summary_path.write_text(json.dumps(summary))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('final reference signature replica0/rank0' in e for e in audit['errors']))

    def fixture_final_step(self, root):
        return json.loads((Path(root) / 'summary.json').read_text())['final_step']

    def test_summarize_returns_auditor_and_raw_hashes(self):
        result = summarize(self.fixture(['retain']))
        self.assertEqual(len(result['actual_auditor_sha256']), 64)
        self.assertEqual(len(result['raw_sha256']), 20)

    def mutate_worker(self, root, filename, predicate):
        path = Path(root) / filename
        events = [json.loads(line) for line in path.read_text().splitlines()]
        changed = False
        for event in events:
            if predicate(event):
                yield event
                changed = True
                break
        if not changed:
            raise AssertionError('fixture event not found: ' + filename)
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))

    def test_duplicate_sample_is_rejected(self):
        root = self.fixture(['replace'])
        path = root / 'g0_A0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        commit = next(x for x in events if x['kind'] == 'commit' and x['step'] == 1)
        commit['sample_ids'][1] = commit['sample_ids'][0]
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('sample IDs' in e for e in audit['errors']))

    def test_missing_candidate_validation_is_rejected(self):
        root = self.fixture(['replace'])
        path = root / 'g0_C0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [x for x in events if x['kind'] != 'candidate_validated']
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('candidate' in e for e in audit['errors']))

    def test_missing_activation_marker_after_commit_is_rejected(self):
        root = self.fixture(['replace'])
        path = root / 'g0_C0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        commit = next(x for x in events if x['kind'] == 'commit' and x['phase'] == 'after')
        commit['global_activation_observed'] = False
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('activation marker' in e for e in audit['errors']))

    def test_commit_before_candidate_activation_is_rejected(self):
        root = self.fixture(['replace'])
        path = root / 'g0_C0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        commit_index = next(i for i, x in enumerate(events)
                            if x['kind'] == 'commit' and x['phase'] == 'after')
        commit = events.pop(commit_index)
        activation_index = next(i for i, x in enumerate(events) if x['kind'] == 'candidate_activated')
        events.insert(activation_index, commit)
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('commit follows activation' in e for e in audit['errors']))

    def test_missing_global_fence_and_resource_node_are_rejected(self):
        root = self.fixture(['retain'])
        path = root / 'g0_B0.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        outer = next(x for x in events if x['kind'] == 'outer_commit')
        outer['global_fenced'] = False
        path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        before_path = root / 'resources_before.json'
        resources = json.loads(before_path.read_text())
        del resources['node3']
        before_path.write_text(json.dumps(resources))
        audit = verify(root)
        self.assertFalse(audit['pass_all'])
        self.assertTrue(any('global fenced' in e for e in audit['errors']))
        self.assertFalse(audit['resources_clean'])


if __name__ == '__main__':
    unittest.main()
