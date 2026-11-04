"""Opt-in, one-cut production adapter. GPU/performance admission is separate.

Training owners alone touch Manager/model/CUDA. The B socket worker only
receives a ticket, fences the reserved owner, and acknowledges that ticket.
All retired PG endpoints wait for every B owner and presend to drain/close.
"""
import json
import socket
import threading
import time
import uuid
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist

from shift_full.overlap_handoff import OverlapHandoff
from shift_full.ownership_control import FenceWire
from shift_full.ownership_protocol import Cut, Ownership
from shift_full.ownership_runtime import ReservedB, completed_quorum, forbid_quorum_recovery
from shift_full.planned_state import validate_state


class OwnershipHandoff(OverlapHandoff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservation = self.contribution = None
        self.initial_audit = None
        self.ownership_active = False
        self.building_manager = False
        self.control_ready = threading.Event()
        self.reserved_ready = threading.Event()
        self.control_done = threading.Event()
        self.control_stop = threading.Event()
        self.control_error = None
        self.wire = self.listener = None
        self.control_thread = None
        self.protocol = None
        if self.role == 'A' and self.rank == 0:
            self.coord.set('ownership_nonce', uuid.uuid4().hex)
        if self.role in ('A', 'B'):
            self.control_thread = threading.Thread(target=self._prepare_control,
                                                   name='shift-owner-control')
            self.control_thread.start()

    def ticket(self):
        return Cut(self.new_epoch, self.cut, self.coord.get('ownership_nonce').decode())

    def _key(self, role, rank, proof):
        return f'ownership/{role}/{rank}/{proof}'

    def proof(self, proof, *, role=None, rank=None, value=None):
        role = role or self.role
        rank = self.rank if rank is None else rank
        packet = dict(cut=asdict(self.ticket()), role=role, rank=rank, proof=proof, value=value)
        encoded = json.dumps(packet, sort_keys=True)
        # Duplicate exact proof is idempotent; conflicting writers fail closed.
        actual = self.coord.compare_set(self._key(role, rank, proof), '', encoded)
        if actual.decode() != encoded:
            raise RuntimeError('conflicting ownership proof')

    def read_proof(self, role, rank, proof):
        packet = json.loads(self.coord.get(self._key(role, rank, proof)))
        if any(packet[k] != v for k, v in dict(cut=asdict(self.ticket()), role=role,
                                              rank=rank, proof=proof).items()):
            raise RuntimeError('stale ownership proof')
        return packet['value']

    def keys(self, roles, proof):
        return [self._key(role, rank, proof) for role in roles for rank in range(self.local_world)]

    def wait_keys(self, keys, seconds=90):
        end = time.monotonic() + seconds
        while not self.coord.check(keys):
            if self.coord.check(['ownership_fatal']):
                raise RuntimeError(self.coord.get('ownership_fatal').decode())
            if time.monotonic() >= end:
                raise TimeoutError('ownership barrier: ' + repr(keys))
            time.sleep(.001)

    def fail(self, exc):
        self.coord.set(f'ownership_failure/{self.role}/{self.rank}', repr(exc))
        self.emit('planned_ownership_pre_activation_failure', step=self.cut, error=repr(exc))

    def failed(self):
        return (self.coord.check(['cancel']) or self.coord.check(['prepare_error']) or
                any(self.coord.check([f'ownership_failure/{r}/{i}'])
                    for r in 'ABC' for i in range(self.local_world)))

    def check(self):
        if self.coord.check(['ownership_fatal']):
            raise RuntimeError('ownership stopped: ' + self.coord.get('ownership_fatal').decode())
        super().check()

    def _prepare_control(self):
        """No Torch, Manager, store, model, RNG, or device access in this worker."""
        try:
            port = 31720 + self.offset + self.rank
            if self.role == 'B':
                self.listener = socket.socket()
                self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.listener.bind((self.args.host_ip, port))
                self.listener.listen(1)
                self.listener.settimeout(.2)
                end = time.monotonic() + 90
                while not self.control_stop.is_set():
                    try:
                        sock, _ = self.listener.accept()
                        break
                    except socket.timeout:
                        if time.monotonic() >= end:
                            raise TimeoutError('A control accept')
                else:
                    return
            else:
                end = time.monotonic() + 90
                while True:
                    if self.control_stop.is_set():
                        return
                    try:
                        sock = socket.create_connection((self.hosts['B'], port), timeout=1)
                        break
                    except OSError:
                        if time.monotonic() >= end:
                            raise TimeoutError('B control connect')
                        self.control_stop.wait(.02)
            self.wire = FenceWire(sock, timeout=10)
            self.control_ready.set()
            if self.role == 'B':
                if not self.reserved_ready.wait(120) or self.control_stop.is_set():
                    raise TimeoutError('B reservation not published')
                self.wire.fence_and_ack(self.reservation.fence, self.rank)
        except BaseException as exc:
            self.control_error = exc
        finally:
            self.control_ready.set()
            self.control_done.set()

    def reserve_descriptor(self, manager, descriptor, local_pgs):
        """Called by the REAL B loop after quorum and before descriptor/sleep."""
        if self.role != 'B' or self.reservation is not None:
            raise RuntimeError('duplicate/wrong ownership reservation')
        completed_quorum(manager, step=self.cut, replica_index=descriptor['replica_index'])
        self.old_replica_index = descriptor['replica_index']
        self.reservation = ReservedB(manager, self.ticket(), local_pgs=local_pgs)
        self.old_epoch_revoked = True
        self.reserved_ready.set()
        self.reservation.descriptor(lambda: self.coord.set(
            f'ownership_descriptor/{self.rank}', json.dumps(descriptor)))
        self.emit('planned_ownership_reserved', step=self.cut, descriptor=descriptor,
                  ticket=asdict(self.ticket()))

    def ready_receiver(self):
        self.join()
        self.emit('planned_local_ready', manifest=self.manifest())
        self.coord.set(f'prepared/C/{self.rank}', '1')
        self.wait_keys(['cut'])
        self.cut = int(self.coord.get('cut'))
        try:
            payload = self.transport().recv_checkpoint(0, '<packed-gpu>', self.cut,
                                                       timedelta(seconds=60))
            validate_state(payload, rank=self.rank, step=self.cut,
                           audit=self.args.audit_state, epoch=self.new_epoch)
            if self.layout_manifest(payload['manifest']) != self.layout_manifest(self.manifest()):
                raise ValueError('state manifest mismatch')
            self.load(payload)
            self.emit('planned_precopy_loaded', step=self.cut)
            self.coord.set(f'preloaded/{self.rank}', str(self.cut))
            self.contribution = self.ownership_recompute(payload['hashes'])
            self.initial_audit = self.contribution.result['initial_audit']
            self.proof('initial_audit', value=self.initial_audit)
            self.proof('computed', value=self.contribution.result)
            self.proof('gradient_audit', value=True)
            self.emit('planned_state_ready', step=self.cut, epoch=self.new_epoch,
                      state_semantics='retained_preouter_contribution',
                      boundary_state_audited_before_forward=True)
        except Exception as exc:
            self.fail(exc)
        self.wait_keys(['ownership_selection'])
        if self.coord.get('ownership_selection') == b'AB':
            return self.cancel_receiver()
        # Gate describes the prospective Manager. The global activation record
        # below, rather than this construction flag, grants dispatch authority.
        self._build_gate('AC')
        return True

    def cancel_receiver(self):
        if self.ownership_active:
            raise RuntimeError('post-activation C cannot roll back')
        if self.contribution is not None and self.contribution.state in ('computed', 'sealed'):
            self.contribution.cancel()
        # State/candidate PG destruction must obey the SAME global B barrier.
        self.wait_keys(self.keys('B', 'closed'))
        self.fallback = True
        self.retired_pgs.extend(self.channels.values())
        self.channels = {}
        if self.candidate is not None:
            self.retired_pgs.append(self.candidate)
            self.candidate = None
        self.proof('cancelled')
        self.start_retired_cleanup()
        self.finish_retired_cleanup()
        self.emit('planned_activation_rejected', step=self.cut)
        return False

    def _finish_presend(self):
        if self.presend is not None:
            self.presend.join(timeout=60)
            if self.presend.is_alive():
                raise TimeoutError('precopy still owns source state')
        self.check()

    def _b_closed(self):
        # Owner has completed the real local finalizer. A control failure uses
        # this slower owner ack, never an invented ack while local work runs.
        if not self.control_done.wait(12):
            raise TimeoutError('control worker still active')
        if self.control_error is not None or not self.reservation.fence.fenced:
            self.fail(self.control_error or RuntimeError('B fence absent'))
            self.reservation.fence.control_fence(self.ticket())
        self.proof('fenced', value=True)
        self.reservation.drain_and_close(drain_local=torch.cuda.synchronize,
                                         finish_presend=self._finish_presend)
        self.proof('drained', value=True)
        self.proof('closed', value=True)
        self.retired_pgs.append(self.reservation.manager._pg)
        self.emit('planned_ownership_b_closed', step=self.cut)

    def _a_closed(self, manager):
        torch.cuda.synchronize()
        self.proof('drained', value=True)
        completed_quorum(manager, step=self.cut)
        self.old_replica_index = manager._participating_replica_rank
        try:
            if not self.control_ready.wait(90) or self.control_error is not None:
                raise self.control_error or TimeoutError('control connection absent')
            self.wire.send('fence', self.ticket(), self.rank)
            self.wire.receive('ack', self.ticket(), self.rank)
            self.proof('fenced', role='B', value=True)
        except Exception as exc:
            self.fail(exc)
        self.wait_keys(self.keys('B', 'fenced') + self.keys('A', 'drained'))
        # Every A owner really drains and every B owner really fences first.
        self.reservation = ReservedB(manager, self.ticket(), local_pgs=self.local_pgs)
        self.reservation.fence.control_fence(self.ticket())
        self.old_epoch_revoked = True
        started = time.monotonic()
        self.reservation.drain_and_close(drain_local=torch.cuda.synchronize,
                                         finish_presend=lambda: None)
        self.retired_pgs.append(manager._pg)
        self.proof('closed', value=True)
        self.emit('planned_old_manager_shutdown_complete', step=self.cut,
                  seconds=time.monotonic()-started)

    def _select(self):
        if self.role == 'A' and self.rank == 0:
            self.protocol = Ownership(self.ticket(), self.local_world)
            ready = self.keys('A', 'drained') + self.keys('B', 'fenced')
            c_ready = self.keys('C', 'gradient_audit')
            self.wait_keys(ready)
            end = time.monotonic() + 90
            while not self.coord.check(c_ready) and not self.failed():
                if time.monotonic() >= end:
                    self.fail(TimeoutError('C contribution deadline'))
                    break
                time.sleep(.001)
            members = 'AB' if self.failed() else 'AC'
            for role, proofs in [('A', ['drained']), ('B', ['fenced']),
                                 ('C', ['initial_audit', 'computed', 'gradient_audit'] if members == 'AC' else [])]:
                for rank in range(self.local_world):
                    for proof in proofs:
                        self.read_proof(role, rank, proof)
                        self.protocol.record(self.ticket(), role, rank, proof)
            self.protocol.choose(members)
            self.coord.compare_set('ownership_selection', '', members)
        self.wait_keys(['ownership_selection'])
        return self.coord.get('ownership_selection').decode()

    def _build_gate(self, members):
        if not self.activated:
            self.gate.activate(self.new_epoch, self.cut, self.cut, True)
        self.gate.members = tuple(members)
        self.fallback = members == 'AB'
        self.args.lighthouse = f'http://{self.control_host}:{31411 if self.fallback else 31410}'
        self.manager_store_port = (31602 if self.fallback else 31600) + self.offset
        if self.fallback:
            self.fallback_store = dist.TCPStore(self.args.host_ip, self.manager_store_port,
                None, self.rank == 0, timeout=timedelta(seconds=60), wait_for_workers=False)

    @staticmethod
    def forbidden_load(*args, **kwargs):
        raise RuntimeError('ownership quorum must not heal/load retained state')

    def prepare_manager(self, make_manager, members, *, prospective=None, binding=None):
        manager = None
        started = time.monotonic()
        try:
            self.building_manager = True
            manager = prospective.manager if prospective is not None else make_manager()
            expected = (self.contribution.result['descriptor']['replica_index']
                        if self.role == 'C' else self.old_replica_index)
            if prospective is None:
                forbid_quorum_recovery(manager, expected)
                manager.start_quorum(allow_heal=False, timeout=timedelta(seconds=10))
                manager.wait_quorum()
            else:
                prospective.validate(binding)
            completed_quorum(manager, step=self.cut, replica_index=expected)
            guard_started = time.monotonic()
            if self.role == 'C':
                self.contribution.seal()
            elif not self.owner_guard():
                raise RuntimeError('new Manager changed retained owner state/gradient')
            self.proof('new_quorum_' + members, value=True)
            self.emit('planned_ownership_new_manager_guard', step=self.cut, members=members,
                      seconds=time.monotonic()-started,guard_seconds=time.monotonic()-guard_started)
        except Exception as exc:
            self.fail(exc)
        finally:
            self.building_manager = False
        return manager

    def activate_manager(self, manager, members):
        ready = self.keys(members, 'new_quorum_' + members)
        if self.role == 'A' and self.rank == 0:
            end = time.monotonic() + 90
            while not self.coord.check(ready):
                if self.failed() and members == 'AC':
                    break
                if time.monotonic() >= end:
                    if members == 'AC':
                        self.fail(TimeoutError('AC owner quorum deadline; choose new AB'))
                        break
                    self.coord.set('ownership_fatal', 'new AB quorum deadline')
                    raise TimeoutError('new AB quorum deadline')
                time.sleep(.001)
            decision = 'AB' if members == 'AC' and self.failed() else members
            if decision == members:
                for role in members:
                    for rank in range(self.local_world):
                        self.read_proof(role, rank, 'new_quorum_' + members)
                self.wait_keys(self.keys('A', 'closed'))
                for rank in range(self.local_world):
                    self.read_proof('A', rank, 'closed')
                    self.protocol.record(self.ticket(), 'A', rank, 'closed')
                if members == 'AB':
                    self.wait_keys(self.keys('B', 'closed'))
                    for rank in range(self.local_world):
                        for proof in ('drained', 'closed'):
                            self.read_proof('B', rank, proof)
                            self.protocol.record(self.ticket(), 'B', rank, proof)
                self.protocol.activate(members)
            else:
                self.protocol.choose('AB')
            self.coord.compare_set('ownership_activation_' + members, '', decision)
        self.wait_keys(['ownership_activation_' + members], seconds=150)
        decision = self.coord.get('ownership_activation_' + members).decode()
        if decision != members:
            if manager is not None:
                manager.shutdown()
                # Preserve candidate endpoints until the B global drain barrier.
                self.retired_pgs.append(manager._pg)
            return False
        if manager is None or not self.coord.check(ready):
            raise RuntimeError('activation without all real new quorums')
        if self.role == 'C':
            self.contribution.activate()
            self.retired_pgs.extend(self.channels.values())
            self.channels = {}
        elif members == 'AB':
            self.retired_pgs.extend(self.channels.values())
            self.channels = {}
            if self.candidate is not None:
                self.retired_pgs.append(self.candidate)
                self.candidate = None
        self.ownership_active = True
        self.relay_done = True
        self.emit('planned_ownership_activated', step=self.cut, members=members,
                  ticket=asdict(self.ticket()))
        self.emit('planned_epoch_activated', step=self.cut, epoch=self.new_epoch,
                  members=members, irreversible_before_outer=True)
        self.start_retired_cleanup()
        return True

    def at_gradient(self, manager, gradients, make_manager, sample_ids, apply_tail):
        # C takes the explicit retained-finalizer path and never enters here.
        if self.role == 'C':
            raise RuntimeError('retained C must not repeat the local finalizer')
        self.emit('planned_local_gradient_ready', step=self.cut)
        self.emit('planned_final_freeze', step=self.cut, epoch=self.base_epoch,
                  boundary='owner_local_ready_before_control_and_selection')
        self.owner_guard = self.capture_owner_guard()
        if self.role == 'B':
            self._b_closed()
        else:
            self._a_closed(manager)
        members = self._select()
        if self.role == 'B' and members == 'AC':
            self.wait_keys(['ownership_activation_AC'])
            members = self.coord.get('ownership_activation_AC').decode()
            if members == 'AC':
                result = self.read_proof('C', self.rank, 'computed')
                actual = self.source_contribution(sample_ids, gradients)
                if any(actual[k] != result[k] for k in actual):
                    self.coord.set('ownership_fatal', f'late B contribution mismatch rank {self.rank}')
                    raise RuntimeError('late retired B contribution mismatch; no rollback permitted')
                self.emit('planned_ownership_late_exact', step=self.cut, exact=True, **actual)
                self.retired_pgs.extend(self.channels.values())
                self.channels = {}
                self.gate.activate(self.new_epoch, self.cut, self.cut, True)
                self.ownership_active = True
                self.emit('planned_retired_alive', step=self.cut)
                self.start_retired_cleanup()
                self.wait_keys(['ac_three'])
                self.finish_retired_cleanup()
                self.emit('planned_standby_verified', step=self.cut)
                return None, True
        self._build_gate(members)
        new = self.prepare_manager(make_manager, members)
        if not self.activate_manager(new, members):
            self._build_gate('AB')
            new = self.prepare_manager(make_manager, 'AB')
            if not self.activate_manager(new, 'AB'):
                raise RuntimeError('new AB activation rejected')
        return new, False

    def finish_receiver(self, make_manager):
        """After the real finalizer is defined, before any main-loop zero_grad."""
        new = self.prepare_manager(make_manager, 'AC')
        if not self.activate_manager(new, 'AC'):
            self.cancel_receiver()
            return None
        return new

    def committed(self):
        if self.ownership_active:
            self.authorize_epoch_commit()
        super().committed()

    def authorize_epoch_commit(self):
        if self.role not in self.gate.members:
            raise RuntimeError('retired owner committed')
        if self.progress['updates'] == self.cut + 1:
            if getattr(self, 'first_owned_commit', False):
                raise RuntimeError('duplicate first owned commit')
            self.first_owned_commit = True
            self.proof('committed', value=self.cut + 1)

    def authorize(self, manager):
        self.check()
        if self.activated and not self.ownership_active:
            raise RuntimeError('new Manager dispatch before global activation')
        return super().authorize(manager)

    def start_retired_cleanup(self):
        if self.cleanup_thread is not None:
            return
        def cleanup():
            try:
                torch.cuda.set_device(self.device)
                control = self.control_client(10)
                keys = self.keys('B', 'closed') + self.keys('A', 'closed')
                if self.fallback:
                    keys += self.keys('C', 'cancelled')
                deadline = time.monotonic() + 120
                while not control.check(keys):
                    if self.cleanup_cancel.wait(.01):
                        return
                    if time.monotonic() >= deadline:
                        raise TimeoutError('global B drain/close cleanup barrier')
                # closed B proofs are published only after actual local and
                # presend drain plus original Manager.shutdown, on its owner.
                for pg in {id(p): p for p in self.retired_pgs}.values():
                    if any(local is not None and (pg is local or getattr(pg, 'parent', None) is local)
                           for local in self.local_pgs):
                        raise RuntimeError('retired PG aliases local training PG')
                    pg.abort(errored=False)
                    pg.shutdown()
                    if pg._pg is not None:
                        raise RuntimeError('retired PG parent still present')
                self.emit('planned_retired_cleanup_complete', step=self.cut,
                          all_b_owners_closed=True, all_parent_absent=True)
            except BaseException as exc:
                self.errors.append(repr(exc))
                self.coord.set('cleanup_error', repr(exc))
        self.cleanup_thread = threading.Thread(target=cleanup, name='shift-retired-cleanup')
        self.cleanup_thread.start()

    def cancel_cleanup_on_exit(self):
        super().cancel_cleanup_on_exit()
        self.control_stop.set()
        self.reserved_ready.set()
        if self.wire is not None:
            self.wire.close()
        if self.listener is not None:
            self.listener.close()
        if self.control_thread is not None:
            self.control_thread.join(timeout=2)
            if self.control_thread.is_alive():
                self.emit('planned_ownership_control_incomplete')
