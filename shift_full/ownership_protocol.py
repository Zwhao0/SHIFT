"""CPU ownership specification, not connected to the training path.

An adapter must prove GPU/quorum quiescence before reporting it here. This
module never calls TorchFT, CUDA, model code or a process-group destructor.
"""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class Cut:
    epoch: int
    step: int
    nonce: str

    def __post_init__(self):
        if (type(self.epoch) is not int or self.epoch < 1 or
                type(self.step) is not int or self.step < 0 or
                not isinstance(self.nonce, str) or not self.nonce):
            raise ValueError('invalid cut identity')


class LocalFence:
    """B reserves the cut on its owner BEFORE descriptor publication/sleep.

    Reservation already forbids the old outer, so the control thread cannot
    race a successful old-outer permission check. A later AB fallback also
    requires a NEW Manager/epoch; the fence is never cleared.
    """
    def __init__(self, cut):
        self.cut = cut
        self.owner = threading.get_ident()
        self.lock = threading.Lock()
        self.reserved = self.fenced = self.drained = False
        self.closed = False

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('training owner required')

    def _cut(self, cut):
        if cut != self.cut:
            raise ValueError('stale epoch/cut/nonce')

    def reserve(self, *, quorum_complete, state_readers_done):
        self._owner()
        with self.lock:
            if self.reserved or not quorum_complete or not state_readers_done:
                raise RuntimeError('cut needs completed non-healing quorum/readers')
            self.reserved = True

    def control_fence(self, cut):
        self._cut(cut)
        with self.lock:
            if not self.reserved:
                raise RuntimeError('descriptor reservation missing')
            self.fenced = True
            return cut  # Only now may the dedicated control client send its ack.

    def preouter(self, cut, *, local_work_drained):
        self._owner()
        self._cut(cut)
        with self.lock:
            if not self.reserved or not local_work_drained:
                raise RuntimeError('local work/reservation incomplete')
            self.drained = True
            return 'park'  # Always; no old-epoch dispatch, even before ack.

    def close_old_manager(self):
        self._owner()
        with self.lock:
            if not self.fenced or not self.drained:
                raise RuntimeError('cannot shut down B while local work runs')
            self.closed = True

    def allow_old_outer(self):
        with self.lock:
            if self.reserved:
                raise RuntimeError('old outer/commit/quorum permanently revoked')
            return True


class Ownership:
    """Single coordinator specification; receive handlers must serialize calls.

    No default success on timeout. A failure before activation selects AB, but
    must wait for B drain/closure. After activation even partial new outer work
    makes an old-B rollback illegal, before or after should_commit.
    """
    def __init__(self, cut, world=8):
        if type(world) is not int or world < 1:
            raise ValueError('invalid role count')
        self.cut, self.world = cut, world
        self.proofs = set()
        self.decision = self.members = None
        self.commits = set()

    def record(self, cut, role, rank, proof):
        if cut != self.cut or role not in ('A', 'B', 'C') or type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError('invalid proof identity')
        allowed = {'A': {'drained', 'closed'}, 'B': {'fenced', 'drained', 'closed'},
                   'C': {'initial_audit', 'computed', 'gradient_audit'}}
        if proof not in allowed[role]:
            raise ValueError('invalid proof kind')
        if proof == 'closed' and (role, rank, 'drained') not in self.proofs:
            raise RuntimeError('manager closure requires local drain')
        if proof == 'closed' and role == 'B' and ('B', rank, 'fenced') not in self.proofs:
            raise RuntimeError('B closure requires fence')
        if proof == 'closed' and role == 'A' and not self._all('B', 'fenced'):
            raise RuntimeError('A closure requires every old B quorum/fence ack')
        if proof in ('computed', 'gradient_audit') and ('C', rank, 'initial_audit') not in self.proofs:
            raise RuntimeError('initial boundary audit must precede recompute')
        if proof == 'gradient_audit' and ('C', rank, 'computed') not in self.proofs:
            raise RuntimeError('gradient audit requires recompute')
        self.proofs.add((role, rank, proof))

    def _all(self, role, *proofs):
        return all((role, rank, proof) in self.proofs for rank in range(self.world) for proof in proofs)

    def choose(self, members):
        if members not in ('AC', 'AB') or self.members is not None:
            raise RuntimeError('invalid/post-activation decision')
        if self.decision == 'AB' and members != 'AB':
            raise RuntimeError('cancel cannot be reversed')
        if members == 'AC' and not (self._all('B', 'fenced') and self._all('A', 'drained') and
                                    self._all('C', 'initial_audit', 'computed', 'gradient_audit')):
            raise RuntimeError('missing role proofs')
        self.decision = members

    def activate(self, members):
        if self.members is not None or self.decision != members or not self._all('A', 'closed'):
            raise RuntimeError('unready/already activated')
        if members == 'AB' and not self._all('B', 'fenced', 'drained', 'closed'):
            raise RuntimeError('fallback must wait for slow B and new epoch')
        self.members = members

    def commit(self, cut, role, rank, step):
        if (cut != self.cut or self.members is None or role not in ('A', 'B', 'C') or role not in self.members or
                type(rank) is not int or not 0 <= rank < self.world or step != cut.step + 1):
            raise RuntimeError('old/wrong owner or step')
        key = role, rank
        if key in self.commits:
            raise RuntimeError('duplicate commit')
        self.commits.add(key)

    def cleanup_allowed(self, role):
        # Aborting even A's old outer can reach B. Both ends wait for B drain.
        return (self.members is not None and role in ('A', 'B') and
                self._all('B', 'fenced', 'drained', 'closed') and
                self._all(role, 'drained', 'closed'))
