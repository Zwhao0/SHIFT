"""Prospective FT quorum; grants preparation only, never dispatch authority.

No native method is replaced. Owner supplies a boundary/identity/version guard;
the real future must finish and pass it before any receiver forward/backward.
"""
import threading
from datetime import timedelta
from shift_full.ownership_runtime import completed_quorum, forbid_quorum_recovery


class ProspectiveQuorum:
    def __init__(self, manager, *, cut, replica_index, binding, unchanged):
        self.owner = threading.get_ident()
        self.manager, self.cut, self.replica_index = manager, cut, replica_index
        self.binding, self.unchanged = binding, unchanged
        self.state = 'new'

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('prospective quorum requires owner')

    def start(self):
        self._owner()
        if self.state != 'new' or not self.manager._use_async_quorum:
            raise RuntimeError('prospective quorum requires fresh async Manager')
        if not self.unchanged(self.binding):
            raise RuntimeError('stale prospective boundary before start')
        forbid_quorum_recovery(self.manager, self.replica_index)
        self.state = 'pending'
        self.manager.start_quorum(allow_heal=False, timeout=timedelta(seconds=10))

    def before_forward(self):
        self._owner()
        if self.state != 'pending':
            raise RuntimeError('prospective future already consumed/invalid')
        try:
            self.manager.wait_quorum()
            completed_quorum(self.manager, step=self.cut.step, replica_index=self.replica_index)
            if not self.unchanged(self.binding):
                raise RuntimeError('stale prospective boundary after future')
        except BaseException:
            self.state = 'invalid'
            raise
        self.state = 'prepared'
        return self.manager

    def validate(self, binding):
        self._owner()
        if self.state != 'prepared' or binding != self.binding:
            self.state = 'invalid'
            raise RuntimeError('stale prospective epoch/step/role/version')
