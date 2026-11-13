"""Bounded cuBLAS PROFILE in the existing SHIFT training schedule."""

import ctypes
import os
from pathlib import Path


class RankProfile:
    def __init__(self, runtime):
        self.runtime = runtime
        native = ctypes.CDLL(None)
        self.prepare = native.flash_v4_prepare_window
        self.prepare.argtypes = [ctypes.c_char_p]
        self.prepare.restype = ctypes.c_int
        self.begin = native.flash_v4_begin_iteration
        self.begin.argtypes = [ctypes.c_uint64, ctypes.c_int]
        self.begin.restype = None
        self.end = native.flash_v4_end_iteration
        self.end.argtypes = [ctypes.c_uint64]
        self.end.restype = None
        self.flush = native.flash_v4_flush
        self.flush.argtypes = []
        self.flush.restype = None
        self.directory = Path(os.environ['FLASH_V4_OUTPUT_DIR'])
        self.delay = None
        logical_rank = runtime.replica * runtime.layout.world + runtime.rank
        if logical_rank == runtime.c.get('flash_inject_rank'):
            self.delay = native.flash_lab_gemm_delay
            self.delay.argtypes = [ctypes.c_int]
            self.delay.restype = ctypes.c_int
        self.injection_offset = runtime.c.get('flash_inject_step', 36) - 1
        self.control = None
        self.epoch = -1
        self.epoch_first = 0
        self.first = 0
        self.opened = False
        self.complete = False

    def note_window(self, value):
        if value == self.control:
            return
        epoch, epoch_first, first = map(int, value.split())
        if epoch != self.runtime.epoch:
            return
        if epoch != self.epoch:
            self.opened = self.complete = False
        self.control = value
        self.epoch, self.epoch_first, self.first = epoch, epoch_first, first

    def fail(self, step, reason):
        if self.opened:
            self.flush()
            self.opened = False
        self.complete = True
        self.runtime.event('flash_profile_failed', first_step=self.first, step=step, reason=reason)

    def schedule(self, step, execute):
        r = self.runtime
        observe = r.epoch == self.epoch and not r.command and not r.in_protocol
        profile = observe and self.first and not self.complete and step >= self.first
        if profile and not self.opened:
            directory = self.directory / f'epoch_{self.epoch}' if self.epoch else self.directory
            try:
                directory.mkdir(parents=True, exist_ok=True)
                prepared = step == self.first and self.prepare(os.fsencode(directory)) == 0
            except OSError:
                prepared = False
            if not prepared:
                self.fail(step, 'missed or unavailable native PROFILE window')
                profile = False
            else:
                self.opened = True
        # The retiring owner stays slow during the handoff. The replacement
        # only sees a new injection after it reads the committed epoch.
        injection_start = self.epoch_first + self.injection_offset
        inject = (r.epoch == self.epoch and self.delay is not None and self.epoch < r.c['flash_inject_epochs']
                  and injection_start <= step < injection_start + 48)
        if profile:
            self.begin(step, 1)
        if inject:
            self.delay(1)
        try:
            return execute()
        finally:
            if inject:
                count = self.delay(0)
                r.event('flash_compute_delay', step=step, delay_launches=count)
            if profile:
                self.end(step)

    def committed(self, step):
        if self.opened and step == self.first + 7:
            self.flush()
            self.opened = False
            self.complete = True
            self.runtime.event('flash_profile_ready', first_step=self.first, last_step=step)

    def skipped(self, step):
        if self.opened:
            self.fail(step, 'training did not commit a profiled step')

    def interrupted(self, step):
        if self.opened:
            self.fail(step, 'training command interrupted PROFILE')
