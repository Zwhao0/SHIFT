"""Linux child-subreaper support for reaping orphaned process-tree members."""

import ctypes
import os
from pathlib import Path
import sys
import time


_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_prctl_fn = None


def _require_linux():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("child subreapers are supported only on Linux")


def _call_prctl(option, arg=0):
    global _prctl_fn
    _require_linux()
    if _prctl_fn is None:
        _prctl_fn = ctypes.CDLL(None, use_errno=True).prctl
        _prctl_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
                              ctypes.c_ulong, ctypes.c_ulong]
        _prctl_fn.restype = ctypes.c_int
    ctypes.set_errno(0)
    arg = ctypes.c_void_p(arg) if isinstance(arg, int) else ctypes.cast(arg, ctypes.c_void_p)
    result = _prctl_fn(option, arg, 0, 0, 0)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def enable_subreaper():
    """Make this process the adoption target for orphaned descendants."""
    _require_linux()
    _call_prctl(_PR_SET_CHILD_SUBREAPER, 1)


def _is_subreaper():
    value = ctypes.c_int()
    _call_prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(value))
    return value.value == 1


def _direct_children():
    """Return (pid, state) pairs for this process's current direct children."""
    parent = os.getpid()
    children = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            # Fields after comm start with state (field 3), then ppid (field 4).
            if int(fields[1]) == parent:
                children.append((int(entry.name), fields[0]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return children


def reap_orphans(exclude_pids=(), timeout=2.0):
    """Reap adopted children that exit within ``timeout`` and return their PIDs.

    The caller must exclude every direct child whose status is owned by a
    ``subprocess.Popen`` object.  ``waitpid`` consumes a child status, so this
    function never scans with ``waitpid(-1, ...)``.  It polls only direct
    children in /proc and calls waitpid only for zombies.
    """
    _require_linux()
    if not _is_subreaper():
        raise RuntimeError("enable_subreaper() must be called before reaping")
    excluded = {int(pid) for pid in exclude_pids}
    deadline = time.monotonic() + max(0.0, float(timeout))
    reaped = []
    while True:
        pending = False
        for pid, state in _direct_children():
            if pid in excluded:
                continue
            pending = True
            if state != "Z":
                continue
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if done:
                reaped.append(done)
        if not pending or time.monotonic() >= deadline:
            return reaped
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
