"""CPU-only Linux tests for orphaned child cleanup."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from shift_full.process_tree import enable_subreaper


@unittest.skipIf(sys.platform.startswith("linux"), "Linux supports child subreapers")
class UnsupportedPlatformTests(unittest.TestCase):
    def test_subreaper_failure_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "only on Linux"):
            enable_subreaper()


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux child subreapers")
class ProcessTreeTests(unittest.TestCase):
    def test_reaps_adopted_orphan_and_preserves_excluded_popen_status(self):
        script = r'''
import json, os, subprocess, sys, time
from shift_full.process_tree import enable_subreaper, reap_orphans

enable_subreaper()
protected = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2); raise SystemExit(23)"])
child_code = chr(10).join(("import os,time", "pid=os.fork()", "if pid == 0: time.sleep(.1); os._exit(41)", "print(pid, flush=True)"))
parent = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
orphan_pid = int(parent.stdout.readline())
parent.wait(timeout=2)
reaped = reap_orphans(exclude_pids=(protected.pid,), timeout=1)
protected_status = protected.wait(timeout=2)
print(json.dumps({"orphan_pid": orphan_pid, "reaped": reaped,
                  "protected_pid": protected.pid, "protected_status": protected_status}))
'''
        result = subprocess.run([sys.executable, "-c", script], check=True,
                                capture_output=True, text=True, timeout=5,
                                cwd=Path(__file__).resolve().parents[1])
        data = json.loads(result.stdout)
        self.assertIn(data["orphan_pid"], data["reaped"])
        self.assertEqual(data["protected_status"], 23)
        self.assertNotIn(data["protected_pid"], data["reaped"])


if __name__ == "__main__":
    unittest.main()
