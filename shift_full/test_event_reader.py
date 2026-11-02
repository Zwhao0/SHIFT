"""Incremental experiment evidence collection regression tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shift_full.event_reader import merge, remote_code


class EventReaderTests(unittest.TestCase):
    def read_packet(self, directory, generation, offsets=None):
        result = subprocess.run(
            [sys.executable, "-c", remote_code(directory, generation, offsets)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_incremental_partial_line_and_generation_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            event_file = root / "events_g0_r0.jsonl"
            supervisor_file = root / "supervisor.jsonl"
            ready = b'{"kind":"ready","time":1}\n'
            supervisor = b'{"kind":"injected","time":2}\n'
            event_file.write_bytes(ready)
            supervisor_file.write_bytes(supervisor)

            first = self.read_packet(root, 0)
            self.assertEqual(first["rows"][event_file.name][0]["kind"], "ready")
            self.assertEqual(first["rows"][supervisor_file.name][0]["kind"], "injected")
            self.assertEqual(first["offsets"][event_file.name], len(ready))

            restored = b'{"kind":"restored","time":3}\n'
            partial = b'{"kind":"commit","time":4'
            with event_file.open("ab") as stream:
                stream.write(restored + partial)
            second = self.read_packet(root, 0, first["offsets"])
            self.assertEqual([row["kind"] for row in second["rows"][event_file.name]], ["restored"])
            self.assertEqual(second["offsets"][event_file.name], len(ready) + len(restored))

            with event_file.open("ab") as stream:
                stream.write(b',"step":4}\n')
            third = self.read_packet(root, 0, second["offsets"])
            self.assertEqual(third["rows"][event_file.name], [{"kind": "commit", "time": 4, "step": 4}])

            next_generation = root / "events_g1_r0.jsonl"
            next_generation.write_bytes(b'{"kind":"ready","time":5}\n')
            switched = self.read_packet(root, 1, third["offsets"])
            self.assertEqual(switched["rows"][next_generation.name][0]["time"], 5)
            self.assertNotIn(event_file.name, switched["rows"])
            self.assertNotIn(event_file.name, switched["offsets"])

    def test_merge_bounds_each_kind_and_keeps_ready(self):
        filename = "events_g0_r0.jsonl"
        cache = merge({}, {"rows": {filename: [
            {"kind": "ready", "time": 1},
            {"kind": "phase", "time": 2},
        ]}})
        commits = [{"kind": "commit", "time": n} for n in range(3, 23)]
        cache = merge(cache, {"rows": {filename: commits}})

        rows = cache[filename]
        self.assertEqual([row["kind"] for row in rows if row["kind"] == "ready"], ["ready"])
        self.assertFalse(any(row["kind"] == "phase" for row in rows))
        self.assertEqual([row["time"] for row in rows if row["kind"] == "commit"], list(range(11, 23)))
        self.assertEqual([row["time"] for row in rows], sorted(row["time"] for row in rows))


if __name__ == "__main__":
    unittest.main()
