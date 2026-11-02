"""Small offline tests for rotation evidence validation helpers."""

import unittest

from .verify_rotation import (
    HASH_FIELDS,
    _common_steps,
    _hashes_ok,
    _max_gap,
    _role_hashes_match,
)


class RotationValidatorTests(unittest.TestCase):
    def test_gap_crossing_window_keeps_real_interval(self):
        self.assertEqual(_max_gap([0.0, 10.0, 20.0], 5.0, 6.0), 10.0)

    def test_missing_and_mismatched_restoration_hashes_fail(self):
        hashes = {name: f"value-{name}" for name in HASH_FIELDS}
        self.assertTrue(_hashes_ok({
            "expected_hashes": hashes,
            "actual_hashes": dict(hashes),
            "hashes_match": True,
        }))
        self.assertFalse(_hashes_ok({
            "expected_hashes": {}, "actual_hashes": {}, "hashes_match": True,
        }))
        mismatched = dict(hashes, model="different")
        self.assertFalse(_hashes_ok({
            "expected_hashes": hashes,
            "actual_hashes": mismatched,
            "hashes_match": False,
        }))

    def test_no_audit_accepts_empty_hash_dicts_only_when_paired(self):
        roles = {
            (rid, rank): {7: {"hashes": {}}}
            for rid in (0, 1) for rank in range(8)
        }
        errors = []
        self.assertEqual(_role_hashes_match(roles, 7, errors, False), (True, 0))
        roles[(1, 7)][7]["hashes"] = None
        errors = []
        self.assertEqual(_role_hashes_match(roles, 7, errors, False), (False, 0))
        self.assertTrue(errors)

    def test_pre_fault_common_step_uses_actual_time_cutoff(self):
        roles = {}
        for rid in (0, 1):
            for rank in range(8):
                rows = [
                    {"kind": "commit", "step": 9, "time": 9.0,
                     "participants": 2, "contributed": True},
                    {"kind": "commit", "step": 10,
                     "time": 10.1 if (rid, rank) == (1, 7) else 9.9,
                     "participants": 2, "contributed": True},
                ]
                roles[(rid, 0, rank)] = rows
        steps, _ = _common_steps(
            roles, {0: 0, 1: 0}, (0, 1), participants=2, before_time=10.0,
        )
        self.assertEqual(steps, [9])


if __name__ == "__main__":
    unittest.main()
