import tempfile
import unittest
from pathlib import Path

from shift_full.reference_audit import (
    REFERENCE_HASH_KEYS,
    REFERENCE_ANCHOR_KEYS,
    parse_reference_dump_steps,
    reference_anchor_path,
    reference_dump_payload,
    should_dump_reference,
    validate_reference_anchor,
)


def anchor_payload(*, schema=2, rank=3, step=7):
    return {
        "schema": schema,
        "model": {},
        "all_buffers": {},
        "optimizer": {},
        "scheduler": {},
        "progress": {"updates": step},
        "torch_rng": object(),
        "cuda_rng": object(),
        "python_rng": object(),
        "tracker_rng": {},
        "source_replica": 0,
        "source_rank": rank,
        "step": step,
        "manager_state": {"step": step, "batches_committed": step * 2},
        "hashes": {key: "0" * 64 for key in REFERENCE_HASH_KEYS},
    }


class ReferenceAuditTests(unittest.TestCase):
    def test_dump_steps_and_filter(self):
        selected = parse_reference_dump_steps("1, 2,11")
        self.assertEqual(selected, frozenset({1, 2, 11}))
        self.assertTrue(should_dump_reference(
            enabled=True, selected_steps=selected, next_step=11,
            replica=0, rank=7, selected_ranks=frozenset(range(8))))
        self.assertFalse(should_dump_reference(
            enabled=True, selected_steps=selected, next_step=10,
            replica=0, rank=7, selected_ranks=frozenset(range(8))))
        self.assertFalse(should_dump_reference(
            enabled=True, selected_steps=None, next_step=11,
            replica=1, rank=7, selected_ranks=frozenset(range(8))))

    def test_dump_steps_reject_empty_duplicate_and_nonpositive_values(self):
        for value in ("", "1,,2", "1,1", "0", "-1", "one"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_reference_dump_steps(value)

    def test_dump_shape_stays_compare_reference_compatible(self):
        payload = reference_dump_payload("grad", "model", "optimizer", "scheduler")
        self.assertEqual(set(payload), {"gradients", "model", "optimizer", "scheduler"})

    def test_rank_anchor_path_is_per_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(reference_anchor_path(directory, 3), Path(directory) / "anchor_r3.pt")

    def test_legacy_partial_anchor_is_routed_to_schema_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "reference_step7_r3.pt"
            old_path.touch()
            self.assertEqual(reference_anchor_path(directory, 3, step=7), old_path)
            with self.assertRaisesRegex(ValueError, "schema 2"):
                validate_reference_anchor({"gradients": {}, "model": {}}, expected_rank=3)

    def test_complete_schema2_anchor_is_accepted(self):
        self.assertEqual(validate_reference_anchor(anchor_payload(), expected_rank=3)["step"], 7)
        self.assertEqual(REFERENCE_ANCHOR_KEYS - set(anchor_payload()), set())

    def test_missing_schema_rejects_old_partial_anchor(self):
        payload = anchor_payload()
        del payload["schema"]
        with self.assertRaisesRegex(ValueError, "schema 2"):
            validate_reference_anchor(payload, expected_rank=3)

    def test_incomplete_schema2_anchor_rejects_missing_rng_or_buffers(self):
        for key in ("torch_rng", "tracker_rng", "all_buffers", "progress", "manager_state"):
            payload = anchor_payload()
            del payload[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "incomplete"):
                validate_reference_anchor(payload, expected_rank=3)

    def test_rank_and_step_must_match(self):
        with self.assertRaisesRegex(ValueError, "rank mismatch"):
            validate_reference_anchor(anchor_payload(rank=2), expected_rank=3)
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            validate_reference_anchor(anchor_payload(), expected_rank=3, expected_step=8)


if __name__ == "__main__":
    unittest.main()
