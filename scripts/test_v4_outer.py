"""CPU-only checks for V4 outer state and fixed sample-role validation."""

import unittest
from types import SimpleNamespace

from shift_full.v4_outer import Outer, validate_progress, validate_sample_role


class OuterProgressTest(unittest.TestCase):
    def test_progress_round_trip_value(self):
        self.assertEqual(validate_progress({"step": 0, "batches_committed": 0}),
                         {"step": 0, "batches_committed": 0})
        state = {"step": 17, "batches_committed": 34}
        self.assertEqual(validate_progress(state), state)

    def test_progress_rejects_mismatch_and_invalid_types(self):
        for state in (
            None,
            {"step": -1, "batches_committed": -2},
            {"step": True, "batches_committed": 2},
            {"step": 1, "batches_committed": 1},
            {"step": 1, "batches_committed": 2.0},
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                validate_progress(state)

    def test_sample_role_is_fixed_per_replica(self):
        self.assertEqual(validate_sample_role(0, 0), 0)
        self.assertEqual(validate_sample_role(1, 1), 1)
        with self.assertRaises(RuntimeError):
            validate_sample_role(0, 1)
        with self.assertRaises(RuntimeError):
            validate_sample_role(1, True)

    def test_sample_role_rejects_invalid_replica(self):
        for replica in (-1, 2, True):
            with self.subTest(replica=replica), self.assertRaises(ValueError):
                validate_sample_role(replica, replica)


class QuorumManagerStub:
    def __init__(self, *, participants=2, participating_rank=0, step=3):
        self.participants = participants
        self.participating_rank_value = participating_rank
        self.step = step
        self.calls = []

    def state_dict(self):
        self.calls.append("state_dict")
        return {"step": self.step, "batches_committed": 2 * self.step}

    def current_step(self):
        self.calls.append("current_step")
        return self.step

    def start_quorum(self, allow_heal):
        self.calls.append(("start_quorum", allow_heal))

    def wait_quorum(self):
        self.calls.append("wait_quorum")

    def num_participants(self):
        self.calls.append("num_participants")
        return self.participants

    def participating_rank(self):
        self.calls.append("participating_rank")
        return self.participating_rank_value


class OuterQuorumTest(unittest.TestCase):
    def make_outer(self, *, async_quorum, replica=0, participants=2,
                   participating_rank=None, step=3):
        if participating_rank is None:
            participating_rank = replica
        value = Outer.__new__(Outer)
        value.manager = QuorumManagerStub(participants=participants,
                                          participating_rank=participating_rank,
                                          step=step)
        value.replica = replica
        value.rank = 0
        value.epoch = 0
        value.event_rows = []
        value.event = lambda kind, **fields: value.event_rows.append((kind, fields))
        value._closed = False
        value._step_started = None
        value._reduced_step = None
        value._effective_timeouts = dict(timeout_seconds=120.0,
                                         quorum_timeout_seconds=120.0,
                                         connect_timeout_seconds=120.0)
        value._async = async_quorum
        value._audit = False
        value._quorum_checked_step = None
        value._quorum_started = None
        value.pg = SimpleNamespace(migration_epoch=0, configure_count=1)
        return value

    def test_async_begin_starts_without_waiting_and_complete_is_idempotent(self):
        outer = self.make_outer(async_quorum=True, step=3)
        outer.begin(3)
        self.assertIn(("start_quorum", False), outer.manager.calls)
        self.assertNotIn("wait_quorum", outer.manager.calls)
        self.assertEqual([kind for kind, _ in outer.event_rows], ["outer_quorum_requested"])

        outer._complete_quorum(3)
        self.assertEqual(outer.manager.calls.count("wait_quorum"), 1)
        first_call_order = outer.manager.calls.index("wait_quorum")
        self.assertLess(first_call_order, outer.manager.calls.index("num_participants"))
        self.assertLess(first_call_order, outer.manager.calls.index("participating_rank"))
        self.assertEqual([kind for kind, _ in outer.event_rows],
                         ["outer_quorum_requested", "outer_quorum_wait", "outer_begin"])
        self.assertEqual(outer.event_rows[-1][1]["participants"], 2)
        self.assertEqual(outer.event_rows[-1][1]["participating_rank"], 0)

        calls_before_repeat = list(outer.manager.calls)
        outer._complete_quorum(3)
        self.assertEqual(outer.manager.calls, calls_before_repeat)
        self.assertEqual([kind for kind, _ in outer.event_rows].count("outer_begin"), 1)

    def test_sync_begin_waits_and_validates_immediately(self):
        outer = self.make_outer(async_quorum=False, replica=1, step=5)
        outer.begin(5)
        self.assertEqual(outer.manager.calls.count("wait_quorum"), 1)
        self.assertEqual([kind for kind, _ in outer.event_rows], ["outer_begin"])
        self.assertEqual(outer.event_rows[0][1]["participating_rank"], 1)

    def test_wrong_participants_and_fixed_sample_role_are_rejected(self):
        for kwargs in (
            dict(async_quorum=True, participants=1, participating_rank=0),
            dict(async_quorum=False, participants=3, participating_rank=0),
            dict(async_quorum=True, replica=0, participants=2, participating_rank=1),
            dict(async_quorum=False, replica=1, participants=2, participating_rank=0),
        ):
            with self.subTest(kwargs=kwargs):
                outer = self.make_outer(step=3, **kwargs)
                if kwargs["async_quorum"]:
                    outer.begin(3)
                    self.assertNotIn("wait_quorum", outer.manager.calls)
                    with self.assertRaises(RuntimeError):
                        outer._complete_quorum(3)
                else:
                    with self.assertRaises(RuntimeError):
                        outer.begin(3)
                self.assertEqual(outer.manager.calls.count("wait_quorum"), 1)

    def test_step_progress_mismatch_is_rejected(self):
        outer = self.make_outer(async_quorum=False, step=4)
        with self.assertRaises(RuntimeError):
            outer.begin(3)
        self.assertNotIn("wait_quorum", outer.manager.calls)

        outer = self.make_outer(async_quorum=True, step=3)
        outer.begin(3)
        outer.manager.step = 4
        with self.assertRaises(RuntimeError):
            outer._complete_quorum(3)
        self.assertEqual(outer.manager.calls.count("wait_quorum"), 1)


if __name__ == "__main__":
    unittest.main()
