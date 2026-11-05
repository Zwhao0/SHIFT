"""CPU-only tests for the live-session command protocol."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shift_full.session_control import (  # noqa: E402
    Command,
    CommandError,
    ReentrantRouteError,
    RouteResult,
    Router,
    StaleCommandError,
)


def payload(command_id="c1", **fields):
    row = {
        "schema": 1,
        "session": "train-session-1",
        "id": command_id,
        "expected_epoch": 0,
        "action": "recover",
        "policy": "v1",
        "source_replica": 0,
        "target": "node-c",
    }
    row.update(fields)
    return row


class FakeRuntime:
    def __init__(self, session="train-session-1", epoch=0, step=3):
        self.session = session
        self.epoch = epoch
        self.progress = {"updates": step}
        self.at_committed_boundary = True
        self.placements = []
        self.stop_calls = 0
        self.placement_error = None

    def validate_placement(self, command):
        self.placements.append(command)
        if self.placement_error is not None:
            raise self.placement_error

    def request_stop(self):
        self.stop_calls += 1
        return "stop-requested"


class RecordingHandler:
    def __init__(self, policy, calls):
        self.policy = policy
        self.calls = calls

    def request(self, runtime, command):
        self.calls.append((self.policy, runtime, command))
        return f"{self.policy}:{command.action}"


class SessionControlTests(unittest.TestCase):
    def test_three_policies_route_sequentially_in_one_session_and_cache_success(self):
        calls = []
        router = Router()
        for policy in ("v1", "v2", "v4"):
            router.register(policy, RecordingHandler(policy, calls))
        runtime = FakeRuntime(step=12)
        commands = [
            Command.from_dict(payload("v1-command")),
            Command.from_dict(payload("v2-command", action="migrate", policy="v2")),
            Command.from_dict(payload("v4-command", action="migrate", policy="v4",
                                      ranks=[7, 5, 4, 6])),
        ]

        results = [router.route(command, runtime) for command in commands]

        self.assertEqual([result.value for result in results],
                         ["v1:recover", "v2:migrate", "v4:migrate"])
        self.assertEqual([call[0] for call in calls], ["v1", "v2", "v4"])
        self.assertEqual(commands[0].ranks, tuple(range(8)))
        self.assertEqual(commands[1].ranks, tuple(range(8)))
        self.assertEqual(commands[2].ranks, tuple(range(4, 8)))
        self.assertEqual(router.route(commands[0], runtime), results[0])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(runtime.placements), 3)

    def test_same_id_with_different_payload_is_rejected(self):
        router = Router()
        calls = []
        router.register("v1", RecordingHandler("v1", calls))
        runtime = FakeRuntime()
        router.route(Command.from_dict(payload("same-id")), runtime)

        changed = Command.from_dict(payload("same-id", target="node-d"))
        with self.assertRaises(CommandError):
            router.route(changed, runtime)

    def test_stale_epoch_and_uncommitted_runtime_are_rejected(self):
        router = Router()
        router.register("v1", RecordingHandler("v1", []))
        with self.assertRaises(StaleCommandError):
            router.route(Command.from_dict(payload(expected_epoch=1)), FakeRuntime(epoch=0))

        runtime = FakeRuntime()
        runtime.at_committed_boundary = False
        with self.assertRaises(RuntimeError):
            router.route(Command.from_dict(payload("not-committed")), runtime)

    def test_parser_rejects_invalid_action_policy_ranks_and_bool_integers(self):
        invalid = [
            payload(action="delete"),
            payload(action="migrate"),
            payload(ranks=[0, 1, 2, 3]),
            payload(ranks=[0, 1, 2, 3, 4, 5, 6, True]),
            payload(expected_epoch=True),
            payload(source_replica=True),
            payload(at_step=True),
            payload(schema=True),
            payload(target="192.0.2.1"),
            payload(session="unsafe/session"),
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(CommandError):
                Command.from_dict(row)

        for row in (
            payload("v4-half", action="migrate", policy="v4", ranks=[0, 1, 4, 5]),
            payload("v4-dup", action="migrate", policy="v4", ranks=[0, 1, 2, 2]),
            payload("v4-bool", action="migrate", policy="v4", ranks=[0, 1, 2, True]),
        ):
            with self.subTest(row=row), self.assertRaises(CommandError):
                Command.from_dict(row)

    def test_future_step_is_pending_then_runs_at_requested_boundary(self):
        calls = []
        router = Router()
        router.register("v1", RecordingHandler("v1", calls))
        command = Command.from_dict(payload("scheduled", at_step=5))
        runtime = FakeRuntime(step=4)

        pending = router.route(command, runtime)
        self.assertEqual(pending, RouteResult("pending", at_step=5, committed_step=4))
        self.assertEqual(calls, [])
        self.assertEqual(runtime.placements, [])

        runtime.progress = {"updates": 5}
        self.assertEqual(router.route(command, runtime).value, "v1:recover")
        self.assertEqual(len(calls), 1)

    def test_placement_error_does_not_cache_success(self):
        calls = []
        router = Router()
        router.register("v1", RecordingHandler("v1", calls))
        runtime = FakeRuntime()
        command = Command.from_dict(payload("placement"))
        runtime.placement_error = LookupError("placement unavailable")

        with self.assertRaisesRegex(LookupError, "placement unavailable"):
            router.route(command, runtime)
        self.assertEqual(calls, [])
        runtime.placement_error = None
        self.assertEqual(router.route(command, runtime).value, "v1:recover")
        self.assertEqual(len(calls), 1)

    def test_handler_failure_can_retry_same_command(self):
        calls = []

        class FailsOnce:
            def request(self, runtime, command):
                calls.append(command.id)
                if len(calls) == 1:
                    raise RuntimeError("transient handler failure")
                return "recovered"

        router = Router()
        router.register("v1", FailsOnce())
        runtime = FakeRuntime()
        command = Command.from_dict(payload("retry"))
        with self.assertRaisesRegex(RuntimeError, "transient"):
            router.route(command, runtime)
        result = router.route(command, runtime)

        self.assertEqual(result, RouteResult("completed", value="recovered"))
        self.assertEqual(calls, ["retry", "retry"])
        self.assertEqual(router.route(command, runtime), result)

    def test_in_flight_rejects_a_different_command(self):
        router = Router()
        runtime = FakeRuntime()
        other = Command.from_dict(payload("other"))
        outer = Command.from_dict(payload("outer"))

        class ReentrantHandler:
            def request(self, active_runtime, command):
                self.assert_outer_is_inflight(active_runtime)
                with self_outer.assertRaises(ReentrantRouteError):
                    router.route(other, active_runtime)
                return "outer-complete"

            @staticmethod
            def assert_outer_is_inflight(active_runtime):
                result = router.route(outer, active_runtime)
                self_outer.assertEqual(result.status, "in_flight")

        self_outer = self
        router.register("v1", ReentrantHandler())
        result = router.route(outer, runtime)
        self.assertEqual(result.value, "outer-complete")

    def test_stop_uses_runtime_callback_without_policy_or_placement(self):
        router = Router()
        runtime = FakeRuntime()
        stop = Command.from_dict({
            "schema": 1,
            "session": "train-session-1",
            "id": "stop-1",
            "expected_epoch": 0,
            "action": "stop",
        })

        result = router.route(stop, runtime)
        self.assertEqual(result, RouteResult("completed", value="stop-requested"))
        self.assertEqual(runtime.stop_calls, 1)
        self.assertEqual(runtime.placements, [])
        self.assertEqual(router.route(stop, runtime), result)
        self.assertEqual(runtime.stop_calls, 1)

        with self.assertRaises(CommandError):
            Command.from_dict({**stop.__dict__, "policy": None})


if __name__ == "__main__":
    unittest.main()
