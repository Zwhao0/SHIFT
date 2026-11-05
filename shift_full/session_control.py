"""CPU-only command parsing and routing for a live training session.

The runtime owns training and placement. This module only validates commands
at a committed boundary and dispatches a request to an already running handler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ipaddress
import re
import threading


_POLICY_ACTIONS = {"v1": "recover", "v2": "migrate", "v4": "migrate"}
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_NODE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_COMMON_FIELDS = {"schema", "session", "id", "expected_epoch", "action", "at_step", "ranks"}
_PLACEMENT_FIELDS = {"policy", "source_replica", "target"}


class CommandError(ValueError):
    """A command is malformed or conflicts with a prior command."""


class StaleCommandError(RuntimeError):
    """A command targets an epoch or boundary that has already passed."""


class ReentrantRouteError(RuntimeError):
    """A route was attempted while another command is being dispatched."""


def _strict_int(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise CommandError(f"{field} must be an integer >= {minimum}")
    return value


def _safe_string(value, field):
    if type(value) is not str or not _SAFE_TOKEN.fullmatch(value):
        raise CommandError(f"{field} must contain only safe identifier characters")
    return value


@dataclass(frozen=True)
class Command:
    schema: int
    session: str
    id: str
    expected_epoch: int
    action: str
    policy: str | None = None
    source_replica: int | None = None
    target: str | None = None
    at_step: int | None = None
    ranks: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, payload):
        """Parse schema 1, rejecting unknown fields and action-policy mismatches."""
        if not isinstance(payload, Mapping):
            raise CommandError("command must be a mapping")
        unknown = set(payload) - (_COMMON_FIELDS | _PLACEMENT_FIELDS)
        if unknown:
            raise CommandError(f"unknown command fields: {sorted(map(str, unknown))}")
        required = {"schema", "session", "id", "expected_epoch", "action"}
        missing = required - set(payload)
        if missing:
            raise CommandError(f"missing command fields: {sorted(missing)}")

        schema = _strict_int(payload["schema"], "schema", minimum=1)
        if schema != 1:
            raise CommandError("only command schema 1 is supported")
        session = _safe_string(payload["session"], "session")
        command_id = _safe_string(payload["id"], "id")
        expected_epoch = _strict_int(payload["expected_epoch"], "expected_epoch")
        action = payload["action"]
        if type(action) is not str or action not in ("recover", "migrate", "stop"):
            raise CommandError("action must be recover, migrate, or stop")

        at_step = payload.get("at_step")
        if at_step is not None:
            at_step = _strict_int(at_step, "at_step")

        if action == "stop":
            present = _PLACEMENT_FIELDS & set(payload)
            if present or "ranks" in payload:
                raise CommandError("stop does not accept policy, source_replica, target, or ranks")
            return cls(schema, session, command_id, expected_epoch, action,
                       at_step=at_step)

        missing = _PLACEMENT_FIELDS - set(payload)
        if missing:
            raise CommandError(f"{action} requires: {', '.join(sorted(missing))}")
        policy = payload["policy"]
        if type(policy) is not str or policy not in _POLICY_ACTIONS:
            raise CommandError("policy must be v1, v2, or v4")
        if _POLICY_ACTIONS[policy] != action:
            raise CommandError(f"policy {policy} does not support action {action}")

        source_replica = _strict_int(payload["source_replica"], "source_replica")
        target = payload["target"]
        if type(target) is not str or not _NODE_NAME.fullmatch(target):
            raise CommandError("target must be an abstract node name")
        try:
            ipaddress.ip_address(target)
        except ValueError:
            pass
        else:
            raise CommandError("target must not be an IP address")

        default_ranks = tuple(range(8)) if policy in ("v1", "v2") else tuple(range(4))
        raw_ranks = payload.get("ranks", list(default_ranks))
        if type(raw_ranks) is not list:
            raise CommandError("ranks must be a list of strict integers")
        parsed_ranks = tuple(sorted(_strict_int(rank, "rank") for rank in raw_ranks))
        if len(set(parsed_ranks)) != len(parsed_ranks):
            raise CommandError("ranks must not contain duplicates")
        if policy in ("v1", "v2"):
            if parsed_ranks != tuple(range(8)):
                raise CommandError(f"{policy} requires all ranks 0..7")
        elif parsed_ranks not in (tuple(range(4)), tuple(range(4, 8))):
            raise CommandError("v4 requires one complete PP stage: ranks 0..3 or 4..7")

        return cls(schema, session, command_id, expected_epoch, action, policy,
                   source_replica, target, at_step, parsed_ranks)

    def payload_key(self):
        """Canonical identity used for duplicate-id detection."""
        return (self.schema, self.session, self.id, self.expected_epoch, self.action,
                self.policy, self.source_replica, self.target, self.at_step, self.ranks)


@dataclass(frozen=True)
class RouteResult:
    status: str
    value: object = None
    at_step: int | None = None
    committed_step: int | None = None


class Router:
    """Dispatch commands to policy handlers without starting or restarting workers.

    The runtime contract is ``session``, ``epoch``, ``progress['updates']``,
    and ``at_committed_boundary is True``. Placement actions also need
    ``validate_placement(command)``; stop needs ``request_stop()``.
    """

    def __init__(self):
        self._handlers = {}
        self._seen = {}
        self._results = {}
        self._inflight = None
        self._lock = threading.Lock()

    def register(self, policy, handler):
        if type(policy) is not str or policy not in _POLICY_ACTIONS:
            raise CommandError("policy must be v1, v2, or v4")
        if not callable(getattr(handler, "request", None)):
            raise TypeError("handler must provide callable request(runtime, command)")
        with self._lock:
            if policy in self._handlers:
                raise CommandError(f"handler already registered for {policy}")
            self._handlers[policy] = handler

    @staticmethod
    def _runtime_boundary(runtime, command):
        if getattr(runtime, "session", None) != command.session:
            raise CommandError("command session does not match runtime session")
        epoch = _strict_int(getattr(runtime, "epoch", None), "runtime.epoch")
        if epoch != command.expected_epoch:
            raise StaleCommandError("expected_epoch does not match runtime epoch")
        if getattr(runtime, "at_committed_boundary", None) is not True:
            raise RuntimeError("runtime is not at a committed boundary")
        progress = getattr(runtime, "progress", None)
        if not isinstance(progress, Mapping):
            raise RuntimeError("runtime.progress must be a mapping")
        committed_step = _strict_int(progress.get("updates"), "runtime.progress.updates")
        if command.at_step is not None:
            if committed_step < command.at_step:
                return committed_step, RouteResult("pending", at_step=command.at_step,
                                                   committed_step=committed_step)
            if committed_step > command.at_step:
                raise StaleCommandError("at_step has already passed")
        return committed_step, None

    def route(self, command, runtime):
        if not isinstance(command, Command):
            raise TypeError("route requires a Command; parse dictionaries with Command.from_dict")
        key = (command.session, command.id)
        payload_key = command.payload_key()

        with self._lock:
            # Check the live session before returning a cached response.
            if getattr(runtime, "session", None) != command.session:
                raise CommandError("command session does not match runtime session")
            prior = self._seen.get(key)
            if prior is not None and prior != payload_key:
                raise CommandError("command id was reused with a different payload")
            self._seen.setdefault(key, payload_key)
            cached = self._results.get(key)
            if cached is not None:
                return cached
            if self._inflight is not None:
                if self._inflight == key:
                    return RouteResult("in_flight")
                raise ReentrantRouteError("another command is already in flight")

            _, pending = self._runtime_boundary(runtime, command)
            if pending is not None:
                return pending
            if command.action != "stop" and command.policy not in self._handlers:
                raise CommandError(f"no handler registered for {command.policy}")
            self._inflight = key

        try:
            if command.action == "stop":
                request_stop = getattr(runtime, "request_stop", None)
                if not callable(request_stop):
                    raise TypeError("runtime must provide request_stop()")
                value = request_stop()
            else:
                validate_placement = getattr(runtime, "validate_placement", None)
                if not callable(validate_placement):
                    raise TypeError("runtime must provide validate_placement(command)")
                validate_placement(command)
                value = self._handlers[command.policy].request(runtime, command)
            result = RouteResult("completed", value=value)
            with self._lock:
                self._results[key] = result
            return result
        finally:
            with self._lock:
                self._inflight = None
