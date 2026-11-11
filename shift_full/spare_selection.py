"""Rank spare reports for one placement. Selection does not reserve or launch GPUs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .session_control import Command
from .session_layout import SessionLayout


@dataclass(frozen=True)
class SpareReport:
    node: str
    instance: str  # Pod UID or agent incarnation, never an IP.
    observed_at: float  # Controller monotonic receipt time; only fresh probes may renew it.
    profiles: frozenset[str]  # Verified model/runtime/topology/network requirements.
    free_gpu_mib: tuple[tuple[str, int], ...]  # Authorized, unoccupied GPU slots by ID.
    ready: bool
    setup_ms: int | None = None


@dataclass(frozen=True)
class Selection:
    node: str
    instance: str
    slots: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    ranked: tuple[Selection, ...]
    rejected: tuple[tuple[str, str], ...]

    @property
    def target(self) -> Selection | None:
        return self.ranked[0] if self.ranked else None


@dataclass(frozen=True)
class ReplacementPlan:
    policy: str
    action: str
    source_replica: int
    ranks: tuple[int, ...]
    world: int
    decision: Decision

    def migration_command(self, *, session, command_id, expected_epoch):
        """Build a v4.3 command after the caller reserves and verifies the target."""
        if self.action != 'migrate':
            raise ValueError('process-exit recovery needs a separate execution path')
        target = self.decision.target
        if target is None:
            raise ValueError('no eligible spare')
        return Command.from_dict(dict(schema=1, session=session, id=command_id,
                                      expected_epoch=expected_epoch, action=self.action, policy=self.policy,
                                      source_replica=self.source_replica, target=target.node,
                                      ranks=list(self.ranks), replica_world_size=self.world))


def plan_replacement(reports, *, fault, source_replica, source_node, world, profile,
                     min_gpu_mib, now, max_age, affected_rank=None, excluded_nodes=()):
    """Convert a manually classified fault (or detector verdict) to a placement plan.

    The live controller supports v2 for DP1 and v4 for a complete PP stage.
    A process-exit plan does not prove that the old workers have been fenced.
    """
    layout = SessionLayout(world)
    if type(source_replica) is not int or source_replica not in (0, 1) or not source_node:
        raise ValueError('a current source replica and node are required')
    if fault == 'process_exit' and affected_rank is None:
        policy, action, ranks = 'v1', 'recover', tuple(range(world))
    elif fault == 'replica_slow' and source_replica == 1 and affected_rank is None:
        policy, action, ranks = 'v2', 'migrate', tuple(range(world))
    elif fault == 'rank_slow' and type(affected_rank) is int and affected_rank in range(world):
        policy, action, ranks = 'v4', 'migrate', layout.stage_ranks(affected_rank // layout.width)
    else:
        raise ValueError('fault or source scope is unsupported in v4.3')

    decision = select_spare(reports, profile=profile, ranks=len(ranks), min_gpu_mib=min_gpu_mib,
                            now=now, max_age=max_age, excluded_nodes={source_node, *excluded_nodes})
    return ReplacementPlan(policy, action, source_replica, ranks, world, decision)


def select_spare(reports, *, profile, ranks, min_gpu_mib, now, max_age, excluded_nodes=()):
    """Filter by eligibility, then prefer shorter setup and greater GPU headroom.

    The caller supplies one verified report per node and excludes faulted nodes.
    Before launch, the controller must reserve and recheck the chosen instance
    and slots against current placement and resource state.
    """
    if not profile or type(ranks) is not int or ranks < 1:
        raise ValueError('a profile and positive rank count are required')
    if type(min_gpu_mib) is not int or min_gpu_mib < 1:
        raise ValueError('a positive per-GPU memory requirement is required')
    if not math.isfinite(now) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError('invalid controller time or report lifetime')

    excluded_nodes = set(excluded_nodes)
    seen = set()
    ranked = []
    rejected = []
    for report in reports:
        if not report.node or not report.instance or report.node in seen:
            raise ValueError('expected one identified report per node')
        seen.add(report.node)
        age = now - report.observed_at
        if report.node in excluded_nodes:
            reason = 'excluded'
        elif not math.isfinite(age) or not 0 <= age <= max_age:
            reason = 'stale_report'
        elif report.ready is not True:
            reason = 'not_ready'
        elif profile not in report.profiles:
            reason = 'incompatible_profile'
        else:
            ids = set()
            for slot, mib in report.free_gpu_mib:
                if not slot or slot in ids or type(mib) is not int or mib < 0:
                    raise ValueError('invalid GPU slot inventory')
                ids.add(slot)
            if report.setup_ms is not None and (type(report.setup_ms) is not int or report.setup_ms < 0):
                raise ValueError('invalid setup estimate')
            eligible = sorted((-mib, slot) for slot, mib in report.free_gpu_mib if mib >= min_gpu_mib)
            if len(eligible) >= ranks:
                chosen = eligible[:ranks]
                score = (math.inf if report.setup_ms is None else report.setup_ms, chosen[-1][0], report.node)
                ranked.append((score, Selection(report.node, report.instance, tuple(slot for _, slot in chosen))))
                continue
            reason = 'insufficient_slots'
        rejected.append((report.node, reason))
    return Decision(tuple(value for _, value in sorted(ranked)), tuple(sorted(rejected)))
