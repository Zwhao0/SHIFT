"""In-memory GPU inventory and claims for a single SHIFT controller."""

from dataclasses import replace

from .spare_selection import Selection, plan_replacement, select_spare


class SparePool:
    def __init__(self, nodes):
        self.nodes = frozenset(nodes)
        self._reports = {}
        self._claims = {}
        self._released_at = {}

    def record(self, report):
        """Record a report after the backend verifies its instance identity."""
        if report.node not in self.nodes:
            raise ValueError('node is outside the standby pool')
        previous = self._reports.get(report.node)
        if previous and report.observed_at < previous.observed_at:
            return
        if report.observed_at > self._released_at.get(report.node, float('-inf')):
            self._reports[report.node] = report

    def _available_reports(self):
        reports = []
        for node, report in self._reports.items():
            claims = [claim for claim in self._claims.values() if claim.node == node]
            if any(claim.instance != report.instance for claim in claims):
                reports.append(replace(report, ready=False))
                continue
            occupied = {slot for claim in claims for slot in claim.slots}
            reports.append(replace(report, free_gpu_mib=tuple(
                (slot, mib) for slot, mib in report.free_gpu_mib if slot not in occupied)))
        return reports

    def decide(self, **options):
        return select_spare(self._available_reports(), **options)

    def track_running(self, attempt, selection):
        """Import the devices that the launcher has verified for a running actor."""
        self._check_new_attempt(attempt)
        if (selection.node not in self.nodes or not selection.instance or not selection.slots
                or any(not slot for slot in selection.slots)
                or len(set(selection.slots)) != len(selection.slots)):
            raise ValueError('running actor needs identified slots in the pool')
        for claim in self._claims.values():
            if claim.node == selection.node and (
                claim.instance != selection.instance or set(claim.slots).intersection(selection.slots)
            ):
                raise ValueError('running actor conflicts with an existing claim')
        self._claims[attempt] = selection

    def _check_new_attempt(self, attempt):
        if attempt in self._claims:
            raise ValueError('attempt already holds a claim')

    def claim(self, attempt, **options):
        self._check_new_attempt(attempt)
        decision = self.decide(**options)
        if decision.target is not None:
            self._claims[attempt] = decision.target
        return decision

    def claim_replacement(self, attempt, **options):
        """Resolve a SHIFT event to a policy, logical ranks and reserved GPUs."""
        self._check_new_attempt(attempt)
        plan = plan_replacement(self._available_reports(), **options)
        if plan.decision.target is not None:
            self._claims[attempt] = plan.decision.target
        return plan

    def release_after_exit(self, attempt, *, now, slots=None):
        """Call only after the backend proves that the prior workload has exited."""
        claim = self._claims[attempt]
        released = set(claim.slots if slots is None else slots)
        if not released or not released.issubset(claim.slots):
            raise ValueError('released slots must be held by this actor')
        remaining = tuple(slot for slot in claim.slots if slot not in released)
        if remaining:
            self._claims[attempt] = Selection(claim.node, claim.instance, remaining)
        else:
            del self._claims[attempt]
        self._reports.pop(claim.node, None)
        self._released_at[claim.node] = now
