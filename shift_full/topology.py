"""Pure-Python planning and validation for replicated MoE rank topologies."""

from collections import deque
from typing import Mapping, Sequence


class TopologyError(ValueError):
    """Malformed topology input or an unsupported generated layout."""


TABLE_4_2 = {
    "A": {"tp": 2, "pp": 2, "ep": 4, "etp": 1},
    "B": {"tp": 4, "pp": 2, "ep": 2, "etp": 1},
    "C": {"tp": 2, "pp": 1, "ep": 4, "etp": 1},
    "D": {"tp": 2, "pp": 1, "ep": 2, "etp": 1},
}


def _is_power_of_two(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0 and value & (value - 1) == 0


def _normalize_groups(name, groups, world_size):
    if groups is None:
        raise TopologyError(f"{name} groups are required")
    normalized = []
    owner = {}
    try:
        iterable = list(groups)
    except TypeError as exc:
        raise TopologyError(f"{name} groups must be an iterable of rank groups") from exc
    if not iterable:
        raise TopologyError(f"{name} groups are empty")
    for group_no, raw_group in enumerate(iterable):
        try:
            group = list(raw_group)
        except TypeError as exc:
            raise TopologyError(f"{name} group {group_no} is not iterable") from exc
        if not group:
            raise TopologyError(f"{name} group {group_no} is empty")
        members = []
        for rank in group:
            if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
                raise TopologyError(f"{name} group {group_no} has out-of-range rank {rank!r}")
            if rank in members:
                raise TopologyError(f"{name} group {group_no} repeats rank {rank}")
            if rank in owner:
                raise TopologyError(
                    f"{name} groups overlap at rank {rank} (groups {owner[rank]} and {group_no})"
                )
            owner[rank] = group_no
            members.append(rank)
        normalized.append(tuple(sorted(members)))
    missing = sorted(set(range(world_size)) - set(owner))
    if missing:
        raise TopologyError(f"{name} groups omit ranks {missing}")
    return tuple(sorted(normalized))


def _groups_by_rank(groups):
    result = {}
    for group in groups:
        for rank in group:
            result[rank] = group
    return result


def _components(world_size, execution_groups):
    adjacency = {rank: set() for rank in range(world_size)}
    for groups in execution_groups.values():
        for group in groups:
            for rank in group:
                adjacency[rank].update(group)
    components = []
    seen = set()
    for start in range(world_size):
        if start in seen:
            continue
        todo = deque([start])
        seen.add(start)
        component = []
        while todo:
            rank = todo.popleft()
            component.append(rank)
            for neighbor in sorted(adjacency[rank] - seen):
                seen.add(neighbor)
                todo.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda group: group[0]))


def _group_positions(execution_groups):
    names = sorted(execution_groups)
    positions = {rank: {} for groups in execution_groups.values() for group in groups for rank in group}
    for name, groups in execution_groups.items():
        for group in groups:
            for index, rank in enumerate(group):
                positions[rank][name] = index
    return names, positions


def analyze_topology(
    world_size,
    dense_dp_groups,
    expert_dp_groups,
    execution_groups,
    rank_to_node=None,
):
    """Build candidate replicas from actual non-DP execution groups.

    Each group family must partition ranks. Groups from different families may
    overlap; their transitive connectivity determines a candidate replica.
    """
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise TopologyError("world_size must be a positive integer")
    dense = _normalize_groups("dense_dp", dense_dp_groups, world_size)
    expert = _normalize_groups("expert_dp", expert_dp_groups, world_size)
    if not isinstance(execution_groups, Mapping) or not execution_groups:
        raise TopologyError("execution_groups must be a non-empty name-to-groups mapping")
    execution = {}
    for raw_name, groups in execution_groups.items():
        name = str(raw_name).upper()
        if name in {"DP", "DENSE_DP", "EXPERT_DP", "EDP"}:
            raise TopologyError(f"{name} is a DP family, not a non-DP execution dependency")
        if name in execution:
            raise TopologyError(f"duplicate execution group name {name}")
        execution[name] = _normalize_groups(f"execution {name}", groups, world_size)

    components = _components(world_size, execution)
    at_least_two = len(components) >= 2
    rank_to_replica = {
        rank: replica_id
        for replica_id, ranks in enumerate(components)
        for rank in ranks
    }
    names, positions = _group_positions(execution)
    rank_to_localrole = {
        rank: {name: positions[rank][name] for name in names}
        for rank in range(world_size)
    }
    role_keys = {
        rank: tuple(rank_to_localrole[rank][name] for name in names)
        for rank in range(world_size)
    }

    rejections = []
    checks = {}
    checks["at_least_two_replicas"] = {"pass": at_least_two, "actual": len(components)}
    if not at_least_two:
        rejections.append("fewer than two candidate replicas; fail-stop recovery is unavailable")
    sizes = [len(ranks) for ranks in components]
    checks["equal_replica_sizes"] = {"pass": len(set(sizes)) == 1, "sizes": sizes}
    if not checks["equal_replica_sizes"]["pass"]:
        rejections.append(f"candidate replica sizes differ: {sizes}")

    per_replica_roles = []
    duplicate_roles = []
    for replica_id, ranks in enumerate(components):
        roles = [role_keys[rank] for rank in ranks]
        if len(set(roles)) != len(roles):
            duplicate_roles.append(replica_id)
        per_replica_roles.append(set(roles))
    roles_match = bool(per_replica_roles) and all(
        roles == per_replica_roles[0] for roles in per_replica_roles[1:]
    ) and not duplicate_roles
    checks["role_coverage_matches"] = {
        "pass": roles_match,
        "role_counts": [len(roles) for roles in per_replica_roles],
        "duplicate_role_replicas": duplicate_roles,
    }
    if not roles_match:
        rejections.append("replicas do not have unique, identical local-role coverage")

    component_for = rank_to_replica
    cut_groups = []
    for name, groups in execution.items():
        for group in groups:
            touched = sorted({component_for[rank] for rank in group})
            if len(touched) != 1:
                cut_groups.append({"group": name, "ranks": list(group), "replicas": touched})
    checks["execution_groups_unbroken"] = {"pass": not cut_groups, "groups": cut_groups}
    if cut_groups:
        rejections.append("an execution group crosses candidate replica boundaries")

    dense_for = _groups_by_rank(dense)
    expert_for = _groups_by_rank(expert)
    intersections = {
        rank: sorted(set(dense_for[rank]) & set(expert_for[rank]))
        for rank in range(world_size)
    }
    recovery_sources = {}
    source_issues = []
    for rank in range(world_size):
        sources = {}
        for replica_id in range(len(components)):
            if replica_id == rank_to_replica[rank]:
                continue
            candidates = [
                source
                for source in intersections[rank]
                if rank_to_replica[source] == replica_id and role_keys[source] == role_keys[rank]
            ]
            if len(candidates) != 1:
                source_issues.append(
                    {"rank": rank, "target_replica": replica_id, "candidates": candidates}
                )
                sources[replica_id] = candidates
            else:
                sources[replica_id] = candidates[0]
        recovery_sources[rank] = sources
    checks["recovery_sources_complete"] = {
        "pass": not source_issues and at_least_two,
        "issues": source_issues,
    }
    if source_issues:
        rejections.append("missing or ambiguous dense∩expert same-role source in another replica")

    normalized_nodes = None
    survivable = {}
    node_local_groups = {}
    if rank_to_node is not None:
        if isinstance(rank_to_node, Mapping):
            normalized_nodes = dict(rank_to_node)
        elif isinstance(rank_to_node, Sequence) and not isinstance(rank_to_node, (str, bytes)):
            if len(rank_to_node) != world_size:
                raise TopologyError("rank_to_node sequence must have one entry per rank")
            normalized_nodes = dict(enumerate(rank_to_node))
        else:
            raise TopologyError("rank_to_node must be a rank mapping or a node sequence")
        missing_nodes = sorted(set(range(world_size)) - set(normalized_nodes))
        invalid_nodes = sorted(rank for rank in normalized_nodes if rank not in range(world_size))
        if missing_nodes or invalid_nodes:
            raise TopologyError(f"rank_to_node mismatch: missing={missing_nodes}, invalid={invalid_nodes}")
        nodes = sorted(set(normalized_nodes.values()), key=str)
        for node in nodes:
            survivable[node] = [
                replica_id
                for replica_id, ranks in enumerate(components)
                if all(normalized_nodes[rank] != node for rank in ranks)
            ]
        node_local_groups = {
            name: sum(len({normalized_nodes[rank] for rank in group}) > 1 for group in groups)
            for name, groups in execution.items()
        }

    group_sizes = lambda groups: sorted(len(group) for group in groups)
    replicas = [
        {
            "id": replica_id,
            "ranks": list(ranks),
            "size": len(ranks),
            "role_count": len(per_replica_roles[replica_id]),
        }
        for replica_id, ranks in enumerate(components)
    ]
    result = {
        "valid": all(check["pass"] for check in checks.values()),
        "world_size": world_size,
        "candidate_replica_count": len(components),
        "replicas": replicas,
        "rank_to_replica": rank_to_replica,
        "rank_to_localrole": rank_to_localrole,
        "recovery_sources": recovery_sources,
        "dense_expert_intersections": intersections,
        "survivable_replicas_by_node_failure": survivable,
        "execution_groups_crossing_nodes": node_local_groups,
        "checks": checks,
        "rejections": rejections,
        "metrics": {
            "dense_dp_group_sizes": group_sizes(dense),
            "expert_dp_group_sizes": group_sizes(expert),
            "dense_expert_intersection_sizes": {
                rank: len(members) for rank, members in intersections.items()
            },
            "execution_group_sizes": {
                name: group_sizes(groups) for name, groups in execution.items()
            },
        },
        "groups": {
            "dense_dp": [list(group) for group in dense],
            "expert_dp": [list(group) for group in expert],
            "execution": {name: [list(group) for group in groups] for name, groups in execution.items()},
        },
    }
    return result


def _groups_varying(coordinates, varying_axis):
    if not coordinates:
        return []
    names = tuple(next(iter(coordinates.values())))
    grouped = {}
    for rank, coord in coordinates.items():
        key = tuple((name, coord[name]) for name in names if name != varying_axis)
        grouped.setdefault(key, []).append(rank)
    return [sorted(group) for group in grouped.values()]


def _power2(value, name):
    if not _is_power_of_two(value):
        raise TopologyError(f"{name}={value!r} must be a positive power of two")


def make_power2_layout(world_size, tp, pp, ep, etp, cp=1, rank_to_node=None):
    """Generate actual dense/expert groups by their separate rank-coordinate orders.

    Dense ranks are TP-fastest, then dense-DP, then PP. Expert ranks are
    ETP-fastest, then EP, expert-DP, then PP. The table layouts use CP=1.
    """
    for name, value in (("world_size", world_size), ("tp", tp), ("pp", pp),
                        ("ep", ep), ("etp", etp), ("cp", cp)):
        _power2(value, name)
    if cp != 1:
        raise TopologyError("the standard rank-order generator is specified for CP=1")
    if world_size % (tp * pp):
        raise TopologyError("world_size is not divisible by TP×PP")
    if world_size % (etp * ep * pp):
        raise TopologyError("world_size is not divisible by ETP×EP×PP")
    dense_dp = world_size // (tp * pp)
    expert_dp = world_size // (etp * ep * pp)
    _power2(dense_dp, "dense_dp_degree")
    _power2(expert_dp, "expert_dp_degree")

    dense_coords = {}
    for pipeline in range(pp):
        for data in range(dense_dp):
            for tensor in range(tp):
                rank = tensor + tp * (data + dense_dp * pipeline)
                dense_coords[rank] = {"tp": tensor, "dp": data, "pp": pipeline}
    expert_coords = {}
    for pipeline in range(pp):
        for data in range(expert_dp):
            for expert in range(ep):
                for tensor in range(etp):
                    rank = tensor + etp * (expert + ep * (data + expert_dp * pipeline))
                    expert_coords[rank] = {"etp": tensor, "ep": expert, "edp": data, "pp": pipeline}
    if set(dense_coords) != set(range(world_size)) or set(expert_coords) != set(range(world_size)):
        raise TopologyError("coordinate decomposition did not cover the rank world")

    dense_dp_groups = _groups_varying(dense_coords, "dp")
    expert_dp_groups = _groups_varying(expert_coords, "edp")
    execution_groups = {
        "TP": _groups_varying(dense_coords, "tp"),
        "PP": _groups_varying(dense_coords, "pp"),
        "CP": [[rank] for rank in range(world_size)],
        "EP": _groups_varying(expert_coords, "ep"),
        "ETP": _groups_varying(expert_coords, "etp"),
    }
    result = analyze_topology(
        world_size, dense_dp_groups, expert_dp_groups, execution_groups, rank_to_node
    )
    min_law = min(dense_dp, expert_dp)
    result["layout"] = "power2_standard"
    result["dimensions"] = {"tp": tp, "pp": pp, "ep": ep, "etp": etp, "cp": cp}
    result["degrees"] = {"dense_dp": dense_dp, "expert_dp": expert_dp}
    result["min_dp_reference"] = min_law
    result["checks"]["generated_min_dp_reference_matches"] = {
        "pass": result["candidate_replica_count"] == min_law,
        "actual": result["candidate_replica_count"],
        "reference": min_law,
    }
    if result["candidate_replica_count"] != min_law:
        result["rejections"].append(
            "coordinate-derived replica count differs from min dense/expert DP reference"
        )
    result["valid"] = all(check["pass"] for check in result["checks"].values())
    return result


def _swap_bits(rank, left, right):
    a, b = (rank >> left) & 1, (rank >> right) & 1
    return rank ^ ((a ^ b) << left) ^ ((a ^ b) << right)


def make_vendor_layout(layout, rank_to_node=None):
    """Plan the measured TP2/PP2/EP2/ETP2, WS16 bit layouts.

    ``legacy`` leaves replica at bit 2 and PP at bit 3. ``node_local_pp``
    swaps those bits so physical nodes (rank // 8) each hold a full replica.
    """
    if layout not in {"legacy", "node_local_pp"}:
        raise TopologyError("layout must be 'legacy' or 'node_local_pp'")
    base = make_power2_layout(16, tp=2, pp=2, ep=2, etp=2, cp=1)
    groups = base["groups"]
    if layout == "node_local_pp":
        transform = lambda group: sorted(_swap_bits(rank, 2, 3) for rank in group)
        dense = [transform(group) for group in groups["dense_dp"]]
        expert = [transform(group) for group in groups["expert_dp"]]
        execution = {
            name: [transform(group) for group in members]
            for name, members in groups["execution"].items()
        }
    else:
        dense = groups["dense_dp"]
        expert = groups["expert_dp"]
        execution = groups["execution"]
    if rank_to_node is None:
        rank_to_node = {rank: rank // 8 for rank in range(16)}
    result = analyze_topology(16, dense, expert, execution, rank_to_node)
    result["layout"] = layout
    result["dimensions"] = {"tp": 2, "pp": 2, "ep": 2, "etp": 2, "cp": 1}
    return result
