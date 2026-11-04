"""Helpers for bounded full-tensor dumps and compact numerical audits."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


REFERENCE_ANCHOR_SCHEMA = 2
REFERENCE_ANCHOR_KEYS = frozenset({
    "schema",
    "model",
    "all_buffers",
    "optimizer",
    "scheduler",
    "progress",
    "torch_rng",
    "cuda_rng",
    "python_rng",
    "tracker_rng",
    "source_replica",
    "source_rank",
    "step",
    "manager_state",
    "hashes",
})
REFERENCE_HASH_KEYS = frozenset({
    "model",
    "optimizer",
    "scheduler",
    "progress",
    "cpu_rng",
    "cuda_rng",
    "python_rng",
    "buffers",
    "tracker_rng",
})


def parse_reference_dump_steps(value: str) -> frozenset[int]:
    """Parse a comma-separated allowlist of committed update numbers."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("steps must be comma-separated positive integers")
    try:
        steps = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("steps must be comma-separated positive integers") from exc
    if any(step < 1 for step in steps):
        raise ValueError("steps must be positive committed update numbers")
    if len(set(steps)) != len(steps):
        raise ValueError("steps must not contain duplicates")
    return frozenset(steps)


def should_dump_reference(
    *,
    enabled: bool,
    selected_steps: frozenset[int] | None,
    next_step: int,
    replica: int,
    rank: int,
    selected_ranks: frozenset[int],
) -> bool:
    """Keep legacy full dumps on replica 0 and the selected local ranks."""
    return bool(
        enabled
        and (selected_steps is None or next_step in selected_steps)
        and replica == 0
        and rank in selected_ranks
    )


def reference_dump_payload(gradients, model, optimizer, scheduler) -> dict:
    """Return the legacy four-key shape consumed by compare_reference."""
    return {
        "gradients": gradients,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def reference_anchor_path(directory: str | Path, rank: int, step: int | None = None) -> Path:
    """Resolve the schema-2 per-rank anchor filename in a shared directory."""
    root = Path(directory)
    path = root / f"anchor_r{rank}.pt"
    # Route legacy filenames through the schema validator so users get a clear
    # incomplete-anchor error instead of an unhelpful missing-file exception.
    legacy = root / f"reference_step{step}_r{rank}.pt" if step is not None else None
    if not path.is_file() and legacy is not None and legacy.is_file():
        return legacy
    return path


def validate_reference_anchor(
    payload: object,
    *,
    expected_rank: int,
    expected_step: int | None = None,
) -> Mapping:
    """Reject incomplete or mismatched anchors before applying any state."""
    if not isinstance(payload, Mapping):
        raise ValueError("reference anchor must be a mapping")
    if payload.get("schema") != REFERENCE_ANCHOR_SCHEMA:
        raise ValueError(
            "reference anchor must use schema 2 with complete model/optimizer/"
            "scheduler/buffers/RNG/progress state; regenerate it with "
            "--reference-anchor-save"
        )
    missing = REFERENCE_ANCHOR_KEYS - set(payload)
    if missing:
        raise ValueError(f"schema-2 reference anchor is incomplete: missing {sorted(missing)}")
    for key in ("model", "all_buffers", "optimizer", "scheduler", "progress",
                "tracker_rng", "manager_state", "hashes"):
        if not isinstance(payload[key], Mapping):
            raise ValueError(f"schema-2 reference anchor field {key!r} must be a mapping")

    step = payload["step"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("reference anchor step must be a non-negative integer")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"reference anchor step mismatch: expected {expected_step}, found {step}")
    if payload["progress"].get("updates") != step:
        raise ValueError("reference anchor progress does not match its step")
    if payload["source_rank"] != expected_rank:
        raise ValueError(
            f"reference anchor rank mismatch: expected {expected_rank}, "
            f"found {payload['source_rank']}"
        )
    manager_state = payload["manager_state"]
    if manager_state.get("step") != step or not isinstance(manager_state.get("batches_committed"), int):
        raise ValueError("reference anchor manager state does not match its step")
    missing_hashes = REFERENCE_HASH_KEYS - set(payload["hashes"])
    if missing_hashes:
        raise ValueError(f"schema-2 reference anchor hashes are incomplete: {sorted(missing_hashes)}")
    if not isinstance(payload["hashes"], Mapping):
        raise ValueError("schema-2 reference anchor hashes must be a mapping")
    return payload
