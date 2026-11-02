#!/usr/bin/env python3
"""Compare baseline and SHIFT reference snapshots without loading all ranks."""

import argparse
import gc
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import torch


RANKS = range(8)
CHUNK_ELEMENTS = 1_000_000
TOLERANCES = {
    "gradients": {"atol": 1e-5, "rtol": 1e-3},
    "model": {"atol": 1e-6, "rtol": 1e-4},
    "optimizer": {"atol": 1e-5, "rtol": 1e-3},
    # No scheduler tolerance was specified; tensor values are exact here.
    "scheduler": {"atol": 0.0, "rtol": 0.0},
}
CATEGORIES = tuple(TOLERANCES)


def parse_steps(value: str) -> list[int]:
    try:
        steps = [int(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must contain non-negative integers")
    return steps


def parse_ranks(value: str) -> list[int]:
    try:
        ranks = [int(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks or any(rank < 0 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must contain non-negative integers")
    if len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("ranks must not contain duplicates")
    return ranks


def _key_order(key: object) -> tuple[str, str]:
    return type(key).__name__, repr(key)


def _child_path(path: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _safe_repr(value: object, limit: int = 200) -> str:
    result = repr(value)
    return result if len(result) <= limit else result[: limit - 3] + "..."


def _equal_leaf(left: object, right: object) -> bool:
    try:
        result = left == right
        if isinstance(result, torch.Tensor):
            return bool(result.all().item())
        return bool(result)
    except Exception:
        return False


def _flat_chunks(tensor: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield logical flat chunks without making a full contiguous copy."""
    count = tensor.numel()
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for start in range(0, count, CHUNK_ELEMENTS):
            yield start, flat.narrow(0, start, min(CHUNK_ELEMENTS, count - start))
        return

    # Gather by logical offsets for strided views; all temporary index tensors
    # are bounded by CHUNK_ELEMENTS rather than by the model tensor size.
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    storage = torch.as_strided(tensor, (storage_elements,), (1,), storage_offset=0)
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    for start in range(0, count, CHUNK_ELEMENTS):
        end = min(start + CHUNK_ELEMENTS, count)
        linear = torch.arange(start, end, dtype=torch.int64)
        offsets = torch.full_like(linear, tensor.storage_offset())
        for size, stride in reversed(list(zip(shape, strides))):
            offsets.add_(linear.remainder(size), alpha=stride)
            linear.div_(size, rounding_mode="floor")
        yield start, storage.index_select(0, offsets)


class TreeComparison:
    def __init__(self, category: str) -> None:
        tolerance = TOLERANCES[category]
        self.category = category
        self.atol = tolerance["atol"]
        self.rtol = tolerance["rtol"]
        self.errors: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {
            "tensor_count": 0,
            "elements": 0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "relative_l2": 0.0,
            "allclose": True,
            "strict_equal": True,
            "_ref_l2_sq": 0.0,
            "_diff_l2_sq": 0.0,
            "_invalid_float_metrics": False,
            "_has_float_tensors": False,
            "_max_abs_invalid": False,
            "_max_rel_invalid": False,
        }

    def _fail(
        self, path: str, reason: str, *, strict_mismatch: bool = True, **details: object
    ) -> None:
        self.stats["allclose"] = False
        if strict_mismatch:
            self.stats["strict_equal"] = False
        error = {"path": path, "reason": reason}
        error.update(details)
        self.errors.append(error)
        self._candidate({"path": path, "kind": reason, **details}, score=-1.0)

    def _candidate(self, value: dict[str, Any], score: float) -> None:
        value["_score"] = score
        self.candidates.append(value)
        self.candidates.sort(key=lambda item: (-item["_score"], item["path"]))
        del self.candidates[10:]

    def _compare_tensor(
        self, left: torch.Tensor, right: torch.Tensor, path: str
    ) -> None:
        self.stats["tensor_count"] += 1
        self.stats["elements"] += left.numel()
        if left.shape != right.shape:
            self._fail(
                path,
                "shape_mismatch",
                baseline_shape=list(left.shape),
                shift_shape=list(right.shape),
            )
            return
        if left.dtype != right.dtype:
            self._fail(
                path,
                "dtype_mismatch",
                baseline_dtype=str(left.dtype),
                shift_dtype=str(right.dtype),
            )
            return
        if left.layout != right.layout:
            self._fail(
                path,
                "layout_mismatch",
                baseline_layout=str(left.layout),
                shift_layout=str(right.layout),
            )
            return

        if left.layout != torch.strided:
            # Checkpoint tensors in the expected bundles are dense. Preserve
            # correctness if another layout appears, without densifying it.
            same = torch.equal(left, right)
            if not same:
                self.stats["strict_equal"] = False
                self._fail(path, "strict_tensor_mismatch", numel=left.numel())
            return

        same = torch.equal(left, right)
        if same:
            if left.is_floating_point() or left.is_complex():
                if not bool(torch.isfinite(left).all().item()):
                    self._fail(path, "nonfinite_float_tensor", strict_mismatch=False,
                               numel=left.numel())
                    return
                if left.is_floating_point():
                    self._record_equal_float(left)
            return

        self.stats["strict_equal"] = False
        if left.is_floating_point():
            self._compare_float_tensor(left, right, path)
        else:
            mismatch_count = 0
            first_bad: int | None = None
            for start, left_chunk in _flat_chunks(left):
                right_chunk = next_chunk(right, start, left_chunk.numel())
                equal = torch.eq(left_chunk, right_chunk)
                bad = ~equal
                count = int(bad.sum().item())
                if count:
                    mismatch_count += count
                    if first_bad is None:
                        first_bad = start + int(bad.nonzero()[0].item())
            if mismatch_count:
                self._fail(
                    path,
                    "strict_tensor_mismatch",
                    numel=left.numel(),
                    mismatched_elements=mismatch_count,
                    first_flat_index=first_bad,
                )

    def _record_equal_float(self, tensor: torch.Tensor) -> None:
        self.stats["_has_float_tensors"] = True
        for _, chunk in _flat_chunks(tensor):
            ref = chunk.to(torch.float64)
            self.stats["_ref_l2_sq"] += float((ref * ref).sum().item())

    def _compare_float_tensor(
        self, left: torch.Tensor, right: torch.Tensor, path: str
    ) -> None:
        self.stats["_has_float_tensors"] = True
        allclose = True
        max_abs = 0.0
        max_rel = 0.0
        max_abs_index: int | None = None
        invalid_metric = False
        ref_l2_sq = 0.0
        diff_l2_sq = 0.0

        for start, left_chunk in _flat_chunks(left):
            right_chunk = next_chunk(right, start, left_chunk.numel())
            ref = left_chunk.to(torch.float64)
            got = right_chunk.to(torch.float64)
            close = torch.isclose(got, ref, atol=self.atol, rtol=self.rtol)
            if not bool(close.all().item()):
                allclose = False

            diff = (got - ref).abs()
            same_inf = torch.isinf(got) & torch.isinf(ref) & (got == ref)
            diff = torch.where(same_inf, torch.zeros_like(diff), diff)
            invalid = ~torch.isfinite(diff)
            if bool(invalid.any().item()):
                invalid_metric = True
                if max_abs_index is None:
                    max_abs_index = start + int(invalid.nonzero()[0].item())
            finite = ~invalid
            if bool(finite.any().item()):
                safe_diff = diff[finite]
                safe_ref = ref[finite]
                chunk_abs, chunk_index = safe_diff.max(dim=0)
                abs_value = float(chunk_abs.item())
                if abs_value > max_abs:
                    max_abs = abs_value
                    finite_positions = finite.nonzero().flatten()
                    max_abs_index = start + int(finite_positions[int(chunk_index.item())].item())
                rel = safe_diff / safe_ref.abs().clamp_min(1e-8)
                max_rel = max(max_rel, float(rel.max().item()))
                ref_l2_sq += float((safe_ref * safe_ref).sum().item())
                diff_l2_sq += float((safe_diff * safe_diff).sum().item())

        self.stats["allclose"] = self.stats["allclose"] and allclose
        if invalid_metric:
            self.stats["_invalid_float_metrics"] = True
            self.stats["_max_abs_invalid"] = True
            self.stats["_max_rel_invalid"] = True
        else:
            self.stats["max_abs"] = max(self.stats["max_abs"], max_abs)
            self.stats["max_rel"] = max(self.stats["max_rel"], max_rel)
        self.stats["_ref_l2_sq"] += ref_l2_sq
        self.stats["_diff_l2_sq"] += diff_l2_sq

        if not allclose:
            score = math.inf if invalid_metric else max_abs
            self._candidate(
                {
                    "path": path,
                    "kind": "float_tensor_difference",
                    "numel": left.numel(),
                    "max_abs": None if invalid_metric else max_abs,
                    "max_rel": None if invalid_metric else max_rel,
                    "relative_l2": (
                        None
                        if invalid_metric
                        else _relative_l2(diff_l2_sq, ref_l2_sq)
                    ),
                    "first_largest_abs_flat_index": max_abs_index,
                    "allclose": False,
                },
                score=score,
            )

    def compare(self, left: object, right: object, path: str) -> None:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                self._fail(path, "tensor_type_mismatch")
                return
            self._compare_tensor(left, right, path)
            return

        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                self._fail(
                    path,
                    "container_type_mismatch",
                    baseline_type=type(left).__name__,
                    shift_type=type(right).__name__,
                )
                return
            left_keys, right_keys = set(left.keys()), set(right.keys())
            for key in sorted(left_keys - right_keys, key=_key_order):
                self._fail(_child_path(path, key), "missing_shift_key")
            for key in sorted(right_keys - left_keys, key=_key_order):
                self._fail(_child_path(path, key), "extra_shift_key")
            for key in sorted(left_keys & right_keys, key=_key_order):
                self.compare(left[key], right[key], _child_path(path, key))
            return

        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):  # type: ignore[arg-type]
                self._fail(
                    path,
                    "sequence_mismatch",
                    baseline_type=type(left).__name__,
                    shift_type=type(right).__name__,
                    baseline_length=len(left) if isinstance(left, (list, tuple)) else None,
                    shift_length=len(right) if isinstance(right, (list, tuple)) else None,
                )
                return
            for index, (a, b) in enumerate(zip(left, right)):  # type: ignore[arg-type]
                self.compare(a, b, f"{path}[{index}]")
            return

        if type(left) is not type(right) or not _equal_leaf(left, right):
            self._fail(
                path,
                "strict_leaf_mismatch",
                baseline_type=type(left).__name__,
                shift_type=type(right).__name__,
                baseline=_safe_repr(left),
                shift=_safe_repr(right),
            )

    def result(self) -> dict[str, Any]:
        stats = self.stats
        if stats["_invalid_float_metrics"]:
            stats["max_abs"] = None
            stats["max_rel"] = None
            stats["relative_l2"] = None
        elif stats["_has_float_tensors"]:
            stats["relative_l2"] = _relative_l2(
                stats["_diff_l2_sq"], stats["_ref_l2_sq"]
            )
        else:
            stats["max_abs"] = None
            stats["max_rel"] = None
            stats["relative_l2"] = None

        clean_stats = {key: value for key, value in stats.items() if not key.startswith("_")}
        candidates = []
        for candidate in self.candidates:
            candidates.append(
                {key: value for key, value in candidate.items() if key != "_score"}
            )
        # strict_equal is diagnostic only; acceptance remains tolerance based.
        passed = bool(stats["allclose"] and not self.errors)
        return {
            "pass": passed,
            "metrics": clean_stats,
            "errors": self.errors,
            "top_differences": candidates,
        }


def next_chunk(tensor: torch.Tensor, start: int, size: int) -> torch.Tensor:
    """Return the same logical flat slice as _flat_chunks for the other tensor."""
    if tensor.is_contiguous():
        return tensor.view(-1).narrow(0, start, size)

    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    storage = torch.as_strided(tensor, (storage_elements,), (1,), storage_offset=0)
    linear = torch.arange(start, start + size, dtype=torch.int64)
    offsets = torch.full_like(linear, tensor.storage_offset())
    for dimension, stride in reversed(list(zip(tensor.shape, tensor.stride()))):
        offsets.add_(linear.remainder(dimension), alpha=stride)
        linear.div_(dimension, rounding_mode="floor")
    return storage.index_select(0, offsets)


def _relative_l2(diff_sq: float, ref_sq: float) -> float | None:
    if not math.isfinite(diff_sq) or math.isnan(ref_sq):
        return None
    denominator = max(math.sqrt(ref_sq), 1e-8)
    value = math.sqrt(diff_sq) / denominator
    return value if math.isfinite(value) else None


def _compare_payloads(baseline: object, shifted: object) -> dict[str, Any]:
    required = set(CATEGORIES)
    root_errors: list[dict[str, Any]] = []
    if not isinstance(baseline, Mapping) or not isinstance(shifted, Mapping):
        return {
            "pass": False,
            "checks": {"payload_mapping": False},
            "errors": [{"path": "$", "reason": "payload_not_mapping"}],
            "by_category": {},
            "top_differences": [],
        }

    baseline_keys, shift_keys = set(baseline), set(shifted)
    if baseline_keys != shift_keys:
        root_errors.append(
            {
                "path": "$",
                "reason": "payload_key_mismatch",
                "missing_shift": [
                    repr(key) for key in sorted(baseline_keys - shift_keys, key=_key_order)
                ],
                "extra_shift": [
                    repr(key) for key in sorted(shift_keys - baseline_keys, key=_key_order)
                ],
            }
        )
    if not required.issubset(baseline_keys) or not required.issubset(shift_keys):
        root_errors.append(
            {
                "path": "$",
                "reason": "required_payload_keys_missing",
                "required": sorted(required),
                "baseline_missing": sorted(required - baseline_keys),
                "shift_missing": sorted(required - shift_keys),
            }
        )

    by_category: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    all_pass = not root_errors
    for category in CATEGORIES:
        comparator = TreeComparison(category)
        if category in baseline and category in shifted:
            comparator.compare(baseline[category], shifted[category], category)
        else:
            comparator._fail(category, "category_missing")
        result = comparator.result()
        by_category[category] = result
        all_pass = all_pass and result["pass"]
        for candidate in result["top_differences"]:
            candidates.append({"category": category, **candidate})

    if root_errors:
        all_pass = False
    def difference_order(item: dict[str, Any]) -> tuple[float, str, str]:
        value = item.get("max_abs")
        if item.get("kind") == "float_tensor_difference" and value is None:
            score = math.inf
        elif isinstance(value, (int, float)):
            score = float(value)
        else:
            score = -1.0
        return -score, item["category"], item["path"]

    candidates.sort(key=difference_order)
    return {
        "pass": all_pass,
        "checks": {
            "payload_keys": not root_errors,
            **{name: item["pass"] for name, item in by_category.items()},
        },
        "errors": root_errors,
        "by_category": by_category,
        "top_differences": candidates[:10],
    }


def _load(path: Path) -> object:
    return torch.load(path, weights_only=False, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("shift_dir", type=Path)
    parser.add_argument("--steps", type=parse_steps, default=[1, 2], metavar="1,2")
    parser.add_argument("--ranks", type=parse_ranks, default=list(RANKS), metavar="0,1,...")
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)

    byrank: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    load_errors: list[dict[str, str]] = []
    failed_pairs: list[str] = []

    for step in args.steps:
        step_results: dict[str, Any] = {}
        byrank[str(step)] = step_results
        for rank in args.ranks:
            name = f"reference_step{step}_r{rank}.pt"
            baseline_path = args.baseline_dir / name
            shift_path = args.shift_dir / name
            absent = [str(path) for path in (baseline_path, shift_path) if not path.is_file()]
            if absent:
                missing.extend(absent)
                step_results[str(rank)] = {
                    "pass": False,
                    "checks": {"files_present": False},
                    "missing_files": absent,
                }
                failed_pairs.append(f"step{step}/rank{rank}")
                continue

            baseline = shifted = None
            try:
                baseline = _load(baseline_path)
                shifted = _load(shift_path)
                result = _compare_payloads(baseline, shifted)
                step_results[str(rank)] = result
                if not result["pass"]:
                    failed_pairs.append(f"step{step}/rank{rank}")
            except Exception as exc:
                error = {
                    "step": step,
                    "rank": rank,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                load_errors.append(error)
                step_results[str(rank)] = {
                    "pass": False,
                    "checks": {"load_and_compare": False},
                    "errors": [error],
                    "top_differences": [],
                }
                failed_pairs.append(f"step{step}/rank{rank}")
            finally:
                del baseline, shifted
                gc.collect()

    checks = {
        "files_present": {"pass": not missing, "missing": missing},
        "loads_succeeded": {"pass": not load_errors, "errors": load_errors},
        "all_rank_step_comparisons_pass": {
            "pass": not failed_pairs,
            "failed_pairs": failed_pairs,
        },
    }
    passed = all(check["pass"] for check in checks.values())
    report = {
        "pass": passed,
        "steps": args.steps,
        "ranks": list(args.ranks),
        "tolerances": TOLERANCES,
        "checks": checks,
        "byrank": byrank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
