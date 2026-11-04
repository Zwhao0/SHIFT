"""Packed, GPU-to-GPU checkpoint transport over a configured torchft PG."""

from __future__ import annotations

import math
import pickle
import time
from datetime import timedelta
from typing import Any, Callable, Optional

import torch
from torch.distributed import Work
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchft.checkpointing.transport import CheckpointTransport
from torchft.process_group import ProcessGroup

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - DTensor exists in supported torch builds
    DTensor = ()  # type: ignore[assignment,misc]


_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_CPU_TENSOR_BYTES = 4 * 1024 * 1024
_VERSION = 1


class PackedGPUTransport(CheckpointTransport[object]):
    """Transfer a packed tensor snapshot over the Manager's reconfigured PG."""

    def __init__(
        self,
        pg: ProcessGroup,
        device: torch.device,
        timeout: timedelta,
        event: Optional[Callable[..., None]] = None,
    ) -> None:
        self._pg = pg
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"PackedGPUTransport requires a CUDA device, got {device}")
        self._timeout = timeout
        self._event = event

    def metadata(self) -> str:
        return "<packed-gpu>"

    def disallow_checkpoint(self) -> None:
        pass

    def shutdown(self, wait: bool = True) -> None:
        pass

    def _emit(
        self,
        kind: str,
        step: int,
        seconds: float,
        gpu_bytes: int,
        control_bytes: int,
        tensors: int,
        cpu_tensor_bytes: int,
    ) -> None:
        if self._event is not None:
            self._event(
                kind,
                step=step,
                seconds=seconds,
                gpu_bytes=gpu_bytes,
                control_bytes=control_bytes,
                tensors=tensors,
                cpu_tensor_bytes=cpu_tensor_bytes,
            )

    def _pack(
        self, state_dict: object, step: int
    ) -> tuple[bytes, list[torch.Tensor], int, int]:
        leaves, treespec = tree_flatten(state_dict)
        dtype_to_group: dict[torch.dtype, int] = {}
        chunks: list[list[torch.Tensor]] = []
        group_numels: list[int] = []
        leaf_meta: list[dict[str, Any]] = []
        cpu_tensor_bytes = 0
        tensor_count = 0

        for value in leaves:
            if isinstance(value, DTensor):
                raise TypeError("PackedGPUTransport does not support DTensor")

            if not isinstance(value, torch.Tensor):
                leaf_meta.append({"kind": "value", "value": value})
                continue

            tensor_count += 1
            if value.layout != torch.strided:
                raise TypeError(
                    f"only strided tensor leaves are supported, got {value.layout}"
                )
            if value.device.type == "cuda":
                dtype = value.dtype
                group = dtype_to_group.get(dtype)
                if group is None:
                    group = len(chunks)
                    dtype_to_group[dtype] = group
                    chunks.append([])
                    group_numels.append(0)

                flat = value.detach().reshape(-1).to(device=self._device)
                offset = group_numels[group]
                chunks[group].append(flat)
                group_numels[group] += flat.numel()
                leaf_meta.append(
                    {
                        "kind": "cuda",
                        "group": group,
                        "offset": offset,
                        "shape": tuple(value.shape),
                    }
                )
            elif value.device.type == "cpu":
                nbytes = value.numel() * value.element_size()
                cpu_tensor_bytes += nbytes
                if cpu_tensor_bytes > _MAX_CPU_TENSOR_BYTES:
                    raise ValueError(
                        "CPU tensor leaves exceed the 4 MiB control-data limit"
                    )
                # Store a compact, independent CPU value in pickle metadata.
                cpu_value = value.detach().contiguous().clone()
                leaf_meta.append({"kind": "cpu", "value": cpu_value})
            else:
                raise TypeError(
                    f"only CUDA and small CPU tensor leaves are supported, got {value.device}"
                )

        # cat creates independent contiguous snapshots, including for a single
        # tensor and for empty tensors. Keep these alive until every send waits.
        groups = [torch.cat(parts, dim=0) for parts in chunks]
        group_meta = [
            {"dtype": chunks[i][0].dtype, "numel": group_numels[i]}
            for i in range(len(chunks))
        ]
        metadata = {
            "version": _VERSION,
            "step": step,
            "treespec": treespec,
            "groups": group_meta,
            "leaves": leaf_meta,
            "tensor_count": tensor_count,
            "cpu_tensor_bytes": cpu_tensor_bytes,
        }
        metadata_bytes = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
        if len(metadata_bytes) > _MAX_METADATA_BYTES:
            raise ValueError(
                f"checkpoint metadata is {len(metadata_bytes)} bytes; "
                f"limit is {_MAX_METADATA_BYTES}"
            )
        return metadata_bytes, groups, tensor_count, cpu_tensor_bytes

    def send_checkpoint(
        self,
        dst_ranks: list[int],
        step: int,
        state_dict: object,
        timeout: timedelta,
    ) -> None:
        start = time.perf_counter()
        leaves, _ = tree_flatten(state_dict)
        versions = [(t, t._version) for t in leaves if isinstance(t, torch.Tensor)]
        metadata_bytes, groups, tensor_count, cpu_tensor_bytes = self._pack(
            state_dict, step
        )
        gpu_bytes = sum(t.numel() * t.element_size() for t in groups)
        control_bytes = len(metadata_bytes) + 8
        length = torch.tensor(
            [len(metadata_bytes)], dtype=torch.int64, device=self._device
        )
        metadata = torch.frombuffer(
            bytearray(metadata_bytes), dtype=torch.uint8
        ).to(self._device)

        works: list[Work] = []
        for dst_rank in dst_ranks:
            # Keep the per-peer sequence identical: length, metadata, then each
            # dtype group in its serialized order. NCCL does not match by tag.
            works.append(self._pg.send([length], dst_rank, tag=1))
            works.append(self._pg.send([metadata], dst_rank, tag=2))
            for index, flat in enumerate(groups):
                if flat.numel() > 0:
                    works.append(self._pg.send([flat], dst_rank, tag=3 + index))

        wait_timeout = timeout if timeout is not None else self._timeout
        for work in works:
            work.wait(wait_timeout)

        torch.cuda.current_stream(self._device).synchronize()
        if any(t._version != version for t, version in versions):
            raise RuntimeError('checkpoint source mutated before packing/send completed')
        if self._event is not None:
            self._event('snapshot_readonly_verified', step=step, tensors=len(versions))
        self._emit(
            "transport_send",
            step,
            time.perf_counter() - start,
            gpu_bytes,
            control_bytes,
            tensor_count,
            cpu_tensor_bytes,
        )

    def recv_checkpoint(
        self,
        src_rank: int,
        metadata: str,
        step: int,
        timeout: timedelta,
    ) -> object:
        del metadata  # The wire header carries the complete transport metadata.
        start = time.perf_counter()
        wait_timeout = timeout if timeout is not None else self._timeout

        length_tensor = torch.empty(1, dtype=torch.int64, device=self._device)
        self._pg.recv([length_tensor], src_rank, tag=1).wait(wait_timeout)
        metadata_length = int(length_tensor.item())
        if metadata_length <= 0 or metadata_length > _MAX_METADATA_BYTES:
            raise ValueError(
                f"invalid checkpoint metadata length {metadata_length}; "
                f"limit is {_MAX_METADATA_BYTES}"
            )

        metadata_tensor = torch.empty(
            metadata_length, dtype=torch.uint8, device=self._device
        )
        self._pg.recv([metadata_tensor], src_rank, tag=2).wait(wait_timeout)
        metadata_bytes = metadata_tensor.cpu().numpy().tobytes()
        wire = pickle.loads(metadata_bytes)
        if not isinstance(wire, dict) or wire.get("version") != _VERSION:
            raise ValueError("unsupported PackedGPUTransport metadata")
        if wire.get("step") != step:
            raise ValueError(f"checkpoint step mismatch: {wire.get('step')} != {step}")

        groups_meta = wire.get("groups")
        leaves_meta = wire.get("leaves")
        treespec = wire.get("treespec")
        if not isinstance(groups_meta, list) or not isinstance(leaves_meta, list):
            raise ValueError("malformed PackedGPUTransport metadata")

        cpu_tensor_bytes = 0
        for leaf in leaves_meta:
            if isinstance(leaf, dict) and leaf.get("kind") == "cpu":
                value = leaf.get("value")
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise ValueError("invalid CPU tensor in checkpoint metadata")
                if value.layout != torch.strided:
                    raise ValueError("only strided CPU tensors are supported")
                cpu_tensor_bytes += value.numel() * value.element_size()
                if cpu_tensor_bytes > _MAX_CPU_TENSOR_BYTES:
                    raise ValueError(
                        "CPU tensor leaves exceed the 4 MiB control-data limit"
                    )
        if cpu_tensor_bytes != wire.get("cpu_tensor_bytes"):
            raise ValueError("CPU tensor byte count does not match metadata")

        groups: list[torch.Tensor] = []
        works: list[Work] = []
        for index, group in enumerate(groups_meta):
            if not isinstance(group, dict):
                raise ValueError(f"invalid tensor group {index}")
            dtype = group.get("dtype")
            numel = group.get("numel")
            if (
                not isinstance(dtype, torch.dtype)
                or not isinstance(numel, int)
                or numel < 0
            ):
                raise ValueError(f"invalid tensor group {index}")
            flat = torch.empty(numel, dtype=dtype, device=self._device)
            groups.append(flat)
            if numel > 0:
                works.append(self._pg.recv([flat], src_rank, tag=3 + index))
        for work in works:
            work.wait(wait_timeout)

        values: list[object] = []
        for leaf in leaves_meta:
            if not isinstance(leaf, dict):
                raise ValueError("invalid checkpoint leaf metadata")
            kind = leaf.get("kind")
            if kind in ("value", "cpu"):
                value = leaf.get("value")
                values.append(value)
            elif kind == "cuda":
                group_index = leaf.get("group")
                offset = leaf.get("offset")
                shape = leaf.get("shape")
                if (
                    not isinstance(group_index, int)
                    or group_index < 0
                    or group_index >= len(groups)
                    or not isinstance(offset, int)
                    or offset < 0
                    or not isinstance(shape, (tuple, list))
                    or any(not isinstance(dim, int) or dim < 0 for dim in shape)
                ):
                    raise ValueError("invalid CUDA tensor leaf metadata")
                numel = math.prod(shape)
                flat = groups[group_index]
                if offset + numel > flat.numel():
                    raise ValueError("CUDA tensor leaf exceeds its packed group")
                values.append(flat.narrow(0, offset, numel).view(tuple(shape)))
            else:
                raise ValueError(f"unknown checkpoint leaf kind: {kind!r}")

        result = tree_unflatten(values, treespec)
        gpu_bytes = sum(t.numel() * t.element_size() for t in groups)
        control_bytes = metadata_length + 8
        torch.cuda.current_stream(self._device).synchronize()
        self._emit(
            "transport_recv",
            step,
            time.perf_counter() - start,
            gpu_bytes,
            control_bytes,
            int(wire.get("tensor_count", len(leaves_meta))),
            cpu_tensor_bytes,
        )
        return result
