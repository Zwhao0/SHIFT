"""Packed, GPU-to-GPU checkpoint transport over a configured torchft PG."""

from __future__ import annotations

import math
import os
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
_CHUNK_BYTES = 64 * 1024 * 1024


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
        self._stage_cpu = os.environ.get('SHIFT_CPU_ADAM')=='1'

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
                staged_cpu_state=getattr(self,'_stage_cpu',False),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device),
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
            staged_cpu=self._stage_cpu and value.device.type=='cpu' and value.numel()*value.element_size()>4096
            if value.device.type == "cuda" or staged_cpu:
                dtype = value.dtype
                group = None if self._stage_cpu else dtype_to_group.get(dtype)
                if group is None:
                    group = len(chunks)
                    dtype_to_group[dtype] = group
                    chunks.append([])
                    group_numels.append(0)

                flat = value.detach().reshape(-1)
                if not staged_cpu:flat=flat.to(device=self._device)
                offset = group_numels[group]
                chunks[group].append(flat)
                group_numels[group] += flat.numel()
                leaf_meta.append(
                    {
                        "kind": "staged_cpu" if staged_cpu else "cuda",
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

        # Legacy cat creates independent contiguous snapshots. Streamed mode
        # borrows tensors under the checked read-only boundary until send waits.
        groups=[]
        for i,parts in enumerate(chunks):
            if self._stage_cpu:
                # Each leaf is one group. Frozen CUDA values are sent directly;
                # CPU moments use a bounded reusable CUDA staging buffer.
                groups.append(parts[0])
            else:groups.append(torch.cat(parts,dim=0))
        group_meta = [
            {"dtype": chunks[i][0].dtype, "numel": group_numels[i],
             "device":chunks[i][0].device.type}
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
            "streamed":self._stage_cpu,
            "chunk_bytes":_CHUNK_BYTES,
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
        versions = (self.frozen_versions if state_dict is None and hasattr(self,'frozen_versions')
                    else [(t,t._version) for t in leaves if isinstance(t,torch.Tensor)])
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
            if not self._stage_cpu:
                for index, flat in enumerate(groups):
                    if flat.numel() > 0:
                        works.append(self._pg.send([flat], dst_rank, tag=3 + index))

        wait_timeout = timeout if timeout is not None else self._timeout
        for work in works:
            work.wait(wait_timeout)

        if self._stage_cpu:
            for index,flat in enumerate(groups):
                count=max(1,_CHUNK_BYTES//flat.element_size())
                scratch=torch.empty(min(count,flat.numel()),dtype=flat.dtype,device=self._device) if flat.device.type=='cpu' else None
                for offset in range(0,flat.numel(),count):
                    length=min(count,flat.numel()-offset)
                    chunk=flat.narrow(0,offset,length)
                    if scratch is not None:
                        scratch[:length].copy_(chunk);chunk=scratch[:length]
                    for dst_rank in dst_ranks:self._pg.send([chunk],dst_rank,tag=3+index).wait(wait_timeout)
                    if scratch is not None:torch.cuda.current_stream(self._device).synchronize()

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
        streamed=wire.get('streamed',False)
        if type(streamed) is not bool or (streamed and wire.get('chunk_bytes')!=_CHUNK_BYTES):
            raise ValueError('unsupported streamed tensor chunk size')
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
            location=group.get('device','cuda') if streamed else 'cuda'
            if location not in ('cpu','cuda'):raise ValueError('invalid streamed tensor location')
            flat = torch.empty(numel, dtype=dtype, device='cpu' if location=='cpu' else self._device)
            groups.append(flat)
            if numel > 0:
                if streamed:
                    count=max(1,_CHUNK_BYTES//flat.element_size())
                    scratch=torch.empty(min(count,numel),dtype=dtype,device=self._device) if location=='cpu' else None
                    for offset in range(0,numel,count):
                        length=min(count,numel-offset)
                        chunk=scratch[:length] if scratch is not None else flat.narrow(0,offset,length)
                        self._pg.recv([chunk],src_rank,tag=3+index).wait(wait_timeout)
                        if scratch is not None:flat.narrow(0,offset,length).copy_(chunk)
                else:works.append(self._pg.recv([flat], src_rank, tag=3 + index))
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
            elif kind in ("cuda","staged_cpu"):
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
                value=flat.narrow(0,offset,numel).view(tuple(shape))
                if kind=='staged_cpu':value=value.to('cpu')
                values.append(value)
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
