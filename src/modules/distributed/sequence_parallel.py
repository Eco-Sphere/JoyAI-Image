from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
import torch.distributed as dist


TensorOrGetter = Union[torch.Tensor, Callable[[], torch.Tensor]]


@dataclass
class SequenceParallelGroup:
    device_group: Optional[dist.ProcessGroup]
    ranks: List[int]
    rank_in_group: int

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    def all_gather(
        self,
        input_tensor: torch.Tensor,
        *,
        dim: int = 0,
        async_op: bool = False,
    ) -> TensorOrGetter:
        if self.world_size <= 1 or self.device_group is None:
            return input_tensor

        dim = dim if dim >= 0 else input_tensor.dim() + dim
        if dim < 0 or dim >= input_tensor.dim():
            raise ValueError(
                f"Invalid gather dim={dim} for input rank={input_tensor.dim()}."
            )

        input_tensor = input_tensor.contiguous()
        gather_list = [
            torch.empty(
                input_tensor.shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            for _ in range(self.world_size)
        ]
        work = dist.all_gather(
            gather_list, input_tensor, group=self.device_group, async_op=async_op
        )

        def _finalize() -> torch.Tensor:
            if async_op:
                work.wait()
            return torch.cat(gather_list, dim=dim)

        return _finalize if async_op else _finalize()


_SP_GROUP: Optional[SequenceParallelGroup] = None


def _read_ulysses_size() -> int:
    value = os.environ.get("ULYSSES_SIZE", "1").strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"ULYSSES_SIZE must be an integer, but got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"ULYSSES_SIZE must be >= 1, but got {parsed}.")
    return parsed


def init_sequence_parallel_group(ulysses_size: Optional[int] = None) -> SequenceParallelGroup:
    global _SP_GROUP

    if not dist.is_available() or not dist.is_initialized():
        _SP_GROUP = SequenceParallelGroup(device_group=None, ranks=[0], rank_in_group=0)
        return _SP_GROUP

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    group_size = ulysses_size if ulysses_size is not None else _read_ulysses_size()
    if group_size <= 1:
        _SP_GROUP = SequenceParallelGroup(
            device_group=None, ranks=[global_rank], rank_in_group=0
        )
        return _SP_GROUP

    if world_size % group_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} is not divisible by ULYSSES_SIZE={group_size}."
        )
    if world_size != group_size:
        raise ValueError(
            "Current sequence parallel path expects WORLD_SIZE == ULYSSES_SIZE. "
            f"Got WORLD_SIZE={world_size}, ULYSSES_SIZE={group_size}."
        )

    local_group: Optional[dist.ProcessGroup] = None
    local_ranks: Optional[List[int]] = None
    for start in range(0, world_size, group_size):
        ranks = list(range(start, start + group_size))
        pg = dist.new_group(ranks=ranks)
        if global_rank in ranks:
            local_group = pg
            local_ranks = ranks

    if local_group is None or local_ranks is None:
        raise RuntimeError("Failed to build sequence parallel process group.")

    _SP_GROUP = SequenceParallelGroup(
        device_group=local_group,
        ranks=local_ranks,
        rank_in_group=local_ranks.index(global_rank),
    )
    return _SP_GROUP


def get_sp_group() -> SequenceParallelGroup:
    global _SP_GROUP
    if _SP_GROUP is None:
        return init_sequence_parallel_group()
    return _SP_GROUP


def get_sequence_parallel_state() -> bool:
    return get_sequence_parallel_world_size() > 1


def get_sequence_parallel_world_size() -> int:
    return get_sp_group().world_size


def get_sequence_parallel_rank() -> int:
    return get_sp_group().rank_in_group


def reset_sequence_parallel_group() -> None:
    global _SP_GROUP
    _SP_GROUP = None

