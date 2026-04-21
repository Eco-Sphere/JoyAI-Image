from __future__ import annotations

from typing import Callable, Union

import torch
import torch.distributed as dist


TensorOrGetter = Union[torch.Tensor, Callable[[], torch.Tensor]]


def seq_all_to_all_4d(
    input_tensor: torch.Tensor,
    *,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: dist.ProcessGroup | None = None,
    use_sync: bool = False,
) -> TensorOrGetter:
    """All-to-all for 4D QKV tensors used by sequence-parallel attention."""
    if input_tensor.dim() != 4:
        raise ValueError(
            f"input_tensor must be 4D, but got shape={tuple(input_tensor.shape)}"
        )

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or group is None
    ):
        return input_tensor

    seq_world_size = dist.get_world_size(group)
    if seq_world_size <= 1:
        return input_tensor

    if scatter_idx == 2 and gather_idx == 1:
        # [B, S/P, H, D] -> [B, S, H/P, D]
        bs, shard_seqlen, hc, hs = input_tensor.shape
        if hc % seq_world_size != 0:
            raise ValueError(
                f"Head count {hc} is not divisible by seq_world_size {seq_world_size}."
            )
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        input_t = (
            input_tensor.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )
        output = torch.empty_like(input_t)
        if use_sync:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            work = dist.all_to_all_single(output, input_t, group=group, async_op=True)

            def _getter() -> torch.Tensor:
                work.wait()
                result = output.reshape(seqlen, bs, shard_hc, hs)
                return result.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

            return _getter

        result = output.reshape(seqlen, bs, shard_hc, hs)
        return result.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

    if scatter_idx == 1 and gather_idx == 2:
        # [B, S, H/P, D] -> [B, S/P, H, D]
        bs, seqlen, shard_hc, hs = input_tensor.shape
        if seqlen % seq_world_size != 0:
            raise ValueError(
                f"Sequence length {seqlen} is not divisible by seq_world_size {seq_world_size}."
            )
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        input_t = (
            input_tensor.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )
        output = torch.empty_like(input_t)
        if use_sync:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            work = dist.all_to_all_single(output, input_t, group=group, async_op=True)

            def _getter() -> torch.Tensor:
                work.wait()
                result = output.reshape(hc, shard_seqlen, bs, hs)
                return result.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

            return _getter

        result = output.reshape(hc, shard_seqlen, bs, hs)
        return result.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

    raise ValueError(
        f"Unsupported all_to_all layout: scatter_idx={scatter_idx}, gather_idx={gather_idx}"
    )

