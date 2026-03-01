import math
from typing import List, Tuple

import torch


def compute_frame_scores_shared(
    query: torch.Tensor,
    key: torch.Tensor,
    frame_tokens: int,
) -> torch.Tensor:
    """
    Compute shared frame scores across batch/heads.

    Args:
        query: [B, L_q, H, D]
        key: [B, L_k, H, D]
        frame_tokens: tokens per frame

    Returns:
        [num_frames] score tensor on the same device/dtype as key.
    """
    if key.numel() == 0 or frame_tokens <= 0 or key.shape[1] < frame_tokens:
        return key.new_empty((0,))

    num_frames = key.shape[1] // frame_tokens
    if num_frames <= 0:
        return key.new_empty((0,))

    valid_tokens = num_frames * frame_tokens
    key = key[:, :valid_tokens]

    # Shared descriptor across query tokens.
    q_desc = query.mean(dim=1).mean(dim=1)  # [B, D]
    k_frames = key.view(key.shape[0], num_frames, frame_tokens, key.shape[2], key.shape[3])
    k_desc = k_frames.mean(dim=2).mean(dim=2)  # [B, F, D]

    scores = (q_desc.unsqueeze(1) * k_desc).sum(dim=-1)  # [B, F]
    scores = scores.mean(dim=0)  # [F]
    return scores


def gather_frames_by_indices(
    key: torch.Tensor,
    value: torch.Tensor,
    frame_indices: List[int],
    frame_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather frames by indices. Indices are relative to `key`/`value`.
    """
    if not frame_indices or key.numel() == 0:
        return key[:, :0], value[:, :0]

    num_frames = key.shape[1] // frame_tokens
    if num_frames <= 0:
        return key[:, :0], value[:, :0]

    valid_set = sorted(i for i in set(frame_indices) if 0 <= i < num_frames)
    if not valid_set:
        return key[:, :0], value[:, :0]

    k_parts = []
    v_parts = []
    for frame_idx in valid_set:
        start = frame_idx * frame_tokens
        end = start + frame_tokens
        k_parts.append(key[:, start:end])
        v_parts.append(value[:, start:end])

    return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)


def select_frame_indices_with_quota(
    sink_scores: torch.Tensor,
    mem_scores: torch.Tensor,
    top_k: int,
    min_sink: int = 1,
    min_mem: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Select frame indices with optional per-region minimum quotas.
    """
    top_k = max(0, int(top_k))
    if top_k == 0:
        return [], []

    selected_sink: List[int] = []
    selected_mem: List[int] = []

    sink_n = int(sink_scores.numel())
    mem_n = int(mem_scores.numel())

    # Quota picks first.
    if sink_n > 0 and min_sink > 0:
        k = min(min_sink, sink_n, top_k)
        selected_sink.extend(torch.topk(sink_scores, k=k).indices.tolist())

    used = len(selected_sink)
    if mem_n > 0 and min_mem > 0 and used < top_k:
        k = min(min_mem, mem_n, top_k - used)
        selected_mem.extend(torch.topk(mem_scores, k=k).indices.tolist())

    used = len(selected_sink) + len(selected_mem)
    remaining = top_k - used
    if remaining > 0:
        candidates = []
        picked_sink = set(selected_sink)
        picked_mem = set(selected_mem)

        for idx in range(sink_n):
            if idx not in picked_sink:
                candidates.append(("sink", idx, float(sink_scores[idx])))
        for idx in range(mem_n):
            if idx not in picked_mem:
                candidates.append(("mem", idx, float(mem_scores[idx])))

        candidates.sort(key=lambda x: x[2], reverse=True)
        for region, idx, _ in candidates[:remaining]:
            if region == "sink":
                selected_sink.append(idx)
            else:
                selected_mem.append(idx)

    selected_sink = sorted(set(selected_sink))
    selected_mem = sorted(set(selected_mem))
    return selected_sink, selected_mem


def split_local_far_near(
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    frame_tokens: int,
    far_frames: int,
    near_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split local tokens into local-far and local-near.
    """
    if frame_tokens <= 0 or k_local.numel() == 0:
        return k_local[:, :0], v_local[:, :0], k_local[:, :0], v_local[:, :0]

    num_frames = k_local.shape[1] // frame_tokens
    valid_tokens = num_frames * frame_tokens
    k_local = k_local[:, :valid_tokens]
    v_local = v_local[:, :valid_tokens]

    near_frames = max(0, min(int(near_frames), num_frames))
    remaining = max(0, num_frames - near_frames)
    far_frames = max(0, min(int(far_frames), remaining))

    far_tokens = far_frames * frame_tokens
    near_tokens = near_frames * frame_tokens

    k_far = k_local[:, :far_tokens] if far_tokens > 0 else k_local[:, :0]
    v_far = v_local[:, :far_tokens] if far_tokens > 0 else v_local[:, :0]
    k_near = k_local[:, -near_tokens:] if near_tokens > 0 else k_local[:, :0]
    v_near = v_local[:, -near_tokens:] if near_tokens > 0 else v_local[:, :0]

    return k_far, v_far, k_near, v_near


def select_top_blocks_shared(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frame_tokens: int,
    block_size: int,
    keep_ratio: float,
    min_blocks: int = 1,
    max_blocks: int = 0,
    per_frame_keep_ratio: List[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Select top-k blocks shared across batch by score.

    When *per_frame_keep_ratio* is provided (one float per frame in key),
    each frame uses its own keep ratio for independent per-frame top-k
    selection.  Otherwise falls back to the original global top-k with
    *keep_ratio*.

    Returns:
        (selected_k, selected_v, selected_block_indices)
    """
    if key.numel() == 0 or frame_tokens <= 0 or block_size <= 0:
        return key[:, :0], value[:, :0], []

    num_frames = key.shape[1] // frame_tokens
    if num_frames <= 0:
        return key[:, :0], value[:, :0], []

    valid_tokens = num_frames * frame_tokens
    key = key[:, :valid_tokens]
    value = value[:, :valid_tokens]

    blocks_per_frame = int(math.ceil(frame_tokens / block_size))
    num_blocks = num_frames * blocks_per_frame
    if num_blocks == 0:
        return key[:, :0], value[:, :0], []

    # --- Vectorized block scoring (shared for both paths) ---
    B, _, H, D = key.shape
    key_frames = key.view(B, num_frames, frame_tokens, H, D)

    padded_tokens_per_frame = blocks_per_frame * block_size
    if padded_tokens_per_frame > frame_tokens:
        pad_tokens = padded_tokens_per_frame - frame_tokens
        key_pad = torch.zeros(
            (B, num_frames, pad_tokens, H, D),
            device=key.device,
            dtype=key.dtype,
        )
        key_frames = torch.cat([key_frames, key_pad], dim=2)

    key_blocks = key_frames.view(B, num_frames, blocks_per_frame, block_size, H, D)

    block_lengths = torch.full(
        (blocks_per_frame,),
        float(block_size),
        device=key.device,
        dtype=torch.float32,
    )
    tail_tokens = frame_tokens - (blocks_per_frame - 1) * block_size
    block_lengths[-1] = float(tail_tokens)

    key_block_sum = key_blocks.sum(dim=3).sum(dim=3)
    denom = block_lengths.view(1, 1, blocks_per_frame, 1) * float(H)
    key_block_desc = key_block_sum / denom.to(dtype=key_block_sum.dtype)

    q_desc = query.mean(dim=1).mean(dim=1)  # [B, D]

    # --- Selection ---
    use_per_frame = False
    if per_frame_keep_ratio is not None:
        if len(per_frame_keep_ratio) == num_frames:
            use_per_frame = True
        else:
            import warnings
            warnings.warn(
                f"per_frame_keep_ratio length {len(per_frame_keep_ratio)} != "
                f"num_frames {num_frames}, falling back to global keep_ratio"
            )

    if use_per_frame:
        # Per-frame top-k via grouped batch: frames sharing the same ratio
        # are batched into a single topk call to minimise kernel launches.
        frame_scores = (
            q_desc.view(B, 1, 1, D) * key_block_desc
        ).sum(dim=-1).mean(dim=0)  # [F, BP]

        # Group frames by n_keep (derived from ratio) — typically only 4 groups.
        from collections import defaultdict
        groups = defaultdict(list)  # n_keep -> [frame_indices]
        for f in range(num_frames):
            ratio_f = float(per_frame_keep_ratio[f])
            n_keep = max(1, min(blocks_per_frame, int(math.ceil(blocks_per_frame * ratio_f))))
            groups[n_keep].append(f)

        selected_idx = []
        for n_keep, frame_list in groups.items():
            batch_scores = frame_scores[frame_list]          # [G, BP]
            top_indices = torch.topk(batch_scores, k=n_keep, dim=1).indices  # [G, n_keep]
            # Convert to flat block indices: frame_idx * blocks_per_frame + block_idx
            offsets = torch.tensor(frame_list, device=top_indices.device).unsqueeze(1) * blocks_per_frame
            flat = (top_indices + offsets).reshape(-1)
            selected_idx.extend(flat.tolist())
        selected_idx.sort()
    else:
        # Original global top-k path
        key_block_desc_flat = key_block_desc.reshape(B, num_blocks, D)
        score_tensor = (
            q_desc.unsqueeze(1) * key_block_desc_flat
        ).sum(dim=-1).mean(dim=0)

        keep_ratio = float(keep_ratio)
        min_blocks = max(1, int(min_blocks))
        k_blocks = max(min_blocks, int(math.ceil(num_blocks * keep_ratio)))
        if max_blocks and max_blocks > 0:
            k_blocks = min(k_blocks, int(max_blocks))
        k_blocks = min(k_blocks, num_blocks)

        top_idx = torch.topk(score_tensor, k=k_blocks).indices
        top_idx, _ = torch.sort(top_idx)
        selected_idx = top_idx.tolist()

    # --- Gather selected blocks from original (un-padded) 1-D sequence ---
    if not selected_idx:
        return key[:, :0], value[:, :0], []

    # Vectorized gather: build flat token indices for all selected blocks.
    token_indices = []
    for idx in selected_idx:
        frame_idx = idx // blocks_per_frame
        block_idx = idx % blocks_per_frame
        frame_base = frame_idx * frame_tokens
        start = frame_base + block_idx * block_size
        end = min(start + block_size, frame_base + frame_tokens)
        token_indices.extend(range(start, end))

    idx_tensor = torch.tensor(token_indices, device=key.device, dtype=torch.long)
    sel_k = key[:, idx_tensor]
    sel_v = value[:, idx_tensor]

    return sel_k, sel_v, selected_idx
