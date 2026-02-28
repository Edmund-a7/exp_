from typing import Optional, Tuple


def compute_prompt_sparse_policy(
    chunk_idx_in_prompt: int,
    total_chunks_in_prompt: int,
    *,
    min_keep_ratio: float = 0.3,
    max_keep_ratio: float = 0.5,
    schedule_mode: str = "increase",
    force_sma_chunk0: bool = True,
) -> Tuple[bool, Optional[float]]:
    """
    Compute sparse policy for one chunk inside a prompt.

    Returns:
        (force_sma, keep_ratio_override)
        - force_sma=True means use SMA for this chunk
        - keep_ratio_override is used by HSA block selection (None means use model default)
    """
    chunk_idx = max(0, int(chunk_idx_in_prompt))
    total_chunks = max(1, int(total_chunks_in_prompt))

    if force_sma_chunk0 and chunk_idx == 0:
        return True, None

    low = min(float(min_keep_ratio), float(max_keep_ratio))
    high = max(float(min_keep_ratio), float(max_keep_ratio))

    sparse_chunks = max(1, total_chunks - 1)  # chunk0 reserved for SMA
    if sparse_chunks == 1:
        frac = 0.0
    else:
        sparse_idx = min(max(chunk_idx - 1, 0), sparse_chunks - 1)
        frac = sparse_idx / float(sparse_chunks - 1)

    mode = str(schedule_mode).lower().strip()
    if mode == "decrease":
        frac = 1.0 - frac

    keep_ratio = low + frac * (high - low)
    return False, float(keep_ratio)
