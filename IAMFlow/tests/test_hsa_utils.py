import os
import sys
import importlib.util

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT_DIR, "wan", "modules", "hsa_utils.py")
SPEC = importlib.util.spec_from_file_location("hsa_utils", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

split_local_far_near = MODULE.split_local_far_near
select_frame_indices_with_quota = MODULE.select_frame_indices_with_quota
select_top_blocks_shared = MODULE.select_top_blocks_shared


def test_select_frame_indices_with_quota_prefers_both_regions():
    sink_scores = torch.tensor([0.90, 0.80])
    mem_scores = torch.tensor([0.95, 0.70, 0.60])

    sink_idx, mem_idx = select_frame_indices_with_quota(
        sink_scores=sink_scores,
        mem_scores=mem_scores,
        top_k=2,
        min_sink=1,
        min_mem=1,
    )

    assert sink_idx == [0]
    assert mem_idx == [0]


def test_select_frame_indices_with_quota_fallback_single_region():
    sink_scores = torch.tensor([0.20, 0.10, 0.05])
    mem_scores = torch.empty(0)

    sink_idx, mem_idx = select_frame_indices_with_quota(
        sink_scores=sink_scores,
        mem_scores=mem_scores,
        top_k=2,
        min_sink=1,
        min_mem=1,
    )

    assert sink_idx == [0, 1]
    assert mem_idx == []


def test_split_local_far_near_9_frames():
    frame_tokens = 1560
    k_local = torch.randn(1, 9 * frame_tokens, 2, 4)
    v_local = torch.randn(1, 9 * frame_tokens, 2, 4)

    k_far, v_far, k_near, v_near = split_local_far_near(
        k_local=k_local,
        v_local=v_local,
        frame_tokens=frame_tokens,
        far_frames=6,
        near_frames=3,
    )

    assert k_far.shape[1] == 6 * frame_tokens
    assert v_far.shape[1] == 6 * frame_tokens
    assert k_near.shape[1] == 3 * frame_tokens
    assert v_near.shape[1] == 3 * frame_tokens


def test_select_top_blocks_shared_keeps_expected_block_count():
    frame_tokens = 1560
    block_size = 64

    # 2 frames => 50 blocks/frame-based with tail (25*2)
    k = torch.randn(1, 2 * frame_tokens, 2, 4)
    v = torch.randn(1, 2 * frame_tokens, 2, 4)
    q = torch.randn(1, 4680, 2, 4)

    k_sel, v_sel, selected = select_top_blocks_shared(
        query=q,
        key=k,
        value=v,
        frame_tokens=frame_tokens,
        block_size=block_size,
        keep_ratio=0.2,
        min_blocks=1,
    )

    assert len(selected) == 10  # ceil(50 * 0.2)
    assert k_sel.shape[1] > 0
    assert v_sel.shape[1] == k_sel.shape[1]
