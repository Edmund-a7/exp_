import os
import importlib.util


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT_DIR, "utils", "hsa_schedule.py")
SPEC = importlib.util.spec_from_file_location("hsa_schedule", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

compute_prompt_sparse_policy = MODULE.compute_prompt_sparse_policy


def test_chunk0_force_sma():
    force_sma, keep_ratio = compute_prompt_sparse_policy(
        chunk_idx_in_prompt=0,
        total_chunks_in_prompt=8,
        min_keep_ratio=0.3,
        max_keep_ratio=0.5,
        schedule_mode="increase",
        force_sma_chunk0=True,
    )
    assert force_sma is True
    assert keep_ratio is None


def test_increase_schedule_spans_min_to_max():
    _, keep_first = compute_prompt_sparse_policy(
        chunk_idx_in_prompt=1,
        total_chunks_in_prompt=6,
        min_keep_ratio=0.3,
        max_keep_ratio=0.5,
        schedule_mode="increase",
        force_sma_chunk0=True,
    )
    _, keep_last = compute_prompt_sparse_policy(
        chunk_idx_in_prompt=5,
        total_chunks_in_prompt=6,
        min_keep_ratio=0.3,
        max_keep_ratio=0.5,
        schedule_mode="increase",
        force_sma_chunk0=True,
    )
    assert abs(keep_first - 0.3) < 1e-6
    assert abs(keep_last - 0.5) < 1e-6


def test_decrease_schedule_spans_max_to_min():
    _, keep_first = compute_prompt_sparse_policy(
        chunk_idx_in_prompt=1,
        total_chunks_in_prompt=6,
        min_keep_ratio=0.3,
        max_keep_ratio=0.5,
        schedule_mode="decrease",
        force_sma_chunk0=True,
    )
    _, keep_last = compute_prompt_sparse_policy(
        chunk_idx_in_prompt=5,
        total_chunks_in_prompt=6,
        min_keep_ratio=0.3,
        max_keep_ratio=0.5,
        schedule_mode="decrease",
        force_sma_chunk0=True,
    )
    assert abs(keep_first - 0.5) < 1e-6
    assert abs(keep_last - 0.3) < 1e-6
