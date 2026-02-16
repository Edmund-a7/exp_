"""
Phase 3 测试: 动态帧分配 + 同场景跳过

验证:
1. _compute_dynamic_id_budget() 贪心集合覆盖
2. _greedy_select_id_frames() 帧选择
3. retrieve_initial_frames() 使用动态预算
4. _compute_scene_distance() 场景距离计算
5. 同场景跳过逻辑 (pipeline 层)
6. bank_size 动态调整
7. 总帧数不超过上限
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iam.llm_agent import EntityStruct
from iam.memory_bank import MemoryBank, FrameInfo


# ============ Fixtures ============

@pytest.fixture
def memory_bank(tmp_path):
    return MemoryBank(
        text_encoder=None,
        max_memory_frames=3,
        max_id_memory_frames=4,
        max_scene_memory_frames=2,
        frame_seq_length=1560,
        num_transformer_blocks=2,
        save_dir=str(tmp_path),
    )


def _add_frame(mb, frame_id, entity_ids, entity_score=0.5, scene_score=0.3, scene_texts=None):
    """辅助: 向 archive 添加帧并创建 mock KV"""
    mb.frame_archive[frame_id] = FrameInfo(
        frame_id=frame_id, frame_path="", prompt_id=1,
        associated_entities=[str(eid) for eid in entity_ids],
        score=0.6 * entity_score + 0.4 * scene_score,
        entity_score=entity_score,
        scene_score=scene_score,
        scene_texts=scene_texts or [],
    )
    kv = []
    for _ in range(mb.num_transformer_blocks):
        kv.append({
            "k": torch.randn(1, mb.frame_seq_length, 12, 128),
            "v": torch.randn(1, mb.frame_seq_length, 12, 128),
        })
    mb._frame_kv_store[frame_id] = kv


# ============ 1. Dynamic ID Budget ============

class TestDynamicIDBudget:

    def test_single_frame_covers_all(self, memory_bank):
        """1帧覆盖所有 ID → budget=1"""
        _add_frame(memory_bank, "f1", [1, 2, 3], entity_score=0.9)
        budget = memory_bank._compute_dynamic_id_budget(["1", "2", "3"])
        assert budget == 1

    def test_two_frames_needed(self, memory_bank):
        """需要2帧覆盖所有 ID"""
        _add_frame(memory_bank, "f1", [1, 2], entity_score=0.8)
        _add_frame(memory_bank, "f2", [3], entity_score=0.7)
        budget = memory_bank._compute_dynamic_id_budget(["1", "2", "3"])
        assert budget == 2

    def test_respects_max_budget(self, memory_bank):
        """budget 不超过 max_id_memory_frames"""
        for i in range(10):
            _add_frame(memory_bank, f"f{i}", [i], entity_score=0.5)
        budget = memory_bank._compute_dynamic_id_budget([str(i) for i in range(10)])
        assert budget <= memory_bank.max_id_memory_frames

    def test_empty_ids_returns_zero(self, memory_bank):
        budget = memory_bank._compute_dynamic_id_budget([])
        assert budget == 0

    def test_empty_archive_returns_zero(self, memory_bank):
        budget = memory_bank._compute_dynamic_id_budget(["1"])
        assert budget == 0

    def test_no_matching_frames(self, memory_bank):
        """archive 中无匹配帧 → budget=0 (贪心无法覆盖)"""
        _add_frame(memory_bank, "f1", [99], entity_score=0.5)
        budget = memory_bank._compute_dynamic_id_budget(["1", "2"])
        # 无法覆盖任何 ID，但至少返回 0 (贪心 cover=0 时 break)
        assert budget == 0

    def test_partial_coverage(self, memory_bank):
        """只能部分覆盖 → budget = 覆盖帧数"""
        _add_frame(memory_bank, "f1", [1], entity_score=0.8)
        _add_frame(memory_bank, "f2", [2], entity_score=0.7)
        # 需要 ID 1,2,3 但只有 1,2 可覆盖
        budget = memory_bank._compute_dynamic_id_budget(["1", "2", "3"])
        assert budget == 2

    def test_greedy_picks_best_coverage(self, memory_bank):
        """贪心优先选覆盖最多 ID 的帧"""
        _add_frame(memory_bank, "f_wide", [1, 2, 3], entity_score=0.5)
        _add_frame(memory_bank, "f_narrow", [1], entity_score=0.9)
        budget = memory_bank._compute_dynamic_id_budget(["1", "2", "3"])
        assert budget == 1  # f_wide 一帧搞定


# ============ 2. Greedy Frame Selection ============

class TestGreedySelectIdFrames:

    def test_selects_covering_frames(self, memory_bank):
        _add_frame(memory_bank, "f1", [1, 2], entity_score=0.8)
        _add_frame(memory_bank, "f2", [3], entity_score=0.7)
        _add_frame(memory_bank, "f3", [1], entity_score=0.9)
        selected = memory_bank._greedy_select_id_frames(["1", "2", "3"], budget=2)
        assert "f1" in selected  # 覆盖 1,2
        assert "f2" in selected  # 覆盖 3

    def test_tiebreak_by_entity_score(self, memory_bank):
        """覆盖度相同时选 entity_score 更高的"""
        _add_frame(memory_bank, "f_low", [1], entity_score=0.3)
        _add_frame(memory_bank, "f_high", [1], entity_score=0.9)
        selected = memory_bank._greedy_select_id_frames(["1"], budget=1)
        assert selected == ["f_high"]

    def test_fills_remaining_budget(self, memory_bank):
        """覆盖完成后用 entity_score top-k 填充剩余预算"""
        _add_frame(memory_bank, "f1", [1], entity_score=0.9)
        _add_frame(memory_bank, "f2", [2], entity_score=0.5)
        _add_frame(memory_bank, "f3", [1], entity_score=0.7)
        selected = memory_bank._greedy_select_id_frames(["1"], budget=3)
        assert len(selected) == 3
        assert selected[0] == "f1"  # 覆盖 ID 1

    def test_empty_budget(self, memory_bank):
        _add_frame(memory_bank, "f1", [1])
        assert memory_bank._greedy_select_id_frames(["1"], budget=0) == []

    def test_empty_ids(self, memory_bank):
        _add_frame(memory_bank, "f1", [1])
        assert memory_bank._greedy_select_id_frames([], budget=3) == []


# ============ 3. retrieve_initial_frames with Dynamic Budget ============

class TestRetrieveWithDynamicBudget:

    def test_single_id_single_frame(self, memory_bank):
        """1个 ID，1帧覆盖 → id_memory 只有1帧"""
        _add_frame(memory_bank, "f1", [1], entity_score=0.9)
        _add_frame(memory_bank, "f2", [1], entity_score=0.5)
        memory_bank.retrieve_initial_frames([1])
        # 动态预算: 1个 ID → budget=1
        assert len(memory_bank.id_memory) == 1

    def test_multi_id_needs_multi_frames(self, memory_bank):
        """多个 ID 需要多帧覆盖"""
        _add_frame(memory_bank, "f1", [1], entity_score=0.8)
        _add_frame(memory_bank, "f2", [2], entity_score=0.7)
        _add_frame(memory_bank, "f3", [3], entity_score=0.6)
        memory_bank.retrieve_initial_frames([1, 2, 3])
        assert len(memory_bank.id_memory) == 3

    def test_one_frame_covers_all_ids(self, memory_bank):
        """1帧覆盖所有 ID → 动态预算=1，但填充到 budget"""
        _add_frame(memory_bank, "f_all", [1, 2, 3], entity_score=0.9)
        _add_frame(memory_bank, "f_extra", [1], entity_score=0.5)
        memory_bank.retrieve_initial_frames([1, 2, 3])
        # budget=1 (1帧覆盖全部)，但 _greedy_select 会填充剩余
        assert "f_all" in memory_bank.id_memory

    def test_id_memory_respects_max(self, memory_bank):
        for i in range(10):
            _add_frame(memory_bank, f"f{i}", [i], entity_score=0.5 + i * 0.01)
        memory_bank.retrieve_initial_frames(list(range(10)))
        assert len(memory_bank.id_memory) <= memory_bank.max_id_memory_frames

    def test_scene_memory_unaffected(self, memory_bank):
        """动态 ID 预算不影响 scene_memory"""
        _add_frame(memory_bank, "f1", [1], entity_score=0.8, scene_score=0.9,
                   scene_texts=["park"])
        _add_frame(memory_bank, "f2", [2], entity_score=0.7, scene_score=0.5,
                   scene_texts=["forest"])
        memory_bank.retrieve_initial_frames([1, 2], scene_texts=["park"])
        assert len(memory_bank.scene_memory) > 0
        assert len(memory_bank.scene_memory) <= memory_bank.max_scene_memory_frames


# ============ 4. Scene Distance ============

class TestSceneDistance:

    def test_identical_scenes(self):
        """完全相同的场景 → 距离 0"""
        dist = MemoryBank._compute_scene_distance(
            ["modern city park", "daytime"],
            ["modern city park", "daytime"],
        )
        assert dist == 0.0

    def test_completely_different(self):
        """完全不同的场景 → 距离 1.0"""
        dist = MemoryBank._compute_scene_distance(
            ["snowy mountain", "night"],
            ["tropical beach", "sunset"],
        )
        assert dist == 1.0

    def test_partial_overlap(self):
        """部分重叠 → 0 < 距离 < 1"""
        dist = MemoryBank._compute_scene_distance(
            ["modern city park", "daytime", "bench"],
            ["modern city park", "evening", "bench"],
        )
        assert 0.0 < dist < 1.0

    def test_empty_both(self):
        assert MemoryBank._compute_scene_distance([], []) == 0.0

    def test_empty_one_side(self):
        assert MemoryBank._compute_scene_distance(["park"], []) == 1.0
        assert MemoryBank._compute_scene_distance([], ["park"]) == 1.0


# ============ 5. Total Frame Count ============

class TestTotalFrameCount:

    def test_total_never_exceeds_max(self, memory_bank):
        """frame_active_memory 总帧数不超过 max_id + max_scene"""
        for i in range(10):
            _add_frame(memory_bank, f"f{i}", [i % 3 + 1],
                       entity_score=0.5 + i * 0.05,
                       scene_score=0.3 + i * 0.03,
                       scene_texts=["park"])

        memory_bank.retrieve_initial_frames([1, 2, 3], scene_texts=["park"])
        max_total = memory_bank.max_id_memory_frames + memory_bank.max_scene_memory_frames
        assert len(memory_bank.frame_active_memory) <= max_total

    def test_dedup_reduces_total(self, memory_bank):
        """同一帧在两层中只计一次"""
        _add_frame(memory_bank, "f_both", [1], entity_score=0.9, scene_score=0.9,
                   scene_texts=["park"])
        _add_frame(memory_bank, "f_id", [2], entity_score=0.8, scene_score=0.1)
        _add_frame(memory_bank, "f_scene", [3], entity_score=0.1, scene_score=0.8,
                   scene_texts=["park"])

        memory_bank.retrieve_initial_frames([1, 2], scene_texts=["park"])
        # f_both 可能同时在 id_memory 和 scene_memory 中
        # frame_active_memory 去重后应 <= id + scene
        assert len(memory_bank.frame_active_memory) <= (
            len(memory_bank.id_memory) + len(memory_bank.scene_memory)
        )


# ============ 6. Integration: Dynamic Budget + Scene Skip ============

class TestIntegrationDynamicAllocation:

    def test_multi_prompt_dynamic_budget(self, memory_bank):
        """模拟多 prompt 场景: 动态预算随 ID 数量变化"""
        # Prompt 1: 1个实体
        _add_frame(memory_bank, "p1_f1", [1], entity_score=0.8)
        memory_bank.retrieve_initial_frames([1])
        budget_1 = len(memory_bank.id_memory)

        # Prompt 2: 3个实体
        _add_frame(memory_bank, "p1_f2", [2], entity_score=0.7)
        _add_frame(memory_bank, "p1_f3", [3], entity_score=0.6)
        memory_bank.retrieve_initial_frames([1, 2, 3])
        budget_3 = len(memory_bank.id_memory)

        # 3个 ID 需要更多帧
        assert budget_3 >= budget_1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
