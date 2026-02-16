"""
Phase 2 测试: 双层记忆 — ID Memory + Scene Memory

验证:
1. FrameInfo 双分数 (entity_score / scene_score)
2. MemoryBank 双层 active memory (id_memory / scene_memory)
3. _build_scene_token_weights() 场景权重构建
4. select_frame_from_chunk() 双路打分
5. update_id_memory() / update_scene_memory() 独立更新
6. get_memory_kv() 去重拼接
7. retrieve_initial_frames() 双路检索
8. 向后兼容: frame_active_memory property
9. save/load JSON 持久化双层记忆
"""

import os
import sys
import json
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iam.llm_agent import EntityStruct
from iam.memory_bank import MemoryBank, FrameInfo


# ============ Fixtures ============

@pytest.fixture
def memory_bank(tmp_path):
    """创建测试用 MemoryBank"""
    return MemoryBank(
        text_encoder=None,
        max_memory_frames=3,
        max_id_memory_frames=4,
        max_scene_memory_frames=2,
        frame_seq_length=1560,
        num_transformer_blocks=2,  # 测试用少量 block
        save_dir=str(tmp_path),
    )


@pytest.fixture
def entities_p1():
    return [
        EntityStruct(entity="young man", attrs=["late 20s", "messy black hair", "denim jacket"], global_id=1)
    ]


@pytest.fixture
def entities_p2():
    return [
        EntityStruct(entity="protagonist", attrs=["denim jacket", "seated on bench"], global_id=1),
        EntityStruct(entity="another man", attrs=["30 years old", "glasses", "grey sweater"], global_id=2),
    ]


PROMPT_P1 = "A realistic video of a modern city park environment. A young man in his late 20s, with messy black hair, wearing a vintage blue denim jacket, sits alone on a park bench."
PROMPT_P2 = "A realistic video of a modern city park environment. The protagonist in the denim jacket remains seated on the bench. Another man, around 30 years old, wearing glasses and a casual grey sweater, walks into the frame."
SCENE_P1 = ["modern city park", "park bench", "daytime"]
SCENE_P2 = ["modern city park", "park bench", "daytime"]


def _make_mock_kv(num_blocks=2, num_frames=3, seq_per_frame=1560):
    """生成 mock chunk KV"""
    B, H, D = 1, 12, 128
    L = num_frames * seq_per_frame
    kv_list = []
    for _ in range(num_blocks):
        kv_list.append({
            "k": torch.randn(B, L, H, D),
            "v": torch.randn(B, L, H, D),
        })
    return kv_list


def _make_mock_crossattn(num_blocks=2, num_tokens=512):
    """生成 mock crossattn cache"""
    B, H, D = 1, 12, 128
    cache = []
    for _ in range(num_blocks):
        cache.append({
            "k": torch.randn(B, num_tokens, H, D),
            "v": torch.randn(B, num_tokens, H, D),
            "is_init": True,
        })
    return cache


# ============ 1. FrameInfo 双分数 ============

class TestFrameInfoDualScores:

    def test_default_scores_zero(self):
        fi = FrameInfo(
            frame_id="p1_c3_f0", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.5,
        )
        assert fi.entity_score == 0.0
        assert fi.scene_score == 0.0

    def test_explicit_scores(self):
        fi = FrameInfo(
            frame_id="p1_c3_f0", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.5,
            entity_score=0.7, scene_score=0.3,
        )
        assert fi.entity_score == 0.7
        assert fi.scene_score == 0.3

    def test_to_dict_includes_dual_scores(self):
        fi = FrameInfo(
            frame_id="p1_c3_f0", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.5,
            entity_score=0.7, scene_score=0.3,
            scene_texts=["snowy forest"],
        )
        d = fi.to_dict()
        assert "entity_score" in d
        assert "scene_score" in d
        assert "scene_texts" in d
        assert d["entity_score"] == 0.7
        assert d["scene_score"] == 0.3
        assert d["scene_texts"] == ["snowy forest"]


# ============ 2. MemoryBank 双层 active memory ============

class TestDualActiveMemory:

    def test_initial_empty(self, memory_bank):
        assert memory_bank.id_memory == []
        assert memory_bank.scene_memory == []
        assert memory_bank.frame_active_memory == []

    def test_frame_active_memory_property_dedup(self, memory_bank):
        """同一帧在两层中只出现一次"""
        memory_bank.id_memory = ["f1", "f2"]
        memory_bank.scene_memory = ["f2", "f3"]
        assert memory_bank.frame_active_memory == ["f1", "f2", "f3"]

    def test_frame_active_memory_preserves_order(self, memory_bank):
        """id_memory 在前，scene_memory 在后"""
        memory_bank.id_memory = ["f3", "f1"]
        memory_bank.scene_memory = ["f2"]
        assert memory_bank.frame_active_memory == ["f3", "f1", "f2"]

    def test_frame_active_memory_setter_compat(self, memory_bank):
        """向后兼容 setter: 写入 id_memory"""
        memory_bank.frame_active_memory = ["f1", "f2"]
        assert memory_bank.id_memory == ["f1", "f2"]
        assert memory_bank.scene_memory == []

    def test_clear_resets_both(self, memory_bank):
        memory_bank.id_memory = ["f1"]
        memory_bank.scene_memory = ["f2"]
        memory_bank.clear()
        assert memory_bank.id_memory == []
        assert memory_bank.scene_memory == []


# ============ 3. Scene Token Weights ============

class TestSceneTokenWeights:

    def test_no_scene_returns_uniform(self, memory_bank):
        w = memory_bank._build_scene_token_weights(None, 512)
        assert w.shape == (512,)
        assert (w == 1.0).all()

    def test_empty_scene_returns_uniform(self, memory_bank):
        w = memory_bank._build_scene_token_weights([], 512)
        assert (w == 1.0).all()

    def test_scene_keywords_boosted(self, memory_bank):
        prompt = "A realistic video of a modern city park. A young man sits on a bench."
        w = memory_bank._build_scene_token_weights(
            ["modern city park"], 512, prompt_text=prompt
        )
        # scene 区域应该有 > 1.0 的权重
        assert w.max().item() > 1.0

    def test_entity_regions_suppressed(self, memory_bank):
        prompt = "A realistic video of a modern city park. A young man sits on a bench."
        entities = [EntityStruct(entity="young man", attrs=["bench"])]
        w = memory_bank._build_scene_token_weights(
            ["modern city park"], 512, prompt_text=prompt, entities=entities
        )
        # entity 区域应该有 < 1.0 的权重
        assert w.min().item() < 1.0

    def test_no_prompt_text_fallback(self, memory_bank):
        """无 prompt 文本时使用开头区域加权"""
        w = memory_bank._build_scene_token_weights(
            ["forest"], 512, prompt_text=None
        )
        # 开头区域应该被加权
        assert w[:50].mean().item() > w[400:].mean().item()


# ============ 4. Dual-Path Scoring ============

class TestDualPathScoring:

    def test_select_returns_dual_scores(self, memory_bank, entities_p1):
        memory_bank.register_entities(entities_p1, prompt_id=1)
        chunk_kv = _make_mock_kv(num_blocks=2)
        crossattn = _make_mock_crossattn(num_blocks=2)

        frame_id, score = memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=chunk_kv,
            crossattn_cache=crossattn,
            prompt_id=1, chunk_id=3,
            current_entity_ids=[1],
            current_entities=entities_p1,
            prompt_text=PROMPT_P1,
            scene_texts=SCENE_P1,
        )

        fi = memory_bank.frame_archive[frame_id]
        assert fi.entity_score != 0.0 or fi.scene_score != 0.0
        # 综合分数 = 0.6*entity + 0.4*scene
        expected = 0.6 * fi.entity_score + 0.4 * fi.scene_score
        assert abs(fi.score - expected) < 1e-4

    def test_select_uses_configurable_fusion_weights(self, tmp_path, entities_p1):
        memory_bank = MemoryBank(
            text_encoder=None,
            max_memory_frames=3,
            max_id_memory_frames=4,
            max_scene_memory_frames=2,
            entity_memory_weight=0.72,
            scene_memory_weight=0.28,
            frame_seq_length=1560,
            num_transformer_blocks=2,
            save_dir=str(tmp_path),
        )
        memory_bank.register_entities(entities_p1, prompt_id=1)

        chunk_kv = _make_mock_kv(num_blocks=2)
        crossattn = _make_mock_crossattn(num_blocks=2)

        frame_id, _ = memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=chunk_kv,
            crossattn_cache=crossattn,
            prompt_id=1, chunk_id=3,
            current_entity_ids=[1],
            current_entities=entities_p1,
            prompt_text=PROMPT_P1,
            scene_texts=SCENE_P1,
        )

        fi = memory_bank.frame_archive[frame_id]
        expected = 0.72 * fi.entity_score + 0.28 * fi.scene_score
        assert abs(fi.score - expected) < 1e-4

    def test_select_without_scene_texts(self, memory_bank, entities_p1):
        """无 scene_texts 时 scene_score 为 0"""
        memory_bank.register_entities(entities_p1, prompt_id=1)
        chunk_kv = _make_mock_kv(num_blocks=2)
        crossattn = _make_mock_crossattn(num_blocks=2)

        frame_id, score = memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=chunk_kv,
            crossattn_cache=crossattn,
            prompt_id=1, chunk_id=3,
            current_entity_ids=[1],
            current_entities=entities_p1,
            prompt_text=PROMPT_P1,
            scene_texts=None,
        )

        fi = memory_bank.frame_archive[frame_id]
        assert fi.scene_score == 0.0

    def test_consensus_score_helper(self, memory_bank):
        """_consensus_score 返回正确形状"""
        chunk_kv = _make_mock_kv(num_blocks=2)
        crossattn = _make_mock_crossattn(num_blocks=2)
        weights = torch.ones(512)

        scores = memory_bank._consensus_score(
            chunk_kv, crossattn, weights,
            available_layers=2, num_candidate_frames=3,
        )
        assert scores.shape == (3,)


# ============ 5. Independent Memory Updates ============

class TestIndependentMemoryUpdates:

    def _add_frame(self, mb, frame_id, entity_score, scene_score):
        """辅助: 直接向 archive 添加帧"""
        mb.frame_archive[frame_id] = FrameInfo(
            frame_id=frame_id, frame_path="", prompt_id=1,
            associated_entities=["1"],
            score=0.6 * entity_score + 0.4 * scene_score,
            entity_score=entity_score,
            scene_score=scene_score,
        )

    def test_update_id_memory_adds(self, memory_bank):
        self._add_frame(memory_bank, "f1", 0.8, 0.2)
        memory_bank.update_id_memory("f1", 0.8)
        assert "f1" in memory_bank.id_memory

    def test_update_scene_memory_adds(self, memory_bank):
        self._add_frame(memory_bank, "f1", 0.2, 0.9)
        memory_bank.update_scene_memory("f1", 0.9)
        assert "f1" in memory_bank.scene_memory

    def test_id_memory_respects_max(self, memory_bank):
        """id_memory 不超过 max_id_memory_frames"""
        for i in range(6):
            fid = f"f{i}"
            score = 0.5 + i * 0.1
            self._add_frame(memory_bank, fid, score, 0.1)
            memory_bank.update_id_memory(fid, score)
        assert len(memory_bank.id_memory) <= memory_bank.max_id_memory_frames

    def test_scene_memory_respects_max(self, memory_bank):
        """scene_memory 不超过 max_scene_memory_frames"""
        for i in range(5):
            fid = f"f{i}"
            score = 0.5 + i * 0.1
            self._add_frame(memory_bank, fid, 0.1, score)
            memory_bank.update_scene_memory(fid, score)
        assert len(memory_bank.scene_memory) <= memory_bank.max_scene_memory_frames

    def test_same_frame_in_both_layers(self, memory_bank):
        """同一帧可以同时在 id_memory 和 scene_memory 中"""
        self._add_frame(memory_bank, "f1", 0.9, 0.8)
        memory_bank.update_id_memory("f1", 0.9)
        memory_bank.update_scene_memory("f1", 0.8)
        assert "f1" in memory_bank.id_memory
        assert "f1" in memory_bank.scene_memory
        # frame_active_memory 去重
        assert memory_bank.frame_active_memory.count("f1") == 1

    def test_id_memory_replaces_lowest(self, memory_bank):
        """新帧 entity_score 更高时替换最低分帧"""
        self._add_frame(memory_bank, "f1", 0.3, 0.1)
        self._add_frame(memory_bank, "f2", 0.5, 0.1)
        self._add_frame(memory_bank, "f3", 0.7, 0.1)
        self._add_frame(memory_bank, "f4", 0.9, 0.1)
        for fid in ["f1", "f2", "f3", "f4"]:
            memory_bank.update_id_memory(fid, memory_bank.frame_archive[fid].entity_score)

        # 现在 id_memory 满了 (4帧)，加入更高分帧
        self._add_frame(memory_bank, "f5", 1.0, 0.1)
        memory_bank.update_id_memory("f5", 1.0)

        assert "f5" in memory_bank.id_memory
        assert "f1" not in memory_bank.id_memory  # 最低分被替换


# ============ 6. get_memory_kv 去重拼接 ============

class TestGetMemoryKV:

    def test_dedup_kv(self, memory_bank):
        """同一帧在两层中只拼接一次"""
        # 手动添加帧 KV
        B, H, D = 1, 12, 128
        seq = memory_bank.frame_seq_length
        for fid in ["f1", "f2", "f3"]:
            kv = []
            for _ in range(memory_bank.num_transformer_blocks):
                kv.append({
                    "k": torch.randn(B, seq, H, D),
                    "v": torch.randn(B, seq, H, D),
                })
            memory_bank._frame_kv_store[fid] = kv
            memory_bank.frame_archive[fid] = FrameInfo(
                frame_id=fid, frame_path="", prompt_id=1,
                associated_entities=["1"], score=0.5,
            )

        # f2 在两层中都有
        memory_bank.id_memory = ["f1", "f2"]
        memory_bank.scene_memory = ["f2", "f3"]

        result = memory_bank.get_memory_kv()
        assert result is not None
        # 去重后 3 帧
        expected_length = 3 * seq
        assert result[0]["k"].shape[1] == expected_length

    def test_empty_memory_returns_none(self, memory_bank):
        assert memory_bank.get_memory_kv() is None


# ============ 7. Dual-Path Retrieval ============

class TestDualPathRetrieval:

    def _populate_archive(self, mb, n=5):
        """填充 archive 用于检索测试"""
        for i in range(n):
            fid = f"p1_c{i+3}_f0"
            mb.frame_archive[fid] = FrameInfo(
                frame_id=fid, frame_path="", prompt_id=1,
                associated_entities=["1"] if i % 2 == 0 else ["2"],
                score=0.5 + i * 0.1,
                entity_score=0.5 + i * 0.1,
                scene_score=0.3 + i * 0.05,
            )
            # 添加 mock KV
            kv = []
            for _ in range(mb.num_transformer_blocks):
                kv.append({
                    "k": torch.randn(1, mb.frame_seq_length, 12, 128),
                    "v": torch.randn(1, mb.frame_seq_length, 12, 128),
                })
            mb._frame_kv_store[fid] = kv

    def test_retrieve_fills_both_layers(self, memory_bank):
        self._populate_archive(memory_bank)
        result = memory_bank.retrieve_initial_frames([1], scene_texts=["park"])
        assert len(memory_bank.id_memory) > 0
        assert len(memory_bank.scene_memory) > 0
        assert len(result) > 0

    def test_retrieve_without_scene(self, memory_bank):
        self._populate_archive(memory_bank)
        result = memory_bank.retrieve_initial_frames([1])
        assert len(memory_bank.id_memory) > 0
        assert memory_bank.scene_memory == []

    def test_retrieve_id_memory_respects_max(self, memory_bank):
        self._populate_archive(memory_bank, n=10)
        memory_bank.retrieve_initial_frames([1], scene_texts=["park"])
        assert len(memory_bank.id_memory) <= memory_bank.max_id_memory_frames

    def test_retrieve_scene_memory_respects_max(self, memory_bank):
        self._populate_archive(memory_bank, n=10)
        memory_bank.retrieve_initial_frames([1], scene_texts=["park"])
        assert len(memory_bank.scene_memory) <= memory_bank.max_scene_memory_frames

    def test_scene_retrieval_prioritizes_query_match(self, memory_bank):
        """Scene Memory 应按当前 scene query 检索，而非仅按历史 scene_score。"""
        memory_bank.frame_archive["f_forest"] = FrameInfo(
            frame_id="f_forest", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.2,
            entity_score=0.2, scene_score=0.1,
            scene_texts=["snowy forest", "pine trees"],
        )
        memory_bank.frame_archive["f_city"] = FrameInfo(
            frame_id="f_city", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.9,
            entity_score=0.9, scene_score=0.95,
            scene_texts=["downtown city", "traffic"],
        )

        result = memory_bank.retrieve_initial_frames([1], scene_texts=["snowy forest"])
        assert "f_forest" in memory_bank.scene_memory
        assert memory_bank.scene_memory[0] == "f_forest"
        assert len(result) > 0

    def test_retrieve_scene_only_keeps_id_memory_empty(self, memory_bank):
        """无实体时应只走 Scene Memory 路径。"""
        self._populate_archive(memory_bank)
        memory_bank.retrieve_initial_frames([], scene_texts=["park"])
        assert memory_bank.id_memory == []
        assert len(memory_bank.scene_memory) > 0


# ============ 8. JSON 持久化 ============

class TestJsonPersistence:

    def test_save_load_dual_memory(self, memory_bank, tmp_path):
        memory_bank.id_memory = ["f1", "f2"]
        memory_bank.scene_memory = ["f3"]
        memory_bank.frame_archive["f1"] = FrameInfo(
            frame_id="f1", frame_path="", prompt_id=1,
            associated_entities=["1"], score=0.5,
            entity_score=0.7, scene_score=0.3,
            scene_texts=["park", "bench"],
        )

        path = str(tmp_path / "test.json")
        memory_bank.save_to_json(path)

        # 加载到新 bank
        new_bank = MemoryBank(text_encoder=None, save_dir=str(tmp_path))
        new_bank.load_from_json(path)

        assert new_bank.id_memory == ["f1", "f2"]
        assert new_bank.scene_memory == ["f3"]
        assert new_bank.frame_archive["f1"].entity_score == 0.7
        assert new_bank.frame_archive["f1"].scene_score == 0.3
        assert new_bank.frame_archive["f1"].scene_texts == ["park", "bench"]

    def test_load_legacy_json(self, memory_bank, tmp_path):
        """旧格式 JSON (只有 frame_active_memory) 向后兼容"""
        path = str(tmp_path / "legacy.json")
        data = {
            "global_registry": {},
            "frame_archive": {},
            "frame_active_memory": ["f1", "f2"],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        memory_bank.load_from_json(path)
        assert memory_bank.id_memory == ["f1", "f2"]
        assert memory_bank.scene_memory == []

    def test_save_includes_frame_active_memory(self, memory_bank, tmp_path):
        """保存的 JSON 包含 frame_active_memory 用于向后兼容"""
        memory_bank.id_memory = ["f1"]
        memory_bank.scene_memory = ["f2"]

        path = str(tmp_path / "test.json")
        memory_bank.save_to_json(path)

        with open(path) as f:
            data = json.load(f)

        assert "frame_active_memory" in data
        assert "id_memory" in data
        assert "scene_memory" in data


# ============ 9. 集成测试: 多 chunk 双路更新 ============

class TestIntegrationMultiChunk:

    def test_multi_chunk_dual_update(self, memory_bank, entities_p1):
        """模拟多个 chunk 的双路打分和更新"""
        memory_bank.register_entities(entities_p1, prompt_id=1)
        crossattn = _make_mock_crossattn(num_blocks=2)

        for chunk_id in range(3, 8):
            chunk_kv = _make_mock_kv(num_blocks=2)
            frame_id, score = memory_bank.select_frame_from_chunk(
                evicted_chunk_kv=chunk_kv,
                crossattn_cache=crossattn,
                prompt_id=1, chunk_id=chunk_id,
                current_entity_ids=[1],
                current_entities=entities_p1,
                prompt_text=PROMPT_P1,
                scene_texts=SCENE_P1,
            )

            fi = memory_bank.frame_archive[frame_id]
            memory_bank.update_id_memory(frame_id, fi.entity_score)
            memory_bank.update_scene_memory(frame_id, fi.scene_score)

        # 验证约束
        assert len(memory_bank.id_memory) <= memory_bank.max_id_memory_frames
        assert len(memory_bank.scene_memory) <= memory_bank.max_scene_memory_frames
        assert len(memory_bank.frame_archive) == 5  # 5 chunks

        # get_memory_kv 应该能正常工作
        kv = memory_bank.get_memory_kv()
        assert kv is not None
        assert len(kv) == memory_bank.num_transformer_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
