"""
IAM_2 端到端测试

测试内容:
1. EntityStructExtractor - 实体提取
2. GlobalIDManager - ID 匹配和分配
3. LLMAgent - 完整流程
4. MemoryBank - 帧管理
5. AgentCausalInferencePipeline - 完整 Pipeline (需要 MemFlow)

测试场景来自 pipeline.md:
- Prompt 1: 引入主角 (young man)
- Prompt 2: 引入第二个角色 (protagonist + another man)
- Prompt 3: 引入第三个角色 (protagonist + man in grey sweater + young woman)
"""

import os
import sys
import json
import pytest
import torch
from typing import List, Dict
from unittest.mock import Mock, MagicMock, patch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IAM_2.llm_agent import (
    EntityStruct,
    EntityStructExtractor,
    GlobalIDManager,
    LLMAgent,
    LLMWrapper
)
from IAM_2.memory_bank import MemoryBank, FrameInfo


# ============ 测试数据 ============

TEST_PROMPTS = [
    "A realistic video of a modern city park environment. A young man in his late 20s, with messy black hair, wearing a vintage blue denim jacket, sits alone on a park bench. He holds a sketchbook in his hands, looking pensive and expectant.",
    "A realistic video of a modern city park environment. The main protagonist in the denim jacket remains seated on the bench. Another man, around 30 years old, wearing glasses and a casual grey sweater, walks into the frame and sits next to him holding a coffee cup.",
    "A realistic video of a modern city park environment. The protagonist and the man in the grey sweater are talking on the bench. A young woman in her late 20s, with long hair and wearing a flowing white dress, approaches them carrying a shoulder bag."
]

EXPECTED_ENTITIES_P1 = [
    {"entity": "young man", "attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook"]}
]

EXPECTED_ENTITIES_P2 = [
    {"entity": "protagonist", "attrs": ["denim jacket", "seated on bench"]},
    {"entity": "another man", "attrs": ["30 years old", "glasses", "grey sweater", "coffee cup"]}
]

EXPECTED_ENTITIES_P3 = [
    {"entity": "protagonist", "attrs": ["denim jacket"]},
    {"entity": "man in grey sweater", "attrs": ["talking on bench"]},
    {"entity": "young woman", "attrs": ["late 20s", "long hair", "white dress", "shoulder bag"]}
]


# ============ Mock LLM 类 ============

class MockLLMWrapper:
    """模拟 LLM 响应用于测试"""

    def __init__(self, model_path: str = "mock"):
        self.model_path = model_path
        self._call_count = 0

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
        """模拟 LLM 生成"""
        self._call_count += 1

        # 判断是实体提取还是匹配
        if "extract" in system_prompt.lower():
            return self._mock_extraction(user_prompt)
        elif "match" in system_prompt.lower():
            return self._mock_matching(user_prompt)
        return "[]"

    def _mock_extraction(self, prompt: str) -> str:
        """模拟实体提取"""
        prompt_lower = prompt.lower()

        if "young man" in prompt_lower and "another" not in prompt_lower:
            return json.dumps([
                {"entity": "young man", "attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook"]}
            ])
        elif "protagonist" in prompt_lower and "another man" in prompt_lower:
            return json.dumps([
                {"entity": "protagonist", "attrs": ["denim jacket", "seated on bench"]},
                {"entity": "another man", "attrs": ["30 years old", "glasses", "grey sweater", "coffee cup"]}
            ])
        elif "protagonist" in prompt_lower and "young woman" in prompt_lower:
            return json.dumps([
                {"entity": "protagonist", "attrs": ["denim jacket"]},
                {"entity": "man in grey sweater", "attrs": ["talking on bench"]},
                {"entity": "young woman", "attrs": ["late 20s", "long hair", "white dress", "shoulder bag"]}
            ])
        return "[]"

    def _mock_matching(self, prompt: str) -> str:
        """模拟实体匹配"""
        prompt_lower = prompt.lower()

        # 从 prompt 中提取新字符描述
        # 新格式: "New character description:\n\"entity: attr1, attr2\"\n..."
        import re

        # 匹配新格式
        desc_match = re.search(r'new character description:\s*"([^"]+)"', prompt_lower)
        if desc_match:
            description = desc_match.group(1)

            # 根据描述匹配
            if "protagonist" in description:
                # protagonist 匹配 young man (ID 1)
                return '{"matched_id": 1}'
            elif "grey sweater" in description:
                # man in grey sweater 匹配 another man (ID 2)
                return '{"matched_id": 2}'
            elif "woman" in description:
                # young woman 是新实体
                return '{"matched_id": null}'

        # 旧格式兼容
        entity_match = re.search(r'entity:\s*([^\n]+)', prompt_lower)
        if entity_match:
            entity_name = entity_match.group(1).strip()

            if "protagonist" in entity_name:
                return '{"matched_id": 1}'
            elif "grey sweater" in entity_name:
                return '{"matched_id": 2}'
            elif "woman" in entity_name:
                return '{"matched_id": null}'

        return '{"matched_id": null}'


# ============ 测试类 ============

class TestEntityStruct:
    """测试 EntityStruct 数据类"""

    def test_create_entity(self):
        entity = EntityStruct(
            entity="young man",
            attrs=["late 20s", "denim jacket"],
            global_id=1
        )
        assert entity.entity == "young man"
        assert entity.attrs == ["late 20s", "denim jacket"]
        assert entity.global_id == 1

    def test_to_dict(self):
        entity = EntityStruct(entity="test", attrs=["attr1"], global_id=1)
        d = entity.to_dict()
        assert d["entity"] == "test"
        assert d["attrs"] == ["attr1"]
        assert d["global_id"] == 1

    def test_from_dict(self):
        d = {"entity": "test", "attrs": ["attr1"], "global_id": 2}
        entity = EntityStruct.from_dict(d)
        assert entity.entity == "test"
        assert entity.global_id == 2


class TestEntityStructExtractor:
    """测试实体提取"""

    def test_extract_single_entity(self):
        """测试提取单个实体"""
        mock_llm = MockLLMWrapper()
        extractor = EntityStructExtractor(llm=mock_llm)

        entities = extractor.extract(TEST_PROMPTS[0])

        assert len(entities) >= 1
        assert entities[0].entity == "young man"
        assert "late 20s" in entities[0].attrs
        assert entities[0].global_id is None  # 提取时不分配 ID

    def test_extract_multiple_entities(self):
        """测试提取多个实体"""
        mock_llm = MockLLMWrapper()
        extractor = EntityStructExtractor(llm=mock_llm)

        entities = extractor.extract(TEST_PROMPTS[1])

        assert len(entities) >= 2
        entity_names = [e.entity for e in entities]
        assert "protagonist" in entity_names
        assert "another man" in entity_names

    def test_parse_response_with_markdown(self):
        """测试解析带 markdown 格式的响应"""
        extractor = EntityStructExtractor(llm=MockLLMWrapper())

        response = '''```json
[{"entity": "test", "attrs": ["attr1"]}]
```'''
        result = extractor._parse_response(response)

        assert len(result) == 1
        assert result[0]["entity"] == "test"


class TestGlobalIDManager:
    """测试 ID 匹配和分配"""

    def test_first_prompt_allocation(self):
        """测试第一个 prompt 直接分配 ID"""
        mock_llm = MockLLMWrapper()
        manager = GlobalIDManager(llm=mock_llm)

        entities = [
            EntityStruct(entity="young man", attrs=["late 20s"])
        ]

        result = manager.assign_ids(entities, {}, is_first_prompt=True)

        assert len(result) == 1
        assert result[0].global_id == 1

    def test_subsequent_prompt_matching(self):
        """测试后续 prompt 的实体匹配"""
        mock_llm = MockLLMWrapper()
        manager = GlobalIDManager(llm=mock_llm)

        # 模拟已有 registry
        global_registry = {
            "1": {
                "name": "man_1",
                "all_entities": ["young man"],
                "all_attrs": ["late 20s", "denim jacket"]
            }
        }

        entities = [
            EntityStruct(entity="protagonist", attrs=["denim jacket"])
        ]

        result = manager.assign_ids(entities, global_registry, is_first_prompt=False)

        assert len(result) == 1
        assert result[0].global_id == 1  # 应该匹配到已有的 ID

    def test_new_entity_detection(self):
        """测试新实体检测 (包含 "another" 等标记)"""
        mock_llm = MockLLMWrapper()
        manager = GlobalIDManager(llm=mock_llm)

        global_registry = {
            "1": {"name": "man_1", "all_entities": ["young man"], "all_attrs": []}
        }

        entities = [
            EntityStruct(entity="another man", attrs=["grey sweater"])
        ]

        result = manager.assign_ids(entities, global_registry, is_first_prompt=False)

        assert len(result) == 1
        assert result[0].global_id == 2  # 应该是新 ID


class TestLLMAgent:
    """测试 LLM Agent 完整流程"""

    def setup_method(self):
        """设置测试环境"""
        self.mock_llm = MockLLMWrapper()

    def test_process_first_prompt(self):
        """测试处理第一个 prompt"""
        agent = LLMAgent.__new__(LLMAgent)
        agent.llm = self.mock_llm
        agent.extractor = EntityStructExtractor(llm=self.mock_llm)
        agent.id_manager = GlobalIDManager(llm=self.mock_llm)

        entities, registry_update = agent.process_prompt(
            prompt=TEST_PROMPTS[0],
            prompt_id=1,
            global_registry={}
        )

        assert len(entities) >= 1
        assert entities[0].global_id == 1
        assert "1" in registry_update
        assert registry_update["1"]["action"] == "create"

    def test_process_subsequent_prompt(self):
        """测试处理后续 prompt"""
        agent = LLMAgent.__new__(LLMAgent)
        agent.llm = self.mock_llm
        agent.extractor = EntityStructExtractor(llm=self.mock_llm)
        agent.id_manager = GlobalIDManager(llm=self.mock_llm)

        # 模拟已有 registry
        global_registry = {
            "1": {
                "name": "man_1",
                "all_entities": ["young man"],
                "all_attrs": ["late 20s", "denim jacket"]
            }
        }

        entities, registry_update = agent.process_prompt(
            prompt=TEST_PROMPTS[1],
            prompt_id=2,
            global_registry=global_registry
        )

        assert len(entities) >= 2

        # 验证 protagonist 匹配到 ID 1
        protagonist = next((e for e in entities if e.entity == "protagonist"), None)
        assert protagonist is not None
        assert protagonist.global_id == 1

        # 验证 another man 获得新 ID
        another_man = next((e for e in entities if "another" in e.entity.lower()), None)
        assert another_man is not None
        assert another_man.global_id == 2


class TestMemoryBank:
    """测试 Memory Bank"""

    def setup_method(self):
        """设置测试环境"""
        self.memory_bank = MemoryBank(
            text_encoder=None,  # 使用模拟编码
            max_memory_frames=3,
            frame_seq_length=1560,
            save_dir="test_frames"
        )

    def teardown_method(self):
        """清理测试环境"""
        import shutil
        if os.path.exists("test_frames"):
            shutil.rmtree("test_frames")
        if os.path.exists("test_mapping.json"):
            os.remove("test_mapping.json")

    def test_register_entities(self):
        """测试实体注册"""
        entities = [
            EntityStruct(entity="young man", attrs=["late 20s", "denim jacket"], global_id=1)
        ]

        self.memory_bank.register_entities(entities, prompt_id=1)

        assert "1" in self.memory_bank.global_registry
        assert "young man" in self.memory_bank.global_registry["1"]["all_entities"]

    def test_update_existing_entity(self):
        """测试更新现有实体"""
        # 第一次注册
        entities1 = [
            EntityStruct(entity="young man", attrs=["late 20s"], global_id=1)
        ]
        self.memory_bank.register_entities(entities1, prompt_id=1)

        # 第二次更新
        entities2 = [
            EntityStruct(entity="protagonist", attrs=["denim jacket"], global_id=1)
        ]
        self.memory_bank.register_entities(entities2, prompt_id=2)

        registry = self.memory_bank.global_registry["1"]
        assert "young man" in registry["all_entities"]
        assert "protagonist" in registry["all_entities"]
        assert len(registry["instances"]) == 2

    def test_frame_selection(self):
        """测试帧选择"""
        # 模拟 chunk KV
        batch_size = 1
        num_frames = 3
        seq_len = num_frames * 1560
        num_heads = 12
        head_dim = 128

        mock_chunk_kv = {
            "k": torch.randn(batch_size, seq_len, num_heads, head_dim),
            "v": torch.randn(batch_size, seq_len, num_heads, head_dim)
        }

        frame_id, score = self.memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=mock_chunk_kv,
            entity_attrs_text="young man late 20s denim jacket",
            prompt_id=1,
            chunk_id=3,
            current_entity_ids=[1]
        )

        assert frame_id == "p1_c3_f0" or frame_id.startswith("p1_c3_f")
        assert score > 0
        assert frame_id in self.memory_bank.frame_archive

    def test_active_memory_update(self):
        """测试 active memory 更新"""
        # 添加几帧
        for chunk_id in range(3, 7):
            mock_kv = {
                "k": torch.randn(1, 4680, 12, 128),
                "v": torch.randn(1, 4680, 12, 128)
            }
            fid, score = self.memory_bank.select_frame_from_chunk(
                mock_kv, "test", prompt_id=1, chunk_id=chunk_id, current_entity_ids=[1]
            )
            self.memory_bank.update_active_memory(fid, score)

        # 应该保持最多 3 帧
        assert len(self.memory_bank.frame_active_memory) <= 3

    def test_retrieve_frames(self):
        """测试帧检索"""
        # 先添加一些帧
        entities = [
            EntityStruct(entity="young man", attrs=["late 20s"], global_id=1)
        ]
        self.memory_bank.register_entities(entities, prompt_id=1)

        for chunk_id in range(3, 6):
            mock_kv = {
                "k": torch.randn(1, 4680, 12, 128),
                "v": torch.randn(1, 4680, 12, 128)
            }
            fid, score = self.memory_bank.select_frame_from_chunk(
                mock_kv, "young man", prompt_id=1, chunk_id=chunk_id, current_entity_ids=[1]
            )
            self.memory_bank.update_active_memory(fid, score)

        # 检索
        retrieved = self.memory_bank.retrieve_initial_frames([1])
        assert len(retrieved) > 0

    def test_save_load_json(self):
        """测试保存和加载"""
        entities = [
            EntityStruct(entity="test", attrs=["attr1"], global_id=1)
        ]
        self.memory_bank.register_entities(entities, prompt_id=1)

        # 保存
        self.memory_bank.save_to_json("test_mapping.json")
        assert os.path.exists("test_mapping.json")

        # 加载
        new_bank = MemoryBank(text_encoder=None, save_dir="test_frames")
        new_bank.load_from_json("test_mapping.json")

        assert "1" in new_bank.global_registry
        assert "test" in new_bank.global_registry["1"]["all_entities"]


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.mock_llm = MockLLMWrapper()

    def teardown_method(self):
        """清理测试环境"""
        import shutil
        if os.path.exists("test_frames"):
            shutil.rmtree("test_frames")

    def test_full_pipeline_flow(self):
        """测试完整流程 (3 个 prompt)"""
        # 初始化组件
        agent = LLMAgent.__new__(LLMAgent)
        agent.llm = self.mock_llm
        agent.extractor = EntityStructExtractor(llm=self.mock_llm)
        agent.id_manager = GlobalIDManager(llm=self.mock_llm)

        memory_bank = MemoryBank(
            text_encoder=None,
            max_memory_frames=3,
            save_dir="test_frames"
        )

        # 处理 Prompt 1
        print("\n=== Processing Prompt 1 ===")
        entities, registry_update = agent.process_prompt(
            prompt=TEST_PROMPTS[0],
            prompt_id=1,
            global_registry=memory_bank.global_registry
        )
        memory_bank.register_entities(entities, prompt_id=1, registry_update=registry_update)

        print(f"Entities: {[e.to_dict() for e in entities]}")
        assert len(entities) >= 1
        assert entities[0].global_id == 1

        # 处理 Prompt 2
        print("\n=== Processing Prompt 2 ===")
        entities, registry_update = agent.process_prompt(
            prompt=TEST_PROMPTS[1],
            prompt_id=2,
            global_registry=memory_bank.global_registry
        )
        memory_bank.register_entities(entities, prompt_id=2, registry_update=registry_update)

        print(f"Entities: {[e.to_dict() for e in entities]}")
        assert len(entities) >= 2

        # 验证 protagonist 匹配到 ID 1
        protagonist = next((e for e in entities if e.entity == "protagonist"), None)
        assert protagonist is not None
        assert protagonist.global_id == 1

        # 验证 another man 获得新 ID 2
        another_man = next((e for e in entities if "another" in e.entity.lower()), None)
        assert another_man is not None
        assert another_man.global_id == 2

        # 处理 Prompt 3
        print("\n=== Processing Prompt 3 ===")
        entities, registry_update = agent.process_prompt(
            prompt=TEST_PROMPTS[2],
            prompt_id=3,
            global_registry=memory_bank.global_registry
        )
        memory_bank.register_entities(entities, prompt_id=3, registry_update=registry_update)

        print(f"Entities: {[e.to_dict() for e in entities]}")
        assert len(entities) >= 3

        # 验证 man in grey sweater 匹配到 ID 2
        grey_sweater = next((e for e in entities if "grey sweater" in e.entity.lower()), None)
        assert grey_sweater is not None
        assert grey_sweater.global_id == 2

        # 验证 young woman 获得新 ID 3
        young_woman = next((e for e in entities if "woman" in e.entity.lower()), None)
        assert young_woman is not None
        assert young_woman.global_id == 3

        # 验证最终 registry
        print("\n=== Final Registry ===")
        print(json.dumps(memory_bank.global_registry, indent=2))

        assert "1" in memory_bank.global_registry  # man_1
        assert "2" in memory_bank.global_registry  # man_2
        assert "3" in memory_bank.global_registry  # woman_1

    def test_mapping_json_format(self):
        """测试 mapping.json 格式符合预期"""
        agent = LLMAgent.__new__(LLMAgent)
        agent.llm = self.mock_llm
        agent.extractor = EntityStructExtractor(llm=self.mock_llm)
        agent.id_manager = GlobalIDManager(llm=self.mock_llm)

        memory_bank = MemoryBank(
            text_encoder=None,
            max_memory_frames=3,
            save_dir="test_frames"
        )

        # 处理所有 prompts
        for i, prompt in enumerate(TEST_PROMPTS):
            entities, registry_update = agent.process_prompt(
                prompt=prompt,
                prompt_id=i + 1,
                global_registry=memory_bank.global_registry
            )
            memory_bank.register_entities(entities, prompt_id=i + 1, registry_update=registry_update)

        # 保存并验证格式
        memory_bank.save_to_json("test_mapping.json")

        with open("test_mapping.json", 'r') as f:
            mapping = json.load(f)

        # 验证结构
        assert "global_registry" in mapping
        assert "frame_archive" in mapping
        assert "frame_active_memory" in mapping

        # 验证 global_registry 结构
        for gid, info in mapping["global_registry"].items():
            assert "name" in info
            assert "all_entities" in info
            assert "all_attrs" in info
            assert "instances" in info

        # 清理
        os.remove("test_mapping.json")


# ============ 运行测试 ============

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
