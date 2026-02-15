"""
Phase 1 测试: Scene Text 提取

验证:
1. EntityStructExtractor 能同时提取 entities + scene
2. 新 JSON 格式解析正确
3. 旧格式 (纯数组) 向后兼容
4. 各种 edge case 的容错
5. LLMAgent.process_prompt() 返回 3-tuple
6. SceneStruct dataclass 基本功能
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iam.llm_agent import (
    EntityStruct,
    SceneStruct,
    EntityStructExtractor,
    GlobalIDManager,
    LLMAgent,
)


# ============ Mock LLM ============

class MockLLM:
    """返回预设 JSON 的 mock LLM"""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
        self.call_count += 1
        if callable(self.response):
            return self.response(system_prompt, user_prompt)
        return self.response


class MockLLMForAgent:
    """模拟完整 Agent 流程的 mock LLM (提取 + 匹配)"""

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
        if "extract" in system_prompt.lower() or "scene" in system_prompt.lower():
            return json.dumps({
                "entities": [
                    {"entity": "young man", "attrs": ["messy black hair", "denim jacket"]}
                ],
                "scene": ["snowy forest", "overcast daylight", "wooden bench"]
            })
        elif "match" in system_prompt.lower():
            return '{"matched_id": null}'
        return "{}"


# ============ SceneStruct Tests ============

class TestSceneStruct:

    def test_create(self):
        s = SceneStruct(scene_texts=["snowy forest", "overcast daylight"])
        assert s.scene_texts == ["snowy forest", "overcast daylight"]

    def test_default_empty(self):
        s = SceneStruct()
        assert s.scene_texts == []

    def test_to_dict(self):
        s = SceneStruct(scene_texts=["park", "bench"])
        assert s.to_dict() == {"scene_texts": ["park", "bench"]}

    def test_from_dict(self):
        s = SceneStruct.from_dict({"scene_texts": ["forest"]})
        assert s.scene_texts == ["forest"]

    def test_from_dict_missing_key(self):
        s = SceneStruct.from_dict({})
        assert s.scene_texts == []


# ============ EntityStructExtractor Tests ============

class TestExtractorNewFormat:
    """测试新的 entities + scene 提取格式"""

    def test_full_extraction(self):
        """正常的 entities + scene 提取"""
        response = json.dumps({
            "entities": [
                {"entity": "young man", "attrs": ["messy black hair", "denim jacket"]}
            ],
            "scene": ["snowy forest", "overcast daylight", "wooden bench"]
        })
        extractor = EntityStructExtractor(llm=MockLLM(response))

        entities, scene_texts = extractor.extract("test prompt")

        assert len(entities) == 1
        assert entities[0].entity == "young man"
        assert entities[0].global_id is None
        assert scene_texts == ["snowy forest", "overcast daylight", "wooden bench"]

    def test_no_entities_with_scene(self):
        """无实体但有场景"""
        response = json.dumps({
            "entities": [],
            "scene": ["empty desert", "sunset"]
        })
        extractor = EntityStructExtractor(llm=MockLLM(response))

        entities, scene_texts = extractor.extract("test")

        assert len(entities) == 0
        assert scene_texts == ["empty desert", "sunset"]

    def test_entities_no_scene(self):
        """有实体但无场景"""
        response = json.dumps({
            "entities": [{"entity": "woman", "attrs": ["red dress"]}],
            "scene": []
        })
        extractor = EntityStructExtractor(llm=MockLLM(response))

        entities, scene_texts = extractor.extract("test")

        assert len(entities) == 1
        assert scene_texts == []

    def test_multiple_entities_and_scene(self):
        """多实体 + 多场景"""
        response = json.dumps({
            "entities": [
                {"entity": "man", "attrs": ["tall", "glasses"]},
                {"entity": "woman", "attrs": ["short hair"]}
            ],
            "scene": ["office", "fluorescent lighting", "whiteboard"]
        })
        extractor = EntityStructExtractor(llm=MockLLM(response))

        entities, scene_texts = extractor.extract("test")

        assert len(entities) == 2
        assert len(scene_texts) == 3


class TestExtractorParsing:
    """测试各种解析 edge case"""

    def _parse(self, response: str):
        extractor = EntityStructExtractor(llm=MockLLM())
        return extractor._parse_response(response)

    def test_markdown_wrapped(self):
        """markdown 代码块包裹"""
        response = '```json\n{"entities": [{"entity": "test", "attrs": []}], "scene": ["park"]}\n```'
        entities, scene = self._parse(response)
        assert len(entities) == 1
        assert scene == ["park"]

    def test_legacy_array_format(self):
        """旧格式: 纯 JSON 数组 → scene 为空"""
        response = '[{"entity": "man", "attrs": ["tall"]}]'
        entities, scene = self._parse(response)
        assert len(entities) == 1
        assert entities[0]["entity"] == "man"
        assert scene == []

    def test_scene_as_string(self):
        """scene 字段为单个字符串而非数组"""
        response = json.dumps({
            "entities": [],
            "scene": "snowy forest"
        })
        entities, scene = self._parse(response)
        assert scene == ["snowy forest"]

    def test_missing_scene_key(self):
        """只有 entities 没有 scene"""
        response = json.dumps({"entities": [{"entity": "man", "attrs": []}]})
        entities, scene = self._parse(response)
        assert len(entities) == 1
        assert scene == []

    def test_missing_entities_key(self):
        """只有 scene 没有 entities"""
        response = json.dumps({"scene": ["forest", "rain"]})
        entities, scene = self._parse(response)
        assert entities == []
        assert scene == ["forest", "rain"]

    def test_extra_text_before_json(self):
        """JSON 前有多余文本"""
        response = 'Here is the result:\n{"entities": [{"entity": "boy", "attrs": []}], "scene": ["beach"]}'
        entities, scene = self._parse(response)
        assert len(entities) == 1
        assert scene == ["beach"]

    def test_unrelated_json_object_returns_safe_empty(self):
        """无关 JSON object 不应导致 extract 崩溃。"""
        response = json.dumps({"foo": "bar", "meta": 1})
        entities, scene = self._parse(response)
        assert entities == []
        assert scene == []

    def test_completely_broken_response(self):
        """完全无法解析的响应"""
        response = "I cannot extract anything from this prompt."
        entities, scene = self._parse(response)
        assert isinstance(entities, list)
        assert isinstance(scene, list)

    def test_scene_fallback_regex(self):
        """_extract_scene_fallback 正则提取"""
        extractor = EntityStructExtractor(llm=MockLLM())
        result = extractor._extract_scene_fallback('"scene": ["forest", "rain"]')
        assert "forest" in result
        assert "rain" in result

    def test_scene_fallback_no_match(self):
        """_extract_scene_fallback 无匹配"""
        extractor = EntityStructExtractor(llm=MockLLM())
        result = extractor._extract_scene_fallback("no scene here")
        assert result == []


# ============ LLMAgent Integration ============

class TestLLMAgentSceneReturn:
    """测试 LLMAgent.process_prompt() 返回 3-tuple"""

    def _make_agent(self, mock_llm):
        agent = LLMAgent.__new__(LLMAgent)
        agent.llm = mock_llm
        agent.extractor = EntityStructExtractor(llm=mock_llm)
        agent.id_manager = GlobalIDManager(llm=mock_llm)
        return agent

    def test_returns_three_tuple(self):
        """process_prompt 返回 (entities, registry_update, scene_texts)"""
        agent = self._make_agent(MockLLMForAgent())

        result = agent.process_prompt(
            prompt="A young man walks in a snowy forest",
            prompt_id=1,
            global_registry={}
        )

        assert len(result) == 3
        entities, registry_update, scene_texts = result
        assert isinstance(entities, list)
        assert isinstance(registry_update, dict)
        assert isinstance(scene_texts, list)

    def test_scene_texts_populated(self):
        """scene_texts 包含场景信息"""
        agent = self._make_agent(MockLLMForAgent())

        entities, registry_update, scene_texts = agent.process_prompt(
            prompt="A young man walks in a snowy forest",
            prompt_id=1,
            global_registry={}
        )

        assert len(scene_texts) > 0
        assert "snowy forest" in scene_texts

    def test_no_entities_still_returns_scene(self):
        """无实体时仍返回 scene_texts"""
        response = json.dumps({"entities": [], "scene": ["empty room", "dim light"]})
        agent = self._make_agent(MockLLM(response))

        entities, registry_update, scene_texts = agent.process_prompt(
            prompt="An empty room with dim light",
            prompt_id=1,
            global_registry={}
        )

        assert len(entities) == 0
        assert registry_update == {}
        assert scene_texts == ["empty room", "dim light"]


# ============ Prompt 设计验证 ============

class TestSystemPromptDesign:
    """验证 system prompt 包含关键指令"""

    def test_prompt_mentions_scene(self):
        assert "scene" in EntityStructExtractor.SYSTEM_PROMPT.lower()

    def test_prompt_mentions_entities(self):
        assert "entities" in EntityStructExtractor.SYSTEM_PROMPT.lower()

    def test_output_format_is_object(self):
        """输出格式应该是 JSON object 而非 array"""
        assert '{"entities"' in EntityStructExtractor.SYSTEM_PROMPT

    def test_scene_excludes_characters(self):
        """prompt 应指示 scene 不包含角色描述"""
        prompt_lower = EntityStructExtractor.SYSTEM_PROMPT.lower()
        assert "do not include character" in prompt_lower or "not include character" in prompt_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
