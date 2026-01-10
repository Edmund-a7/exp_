"""
IAM_2 LLM Agent 模块

包含4个类:
- EntityStruct: 实体数据结构
- EntityStructExtractor: 使用LLM从prompt提取实体和属性
- GlobalIDManager: 使用LLM进行实体匹配和ID分配
- LLMAgent: 协调器，每个prompt的chunk 1触发

流程:
1. 每个prompt，使用EntityStructExtractor提取实体和属性
2. 第一个prompt，直接分配新ID
3. 后续prompt，使用GlobalIDManager匹配或分配新ID
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class EntityStruct:
    """实体数据结构"""
    entity: str
    attrs: List[str] = field(default_factory=list)
    global_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "attrs": self.attrs,
            "global_id": self.global_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EntityStruct":
        return cls(
            entity=data.get("entity", ""),
            attrs=data.get("attrs", []),
            global_id=data.get("global_id")
        )


class LLMWrapper:
    """LLM封装类，支持多种模型"""

    def __init__(self, model_path: str = "../Qwen3-0.6B", device: str = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            self._device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        print(f"[LLMWrapper] Loading model from {self.model_path} on {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True
        )
        if self._device != "cuda":
            self._model = self._model.to(self._device)

        print(f"[LLMWrapper] Model loaded successfully")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.1) -> str:
        """生成LLM响应"""
        self._load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 兼容不同模型的 chat template
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # 某些模型不支持 enable_thinking 参数
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return response


class EntityStructExtractor:
    """
    实体提取器
    功能: 从prompt文本中提取实体和属性
    使用LLM: Qwen 系列或其他兼容模型
    """

    SYSTEM_PROMPT = """Extract human characters from the video prompt.

RULES:
1. ONLY extract human/person entities (man, woman, protagonist, etc.)
2. DO NOT extract objects or locations
3. Extract visual attributes as a string list
4. Keep entity names short

OUTPUT FORMAT (JSON array only, no explanation):
[{"entity": "<entity_name>", "attrs": ["<attr1>", "<attr2>", ...]}]

If no humans found, return: []"""

    def __init__(self, llm: Optional[LLMWrapper] = None, model_path: str = "../Qwen3-0.6B"):
        self.llm = llm or LLMWrapper(model_path)

    def extract(self, prompt: str) -> List[EntityStruct]:
        """
        从prompt中提取实体和属性

        Args:
            prompt: 视频生成的文本prompt

        Returns:
            实体列表，global_id均为None
        """
        response = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=1024,
            temperature=0.1
        )

        entities_data = self._parse_response(response)
        return [
            EntityStruct(
                entity=e.get("entity", ""),
                attrs=e.get("attrs", []),
                global_id=None
            )
            for e in entities_data
        ]

    def _parse_response(self, response: str) -> List[Dict]:
        """解析LLM响应，带容错处理"""
        try:
            # 移除markdown代码块标记
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()

            # 找到JSON数组
            start = response.find('[')
            end = response.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复常见错误：提取所有 {"entity": ..., "attrs": ...} 模式
                    return self._extract_entities_fallback(json_str)
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON, raw response: {response[:500]}...")
            # 最后尝试用正则提取
            return self._extract_entities_fallback(response)

    def _extract_entities_fallback(self, text: str) -> List[Dict]:
        """
        后备方案：用正则从混乱的输出中提取实体
        处理类似: {"entity": "xxx", "attrs": [...], "entity": "yyy", ...} 的错误格式
        """
        entities = []

        # 方法1: 尝试匹配 "entity": "xxx" 和对应的 "attrs": [...]
        entity_pattern = r'"entity"\s*:\s*"([^"]+)"'
        attrs_pattern = r'"attrs"\s*:\s*\[([^\]]*)\]'

        entity_matches = list(re.finditer(entity_pattern, text))
        attrs_matches = list(re.finditer(attrs_pattern, text))

        # 按位置配对
        for i, entity_match in enumerate(entity_matches):
            entity_name = entity_match.group(1)

            # 找到这个entity之后最近的attrs
            entity_pos = entity_match.end()
            best_attrs = []

            for attrs_match in attrs_matches:
                if attrs_match.start() > entity_pos:
                    # 解析 attrs 列表
                    attrs_str = attrs_match.group(1)
                    try:
                        # 尝试解析为列表
                        attrs_list = json.loads(f"[{attrs_str}]")
                        best_attrs = [str(a) for a in attrs_list if a]
                    except:
                        # 用正则提取引号内的字符串
                        best_attrs = re.findall(r'"([^"]+)"', attrs_str)
                    break

            if entity_name and entity_name not in ['<name>', 'name']:
                entities.append({
                    "entity": entity_name,
                    "attrs": best_attrs
                })

        return entities


class GlobalIDManager:
    """
    全局ID管理器
    功能: 实体ID匹配和分配
    策略:
    - 第一个prompt: 直接分配新ID
    - 后续prompt: LLM判断是否匹配现有实体
    """

    MATCHING_SYSTEM_PROMPT = """Match a new character to existing characters.

TASK: Given a new character description and existing character registry, determine if they refer to the same person.

MATCHING RULES:
1. Words like "protagonist", "main character", "he", "she" usually refer to previously introduced characters
2. Matching clothing or appearance attributes indicates the same person
3. Words like "another", "other", "new", "different" indicate a NEW person - return null

OUTPUT FORMAT (JSON only, no explanation):
{"matched_id": <number or null>}"""

    # 明确新实体的标记词
    NEW_ENTITY_MARKERS = ["another", "other", "new", "different", "second", "third"]

    def __init__(self, llm: Optional[LLMWrapper] = None, model_path: str = "../Qwen3-0.6B"):
        self.llm = llm or LLMWrapper(model_path)
        self._next_id = 1

    def assign_ids(self,
                   entities: List[EntityStruct],
                   global_registry: Dict[str, Dict],
                   is_first_prompt: bool) -> List[EntityStruct]:
        """
        为实体分配全局ID

        Args:
            entities: 待分配ID的实体列表
            global_registry: 现有的全局注册表
            is_first_prompt: 是否为第一个prompt

        Returns:
            分配了global_id的实体列表
        """
        # 更新next_id为当前最大ID+1
        if global_registry:
            max_id = max(int(k) for k in global_registry.keys())
            self._next_id = max(self._next_id, max_id + 1)

        if is_first_prompt:
            # 第一个prompt，直接分配新ID
            for entity in entities:
                entity.global_id = self._allocate_new_id()
        else:
            # 后续prompt，使用LLM匹配
            for entity in entities:
                matched_id = self._match_or_allocate(entity, global_registry)
                entity.global_id = matched_id

        return entities

    def _allocate_new_id(self) -> int:
        """分配新ID"""
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def _match_or_allocate(self,
                           entity: EntityStruct,
                           global_registry: Dict[str, Dict]) -> int:
        """
        匹配现有实体或分配新ID

        Args:
            entity: 待匹配的实体
            global_registry: 全局注册表

        Returns:
            匹配的ID或新分配的ID
        """
        entity_lower = entity.entity.lower()

        # 1. 检查是否包含明确的新实体标记
        is_explicitly_new = any(marker in entity_lower for marker in self.NEW_ENTITY_MARKERS)
        if is_explicitly_new:
            new_id = self._allocate_new_id()
            print(f"[DEBUG] _match_or_allocate: entity='{entity.entity}' has NEW marker, allocated new_id={new_id}")
            return new_id

        if not global_registry:
            new_id = self._allocate_new_id()
            print(f"[DEBUG] _match_or_allocate: entity='{entity.entity}', registry empty, allocated new_id={new_id}, _next_id now={self._next_id}")
            return new_id

        # 2. 拼接 entity + attrs 为完整描述
        entity_desc = f"{entity.entity}: {', '.join(entity.attrs)}" if entity.attrs else entity.entity

        # 3. 构建registry信息 (同样拼接为完整描述)
        registry_info = self._format_registry_for_llm(global_registry)

        user_prompt = f"""New character description:
"{entity_desc}"

Existing characters:
{registry_info}

Does the new character match any existing one? If yes, return the ID. If no, return null.
Output JSON only: {{"matched_id": <number or null>}}"""

        response = self.llm.generate(
            system_prompt=self.MATCHING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=256,
            temperature=0.1
        )

        matched_id = self._parse_matching_response(response)

        print(f"[DEBUG] _match_or_allocate: entity='{entity.entity}', matched_id={matched_id}, registry_keys={list(global_registry.keys())}, _next_id={self._next_id}")
        if matched_id is not None and str(matched_id) in global_registry:
            return matched_id
        else:
            new_id = self._allocate_new_id()
            print(f"[DEBUG] _match_or_allocate: allocated new_id={new_id}")
            return new_id

    def _format_registry_for_llm(self, global_registry: Dict[str, Dict]) -> str:
        """格式化registry信息为entity: attrs描述格式"""
        lines = []
        for gid, info in global_registry.items():
            entities = info.get("all_entities", [])
            attrs = info.get("all_attrs", [])
            # 拼接为 "ID X: entity1/entity2: attr1, attr2, ..."
            entity_names = "/".join(entities)
            attrs_str = ", ".join(attrs) if attrs else "no attributes"
            lines.append(f"ID {gid}: {entity_names}: {attrs_str}")
        return "\n".join(lines)

    def _parse_matching_response(self, response: str) -> Optional[int]:
        """解析匹配响应"""
        try:
            # 移除markdown代码块
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            response = response.strip()

            # 找到JSON对象
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
                data = json.loads(json_str)
                matched_id = data.get("matched_id")
                if matched_id is not None:
                    return int(matched_id)
            return None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


class LLMAgent:
    """
    LLM Agent协调器
    触发: 每个prompt的chunk 1
    流程:
    1. 调用EntityStructExtractor提取实体
    2. 调用GlobalIDManager分配/匹配ID
    3. 返回实体列表和registry更新信息
    """

    def __init__(self, model_path: str = "../Qwen3-0.6B"):
        # 共享同一个LLM实例
        self.llm = LLMWrapper(model_path)
        self.extractor = EntityStructExtractor(llm=self.llm)
        self.id_manager = GlobalIDManager(llm=self.llm)

    def process_prompt(self,
                       prompt: str,
                       prompt_id: int,
                       global_registry: Dict[str, Dict]) -> Tuple[List[EntityStruct], Dict[str, Any]]:
        """
        处理prompt，提取实体并分配ID

        Args:
            prompt: prompt文本
            prompt_id: prompt序号 (从1开始)
            global_registry: 现有的全局注册表

        Returns:
            (entities, registry_update) - 实体列表和需要更新的registry信息
        """
        # 1. 提取实体
        entities = self.extractor.extract(prompt)

        if not entities:
            return [], {}

        # 2. 分配ID
        is_first_prompt = (prompt_id == 1)
        entities = self.id_manager.assign_ids(entities, global_registry, is_first_prompt)

        # 3. 构建registry更新
        registry_update = self._build_registry_update(entities, prompt_id, global_registry)

        return entities, registry_update

    def _build_registry_update(self,
                               entities: List[EntityStruct],
                               prompt_id: int,
                               existing_registry: Dict[str, Dict]) -> Dict[str, Any]:
        """
        构建registry更新信息

        Args:
            entities: 已分配ID的实体列表
            prompt_id: prompt序号
            existing_registry: 现有registry

        Returns:
            需要更新/新增到registry的信息
        """
        update = {}

        for entity in entities:
            gid = str(entity.global_id)

            if gid in existing_registry:
                # 更新现有实体
                update[gid] = {
                    "action": "update",
                    "new_entity": entity.entity,
                    "new_attrs": entity.attrs,
                    "prompt_id": prompt_id
                }
            else:
                # 新增实体
                entity_type = self._infer_entity_type(entity.entity)
                type_count = sum(
                    1 for k, v in existing_registry.items()
                    if v.get("name", "").startswith(entity_type)
                )
                type_count += sum(
                    1 for k, v in update.items()
                    if v.get("name", "").startswith(entity_type)
                )

                update[gid] = {
                    "action": "create",
                    "name": f"{entity_type}_{type_count + 1}",
                    "all_entities": [entity.entity],
                    "all_attrs": entity.attrs.copy(),
                    "instances": [{
                        "prompt_id": prompt_id,
                        "entity": entity.entity,
                        "attrs": entity.attrs.copy()
                    }]
                }

        return update

    def _infer_entity_type(self, entity_name: str) -> str:
        """推断实体类型"""
        entity_lower = entity_name.lower()
        if any(w in entity_lower for w in ["woman", "girl", "lady", "female", "she"]):
            return "woman"
        elif any(w in entity_lower for w in ["man", "boy", "guy", "male", "he", "protagonist"]):
            return "man"
        else:
            return "person"


# ============ 测试代码 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM Agent")
    parser.add_argument("--model", type=str, default="../Qwen3-0.6B",
                        help="Path to LLM model (e.g., Qwen/Qwen2.5-7B-Instruct)")
    args = parser.parse_args()

    # 测试用prompts (来自pipeline.md)
    prompts = [
        "A realistic video of a modern city park environment. A young man in his late 20s, with messy black hair, wearing a vintage blue denim jacket, sits alone on a park bench. He holds a sketchbook in his hands, looking pensive and expectant.",
        "A realistic video of a modern city park environment. The main protagonist in the denim jacket remains seated on the bench. Another man, around 30 years old, wearing glasses and a casual grey sweater, walks into the frame and sits next to him holding a coffee cup.",
        "A realistic video of a modern city park environment. The protagonist and the man in the grey sweater are talking on the bench. A young woman in her late 20s, with long hair and wearing a flowing white dress, approaches them carrying a shoulder bag."
    ]

    print("=" * 60)
    print(f"Testing LLM Agent with model: {args.model}")
    print("=" * 60)

    agent = LLMAgent(model_path=args.model)
    global_registry = {}

    for i, prompt in enumerate(prompts):
        prompt_id = i + 1
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_id}:")
        print(f"{'='*60}")
        print(prompt)

        entities, registry_update = agent.process_prompt(prompt, prompt_id, global_registry)

        print(f"\nExtracted entities:")
        for e in entities:
            print(f"  - {e.entity} (ID: {e.global_id})")
            print(f"    Attrs: {e.attrs}")

        # 应用更新到registry
        for gid, info in registry_update.items():
            if info["action"] == "create":
                global_registry[gid] = {
                    "name": info["name"],
                    "all_entities": info["all_entities"],
                    "all_attrs": info["all_attrs"],
                    "instances": info["instances"]
                }
            elif info["action"] == "update":
                if gid in global_registry:
                    reg = global_registry[gid]
                    if info["new_entity"] not in reg["all_entities"]:
                        reg["all_entities"].append(info["new_entity"])
                    for attr in info["new_attrs"]:
                        if attr not in reg["all_attrs"]:
                            reg["all_attrs"].append(attr)
                    reg["instances"].append({
                        "prompt_id": info["prompt_id"],
                        "entity": info["new_entity"],
                        "attrs": info["new_attrs"]
                    })

        print(f"\nCurrent global_registry:")
        print(json.dumps(global_registry, indent=2, ensure_ascii=False))
