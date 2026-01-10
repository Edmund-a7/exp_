# IAM 单 Prompt 模式技术说明

## 核心机制

即使在单 prompt 模式下，IAM 的 LLM Agent 和 Memory Bank 仍然**完整工作**。

### 🔍 执行流程

#### 1. 初始化阶段（第 1 个 Chunk）
```python
# agent_causal_inference.py, line 190-194
self._process_prompt_start(
    prompt_text=text_prompts_list[0][0],
    prompt_id=1,
    is_first_prompt=True
)
```

**LLM Agent 动作**：
- 调用 `llm_agent.process_prompt()` 提取实体
- 分配 global_id（第一个 prompt，直接分配）
- 示例输入：`"A young man in a park with a dog"`
- 示例输出：
  ```python
  [
      EntityStruct(entity="young man", attrs=["in a park"], global_id=1),
      EntityStruct(entity="dog", attrs=["with young man"], global_id=2)
  ]
  ```

**Memory Bank 动作**：
- 注册实体到 `global_registry`
- 初始化 `frame_active_memory` 为空（首个 prompt 无历史帧）

---

#### 2. 生成阶段（Chunk 1-2）
- 正常生成视频帧
- KV Cache 逐步填充：Sink + Local
- **不触发帧驱逐**（Local 未满）

---

#### 3. 记忆管理阶段（Chunk 3+）
```python
# agent_causal_inference.py, line 296-299
if self.current_chunk_id >= 3 and self.current_entities:
    self._process_chunk_eviction(
        current_start_frame=current_start_frame,
        current_num_frames=current_num_frames
    )
```

**每个 Chunk 执行**：

##### a) 驱逐旧 Chunk
- Local 窗口满载（6 帧 = 2 chunks）
- 最早的 chunk 被驱逐（如 Chunk 1 → Chunk 3 时驱逐）

##### b) IAM 帧选择
```python
# agent_causal_inference.py, line 416-422
entity_ids = self.agent_memory_bank.get_entity_ids(self.current_entities)
frame_id, score = self.agent_memory_bank.select_frame_from_chunk(
    evicted_chunk_kv=evicted_chunk_kv,
    crossattn_cache=self.crossattn_cache,
    prompt_id=self.current_prompt_id,
    chunk_id=self.current_chunk_id,
    current_entity_ids=entity_ids  # [1, 2] (young man + dog)
)
```

**选帧机制**：
1. 构建 query text：`"young man in a park dog with young man"`
2. 对被驱逐 chunk 的 3 帧计算交叉注意力分数
3. 选择得分最高的帧
4. 示例：
   ```
   Chunk 1 Frame 0: score = 0.85
   Chunk 1 Frame 1: score = 0.91  ← 选中
   Chunk 1 Frame 2: score = 0.78
   ```

##### c) 更新 Active Memory
```python
# memory_bank.py
self.agent_memory_bank.update_active_memory(frame_id, score)
```

- 维护 Top-3 记忆帧（`max_memory_frames=3`）
- 如果新帧 score 高于当前最低，替换
- 示例进程：
  ```
  Chunk 3: ["p1_c1_f1"] (1 帧)
  Chunk 4: ["p1_c2_f0", "p1_c1_f1"] (2 帧)
  Chunk 5: ["p1_c3_f1", "p1_c2_f0", "p1_c1_f1"] (3 帧，满载)
  Chunk 6: ["p1_c4_f2", "p1_c3_f1", "p1_c2_f0"] (替换最低分)
  ```

##### d) 注入到 KV Bank
```python
# agent_causal_inference.py, line 393
self._inject_iam_memory_to_bank()
```

- 将 active memory 的 3 帧注入到 KV Bank
- 模型在生成时读取这些记忆帧（`q_bank=True`）
- 保持实体外观和动作的一致性

---

## 📊 数据结构示例

### 单 Prompt 完整执行后的状态

**Prompt**: `"A young man in a park playing with a dog"`

**global_registry**:
```json
{
  "1": {
    "name": "man_1",
    "all_entities": ["young man"],
    "all_attrs": ["in a park", "playing with dog"],
    "instances": [
      {"prompt_id": 1, "entity": "young man", "attrs": ["in a park", "playing with dog"]}
    ]
  },
  "2": {
    "name": "dog_1",
    "all_entities": ["dog"],
    "all_attrs": ["with young man", "playing"],
    "instances": [
      {"prompt_id": 1, "entity": "dog", "attrs": ["with young man", "playing"]}
    ]
  }
}
```

**frame_archive** (假设生成 40 chunks = 120 帧):
```json
{
  "p1_c1_f1": {"prompt_id": 1, "associated_entities": ["1", "2"], "score": 0.91},
  "p1_c2_f0": {"prompt_id": 1, "associated_entities": ["1", "2"], "score": 0.89},
  "p1_c3_f1": {"prompt_id": 1, "associated_entities": ["1", "2"], "score": 0.93},
  // ... 37 more frames
}
```

**frame_active_memory** (Top 3):
```json
["p1_c37_f2", "p1_c25_f1", "p1_c18_f0"]
```

---

## 🔄 与交互式模式的差异

| 阶段 | 单 Prompt | 交互式 (多 Prompt) |
|------|-----------|-------------------|
| **Prompt 1 Chunk 1** | LLM Agent 提取实体 | LLM Agent 提取实体 |
| **Prompt 1 Chunk 3+** | Memory Bank 选帧 | Memory Bank 选帧 |
| **Prompt 2 Chunk 1** | ❌ 无 | ✅ LLM Agent **匹配**实体 |
|  |  | ✅ Memory Bank **检索**历史帧 |
| **Prompt 2 Chunk 3+** | ❌ 无 | Memory Bank 选帧（含跨 prompt 帧） |

**关键区别**：
- **单 Prompt**：只有一次 LLM 调用（提取），不涉及匹配和检索
- **交互式**：每个新 prompt 都需要 LLM 匹配实体 ID，并从历史帧检索相关记忆

---

## 🎯 单 Prompt 模式的价值

即使没有跨场景切换，IAM 仍然带来显著价值：

### 1. 实体一致性
- 自动识别 prompt 中的关键实体（人物、物体）
- 维护这些实体在整个视频中的外观一致性

### 2. 智能记忆管理
- 不是简单的滑动窗口（如原 MemFlow）
- 基于内容相关性选择最重要的帧
- 示例：人物转身的关键帧、动作高潮帧

### 3. 长视频生成
- 即使 120 帧（40 chunks），只保留 3 个关键记忆帧
- 降低 KV Cache 负担，提高生成质量

### 4. 对比原 MemFlow
```
MemFlow (原始):
  - Sink: 3 帧（固定）
  - Bank: 3 帧（按时序自动更新，如 SMA）
  - 无实体感知

IAM (单 Prompt):
  - Sink: 3 帧（固定）
  - Bank: 3 帧（按实体相关性选择）
  - ✅ 实体感知选帧
```

---

## 🧪 实验建议

### 对比实验
```bash
# 1. 原 MemFlow (无 IAM)
bash inference.sh  # 使用 MemFlow 的 causal_inference.py

# 2. IAM 单 Prompt
bash agent_inference.sh  # 使用 AgentCausalInferencePipeline
```

**观察指标**：
- 角色外观一致性（服装、发型、面部特征）
- 物体一致性（颜色、形状）
- 动作连贯性

### 示例 Prompt（适合测试）
```
A young woman with long red hair, wearing a blue dress,
walking through a forest. She encounters a white rabbit
and follows it through the trees.
```

**期望效果**：
- 女性的红色长发在整个视频中保持一致
- 蓝色裙子颜色不变
- 白兔的外观（尤其在远近变化时）保持稳定

---

## 📝 总结

**单 Prompt 模式是 IAM 的"简化版"而非"禁用版"**：
- ✅ LLM Agent 工作（提取实体）
- ✅ Memory Bank 工作（选帧与驱逐）
- ❌ 不涉及跨 prompt 的实体匹配
- ❌ 不需要历史帧检索

**适用场景**：
- 需要高质量单场景视频生成
- 希望保持实体一致性但不涉及场景切换
- 介于原始 MemFlow 和完整交互式模式之间的折中方案