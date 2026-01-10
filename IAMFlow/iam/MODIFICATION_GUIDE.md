# IAM_Flow 修改指南

## 概述

IAM_Flow 是基于 MemFlow 的改进版本，将 MemFlow 的帧选择逻辑完全替换为 IAM (Intelligent Agent Memory) 的 entity-based 帧选择。

**核心改动**: 删除了 MemFlow 的两种帧选择机制：
1. **NAM (Narrative Attention Memory)**: 使用整个 prompt 文本做交叉注意力选帧
2. **SMA (Sparse Memory Attention)**: 使用视觉特征做 top-k chunk routing

**替代方案**: IAM 使用 LLM Agent 提取实体，基于 entity-attrs 子串做交叉注意力选帧。

---

## 目录结构

```
IAM_Flow/
├── iam/                                    # [新增] IAM 模块
│   ├── __init__.py                         # 模块导出
│   ├── llm_agent.py                        # LLM Agent (实体提取、ID 匹配)
│   ├── memory_bank.py                      # Memory Bank (帧选择、记忆管理)
│   ├── test_iam.py                         # 测试文件
│   └── MODIFICATION_GUIDE.md               # 本文档
├── pipeline/
│   ├── agent_causal_inference.py           # [新增] IAM 整合 Pipeline
│   ├── interactive_causal_inference.py     # 父类 Pipeline
│   └── causal_inference.py
├── wan/
│   └── modules/
│       └── causal_model.py                 # [修改] 删除 NAM 和 SMA 代码
└── ...
```

---

## 代码修改详情

### 1. `wan/modules/causal_model.py`

#### 1.1 删除 NAM 相关代码

**删除的方法**:
- `_apply_cache_updates_before()` - NAM 入口方法
- `compress_kv_bank()` - NAM 帧选择算法（基于 prompt 文本的交叉注意力）

**删除的调用** (原 `_forward_inference` 第 1087-1088 行):
```python
# 原代码 (已删除):
# if kv_bank is not None and q_bank:
#     self._apply_cache_updates_before(kv_bank, crossattn_cache)

# 现在只保留注释:
# IAM: NAM 帧选择已删除，由 IAM 的 entity-based 帧选择替代
```

#### 1.2 删除 SMA 相关代码

**删除的方法**:
- `dynamic_topk_routing_attention()` - SMA 的 top-k chunk routing 算法

**修改后的代码** (第 378-382 行):
```python
k_bank = bank_k[:, :local_end_index_bank_]
v_bank = bank_v[:, :local_end_index_bank_]
# IAM: SMA (dynamic_topk_routing_attention) 已删除，直接拼接 k_bank
k_cat = torch.cat([k_sink, k_bank, k_local], dim=1)
v_cat = torch.cat([v_sink, v_bank, v_local], dim=1)
```

原代码包含 SMA 分支判断：
```python
# 原代码 (已删除):
# if not self.SMA:
#     k_cat = torch.cat([k_sink, k_bank, k_local], dim=1)
#     v_cat = torch.cat([v_sink, v_bank, v_local], dim=1)
# else:
#     k_global, v_global = self.dynamic_topk_routing_attention(...)
#     k_cat = torch.cat([k_global, k_local], dim=1)
#     v_cat = torch.cat([v_global, v_local], dim=1)
```

---

### 2. `pipeline/agent_causal_inference.py` (新文件)

继承 `InteractiveCausalInferencePipeline`，整合 IAM 模块。

**核心组件**:
- `LLMAgent`: 实体提取和 ID 匹配
- `MemoryBank`: 帧选择和记忆管理

**关键逻辑**:

```python
class AgentCausalInferencePipeline(InteractiveCausalInferencePipeline):
    def __init__(self, ...):
        # 初始化 LLM Agent
        self.llm_agent = LLMAgent(model_path=llm_model_path)

        # 初始化 Memory Bank
        self.agent_memory_bank = MemoryBank(
            text_encoder=self.text_encoder,
            max_memory_frames=max_memory_frames,
            ...
        )

    def inference(self, ...):
        # 1. 始终设置 update_bank=False，阻止 MemFlow 自动更新 bank
        self.generator(..., update_bank=False, ...)

        # 2. chunk >= 3 时触发 IAM 帧选择
        if self.current_chunk_id >= 3:
            self._process_chunk_eviction(...)  # IAM 选帧
            self._inject_iam_memory_to_bank()  # 注入 kv_bank
```

---

### 3. `iam/` 模块

#### 3.1 `llm_agent.py`
- `EntityStruct`: 实体数据结构
- `EntityStructExtractor`: 从 prompt 提取实体
- `GlobalIDManager`: 实体 ID 匹配和分配
- `LLMAgent`: 协调器

#### 3.2 `memory_bank.py`
- `FrameInfo`: 帧信息数据结构
- `MemoryBank`: 帧选择、记忆管理、KV 注入

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      IAM_Flow 工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prompt 开始 (chunk 1)                                       │
│       ↓                                                     │
│  LLMAgent.process_prompt()                                  │
│       ├── EntityStructExtractor: 提取实体                    │
│       └── GlobalIDManager: 分配/匹配 ID                      │
│       ↓                                                     │
│  MemoryBank.register_entities()                             │
│       ↓                                                     │
│  (非首个 prompt) MemoryBank.retrieve_initial_frames()        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chunk 生成循环                                              │
│       ↓                                                     │
│  self.generator(..., update_bank=False)  # 阻止 MemFlow     │
│       ↓                                                     │
│  (chunk >= 3) _process_chunk_eviction()                     │
│       ├── 获取被驱逐 chunk 的 KV                             │
│       ├── 构建 entity-attrs 查询文本                         │
│       └── MemoryBank.select_frame_from_chunk() 选帧          │
│       ↓                                                     │
│  _inject_iam_memory_to_bank()                               │
│       └── 将 IAM 选择的帧 KV 写入 kv_bank                    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Forward 时 (causal_model.py)                               │
│       ↓                                                     │
│  k_cat = [k_sink, k_bank, k_local]  # 直接拼接，无 SMA      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 使用方法

```python
from pipeline.agent_causal_inference import AgentCausalInferencePipeline

# 创建 pipeline
pipeline = AgentCausalInferencePipeline(
    args=config,
    device=device,
    llm_model_path="path/to/Qwen3-0.6B",
    max_memory_frames=3,
    save_dir="data/agent_frames"
)

# 推理
video = pipeline.inference(
    noise=noise,
    text_prompts_list=[["prompt1"], ["prompt2"], ["prompt3"]],
    switch_frame_indices=[20, 40],
    save_mapping=True,
    mapping_path="mapping.json"
)
```

---

## MemFlow vs IAM 对比

| 方面 | MemFlow (已删除) | IAM |
|------|------------------|-----|
| **NAM 帧选择** | `compress_kv_bank()` 基于整个 prompt | 无 |
| **SMA 帧选择** | `dynamic_topk_routing_attention()` 基于视觉特征 | 无 |
| **IAM 帧选择** | 无 | `MemoryBank.select_frame_from_chunk()` 基于 entity-attrs |
| **实体追踪** | 无 | LLM Agent + global_registry |
| **帧选择查询** | 整个 prompt 文本 / 视觉 query | entity-attrs 子串 |
| **记忆管理** | 自动 (bank 更新) | 手动 (`_inject_iam_memory_to_bank()`) |
| **输出** | 无 | mapping.json (实体追踪记录) |

---

## 验证方法

### 1. 查看日志输出

```
[AgentPipeline] Prompt 1 entities:
  - young man (ID: 1): ['late 20s', 'messy black hair', 'denim jacket']
[AgentPipeline] IAM selected frame p1_c3_f2 with score 0.8534
[AgentPipeline] Active memory: ['p1_c3_f2']
[AgentPipeline] Injected 1560 tokens from IAM to kv_bank
```

### 2. 检查 mapping.json

```json
{
  "global_registry": {
    "1": {
      "name": "person_1",
      "all_entities": ["young man", "protagonist"],
      "all_attrs": ["late 20s", "messy black hair", "denim jacket"],
      "instances": [...]
    }
  },
  "frame_archive": {
    "p1_c3_f2": {
      "prompt_id": 1,
      "chunk_id": 3,
      "frame_idx": 2,
      "associated_entities": ["1"],
      "score": 0.8534
    }
  },
  "frame_active_memory": ["p1_c3_f2", "p2_c4_f1", "p2_c5_f0"]
}
```

---

## 回滚方法

如需恢复 MemFlow 原有逻辑：

使用原始 `MemFlow/` 目录下的代码，而非 `IAM_Flow/`。

```python
# 使用 MemFlow 原版
from MemFlow.pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline

# 而非 IAM_Flow
# from IAM_Flow.pipeline.agent_causal_inference import AgentCausalInferencePipeline
```
