# ID-Centric Memory 实现方案

> 基于 [ID-Centric Memory 框架设计](./ID-Centric%20Memory%20-%20基于实体ID的LLM驱动长视频记忆管理框架.md) 的工程实现计划。
> Benchmark 部分不在本方案范围内。

---

## 现状分析

### 已实现（可直接复用）

| 模块 | 文件 | 说明 |
|------|------|------|
| LLM Agent | `iam/llm_agent.py` | 实体提取 + Global ID 管理，Qwen3-0.6B via vLLM |
| Scene 文本提取 | `iam/llm_agent.py` | **Phase 1 已完成**，LLM 同时输出 entities + scene |
| Memory Bank | `iam/memory_bank.py` | frame_archive / global_registry / active_memory |
| SPT | `utils/transition_scheduler.py` | 5 种调度器 + Adaptive 语义距离自适应窗口 |
| 多层共识打分 | `iam/memory_bank.py` | layers 0/15/29，权重 0.5/0.3/0.2 |
| 快速帧选择 | `iam/memory_bank.py` | 均值池化点积，~4000x 加速 |
| KV cache 三层架构 | `wan/modules/causal_model.py` | sink + bank + local |
| Agent Pipeline | `pipeline/agent_causal_inference.py` | 完整集成，替代 MemFlow NAM |
| TCAT 稀疏注意力 | `wan/modules/causal_model.py` | per-region top-k + EMA 平滑 |

### 核心缺失

| 模块 | 框架设计 | 当前状态 |
|------|---------|---------|
| 双层记忆 | ID Memory + Scene Memory 独立检索 | 单一 active_memory，无分层 |
| Scene 检索 | scene text 直接对 Frame Archive 打分 | 不存在 |
| 同场景跳过 | 语义距离 ≤ 阈值 → 零开销 | 不存在 |
| 动态帧分配 | 按 ID 覆盖度动态调整 1~4 帧 | 固定 3 帧 top-k |
| VLM 视觉验证 | 第一个 chunk 后校验 entity attrs | 不存在 |
| Block-wise Sparse Attention | 帧级粗筛 + token block 细筛 | TCAT 机制不同 |

---

## 实现分 Phase

```
Phase 1 ✅ → Phase 2 ✅ → Phase 3 ✅ → Phase 4 → Phase 5
Scene提取     双层记忆      动态分配      VLM验证    Sparse Attn
(已完成)      (已完成)     +场景跳过     (闭环)     (加速)
                           (已完成)
```

依赖关系：Phase 2 是后续所有 Phase 的基础。Phase 3/4/5 之间相互独立，可并行开发。

---

## Phase 1: Scene Text 提取 ✅ 已完成

**改动文件:** `iam/llm_agent.py`, `iam/__init__.py`, `pipeline/agent_causal_inference.py`, 测试文件

**完成内容:**
- 新增 `SceneStruct` dataclass
- `EntityStructExtractor` prompt 改为同时输出 `{"entities": [...], "scene": [...]}`
- 解析器支持新格式，向后兼容旧数组格式
- `LLMAgent.process_prompt()` 返回 `(entities, registry_update, scene_texts)` 三元组
- `AgentCausalInferencePipeline` 新增 `current_scene_texts` 状态
- 25 个单元测试全部通过

---

## Phase 2: 双层记忆 — ID Memory + Scene Memory ✅ 已完成

**目标:** 将单一 active_memory 拆分为 ID Memory（管"谁"）和 Scene Memory（管"在哪"），各自用不同的 text query 独立检索。

**改动文件:** `iam/memory_bank.py`, `pipeline/agent_causal_inference.py`, `tests/test_dual_memory.py`

**完成内容:**
- `FrameInfo` 新增 `entity_score` / `scene_score` 双分数字段
- `MemoryBank` 拆分为 `id_memory` (max 4帧) + `scene_memory` (max 2帧)，`frame_active_memory` 改为去重合并的 property
- 新增 `_build_scene_token_weights()` 场景权重构建（与 entity 权重互补）
- `select_frame_from_chunk()` 双路打分: entity_scores + scene_scores → 综合分数 0.6/0.4
- 提取 `_consensus_score()` 通用多层共识打分方法，消除代码重复
- 新增 `update_id_memory()` / `update_scene_memory()` 独立 top-k 更新
- `retrieve_initial_frames()` 双路检索: ID 匹配 + scene_score 排序
- Pipeline `_process_chunk_eviction()` 改为双路更新
- Pipeline `_process_prompt_start()` 传入 scene_texts 进行双路检索
- JSON 持久化支持双层记忆，向后兼容旧格式
- 32 个单元测试全部通过，61 个总测试零回归

### 2.1 数据结构重构

`FrameInfo` 增加双分数:

```python
@dataclass
class FrameInfo:
    frame_id: str
    frame_path: str
    prompt_id: int
    associated_entities: List[str]
    entity_score: float          # 新增: entity 维度分数
    scene_score: float           # 新增: scene 维度分数
    score: float                 # 保留: 综合分数 (向后兼容)
    kv_cache: Optional[...] = None
```

`MemoryBank` 拆分 active memory:

```python
class MemoryBank:
    def __init__(self, ...,
                 max_id_memory_frames: int = 4,
                 max_scene_memory_frames: int = 2):
        # 替代原来的 frame_active_memory
        self.id_memory: List[str] = []        # entity 驱动，1~4 帧
        self.scene_memory: List[str] = []     # scene 驱动，1~2 帧
        # 保留 frame_active_memory 作为兼容属性
```

### 2.2 双路打分

Scene 打分需要一个新的权重构建方法，与 entity 权重互补:

```python
def _build_scene_token_weights(self, scene_texts, num_tokens, prompt_text):
    """
    构建 scene token 权重向量
    - scene 关键词区域权重高 (2.5x)
    - entity 区域权重低 (0.5x)
    - 与 _build_entity_token_weights 互补
    """
```

`select_frame_from_chunk()` 改为同时计算两个分数:

```python
def select_frame_from_chunk(self, ..., scene_texts=None):
    # 1. entity 路径: 用 entity+attrs token 权重打分
    entity_scores = self._score_frames(chunk_kv, crossattn_cache, entity_weights)

    # 2. scene 路径: 用 scene text token 权重打分
    scene_scores = self._score_frames(chunk_kv, crossattn_cache, scene_weights)

    # 3. 综合分数 (用于 frame_archive 排序)
    combined_scores = 0.6 * entity_scores + 0.4 * scene_scores

    # 4. 存储双分数到 FrameInfo
    frame_info.entity_score = entity_scores[best_idx]
    frame_info.scene_score = scene_scores[best_idx]
```

> 注: 底层打分函数 `_compute_frame_scores_fast()` 可以复用，只需传入不同的 weights 向量。

### 2.3 独立的 Active Memory 更新

```python
def update_id_memory(self, frame_id: str, entity_score: float):
    """按 entity_score top-k 更新 ID Memory 槽"""

def update_scene_memory(self, frame_id: str, scene_score: float):
    """按 scene_score top-k 更新 Scene Memory 槽"""
```

同一帧可能同时被两层选中（设计文档明确允许）。

### 2.4 KV 输出拼接

```python
def get_memory_kv(self, device=None):
    """拼接 id_memory + scene_memory 的 KV，注入 kv_bank"""
    # 去重: 同一帧被两层选中时只出现一次
    all_frame_ids = list(dict.fromkeys(self.id_memory + self.scene_memory))
    # 按时间排序后拼接
    ...
```

### 2.5 Pipeline 集成

`_process_prompt_start()`:
- ID Memory: 基于 Global ID 从 frame_archive 检索 → 更新 id_memory
- Scene Memory: 用 scene_texts 对 frame_archive 打分 → 更新 scene_memory

`_process_chunk_eviction()`:
- 双路打分 → 分别更新 id_memory 和 scene_memory

`_inject_iam_memory_to_bank()`:
- 调用 `get_memory_kv()` 获取拼接后的 KV → 写入 kv_bank

### 2.6 测试计划

- 单元测试: 双路打分、独立更新、去重拼接
- 集成测试: 3-prompt 场景验证 id_memory 和 scene_memory 分别包含正确的帧
- 回归测试: 确保现有 29 个测试不受影响

---

## Phase 3: 动态帧分配 + 同场景跳过 ✅ 已完成

**目标:** ID Memory 帧数根据 ID 覆盖度动态调整；场景未变时跳过 Scene Memory 检索。

**改动文件:** `iam/memory_bank.py`, `pipeline/agent_causal_inference.py`, `configs/agent_*.yaml`, `tests/test_dynamic_allocation.py`

**完成内容:**
- 新增 `_compute_dynamic_id_budget()` 贪心集合覆盖算法，动态计算 ID Memory 帧预算
- 新增 `_greedy_select_id_frames()` 贪心帧选择，覆盖最多 ID + entity_score 打破平局
- `retrieve_initial_frames()` 改用动态预算替代固定 top-k
- 新增 `_compute_scene_distance()` 基于 token Jaccard 的场景距离计算
- Pipeline 新增 `prev_scene_texts` 状态 + `scene_skip_threshold` (默认 0.3)
- `_process_prompt_start()` 同场景跳过: 距离 ≤ 阈值时保留 scene_memory 不重新检索
- `bank_size` 从固定 3 改为动态上限 6 (max_id=4 + max_scene=2)，config YAML 同步更新
- 31 个新测试全部通过，92 个总测试零回归

### 3.1 ID 覆盖度动态帧分配

```python
def _compute_dynamic_id_budget(self, required_entity_ids: List[str]) -> int:
    """
    根据 ID 覆盖度计算 ID Memory 需要的帧数

    策略:
    1. 优先选覆盖所有 ID 的帧 (1帧搞定 → budget=1)
    2. 不够则贪心选覆盖最多未覆盖 ID 的帧
    3. 直到所有 ID 被覆盖，budget = 选出的帧数
    4. 上限 max_id_memory_frames (默认4)
    """
```

帧选择的贪心算法:

```
uncovered_ids = set(required_entity_ids)
selected = []
while uncovered_ids and len(selected) < max_budget:
    best_frame = argmax(|frame.associated_entities ∩ uncovered_ids|, then entity_score)
    selected.append(best_frame)
    uncovered_ids -= set(best_frame.associated_entities)
```

### 3.2 同场景跳过优化

```python
# AgentCausalInferencePipeline
def _process_prompt_start(self, ...):
    ...
    # Scene 路径: 计算前后 scene text 语义距离
    if not is_first_prompt:
        scene_distance = self._compute_scene_distance(
            self.prev_scene_texts, scene_texts
        )
        if scene_distance <= self.scene_skip_threshold:  # 默认 0.3
            # 场景未变，Scene Memory 保持不动，零开销
            print(f"[DEBUG] Scene unchanged (dist={scene_distance:.3f}), skipping retrieval")
        else:
            # 场景切换，重新检索 Scene Memory
            self._retrieve_scene_memory(scene_texts)

    self.prev_scene_texts = scene_texts
```

语义距离计算复用 text_encoder:

```python
def _compute_scene_distance(self, old_texts, new_texts):
    """
    计算两组 scene text 的语义距离
    方案: 拼接为句子 → text_encoder 编码 → cosine distance
    """
    old_str = ", ".join(old_texts)
    new_str = ", ".join(new_texts)
    old_emb = self.text_encoder([old_str])["prompt_embeds"].mean(dim=(0,1))
    new_emb = self.text_encoder([new_str])["prompt_embeds"].mean(dim=(0,1))
    return 1.0 - F.cosine_similarity(old_emb.unsqueeze(0), new_emb.unsqueeze(0)).item()
```

### 3.3 KV Cache 结构调整

```
KV cache = sink + ID mem + scene mem + local
           (2)    (动态1~4)  (1~2)      (6)
           ─────────────────────────────────
                        总计 ~12 帧
```

总帧数预算与原来的 3+3+6 相当，但分配更智能。`bank_size` 参数需要从固定 3 改为动态上限 6 (max_id=4 + max_scene=2)。

### 3.4 测试计划

- 单元测试: 贪心覆盖算法、动态 budget 计算
- 场景跳过: 相同 scene → 跳过，不同 scene → 重新检索
- 帧预算: 验证总帧数不超过上限

---

## Phase 4: VLM 视觉验证

**目标:** 每个 prompt 的第一个 chunk 生成后，用 VLM 从实际帧中提取 entity attrs，与 LLM 提取结果校验，形成闭环。

**新增文件:** `iam/vlm_agent.py`
**改动文件:** `pipeline/agent_causal_inference.py`

### 4.1 VLMAgent 设计

```python
class VLMAgent:
    """
    视觉验证 Agent
    模型: Qwen2.5-VL 或类似 VLM
    触发: 每个 prompt 的第一个 chunk 生成后
    """

    def __init__(self, model_path, device="cuda"):
        self.model = ...  # VLM 模型

    def verify_entities(self,
                        frame_pixels: torch.Tensor,
                        expected_entities: List[EntityStruct]
                        ) -> List[EntityStruct]:
        """
        从生成帧中提取实际 entity & attrs，与预期对比

        Returns:
            verified_entities: 校验后的实体列表 (attrs 可能被更新)
        """
```

### 4.2 Pipeline 集成

```python
# 在第一个 chunk 生成后触发
if self.current_chunk_id == 1 and self.vlm_agent is not None:
    # 1. VAE decode 当前 chunk → 像素帧
    chunk_pixels = self.vae.decode_to_pixel(denoised_pred)

    # 2. VLM 验证
    verified = self.vlm_agent.verify_entities(chunk_pixels, self.current_entities)

    # 3. 属性冲突解决: 保留最新 attrs
    self._update_registry_from_vlm(verified)
```

### 4.3 显存管理

VLM 模型较大，需要与 diffusion 模型交替占用显存:

```python
# 方案: 用 DynamicSwapInstaller 在 CPU/GPU 间切换
# diffusion forward 时 VLM 在 CPU
# VLM verify 时 diffusion 暂停 (chunk 间隙)
```

### 4.4 可选开关

```python
AgentCausalInferencePipeline(
    ...,
    use_vlm=False,           # 默认关闭
    vlm_model_path=None,
)
```

### 4.5 测试计划

- Mock VLM 测试: 验证属性冲突解决逻辑
- 集成测试: 验证 VLM 检测到属性漂移后 registry 正确更新

---

## Phase 5: Block-wise Sparse Attention

**目标:** 对 ID Memory + Scene Memory 帧施加 block-wise 稀疏注意力，控制计算开销不随记忆规模线性增长。

**改动文件:** `wan/modules/causal_model.py`

### 5.1 两级筛选

```
输入: mem_frames KV (动态 1~6 帧, 每帧 1560 tokens)
      current_chunk query (3 帧, 4680 tokens)

第一级 - 帧级粗筛:
  K_frame_mean = mean_pool(K_mem, per_frame)  → [F, H, D]
  Q_chunk_mean = mean_pool(Q_chunk)           → [H, D]
  frame_scores = Q_chunk_mean · K_frame_mean  → [F]
  selected_frames = top_k(frame_scores, k=3)

第二级 - Token block 细筛:
  对 selected_frames 内的 tokens 按 block_size=64 分组
  block_scores = Q_chunk · K_block_mean       → [num_blocks]
  selected_blocks = top_k(block_scores, k_b)

输出: 仅对 selected_blocks 执行精确注意力
```

### 5.2 与现有 TCAT 的关系

TCAT 是 per-region (sink/mem/local) 的 chunk 级 top-k，粒度为 chunk (1560 tokens)。
Block-wise Sparse 是 mem 内部的 token block 级 top-k，粒度为 block (64 tokens)。

两者可以共存: TCAT 决定 sink/mem/local 各取多少 chunk，Block-wise 在 mem 内部进一步细筛。

### 5.3 实现位置

在 `CausalWanSelfAttention.forward()` 的 inference 路径中，bank 拼接前插入 sparse 筛选:

```python
if self.sparse_mem_attention and bank_length > 0:
    k_bank, v_bank = self._blockwise_sparse_select(
        q_current, k_bank, v_bank,
        frame_top_k=3, block_size=64, block_top_k=8
    )
```

### 5.4 测试计划

- 正确性: sparse 输出与 dense 输出的 cosine similarity > 0.95
- 性能: 当 mem 帧数 > 3 时，sparse 比 dense 快

---

## TODO 汇总

```
Phase 1: Scene Text 提取 ✅ 已完成
  ✅ 1.1 新增 SceneStruct dataclass
  ✅ 1.2 修改 EntityStructExtractor prompt + 解析逻辑
  ✅ 1.3 LLMAgent.process_prompt() 返回 scene_texts
  ✅ 1.4 AgentPipeline 增加 current_scene_texts 状态
  ✅ 1.5 单元测试 (25 tests passed)

Phase 2: 双层记忆 ✅ 已完成
  ✅ 2.1 FrameInfo 增加 entity_score / scene_score
  ✅ 2.2 MemoryBank 拆分 id_memory + scene_memory
  ✅ 2.3 实现 _build_scene_token_weights()
  ✅ 2.4 select_frame_from_chunk() 双路打分
  ✅ 2.5 update_id_memory() + update_scene_memory()
  ✅ 2.6 get_memory_kv() 去重拼接双层 KV
  ✅ 2.7 retrieve_initial_frames() 分别检索 ID/Scene 帧
  ✅ 2.8 Pipeline: _process_prompt_start() 双路检索
  ✅ 2.9 Pipeline: _process_chunk_eviction() 双路更新
  ✅ 2.10 Pipeline: _inject_iam_memory_to_bank() 拼接注入 (通过 property 自动去重)
  ✅ 2.11 单元测试 + 集成测试 + 回归测试 (32 new, 61 total)

Phase 3: 动态分配 + 场景跳过 ✅ 已完成
  ✅ 3.1 _compute_dynamic_id_budget() ID 覆盖度计算 (贪心集合覆盖)
  ✅ 3.2 _greedy_select_id_frames() 贪心帧选择: 覆盖所有 ID 的最小帧集
  ✅ 3.3 retrieve_initial_frames() 使用动态预算替代固定 top-k
  ✅ 3.4 prev_scene_texts 状态 + _compute_scene_distance() (token Jaccard)
  ✅ 3.5 同场景跳过逻辑 (阈值可配置，默认 0.3)
  ✅ 3.6 bank_size 从固定 3 改为动态上限 6 (pipeline 自动覆盖 + config 更新)
  ✅ 3.7 测试: 31 new tests, 92 total, 零回归

Phase 4: VLM 视觉验证 [P2 - 闭环增强]
  □ 4.1 新增 iam/vlm_agent.py (VLMAgent 类)
  □ 4.2 Pipeline: 第一个 chunk 后触发 VLM 验证
  □ 4.3 属性冲突解决 + Registry 更新
  □ 4.4 use_vlm 开关 + 显存管理 (CPU/GPU swap)
  □ 4.5 测试: Mock VLM 验证属性漂移检测

Phase 5: Block-wise Sparse Attention [P2 - 加速]
  □ 5.1 帧级粗筛: mem frames K 均值池化 → top-k 帧
  □ 5.2 Token block 级细筛: block_size=64 → top-k_b blocks
  □ 5.3 集成到 CausalWanSelfAttention forward
  □ 5.4 与 TCAT / 动态帧数配合测试
  □ 5.5 正确性验证 + 性能对比
```

---

## 风险与注意事项

1. **显存预算**: 双层记忆最多 6 帧 (id=4 + scene=2)，比原来的 3 帧多一倍。需要确认 bank_size 扩大后显存是否够用。建议先在 1.3B 模型上验证。

2. **Scene 打分质量**: scene token 权重依赖 prompt 中 scene 描述的位置。如果 prompt 格式不规范（scene 和 entity 混在一起），权重可能不准。可以考虑用 LLM 提取的 scene_texts 重新编码为独立 query，而非从 crossattn_cache 中加权。

3. **VLM 延迟**: VLM 推理 + VAE decode 会增加每个 prompt 首 chunk 的延迟。需要评估是否值得。可以先作为离线验证工具，不阻塞生成流程。

4. **LLM 提取质量**: Qwen3-0.6B 对 scene 提取的准确度需要实测验证。如果不够好，可能需要换更大的模型或调整 prompt。
