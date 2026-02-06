# IAMFlow 三项改进实现方案

> 目标：扩大 IAMFlow 相对 MemFlow 的 VBench 涨点幅度

## 改进一览

| # | 改进 | 预期影响指标 | 改动范围 |
|---|------|-------------|---------|
| 1 | Adaptive SPT — 按语义距离动态调整过渡窗口 | imaging_quality, dynamic_degree | transition_scheduler.py, pipeline |
| 2 | Memory Bank 扩容（3→6帧） | subject_consistency, background_consistency | config, memory_bank.py, pipeline |
| 3 | 多层共识快速打分（等价重写 + 3层投票） | subject_consistency | memory_bank.py |

---

## 改进 1：Adaptive SPT — 按语义距离动态调整过渡窗口

### 问题

当前 SPT 对所有 prompt 切换使用固定参数（cosine, window=9帧, delay=3帧）：
- "走路→跑步"（小变化）：9帧过渡太长，混合期内运动方向模糊，压低 dynamic_degree
- "公园→雪山"（大变化）：9帧可能不够，前几帧出现混合伪影，压低 imaging_quality

### 方案

prompt 切换时，计算新旧 prompt 的 text embedding cosine distance，动态调整 `window_frames`。

### 1.1 新增 AdaptiveScheduler

**文件**: `utils/transition_scheduler.py`

```python
class AdaptiveScheduler:
    """根据语义距离动态选择窗口大小的调度器包装"""

    def __init__(self,
                 base_scheduler_type: str = "cosine",
                 min_window: int = 3,      # 最短 1 chunk
                 max_window: int = 15,     # 最长 5 chunks
                 delay_frames: int = 3,
                 **scheduler_kwargs):
        self.base_scheduler_type = base_scheduler_type
        self.min_window = min_window
        self.max_window = max_window
        self.delay_frames = delay_frames
        self.scheduler_kwargs = scheduler_kwargs
        self._active_scheduler: Optional[TransitionScheduler] = None

    def update_for_switch(self, semantic_distance: float):
        """
        prompt 切换时调用，根据语义距离创建对应窗口的 scheduler

        Args:
            semantic_distance: cosine distance (0=相同, 1=完全不同)
        """
        t = max(0.0, min(1.0, semantic_distance))
        window = int(self.min_window + t * (self.max_window - self.min_window))
        window = max(3, (window // 3) * 3)  # 对齐到 chunk 边界

        self._active_scheduler = create_scheduler(
            scheduler_type=self.base_scheduler_type,
            window_frames=window,
            delay_frames=self.delay_frames,
            **self.scheduler_kwargs
        )

    def get_alpha(self, frames_since_switch: int) -> Optional[float]:
        return self._active_scheduler.get_alpha(frames_since_switch)

    def is_complete(self, frames_since_switch: int) -> bool:
        return self._active_scheduler.is_complete(frames_since_switch)
```

### 1.2 计算语义距离

**文件**: `pipeline/interactive_causal_inference.py`

复用已有的 `cond_list`（`text_encoder` 输出），不需要额外编码：

```python
def _compute_prompt_distance(self, cond_old, cond_new) -> float:
    """计算两个 prompt embedding 的 cosine distance"""
    emb_old = cond_old["context"].mean(dim=(0, 1))  # [D]
    emb_new = cond_new["context"].mean(dim=(0, 1))  # [D]
    cos_sim = F.cosine_similarity(emb_old.unsqueeze(0), emb_new.unsqueeze(0))
    return (1.0 - cos_sim.item())  # 0=相同, 1=完全不同
```

计算成本：一次 mean + 一次 cosine_similarity，可忽略不计。

### 1.3 Pipeline 集成

**文件**: `pipeline/agent_causal_inference.py` (约 L264-L283)

在 prompt 切换检测处，先算距离再 soft_switch：

```python
# 现有代码 L264:
if next_switch_pos is not None and current_start_frame >= next_switch_pos:
    segment_idx += 1

    if self.spt_enabled:
        # 新增: 计算语义距离并更新 scheduler
        if hasattr(self.transition_scheduler, 'update_for_switch'):
            dist = self._compute_prompt_distance(
                cond_list[segment_idx - 1], cond_list[segment_idx]
            )
            self.transition_scheduler.update_for_switch(dist)
            print(f"[SPT-Adaptive] distance={dist:.3f}, "
                  f"window={self.transition_scheduler._active_scheduler.window_frames}")

        self._soft_switch()
```

### 1.4 配置

**文件**: `configs/agent_interactive_inference.yaml`

```yaml
spt_enabled: true
spt:
  scheduler_type: adaptive    # 新增类型
  base_scheduler: cosine      # 底层曲线
  min_window: 3               # 最短 1 chunk
  max_window: 15              # 最长 5 chunks
  delay_frames: 3
```

### 1.5 改动文件清单

| 文件 | 改动 |
|------|------|
| `utils/transition_scheduler.py` | 新增 `AdaptiveScheduler` 类 |
| `pipeline/interactive_causal_inference.py` | 新增 `_compute_prompt_distance()`，`__init__` 支持 adaptive |
| `pipeline/agent_causal_inference.py` | prompt 切换处调用 `update_for_switch` |
| `configs/agent_interactive_inference.yaml` | 新增 adaptive 配置项 |

---

## 改进 2：Memory Bank 扩容（3→6帧）

### 问题

当前 `max_memory_frames=3`，`bank_size=3`。LLM Agent 的实体感知选帧再精准，也只能保留 3 帧。当多个场景/实体交替出现时，旧场景的帧很快被覆盖，实体追踪的优势被容量瓶颈浪费。

### 方案

将 Memory Bank 从 3 帧扩到 6 帧，让 IAMFlow 能同时保留多个场景的关键帧。MemFlow 在 6 帧里盲选会塞入冗余帧，而 IAMFlow 能精确分配配额给不同实体——**容量越大，智能选帧 vs 盲选的差距越大**。

### 2.1 配置改动

**文件**: `configs/agent_interactive_inference.yaml`

```yaml
model_kwargs:
  bank_size: 6          # 3 → 6
  record_interval: 3    # 不变

max_memory_frames: 6    # 3 → 6
```

这两个值需要同步改：
- `bank_size`: 控制 kv_bank 的物理大小（pipeline 初始化时分配显存）
- `max_memory_frames`: 控制 MemoryBank 的逻辑容量（选帧时的上限）

### 2.2 显存影响

当前 kv_bank 大小 = `bank_size × frame_seq_length × 2(k,v) × 30(blocks)`

每帧 KV 占用（bf16）：`1560 tokens × 24 heads × 64 dim × 2 bytes = ~5.7MB/block`
30 blocks × 2(k,v) = ~342MB/帧

- bank_size=3: ~1.0 GB
- bank_size=6: ~2.0 GB（增加 ~1.0 GB）

H100 80GB 上可以接受。如果显存紧张，可配合 SMA 做稀疏 attention 来控制计算量（SMA 从 6 帧中选 top-3 参与 attention，计算量不变）。

### 2.3 驱逐策略调整

当前 `max_memory_frames=3` 时，驱逐逻辑比较简单——满了就替换分数最低的帧。扩到 6 帧后，建议增加**实体多样性约束**：

```python
def _select_frame_to_evict(self) -> str:
    """选择要驱逐的帧，优先驱逐实体冗余的帧"""
    if len(self.frame_active_memory) < self.max_memory_frames:
        return None  # 还没满，不需要驱逐

    # 统计每个实体被多少帧覆盖
    entity_coverage = {}  # entity_id → [frame_ids]
    for fid in self.frame_active_memory:
        info = self.frame_archive[fid]
        for eid in info.associated_entities:
            entity_coverage.setdefault(eid, []).append(fid)

    # 优先驱逐：所关联实体都有其他帧覆盖的、且分数最低的帧
    candidates = []
    for fid in self.frame_active_memory:
        info = self.frame_archive[fid]
        all_covered = all(
            len(entity_coverage.get(eid, [])) > 1
            for eid in info.associated_entities
        )
        candidates.append((fid, info.score, all_covered))

    # 先选冗余帧中分数最低的；如果没有冗余帧，选全局分数最低的
    redundant = [c for c in candidates if c[2]]
    if redundant:
        return min(redundant, key=lambda x: x[1])[0]
    else:
        return min(candidates, key=lambda x: x[1])[0]
```

这样 6 帧的 Memory Bank 能尽量覆盖不同实体，而不是被同一个实体的多帧占满。

### 2.4 改动文件清单

| 文件 | 改动 |
|------|------|
| `configs/agent_interactive_inference.yaml` | `bank_size: 6`, `max_memory_frames: 6` |
| `iam/memory_bank.py` | 新增 `_select_frame_to_evict()` 实体多样性驱逐 |
| pipeline 无需改动 | kv_bank 大小由 config 驱动，自动适配 |

---

## 改进 3：多层共识快速打分

### 问题

当前帧选择只用第 0 层（最浅层）的 KV 打分。第 0 层主要编码颜色/纹理等低级特征，容易出现"纹理相似但语义不同"的误选（比如两个穿同色衣服的不同人）。

### 方案

分两步：先把打分算法等价重写降低单层成本，再扩展到 3 层投票。

### 3.1 等价重写：从 O(512×L×D) 到 O(512×D + F×D)

当前打分流程（`_compute_frame_scores_with_entity_focus`，memory_bank.py:392-445）：

```
Step 1: attn = bmm(Q, K^T)           # [B*H, 512, L]  ← 大矩阵
Step 2: weighted = attn * weights     # [B*H, 512, L]
Step 3: sum over 512 text tokens      # [B*H, L]
Step 4: mean over heads               # [B, L]
Step 5: mean per frame                # [num_frames]
```

核心观察：Step 2-3 是"先乘权重再求和"，等价于"先用权重加权求和 Q，再点积"：

```
原始: sum_i( w_i * Q_i · K^T )  =  (sum_i(w_i * Q_i)) · K^T
```

等价重写：

```
Step 1: q_agg = weights @ Q           # [B*H, 1, D]   ← 512个向量加权合并成1个
Step 2: k_per_frame = mean(K, frame)  # [B*H, F, D]   ← L个token按帧分组取均值
Step 3: score = q_agg @ k_per_frame^T # [B*H, 1, F]   ← 直接得到每帧分数
Step 4: mean over heads               # [num_frames]
```

复杂度对比（B=1, H=24, D=64, 512 text tokens, L=4680 即 3帧×1560）：

| | 原始 | 重写 |
|---|------|------|
| 主要运算 | bmm [24, 512, 4680] | bmm [24, 1, 3] |
| 乘法次数 | 24×512×4680×64 ≈ 37亿 | 24×(512×64 + 1×3×64) ≈ 79万 |
| **加速比** | | **~4700x** |

### 3.2 快速打分实现

**文件**: `iam/memory_bank.py`

替换 `_compute_frame_scores_with_entity_focus`：

```python
def _compute_frame_scores_fast(self,
                                chunk_kv: Dict[str, torch.Tensor],
                                crossattn_cache_block: Dict[str, torch.Tensor],
                                entity_weights: torch.Tensor) -> torch.Tensor:
    """
    等价快速打分：先聚合Q，再与帧均值K点积

    Args:
        chunk_kv: {"k": [B, L, H, D]}
        crossattn_cache_block: {"k": [B, 512, H, D]}
        entity_weights: [512] 实体权重向量

    Returns:
        [num_frames] 每帧分数
    """
    chunk_k = chunk_kv["k"]  # [B, L, H, D]
    text_q = crossattn_cache_block["k"]  # [B, 512, H, D]
    B, L, H, D = chunk_k.shape
    num_frames = L // self.frame_seq_length

    # Step 1: 加权聚合 Q → [B, H, D]
    w = entity_weights.to(text_q.device)  # [512]
    w = w / (w.sum() + 1e-8)
    # text_q: [B, 512, H, D] → einsum → [B, H, D]
    q_agg = torch.einsum('bshd,s->bhd', text_q, w)

    # Step 2: K 按帧分组取均值 → [B, F, H, D]
    k_frames = chunk_k.view(B, num_frames, self.frame_seq_length, H, D)
    k_agg = k_frames.mean(dim=2)  # [B, F, H, D]

    # Step 3: 点积 → [B, H, F]
    scores = torch.einsum('bhd,bfhd->bhf', q_agg, k_agg) * (D ** -0.5)

    # Step 4: mean over heads → [F]
    scores = scores.mean(dim=(0, 1))  # [F]
    return scores
```

### 3.3 多层共识投票

用快速打分后，跑 3 层的成本比原来跑 1 层还低。选 3 个代表层：

| 层组 | 代表层 | 编码内容 | 权重 |
|------|--------|---------|------|
| 浅层 (0-9) | layer 0 | 颜色、纹理 | 0.2 |
| 中层 (10-19) | layer 15 | 物体形状、结构 | 0.3 |
| 深层 (20-29) | layer 29 | 语义（"这是一只狗"） | 0.5 |

深层权重最高，因为实体识别本质上是语义任务。

```python
# 配置
CONSENSUS_LAYERS = [0, 15, 29]
CONSENSUS_WEIGHTS = [0.2, 0.3, 0.5]

def select_frame_from_chunk(self, evicted_chunk_kv, crossattn_cache,
                            prompt_id, chunk_id, current_entity_ids,
                            current_entities=None, prompt_text=None):
    """多层共识版帧选择"""
    entity_weights = self._build_entity_token_weights(
        current_entities, 512, prompt_text
    )

    # 对每个代表层做快速打分
    all_scores = []
    for layer_idx, layer_weight in zip(CONSENSUS_LAYERS, CONSENSUS_WEIGHTS):
        scores = self._compute_frame_scores_fast(
            evicted_chunk_kv[layer_idx],
            crossattn_cache[layer_idx],
            entity_weights
        )
        all_scores.append(scores * layer_weight)

    # 加权求和
    final_scores = sum(all_scores)
    best_frame_idx = final_scores.argmax().item()
    # ... 后续逻辑不变
```

### 3.4 改动文件清单

| 文件 | 改动 |
|------|------|
| `iam/memory_bank.py` | 新增 `_compute_frame_scores_fast()`，修改 `select_frame_from_chunk()` 为多层共识 |
| 配置无需改动 | 层索引和权重可硬编码或加入 config |

---

## 实施顺序

建议按以下顺序实施，每步独立可测：

### 第一步：改进 3（多层共识打分）

**理由**：改动最小（只改 memory_bank.py），不影响其他模块，可以立刻跑 benchmark 验证。

验证方法：
- 对比单层 vs 3层共识的帧选择结果（打印 log 对比）
- 跑 VBench 看 subject_consistency 是否提升

### 第二步：改进 2（Memory Bank 扩容）

**理由**：只改配置 + 驱逐策略，与改进 3 叠加后，实体感知选帧的优势在更大池子里充分发挥。

验证方法：
- 对比 bank=3 vs bank=6 的 VBench 全指标
- 观察 dynamic_degree 是否下降过多（如果下降严重，说明需要配合 SMA 控制参与 attention 的帧数）

### 第三步：改进 1（Adaptive SPT）

**理由**：改动涉及 scheduler + pipeline 两层，依赖前两步的基线数据做对比。

验证方法：
- 打印每次切换的 semantic_distance 和分配的 window_frames
- 对比固定窗口 vs adaptive 的 imaging_quality 和 dynamic_degree

---

## 全部改动文件汇总

| 文件 | 改进 | 改动类型 |
|------|------|---------|
| `iam/memory_bank.py` | 2, 3 | 新增快速打分、多层共识、实体多样性驱逐 |
| `utils/transition_scheduler.py` | 1 | 新增 AdaptiveScheduler |
| `pipeline/interactive_causal_inference.py` | 1 | 新增 `_compute_prompt_distance()`，`__init__` 支持 adaptive |
| `pipeline/agent_causal_inference.py` | 1 | prompt 切换处调用 `update_for_switch` |
| `configs/agent_interactive_inference.yaml` | 1, 2 | adaptive SPT 配置、bank_size/max_memory_frames 扩容 |
