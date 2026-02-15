
## 论文设计总览

本项目围绕**流式交互式长视频生成**，在 LongLive 和 MemFlow 的基础上提出系统性改进。核心 insight：**实体一致性和场景一致性是两种本质不同的需求，需要解耦的检索信号和独立的记忆层分别维护。**

### 三大贡献

1. **分层记忆机制**：LLM+VLM 驱动的显式实体 ID 管理 + ID Memory / Scene Memory 双层记忆 + 动态记忆帧分配
2. **加速方法**：Soft Prompt Transition、快速关键帧提取、Block-wise Sparse Attention、量化、LightVAE
3. **Benchmark**：面向流式交互长视频的 600 prompt 评测体系，涵盖 Quality / Temporal Consistency / Instruction Compliance

### 系统流程概览

```
Prompt 输入
    │
    ▼
┌─────────────────────────────────┐
│  LLM Agent（每个 prompt 触发）     │
│  · 提取 entity & attrs          │
│  · 提取 scene text              │
│  · 为 entity 分配/复用 Global ID  │
│    （scene 无需 ID）              │
│  · 更新 Global Registry          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Prompt 切换处理                  │
│  · Soft Prompt Transition       │
│                                 │
│  Entity 路径（ID 驱动）：          │
│  · 基于 Global ID 检索           │
│    Frame Archive → ID Memory    │
│  · 动态调整帧数（ID 覆盖度）       │
│                                 │
│  Scene 路径（相似度驱动）：         │
│  · 计算前后 scene text 语义距离    │
│  · 距离 > 阈值 → 重新检索         │
│    Frame Archive → Scene Memory │
│  · 距离 ≤ 阈值 → 保持不动，零开销  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  逐 Chunk 生成                    │
│  KV cache = sink + ID mem       │
│            + scene mem + local   │
│  · Block-wise Sparse Attention  │
│    对 mem frames 施加             │
└──────────────┬──────────────────┘
               │
               ▼ （第一个 chunk 生成后）
┌─────────────────────────────────┐
│  VLM 视觉验证                     │
│  · 从生成帧提取实际 entity & attrs │
│  · 与 LLM 提取结果校验/更新        │
│  · 冲突时保留最新 attrs            │
└──────────────┬──────────────────┘
               │
               ▼ （每个 chunk 生成后）
┌─────────────────────────────────┐
│  Memory Bank 更新                 │
│  · 关键帧提取（加速版均值池化）      │
│  · 存入 Frame Archive            │
│  · 更新 Active Memory 槽         │
└─────────────────────────────────┘
```

---

## 一、Baseline 回顾

### LongLive（2510）

LongLive 采用短窗口注意力机制，KV cache 由 3 帧 sink 与过去 9 帧组成：

- **Frame Sink**：固定首个 chunk 的 3 帧 token，加入 KV cache
- **Short Window**：短窗口注意力，仅关注局部帧
- **KV Recache**：每次 prompt 切换时，将上一段生成的 token 和新 prompt 重新做一次前向过程，得到新的 KV cache

### MemFlow（2512）

MemFlow 在 LongLive 基础上引入动态记忆机制，维护 3 帧 token 的记忆库：

- **NAM（叙事自适应记忆）**：每个 chunk 生成时，根据各帧与当前 prompt 的交叉注意力提取关键帧，动态更新记忆库
- **SMA（稀疏记忆激活）**：将 sink 和 bank 拼接后，计算当前 query 与各帧的点积，取 top-3 帧参与注意力计算；最终 KV cache = top-3 + 9 帧 local window

---

## 二、核心创新：分层记忆机制

### Key Insight

现有方法（包括 MemFlow）用单一的交叉注意力同时服务实体一致性和场景一致性，但 prompt "一个男人走在雪地森林中" 的注意力大概率被"男人"主导，"雪地森林"的信号被淹没。

本项目的核心洞察：**实体一致性和场景一致性是两种本质不同的需求，需要解耦的检索信号和独立的记忆层分别维护。** 通过 LLM+VLM 显式建模实体身份，为每个实体分配全局唯一 ID，并将记忆分为 ID Memory（管"谁"）和 Scene Memory（管"在哪"），各自用不同的文本 query 独立检索。

### 2.1 LLM + VLM Agent

#### LLM Agent（每个 prompt 触发一次，约每 10s）

1. 从 prompt 中提取 entity & attributes，根据 Global Registry 分配/复用全局 ID
2. 从 prompt 中提取 scene text（**无需分配 ID**，纯文本即可）
3. 更新 Global Registry（仅管理 entity）

**为什么 entity 需要 ID 而 scene 不需要？**
- Entity 存在**指代消歧**问题：同一个人在不同 prompt 中可能叫 "young man"、"protagonist"、"he"，文本相似度无法匹配，必须靠 LLM 判断身份并分配 ID
- Scene 不存在这个问题："snowy forest" 就是 "snowy forest"，场景本质上是**连续的视觉属性**而非离散身份，文本相似度天然可用

```json
{
  "entities": [
    {"id": 1, "entity": "young man", "attrs": ["messy black hair", "denim jacket"]}
  ],
  "scene": ["snowy forest", "overcast daylight", "wooden bench", "pine trees"]
}
```

#### VLM 视觉验证（每个 prompt 的第一个 chunk 生成后触发）

- 从实际生成帧中提取 entity & attrs，与 LLM 提取结果校验
- 将视觉验证结果更新到 Global Registry
- **属性冲突解决**：相同全局 ID 的 attrs 存在冲突时，保留最新的（流式交互中 prompt 修改即意味着属性更新）

VLM 验证将记忆管理从"基于文本期望"升级为"基于视觉事实"，形成闭环。

### 2.2 分层记忆：ID Memory + Scene Memory

| 维度        | ID Memory                                               | Scene Memory                                   |
| --------- | ------------------------------------------------------- | ---------------------------------------------- |
| 管什么       | 实体外貌一致性                                                 | 场景/背景/光照一致性                                    |
| 是否需要全局 ID | ✅ 需要（解决指代消歧）                                            | ❌ 不需要（文本相似度天然可用）                               |
| 检索机制      | Global ID → 定位关联帧 → entity text 打分                      | scene text 直接对 Frame Archive 打分                |
| 检索 query  | 实体 entity & attrs 文本 token                              | 场景 scene text 文本 token                         |
| 解决的问题     | identity drift, character overlap, duplicated instances | background drift, lighting shift, layout drift |
| 更新触发      | 每个 chunk（基于 ID 覆盖度）                                     | 仅当场景变化时（语义距离 > 阈值）                             |

#### ID Memory 检索

$$S^l_{\text{id}} = \text{Aggregate}\left(\text{Softmax}\left(\frac{Q^l_{\text{entity}}(K^l_{m,i})^\top}{\sqrt{d}}\right)\right)$$

$Q^l_{\text{entity}}$ = 实体文本 token（"young man, messy black hair, denim jacket"）

#### Scene Memory 检索

$$S^l_{\text{scene}} = \text{Aggregate}\left(\text{Softmax}\left(\frac{Q^l_{\text{scene}}(K^l_{m,i})^\top}{\sqrt{d}}\right)\right)$$

$Q^l_{\text{scene}}$ = 场景文本 token（"snowy forest, overcast daylight, wooden bench"）

Scene Memory 不经过任何 ID 映射，直接用当前 prompt 的 scene text 作为 query 对 Frame Archive 所有帧的 KV 打分。这种纯相似度检索天然支持**场景重入**：视频经历 森林→城市→森林 时，第三段的 scene query 会自动从 Frame Archive 中捞回第一段的森林帧，无需手动记录"这是同一个场景"。

**同场景跳过优化**：Prompt 切换时，先计算前后 scene text 的语义距离。若距离 ≤ 阈值（场景未变），scene_memory 保持不动，跳过检索，零开销；若距离 > 阈值（场景切换），才触发重新检索。

两者独立打分，各自选 top-k 帧，分别填入 ID Memory 和 Scene Memory 槽。同一历史帧可能同时被两层选中。

### 2.3 动态记忆帧分配

不再使用 MemFlow 的固定 3 帧记忆，而是根据 **ID 覆盖度**动态调整 ID Memory 的帧数：

- 从 prompt 提取出 Global ID 后，检查当前 Frame Archive 中对这些 ID 的覆盖情况
- 实体多且覆盖不足 → 多分配 ID mem 帧
- 实体少且覆盖充分 → 少分配 ID mem 帧

可以是优先选择覆盖 prompt 中 所有id的那一帧，如果不行的话，选择不包含其他 id，且覆盖所需id最多的、分数最高的；直到选的帧一共能包含所有 id；
或者定义$$C=\frac{Frame覆盖的 ID}{prompt需要的 ID}$$
这直接回应了 MemFlow 消融实验中"b=3 最优但 b=6/9 反而差"的问题——不是固定大小的问题，而是应该根据需求动态调整。

### 2.4 KV Cache 结构

```
KV cache = sink + ID mem + scene mem + local
           (2)    (动态1~4)  (1~2)      (6)
           ─────────────────────────────────
                        总计 ~12 帧
```

| 组件 | 帧数 | 说明 |
|------|------|------|
| Sink | 2 | 首 chunk 锚定帧 |
| ID Memory | 动态 1~4 | 基于 ID 覆盖度，实体多则多分配 |
| Scene Memory | 1~2 | 场景变化少时 1 帧，场景复杂时 2 帧 |
| Local | 6 | 短窗口，保持时序连贯 |

总帧数预算与原来的 3+3+6 相当，但分配更智能。

### 2.5 数据库结构

#### Global Registry（全局注册表）

```json
{
  "1": {
    "name": "man_1",
    "all_entities": ["young man", "protagonist"],
    "all_attrs": ["late 20s", "messy black hair", "denim jacket"],
    "instances": [
      {"prompt_id": 1, "entity": "young man", "attrs": ["late 20s", "messy black hair", "denim jacket"]},
      {"prompt_id": 2, "entity": "protagonist", "attrs": ["denim jacket", "seated on bench"]}
    ]
  }
}
```

#### Frame Archive（帧存档）

存储被提取的关键帧，同时标注实体关联和场景关联：

```json
{
  "p1_c3_f0": {
    "prompt_id": 1,
    "associated_entities": ["1"],
    "entity_score": 0.91,
    "scene_score": 0.75
  }
}
```

#### Frame Active Memory（活跃记忆槽）

```json
{
  "id_memory": ["p2_c5_f1", "p1_c3_f0"],
  "scene_memory": ["p3_c4_f0"]
}
```

### 2.6 Memory Bank 更新流程

**每次 Prompt 切换时：**
1. LLM Agent 提取 entity & attrs（分配 Global ID）+ scene text（无 ID）
2. **Entity 路径**：基于 Global ID 从 Frame Archive 检索 → 更新 ID Memory 槽（动态帧数）
3. **Scene 路径**：计算前后 scene text 语义距离
   - 距离 > 阈值 → 用 scene text 直接检索 Frame Archive → 更新 Scene Memory 槽
   - 距离 ≤ 阈值 → Scene Memory 保持不动，跳过检索
4. Soft Prompt Transition 平滑过渡 KV cache

**第一个 Chunk 生成后：**
- VLM 视觉验证，校验/更新 Global Registry 中的 entity & attrs

**每个 Chunk 生成后：**
- 关键帧提取（加速版均值池化），同时计算 entity_score 和 scene_score
- 存入 Frame Archive，更新 Active Memory 槽

---

## 三、加速方法

### 1. Soft Prompt Transition（软提示过渡）

解决 prompt 切换时 KV cache 突变的问题：

- 计算前后 prompt 编码后的**语义距离**，决定过渡窗口长度
- 保存旧的 KV cache，在过渡窗口中逐步混合新旧 KV cache：

$$K_{\text{mixed}} = (1-\alpha) \times K_{\text{old}} + \alpha \times K_{\text{new}}$$

这种软过渡避免了 LongLive 中 KV Recache 的完整前向重计算，同时保证了 prompt 切换时的视觉连贯性。

### 2. 快速关键帧提取

**原始 MemFlow 方法：**
- 取第 0 个 block 的 KV
- Q 为完整 prompt token `[512, H, D]`，K 为三帧 token `[4680, H, D]`
- 计算点积 `[512, 4680]`，再聚合为分数 `[1, 3]`

**加速方法：**
- 取第 0、15、29 个 block 的 KV（多层采样）
- 对 Q 和每帧对应的 token 分别取均值后再点积
- Q_mean: `[1, H, D]`，K_mean: `[3, H, D]`，点积 `[1, 3]`
- 计算量大幅降低

### 3. Block-wise Sparse Attention（对 mem frames）

借鉴 Light Forcing 的 HSA（Hierarchical Sparse Attention），对 ID Memory + Scene Memory 帧施加 block-wise 稀疏注意力：

1. **帧级粗筛**：将 mem frames 的 key 做均值池化得到帧级表示，与当前 chunk 的 query 计算相似度，选出 top-k 帧
2. **Token 级细筛**：在选出的帧内，将 token 组织为固定大小的 block，计算 block 级相似度，选出 top-k_b 个最相关的 block
3. 仅对选中的 block 执行精确注意力计算

当动态记忆帧数增大时（如实体多、ID 覆盖不足），sparse attention 控制计算开销不随记忆规模线性增长。

> 注：原 MemFlow 的 SMA 是从 sink+bank 中取 top-3 帧。本方案用 block-wise sparse 替代，粒度更细（token block 级而非帧级），且与动态帧数机制配合更好。

### 4. 量化

- **线性层量化**：参考 TurboDiffusion 的 W8A8 PTQ 量化
- **KV Cache 量化**：考虑 FP8 或 INT8 的 PTQ 量化

### 5. 其他加速手段

- 轻量化 VAE：LightVAE
- 效仿 Block Cascading 的多卡并行加速
- 每生成一个 chunk 即进行 VAE 解码，节约 VAE 解码时间
- 使用 vLLM 框架：原生 Transformer 时间 35s → vLLM 时间 3～4s


---

## 四、Benchmark 设计

### 设计目标

针对流式交互长视频的特点——每段 prompt 都会在前一段 prompt 上做一定修改——设计更全面的评测体系。参考 VBench-Long，计划覆盖 300-600 条测试数据。

### 关注的典型问题

- **Duplicated Instances**：画面中出现多余的重复角色
- **Character Overlap**：两个角色融合在一起
- **Identity Attribute Drift**：属性漂移（服装颜色变化、面部特征改变）、身份交换（男性变成女性外貌）
- 镜头、背景、光照等其他维度

### 评测维度设计

#### Quality（画质）

- subject_consistency
- background_consistency
- temporal_flickering
- motion_smoothness
- VTSS
- **长视频融合策略**：VDE Decay（画质衰减率）

#### Temporal Consistency（时序一致性）

- **边界平滑一致性**：第 $i$ 段最后帧和第 $i+1$ 段开始帧的光流变化
  - 融合策略：均值
- **条件相邻一致性**：利用 MLLM 判断 prompt 是否要求换人/换场景，对未切换部分计算第 $i$ 段和第 $i+1$ 段的特征相似度
  - 融合策略：均值
- **条件长程一致性**：利用 MLLM 判断 prompt 中的同一主体，计算该主体在不同段视频中的相似度
  - 融合策略：逆序加权

#### Instruction Compliance（指令遵循）

- **分段语义对齐**：各段视频和 prompt 的 CLIP Score
  - 融合策略：均值
- **动态轨迹对齐**：第 $i+1$ 段与第 $i$ 段视频的向量差值，和 prompt 向量差值之间的余弦相似度
  - 融合策略：均值
- **VLM 评分**：输入完整视频和指令列表，让 GPT-4o / Qwen2.5-VL 判断"视频是否在正确的时间点执行了正确的动作"

### 与现有 Benchmark 的关系

VDE 和 VBench-Long 的共同点：都在短视频打分基础上增加了长视频融合方法。
- VDE：根据分数相对变化率，线性衰减加权求和
- VBench-Long：抽帧组成新视频，通过分位数加权求和

本项目的 Benchmark 在此基础上，进一步引入了条件一致性（区分有意切换和无意漂移）和指令遵循维度，更贴合流式交互场景的评测需求。

---

## 五、验证流程

### 1. Prompt 扩展

- 编写 `configs/prompts_12_quick.yaml`
- 运行 `python batch_generator.py --config configs/prompts_12_quick.yaml`
- 在 `outputs/batch_{时间戳}` 下生成 ghost / overlap / attrs / identity 四类测试数据，以及按顺序排序的完整 prompt

### 2. 视频生成

使用模型生成测试视频。

### 3. VBench-Long 评测

- 修改 `videos_path`
- 运行 `bash vbench2_beta_long/evaluate_long_custom.sh`

---

## 总结

本项目在 MemFlow 的动态记忆机制基础上，提出了三个层面的系统性改进：

**记忆层面**：从隐式视觉相似度匹配升级为 LLM 驱动的显式实体 ID 管理。LLM Agent 为每个 prompt 提取实体及属性并分配全局 ID，使记忆系统能够跨 prompt 追踪同一实体的身份，解决了角色重复、身份漂移等问题。关键设计选择：Entity 采用 ID 系统（因为存在指代消歧需求，同一人可能被称为 "young man"、"protagonist"、"he"），Scene 采用纯相似度检索（因为场景是连续视觉属性，文本相似度天然可用，且同场景连续出现时可跳过检索实现零开销）。记忆库由 Global Registry（仅管理 entity）、Frame Archive、Frame Active Memory 三层结构组成，实现了从全局注册到关键帧存档再到活跃记忆激活的完整链路。

**加速层面**：通过 Soft Prompt Transition 替代 KV Recache 的完整前向重计算，用语义距离自适应控制过渡窗口；通过多层采样和均值池化大幅降低关键帧提取的计算量；结合 vLLM 框架实现约 10 倍的推理加速。量化方案（W8A8 线性层、FP8/INT8 KV Cache）作为后续优化方向。

**评测层面**：针对流式交互长视频的特点，设计了涵盖 Quality、Temporal Consistency、Instruction Compliance 三大维度的评测体系，引入条件一致性（区分有意切换与无意漂移）和动态轨迹对齐等新指标，弥补了现有 Benchmark 对交互式场景评测能力的不足。
