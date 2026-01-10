# IAM Flow - 推理模式使用指南

IAM Flow 提供两种视频生成模式：**单 Prompt 生成** 和 **交互式多 Prompt 生成**。

## 📌 模式对比

| 特性 | agent_inference | agent_interactive_inference |
|------|----------------|----------------------------|
| **Prompt 数量** | 单个 | 多个（支持切换） |
| **使用场景** | 单场景视频生成 | 复杂多场景视频 |
| **LLM Agent** | ✅ 使用（提取实体+属性） | ✅ 使用（实体提取+ID匹配+跨prompt跟踪） |
| **Memory Bank** | ✅ 使用（帧选择与驱逐） | ✅ 使用（完整的记忆管理） |
| **实体跟踪** | 单 prompt 内一致性 | 跨 prompt 一致性 |
| **帧检索** | ❌ 无需（没有历史prompt） | ✅ 从历史帧检索 |
| **Mapping 文件** | ❌ 默认不生成 | ✅ 生成 mapping.json |

### 💡 两种模式都使用 IAM 的核心能力

**共同点**：
- ✅ **LLM Agent** 会提取第一个 prompt 的实体和属性
- ✅ **Memory Bank** 从第 3 个 chunk 开始进行帧选择和驱逐
- ✅ 维护 KV Bank 用于记忆关键帧
- ✅ 保持实体在视频中的一致性

**主要区别**：
- **agent_inference**：只处理单个 prompt，不需要跨场景的实体匹配
- **agent_interactive_inference**：处理多个 prompt，需要跨场景匹配和跟踪实体（如"主角"在不同场景中的一致性）

---

## 🚀 快速开始

### 1️⃣ 单 Prompt 生成 (agent_inference)

**适用场景**: 从单个文本描述生成视频，类似 MemFlow 的原始功能。

```bash
# 运行推理
bash agent_inference.sh

# 或手动指定配置
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  agent_inference.py \
  --config_path configs/agent_inference.yaml
```

**配置文件**: `configs/agent_inference.yaml`
- `data_path`: 文本 prompt 文件路径（`.txt` 格式，每行一个 prompt）
- `output_folder`: 视频输出目录
- `num_output_frames`: 生成帧数（默认 120）

**示例**:
```bash
# 准备 prompts 文件
echo "A young man walking through a park at sunset" > prompts/my_prompt.txt

# 修改配置
# configs/agent_inference.yaml:
#   data_path: prompts/my_prompt.txt
#   output_folder: videos/my_output

# 运行
bash agent_inference.sh
```

---

### 2️⃣ 交互式多 Prompt 生成 (agent_interactive_inference)

**适用场景**: 生成包含多个场景切换的长视频，支持实体一致性跟踪。

```bash
# 运行推理
bash agent_interactive_inference.sh

# 或手动指定参数
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=29502 \
  agent_interactive_inference.py \
  --config_path configs/agent_interactive_inference.yaml \
  --llm_model_path ../Qwen3-0.6B \
  --max_memory_frames 3 \
  --save_dir data/agent_frames
```

**配置文件**: `configs/agent_interactive_inference.yaml`
- `data_path`: JSONL 格式的多 prompt 文件
- `switch_frame_indices`: 切换帧索引（如 `40, 80, 120, 160, 200`）
- `num_output_frames`: 总帧数（默认 240）
- `llm_model_path`: LLM 模型路径（用于实体提取）
- `max_memory_frames`: 最大记忆帧数

**示例**:
```bash
# 准备 JSONL prompts 文件
# prompts/interactive_example.jsonl:
# {"prompts": ["A young man in a park...", "The man walks to a bench...", "Another person approaches..."]}

# 运行
bash agent_interactive_inference.sh

# 输出
# - videos/iam_output/rank0-0-0_iam_lora.mp4
# - videos/iam_output/mapping_0.json (实体跟踪信息)
```

---

## 📁 配置文件说明

### 共享配置项
```yaml
# 模型架构
model_name: Wan2.1-T2V-1.3B
num_output_frames: 120  # 单 prompt: 120, 交互式: 240

# 检查点
generator_ckpt: checkpoints/base.pt
lora_ckpt: checkpoints/lora.pt

# LoRA 设置
adapter:
  type: "lora"
  rank: 256
  alpha: 256
```

### 交互式特有配置
```yaml
# 多 prompt 设置
switch_frame_indices: 40, 80, 120, 160, 200  # prompt 切换位置

# IAM Agent 设置
llm_model_path: ../Qwen3-0.6B        # LLM 模型路径
max_memory_frames: 3                 # 记忆帧数量
save_dir: data/agent_frames          # 帧数据保存目录
```

---

## 🔍 输出文件

### 单 Prompt 模式
```
videos/iam_single_prompt/
  ├── rank0-0-0_iam_lora.mp4  # 生成的视频
  └── rank0-1-0_iam_lora.mp4
```

### 交互式模式
```
videos/iam_output/
  ├── rank0-0-0_iam_lora.mp4     # 生成的视频
  ├── mapping_0.json              # 实体跟踪信息
  └── ...

data/agent_frames/
  ├── p1_c1_f0.pt                 # 保存的帧数据
  └── ...
```

**mapping.json 结构**:
```json
{
  "global_registry": {
    "1": {
      "name": "man_1",
      "all_entities": ["young man", "protagonist"],
      "all_attrs": ["late 20s", "denim jacket", ...],
      "instances": [...]
    }
  },
  "frame_archive": {...},
  "frame_active_memory": [...]
}
```

---

## 🛠️ 高级用法

### 自定义 prompt 数据

**单 Prompt 格式** (`.txt`):
```
A beautiful sunset over the ocean
A cat playing with a ball
```

**交互式格式** (`.jsonl`):
```json
{"prompts": ["Scene 1 description", "Scene 2 description", "Scene 3 description"]}
{"prompts": ["Another video scene 1", "Another video scene 2"]}
```

### 调整 GPU 数量
```bash
# 使用 4 个 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  agent_inference.py \
  --config_path configs/agent_inference.yaml
```

---

## 📖 详细流程说明

详细的技术流程和实现细节请参考：
- [document/pipeline.md](document/pipeline.md) - 完整流程详解
- [iam/MODIFICATION_GUIDE.md](iam/MODIFICATION_GUIDE.md) - IAM 模块使用指南

---

## ❓ 常见问题

**Q: 单 Prompt 模式真的使用了 IAM 的 LLM Agent 和 Memory Bank 吗？**
A: **是的！** 两种模式都完整使用 IAM 能力：
- **LLM Agent**：在第一个 prompt 时提取实体和属性（`_process_prompt_start`）
- **Memory Bank**：从第 3 个 chunk 开始进行帧选择和驱逐（`_process_chunk_eviction`）
- 主要区别在于单 prompt 模式不涉及跨场景的实体匹配和帧检索

**Q: 何时使用单 Prompt 模式？**
A: 当你只需要从一段文本生成视频，不涉及场景切换或实体跨场景跟踪时。即使是单个 prompt，IAM 仍会：
- 提取并跟踪实体（如"主角"、"背景物体"）
- 维护关键记忆帧以保持视频一致性

**Q: 何时使用交互式模式？**
A: 当你需要生成包含多个场景的长视频，且希望保持角色/物体在不同场景中的一致性时。例如：
- Scene 1: "A young man in a park"
- Scene 2: "The protagonist walks to a bench"（需要匹配 Scene 1 的主角）
- Scene 3: "Another person approaches him"（需要跟踪两个人）

**Q: 两种模式的性能差异？**
A: 单 Prompt 模式略快，因为：
- 不需要跨 prompt 的实体匹配（节省 LLM 推理时间）
- 不需要从历史帧检索（跳过帧检索步骤）
- 但两者都使用完整的 Memory Bank 帧选择机制

**Q: 可以在单 Prompt 模式下禁用 IAM 功能吗？**
A: 如果不需要 IAM 的实体跟踪和记忆管理功能，建议直接使用 MemFlow 的原始 `inference.py`。但保留 IAM 能力即使对单 prompt 也有益处，可以提高生成视频的一致性。
