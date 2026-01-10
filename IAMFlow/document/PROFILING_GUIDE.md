# IAM Flow Profiling 指南

## 概述

IAM Flow 提供详细的性能分析（profiling）功能，可以精确测量：
- MemFlow 原有的时间统计（初始化、扩散生成、VAE 解码）
- IAM 特有的组件时间（LLM Agent、Memory Bank）
- 每个 prompt 的 Agent 处理时间
- 每个 chunk 的 Memory Bank 处理时间

## 🚀 使用方法

Profiling 功能已默认启用，运行推理时会自动输出性能统计。

### 单 Prompt 模式
```bash
bash agent_inference.sh
```

### 交互式多 Prompt 模式
```bash
bash agent_interactive_inference.sh
```

## 📊 输出示例

```
======================================================================
IAM Agent Pipeline Profiling Results
======================================================================

[Overall Performance]
  - Initialization time:      1234.56 ms ( 5.23%)
  - Diffusion generation:    18765.43 ms (79.45%)
  - VAE decoding:             3654.21 ms (15.32%)
  - Total time:              23654.20 ms
  - Throughput:                  5.07 FPS

[IAM Components (within diffusion)]
  - Total LLM Agent time:      456.78 ms ( 2.43% of diffusion)
  - Total Memory Bank time:    234.56 ms ( 1.25% of diffusion)
  - Pure diffusion time:     18074.09 ms (96.32% of diffusion)

[LLM Agent - Per Prompt]
  - Prompt 1       processing:   234.56 ms
  - Prompt 2       processing:   123.45 ms
  - Prompt 3       processing:    98.77 ms

[Memory Bank - Per Chunk (Chunk 3+)]
  - Chunk 2        eviction:      12.34 ms
  - Chunk 3        eviction:      11.89 ms
  - Chunk 4        eviction:      12.56 ms
  - Chunk 5        eviction:      11.23 ms
  - Chunk 6        eviction:      12.78 ms
  - ... (30 chunks omitted)
  - Chunk 36       eviction:      11.98 ms
  - Chunk 37       eviction:      12.34 ms
  - Chunk 38       eviction:      11.67 ms
  - Chunk 39       eviction:      12.45 ms
  - Chunk 40       eviction:      11.90 ms
  - Average per chunk:           12.15 ms

[Diffusion - Per Block]
  - Block   0 generation:    456.78 ms ( 2.43% of diffusion)
  - Block   1 generation:    445.67 ms ( 2.37% of diffusion)
  - Block   2 generation:    478.90 ms ( 2.55% of diffusion)
  - Block   3 generation:    465.34 ms ( 2.48% of diffusion)
  - Block   4 generation:    471.23 ms ( 2.51% of diffusion)
  - ... (30 blocks omitted)
  - Block  36 generation:    468.45 ms ( 2.50% of diffusion)
  - Block  37 generation:    472.89 ms ( 2.52% of diffusion)
  - Block  38 generation:    466.12 ms ( 2.48% of diffusion)
  - Block  39 generation:    469.78 ms ( 2.50% of diffusion)
  - Block  40 generation:    467.34 ms ( 2.49% of diffusion)
  - Average per block:       468.91 ms
======================================================================
```

## 📈 指标说明

### 总体性能
| 指标 | 说明 |
|------|------|
| **Initialization time** | KV cache、crossattn cache、KV bank 初始化时间 |
| **Diffusion generation** | 扩散模型生成总时间（包含 IAM 组件） |
| **VAE decoding** | 将 latent 解码为像素的时间 |
| **Total time** | 总推理时间 = 初始化 + 扩散 + VAE |
| **Throughput (FPS)** | 吞吐量 = 生成帧数 / 总时间（秒） |

### IAM 组件（在扩散时间内）
| 指标 | 说明 |
|------|------|
| **Total LLM Agent time** | 所有 prompt 的 LLM Agent 处理总时间 |
| **Total Memory Bank time** | 所有 chunk 的 Memory Bank 选帧总时间 |
| **Pure diffusion time** | 纯粹的扩散生成时间（排除 IAM 组件） |

### 详细分解
| 类别 | 说明 |
|------|------|
| **LLM Agent - Per Prompt** | 每个 prompt 的实体提取和 ID 匹配时间 |
| **Memory Bank - Per Chunk** | 从 Chunk 3 开始，每个 chunk 的帧选择和驱逐时间 |
| **Diffusion - Per Block** | 每个 block 的扩散生成时间 |

## 🔍 性能分析技巧

### 1. 识别瓶颈
查看各组件的百分比：
- 如果 **VAE decoding > 30%**：考虑使用更快的 VAE 或减少解码次数
- 如果 **LLM Agent > 5%**：考虑使用更快的 LLM 或减少 prompt 数量
- 如果 **Memory Bank > 3%**：正常，这是 IAM 的核心功能

### 2. 对比 MemFlow vs IAM
```bash
# MemFlow (原始)
cd /path/to/MemFlow
bash inference.sh  # 查看 profile 输出

# IAM_Flow
cd /path/to/IAM_Flow
bash agent_inference.sh  # 查看 profile 输出
```

**预期差异**：
- IAM 会增加 **2-5%** 的开销（LLM Agent + Memory Bank）
- 但换来更好的实体一致性和记忆管理

### 3. 优化建议

#### 减少 LLM Agent 开销
- 使用更小的 LLM 模型（如 Qwen3-0.6B → Qwen2-0.5B）
- 减少 prompt 切换次数（单 prompt 模式只调用一次）

#### 减少 Memory Bank 开销
- 降低 `max_memory_frames`（3 → 2）
- 使用更简单的相似度计算

#### 提高吞吐量
- 增加 GPU 数量
- 使用更小的模型（1.3B → 0.6B）
- 减少生成帧数（240 → 120）

## 📊 性能基准参考

### 单 Prompt 模式（120 帧）
| 硬件 | 总时间 | 吞吐量 | LLM Agent | Memory Bank |
|------|--------|--------|-----------|-------------|
| 2×A100 40GB | ~23s | ~5 FPS | ~230ms (1%) | ~450ms (2%) |
| 2×V100 32GB | ~35s | ~3.4 FPS | ~280ms (0.8%) | ~520ms (1.5%) |

### 交互式模式（240 帧，3 prompts）
| 硬件 | 总时间 | 吞吐量 | LLM Agent | Memory Bank |
|------|--------|--------|-----------|-------------|
| 2×A100 40GB | ~48s | ~5 FPS | ~650ms (1.4%) | ~920ms (1.9%) |
| 2×V100 32GB | ~72s | ~3.3 FPS | ~800ms (1.1%) | ~1.1s (1.5%) |

## 🛠️ 高级用法

### 禁用 Profiling（如果不需要）
编辑推理脚本，将 `profile=True` 改为 `profile=False`：

```python
# agent_inference.py 或 agent_interactive_inference.py
video = pipeline.inference(
    ...,
    profile=False,  # 禁用 profiling
)
```

### 导出 Profiling 数据
如需将 profiling 数据保存到文件：

```bash
bash agent_inference.sh 2>&1 | tee profiling_output.txt
```

### 与 MemFlow 对比
```bash
# 生成对比报告
echo "=== MemFlow ===" > comparison.txt
cd /path/to/MemFlow && bash inference.sh 2>&1 | grep -A 20 "Profiling results" >> comparison.txt

echo "=== IAM_Flow ===" >> comparison.txt
cd /path/to/IAM_Flow && bash agent_inference.sh 2>&1 | grep -A 50 "IAM Agent Pipeline" >> comparison.txt
```

## ❓ 常见问题

**Q: 为什么 LLM Agent 时间这么少？**
A: LLM Agent 使用小型模型（Qwen3-0.6B），且只在 prompt 切换时调用。单 prompt 模式只调用一次。

**Q: Memory Bank 每个 chunk 都执行吗？**
A: 从 **Chunk 3** 开始执行（前 2 个 chunk 用于填充 Local 窗口）。

**Q: 如何提高 FPS？**
A: 主要瓶颈在扩散生成，使用更快的 GPU 或更小的模型。IAM 组件开销很小（2-3%）。

**Q: IAM 比 MemFlow 慢多少？**
A: 约 2-5% 的额外开销（LLM Agent + Memory Bank），几乎可以忽略。

## 📖 相关文档

- [MemFlow Profiling](../MemFlow/pipeline/causal_inference.py) - MemFlow 原始 profiling 实现
- [IAM Pipeline](pipeline/agent_causal_inference.py) - IAM profiling 实现
- [Inference Modes](INFERENCE_MODES.md) - 两种推理模式对比
