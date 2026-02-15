# IAMFlow Linear INT8 Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `exp_/IAMFlow` 中实现与 TurboDiffusion 同风格的线性层 INT8 量化（离线量化权重 + 在线量化激活），用于推理显存降低与速度提升。

**Architecture:** 复用 TurboDiffusion 的 `Int8Linear` 思路，只替换 `generator.model.blocks` 内的 `nn.Linear`，避免影响 `head/text_embedding/time_embedding` 等非主干层。实现两条路径：1) 离线将 checkpoint 转换为 quant checkpoint；2) 推理时识别 quant checkpoint 并按正确顺序替换模块后加载。新增统一量化加载 helper，复用到四个推理入口脚本。

**Tech Stack:** PyTorch 2.9+, CUDA, TurboDiffusion quant/gemm kernels（或兼容封装）、OmegaConf、pytest。

### Task 1: 建立可复用量化模块（仅替换 blocks 内 Linear）

**Files:**
- Create: `exp_/IAMFlow/utils/quant_linear.py`
- Create: `exp_/IAMFlow/tests/test_quant_linear_replace.py`

**Step 1: Write the failing test**

```python
import torch
import torch.nn as nn

from utils.quant_linear import collect_replace_targets

class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.head = nn.Linear(16, 8)


def test_collect_targets_only_under_blocks():
    model = ToyModel()
    names = collect_replace_targets(model, scope_attr="blocks")
    assert all(name.startswith("blocks") for name in names)
    assert not any(name.startswith("head") for name in names)
```

**Step 2: Run test to verify it fails**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py::test_collect_targets_only_under_blocks -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.quant_linear'`

**Step 3: Write minimal implementation**

```python
# utils/quant_linear.py
import torch
import torch.nn as nn

def collect_replace_targets(model: nn.Module, scope_attr: str = "blocks", skip_keywords=()):
    scope = getattr(model, scope_attr)
    targets = []
    for name, module in scope.named_modules():
        if isinstance(module, nn.Linear) and not any(k in name for k in skip_keywords):
            full_name = f"{scope_attr}.{name}" if name else scope_attr
            targets.append(full_name)
    return targets
```

**Step 4: Run test to verify it passes**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py::test_collect_targets_only_under_blocks -v`
Expected: PASS

**Step 5: Commit**

```bash
git add exp_/IAMFlow/utils/quant_linear.py exp_/IAMFlow/tests/test_quant_linear_replace.py
git commit -m "feat: add linear replacement target collector for quantization"
```

### Task 2: 接入 Int8Linear 并处理 dtype 陷阱

**Files:**
- Modify: `exp_/IAMFlow/utils/quant_linear.py`
- Modify: `exp_/IAMFlow/tests/test_quant_linear_replace.py`

**Step 1: Write the failing test**

```python
def test_int8_scale_kept_float32_after_to_dtype():
    from utils.quant_linear import QuantLinearAdapter
    layer = QuantLinearAdapter.from_linear(nn.Linear(16, 16), quantize=False)
    layer = layer.to(dtype=torch.bfloat16)
    assert layer.scale.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py::test_int8_scale_kept_float32_after_to_dtype -v`
Expected: FAIL because `scale` is cast to `torch.bfloat16`

**Step 3: Write minimal implementation**

```python
class QuantLinearAdapter(Int8Linear):
    # Int8Linear 可来自 turbodiffusion.ops 或本地实现
    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "scale") and self.scale is not None:
            self.scale.data = self.scale.data.float()
        return self
```

并增加：

```python
def replace_linear_with_int8(model, scope_attr="blocks", quantize=True, skip_keywords=()):
    # 遍历 scope.named_modules()，把 nn.Linear 替换为 QuantLinearAdapter.from_linear(...)
    ...
```

**Step 4: Run test to verify it passes**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add exp_/IAMFlow/utils/quant_linear.py exp_/IAMFlow/tests/test_quant_linear_replace.py
git commit -m "feat: add int8 linear adapter and dtype-safe scale handling"
```

### Task 3: 新增 checkpoint 量化转换脚本

**Files:**
- Create: `exp_/IAMFlow/scripts/quantize_generator_ckpt.py`
- Create: `exp_/IAMFlow/tests/test_quant_ckpt_convert.py`

**Step 1: Write the failing test**

```python
import torch

def test_convert_checkpoint_preserves_top_keys(tmp_path):
    src = tmp_path / "src.pt"
    dst = tmp_path / "dst.pt"
    torch.save({"generator": {"model.blocks.0.self_attn.q.weight": torch.randn(4,4)}}, src)
    from scripts.quantize_generator_ckpt import convert_checkpoint
    out = convert_checkpoint(str(src), str(dst), checkpoint_key="generator")
    assert "generator" in out
    assert any("int8_weight" in k for k in out["generator"].keys())
```

**Step 2: Run test to verify it fails**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_ckpt_convert.py::test_convert_checkpoint_preserves_top_keys -v`
Expected: FAIL with missing converter script/function

**Step 3: Write minimal implementation**

```python
# scripts/quantize_generator_ckpt.py
# 1) 构建 CausalInferencePipeline
# 2) 加载原 fp checkpoint 到 pipeline.generator
# 3) 调用 replace_linear_with_int8(..., quantize=True)
# 4) 回写到 state_dict[checkpoint_key] 并保存
```

CLI 样例：

```bash
python scripts/quantize_generator_ckpt.py \
  --input checkpoints/base.pt \
  --output checkpoints/base-int8.pt \
  --checkpoint_key generator \
  --scope_attr blocks
```

**Step 4: Run test to verify it passes**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_ckpt_convert.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add exp_/IAMFlow/scripts/quantize_generator_ckpt.py exp_/IAMFlow/tests/test_quant_ckpt_convert.py
git commit -m "feat: add generator checkpoint int8 quantization converter"
```

### Task 4: 统一推理入口量化加载顺序（4个入口）

**Files:**
- Modify: `exp_/IAMFlow/inference.py`
- Modify: `exp_/IAMFlow/interactive_inference.py`
- Modify: `exp_/IAMFlow/agent_inference.py`
- Modify: `exp_/IAMFlow/agent_interactive_inference.py`
- Modify: `exp_/IAMFlow/utils/quant_linear.py`

**Step 1: Write the failing test**

```python
def test_quantized_ckpt_load_order():
    # 伪测试：确保顺序为
    # (1) replace_linear_with_int8(quantize=False)
    # (2) load_state_dict(quant ckpt)
    # (3) pipeline.to(dtype=torch.bfloat16) 后 scale 仍 float32
    assert False
```

**Step 2: Run test to verify it fails**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py::test_quantized_ckpt_load_order -v`
Expected: FAIL

**Step 3: Write minimal implementation**

新增统一 helper（建议放 `utils/quant_linear.py`）：

```python
def maybe_prepare_quantized_generator(pipeline, config, local_rank):
    qcfg = getattr(config, "quantization", None)
    if not qcfg or not qcfg.enabled:
        return

    if getattr(config, "adapter", None):
        raise ValueError("INT8 quantization is incompatible with runtime LoRA injection. Please disable adapter or pre-merge LoRA.")

    if qcfg.checkpoint_is_quantized:
        replace_linear_with_int8(pipeline.generator.model, scope_attr="blocks", quantize=False)
    # 加载 checkpoint 后，如果不是预量化 checkpoint，再执行 quantize=True 的替换
```

**Step 4: Run tests to verify they pass**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add exp_/IAMFlow/inference.py exp_/IAMFlow/interactive_inference.py exp_/IAMFlow/agent_inference.py exp_/IAMFlow/agent_interactive_inference.py exp_/IAMFlow/utils/quant_linear.py
git commit -m "feat: wire int8 quantization into all inference entrypoints"
```

### Task 5: 配置与文档对齐

**Files:**
- Modify: `exp_/IAMFlow/configs/inference.yaml`
- Modify: `exp_/IAMFlow/configs/interactive_inference.yaml`
- Modify: `exp_/IAMFlow/configs/agent_inference.yaml`
- Modify: `exp_/IAMFlow/configs/agent_interactive_inference.yaml`
- Modify: `exp_/IAMFlow/README.md`

**Step 1: Write the failing test**

```python
def test_quantization_config_defaults_present():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/inference.yaml")
    assert "quantization" in cfg
    assert cfg.quantization.enabled is False
```

**Step 2: Run test to verify it fails**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py::test_quantization_config_defaults_present -v`
Expected: FAIL

**Step 3: Write minimal implementation**

在四个 config 追加：

```yaml
quantization:
  enabled: false
  checkpoint_is_quantized: false
  scope_attr: blocks
  skip_keywords: ["proj_l"]
```

README 增加：
- 如何离线量化 checkpoint
- 如何启用推理量化
- LoRA 与 INT8 的兼容性说明（默认互斥）

**Step 4: Run tests to verify they pass**

Run: `cd exp_/IAMFlow && pytest tests/test_quant_linear_replace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add exp_/IAMFlow/configs/inference.yaml exp_/IAMFlow/configs/interactive_inference.yaml exp_/IAMFlow/configs/agent_inference.yaml exp_/IAMFlow/configs/agent_interactive_inference.yaml exp_/IAMFlow/README.md
git commit -m "docs: add quantization config and usage guide"
```

### Task 6: GPU 验证与回归

**Files:**
- Modify: `exp_/IAMFlow/document/`（可选，记录 profiling 数据）

**Step 1: 运行最小功能验证（单 prompt）**

Run:
```bash
cd exp_/IAMFlow
python inference.py --config_path configs/inference.yaml
```
Expected: 能成功产出视频；日志中显示 quantization 已生效。

**Step 2: 运行交互模式验证**

Run:
```bash
cd exp_/IAMFlow
python interactive_inference.py --config_path configs/interactive_inference.yaml
```
Expected: 多段 prompt 切换不报错，输出视频正常。

**Step 3: 运行 Agent 推理验证**

Run:
```bash
cd exp_/IAMFlow
python agent_inference.py --config_path configs/agent_inference.yaml
python agent_interactive_inference.py --config_path configs/agent_interactive_inference.yaml
```
Expected: IAM Agent 路径不因 INT8 破坏。

**Step 4: 记录性能指标**

记录以下对比：
- `max_memory_allocated`
- 端到端耗时
- 每 block 扩散时间（若 `profile=True`）

**Step 5: Commit**

```bash
git add exp_/IAMFlow/document
git commit -m "chore: add int8 quantization profiling results"
```

## Notes / Guardrails

- 仅替换 `generator.model.blocks` 内 `nn.Linear`，与 TurboDiffusion 保持一致，避免输出头和时间/文本嵌入层的数值回归风险。
- `Int8Linear.scale` 必须保持 `float32`；不要让 `pipeline.to(dtype=torch.bfloat16)` 把它降精度。
- LoRA 与 INT8 默认互斥：若要两者同时使用，先离线 merge LoRA 到 base，再执行 checkpoint 量化。
- 优先保证功能正确，再做性能优化；性能目标建议分两档：
  1) 显存降低 ≥ 20%
  2) 端到端时延不劣于 FP16/BF16 超过 10%
