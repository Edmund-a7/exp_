# TCAT Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 TCAT 选择逻辑与方案一致，并确保配置可传递、生效且边界行为明确。

**Architecture:** 主要集中在 `CausalWanSelfAttention.tcat_routing_attention` 的选择/重排逻辑，以及 `WanDiffusionWrapper` 的配置透传；必要时调整 IAM memory 注入顺序以保证“时间顺序”。

**Tech Stack:** Python, PyTorch, IAMFlow.

### Task 1: 让 TCAT 配置正确透传

**Files:**
- Modify: `IAMFlow/utils/wan_wrapper.py`

**Step 1: 写失败测试（可选）**
```python
# tests/test_tcat_config.py
# 断言 WanDiffusionWrapper 能接受 tcat_* 参数并传入 CausalWanModel
```

**Step 2: 运行测试验证失败**
Run: `pytest tests/test_tcat_config.py -v`
Expected: FAIL（参数不被接受或不生效）

**Step 3: 最小实现**
- 给 `WanDiffusionWrapper.__init__` 增加 `tcat_enabled/tcat_sink_k/tcat_mem_k/tcat_local_k` 参数
- 在 `CausalWanModel.from_pretrained(...)` 中透传这些参数
- 若非因果分支，显式忽略或过滤这些参数

**Step 4: 运行测试验证通过**
Run: `pytest tests/test_tcat_config.py -v`
Expected: PASS

**Step 5: Commit（人工执行）**
```bash
git add IAMFlow/utils/wan_wrapper.py tests/test_tcat_config.py
git commit -m "fix(iamflow): pass tcat config to causal model"
```

### Task 2: 实现“总数≤6全选 + 剩余配额再分配”

**Files:**
- Modify: `IAMFlow/wan/modules/causal_model.py`

**Step 1: 写失败测试（可选）**
```python
# tests/test_tcat_selection.py
# 构造 region chunk 数不足的场景，验证会把剩余额度补给其他 region
```

**Step 2: 运行测试验证失败**
Run: `pytest tests/test_tcat_selection.py -v`
Expected: FAIL（当前不会再分配）

**Step 3: 最小实现**
- 计算每个 region 的 chunk 数与 top_k
- 若总 chunk 数 ≤ 6：直接拼接全量
- 先按区域 top‑k 选出“基础集合”
- 统计 leftover，从“未被选中的候选 chunks”里按分数降序补齐
- 保持最终输出按时间顺序（索引升序）

**Step 4: 运行测试验证通过**
Run: `pytest tests/test_tcat_selection.py -v`
Expected: PASS

**Step 5: Commit（人工执行）**
```bash
git add IAMFlow/wan/modules/causal_model.py tests/test_tcat_selection.py
git commit -m "fix(iamflow): tcat redistribute leftover budget"
```

### Task 3: Mem 区“时间顺序”修正

**Files:**
- Modify: `IAMFlow/iam/memory_bank.py`
- Modify: `IAMFlow/pipeline/agent_causal_inference.py`

**Step 1: 写失败测试（可选）**
```python
# tests/test_tcat_mem_order.py
# 注入多帧记忆时验证按时间字段排序写入 kv_bank
```

**Step 2: 运行测试验证失败**
Run: `pytest tests/test_tcat_mem_order.py -v`
Expected: FAIL（当前注入顺序依赖 active_memory 列表）

**Step 3: 最小实现**
- 在获取 memory_kv 之前/之后，按照 frame_id 解析时间并排序
- 再写入 kv_bank，保证索引升序等价于时间顺序

**Step 4: 运行测试验证通过**
Run: `pytest tests/test_tcat_mem_order.py -v`
Expected: PASS

**Step 5: Commit（人工执行）**
```bash
git add IAMFlow/iam/memory_bank.py IAMFlow/pipeline/agent_causal_inference.py tests/test_tcat_mem_order.py
git commit -m "fix(iamflow): enforce temporal order for mem bank"
```

### Task 4: 全局共享 top‑k（非 per‑head）

**Files:**
- Modify: `IAMFlow/wan/modules/causal_model.py`

**Step 1: 写失败测试（可选）**
```python
# tests/test_tcat_global_topk.py
# 断言所有 head 使用相同 chunk 索引
```

**Step 2: 运行测试验证失败**
Run: `pytest tests/test_tcat_global_topk.py -v`
Expected: FAIL（当前是 per-head）

**Step 3: 最小实现**
- 对 `scores` 在 head 维求平均或最大值，得到 `[B, num_chunks]`
- 用全局 top‑k 选 index，并扩展到 head 维

**Step 4: 运行测试验证通过**
Run: `pytest tests/test_tcat_global_topk.py -v`
Expected: PASS

**Step 5: Commit（人工执行）**
```bash
git add IAMFlow/wan/modules/causal_model.py tests/test_tcat_global_topk.py
git commit -m "fix(iamflow): global topk selection for tcat"
```

### Task 5: 处理 chunk_size 不整除

**Files:**
- Modify: `IAMFlow/wan/modules/causal_model.py`

**Step 1: 写失败测试（可选）**
```python
# tests/test_tcat_chunk_remainder.py
# 构造 L_region % chunk_size != 0 的情况，断言不报错且回退为全选
```

**Step 2: 运行测试验证失败**
Run: `pytest tests/test_tcat_chunk_remainder.py -v`
Expected: FAIL（当前 view 报错）

**Step 3: 最小实现**
- 检测 `L_region % chunk_size != 0`：直接保留该 region 全量

**Step 4: 运行测试验证通过**
Run: `pytest tests/test_tcat_chunk_remainder.py -v`
Expected: PASS

**Step 5: Commit（人工执行）**
```bash
git add IAMFlow/wan/modules/causal_model.py tests/test_tcat_chunk_remainder.py
git commit -m "fix(iamflow): guard tcat chunk remainder"
```
