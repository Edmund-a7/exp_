# IAM Flow Pipeline 说明

## 两种推理模式

### 1. agent_inference.py - 单 Prompt 生成
- **用途**: 从单个文本 prompt 生成视频，类似 MemFlow 的 inference.py
- **特点**:
  - 使用 AgentCausalInferencePipeline（包含 IAM 能力）
  - 不涉及 prompt 切换
  - 适合简单的文本到视频生成任务
- **配置文件**: `configs/agent_inference.yaml`
- **运行**: `bash agent_inference.sh`

### 2. agent_interactive_inference.py - 交互式多 Prompt 生成
- **用途**: 多 prompt 交互式视频生成，支持场景切换
- **特点**:
  - 使用 AgentCausalInferencePipeline 的完整功能
  - LLM Agent 进行实体提取和 ID 匹配
  - Memory Bank 进行帧选择和记忆管理
  - 支持在指定帧切换 prompt
- **配置文件**: `configs/agent_interactive_inference.yaml`
- **运行**: `bash agent_interactive_inference.sh`

---

## 项目完整流程详解（交互式模式）
基本设定:
每个 prompt 生成 10 个 chunk，每个 chunk 3 帧
KV Cache 结构: [Sink(3帧) + Mem(3帧) + Local(6帧)]
Sink = 第一个 chunk（固定不变）
Local = 滑动窗口（6 帧 = 2 个 chunk）
Mem = 记忆帧（3 帧，按帧更新）
Chunk 3 开始驱逐并选帧
═══════════════════════════════════════════════════════════════
Prompt 1: 引入主角
═══════════════════════════════════════════════════════════════
Prompt 文本: "A young man in his late 20s, with messy black hair, wearing a vintage blue denim jacket, sits alone on a park bench..."
Prompt 1, Chunk 1
LLM Agent（每个 prompt 触发一次）
Step 1: EntityStructExtractor.extract()

# 输入: prompt 文本
# 输出: List[EntityStruct]，global_id 为 None
[
    EntityStruct(
        entity="young man",
        attrs=["late 20s", "messy black hair", "denim jacket", "holding sketchbook"],
        global_id=None  # 尚未分配
    )
]
Step 2: GlobalIDManager.assign_ids(is_first_prompt=True)

# 第一个 prompt，直接分配新 ID
entity.global_id = 1  # 调用 _allocate_new_id()
Step 3: 更新 mapping.json 的 global_registry

"global_registry": {
    "1": {
        "name": "man_1",
        "all_entities": ["young man"],
        "all_attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook"],
        "instances": [
            {
                "prompt_id": 1,
                "entity": "young man",
                "attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook"]
            }
        ]
    }
}
Memory Bank
frame_archive: {}（空）
frame_active_memory: []（空）
Video Model
KV Cache 组成	内容
Sink	0 → 3 帧（chunk 1 存入）
Mem	0 帧
Local	0 → 3 帧（chunk 1）
输入: [Sink(0) + Mem(0) + Local(0)]
生成 chunk 1（3 帧）
Sink ← chunk 1（固定）
Local ← chunk 1
Prompt 1, Chunk 2
LLM Agent
不触发（同一个 prompt 只在 chunk 1 触发）
Memory Bank
无驱逐，无选帧
frame_active_memory: []
Video Model
KV Cache 组成	内容
Sink	3 帧（C1）
Mem	0 帧
Local	6 帧（C1 + C2）
输入: [Sink(3) + Mem(0) + Local(3)]
生成 chunk 2（3 帧）
Local 满载：C1 + C2 = 6 帧
Prompt 1, Chunk 3 ⭐ 首次驱逐
Memory Bank
驱逐 chunk 1（3 帧）
Query text 生成:

query = "young man late 20s messy black hair denim jacket holding sketchbook"
交叉注意力计算: 对 C1 的 3 帧计算 score，选最高分帧
存入 frame_archive:

"frame_archive": {
    "p1_c1_f2": {
        "frame_path": "data/p1_c1_f2.pt",
        "prompt_id": 1,
        "associated_entities": ["1"],
        "score": 0.87
    }
}
更新 frame_active_memory: ["p1_c1_f2"]（1 帧）
Video Model
KV Cache 组成	内容
Sink	3 帧（C1）
Mem	0 → 1 帧（生成后选帧）
Local	6 帧（C2 + C3）
输入: [Sink(3) + Mem(0) + Local(6)]
生成 chunk 3（3 帧）
Local: C2 + C3
Prompt 1, Chunk 4
Memory Bank
驱逐 C2，选帧 p1_c2_f1，score = 0.91
frame_active_memory: ["p1_c2_f1", "p1_c1_f2"]（2 帧）
Video Model
输入: [Sink(3) + Mem(1) + Local(6)]
Local: C3 + C4
Prompt 1, Chunk 5
Memory Bank
驱逐 C3，选帧 p1_c3_f0，score = 0.93
frame_active_memory: ["p1_c3_f0", "p1_c2_f1", "p1_c1_f2"]（3 帧，达到上限）
Video Model
输入: [Sink(3) + Mem(2) + Local(6)]
Prompt 1, Chunk 6-10
Memory Bank
每个 chunk 继续驱逐、选帧、比较 score、更新 top 3。 例如 Chunk 6:
驱逐 C4，选帧 p1_c4_f2，score = 0.89
0.89 > 0.87（最低分 p1_c1_f2）
frame_active_memory: ["p1_c3_f0", "p1_c2_f1", "p1_c4_f2"]（替换）
Video Model
输入: [Sink(3) + Mem(3) + Local(6)] ✅ 完整结构
Prompt 1 结束时 mapping.json 状态

{
    "global_registry": {
        "1": {
            "name": "man_1",
            "all_entities": ["young man"],
            "all_attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook"],
            "instances": [
                {"prompt_id": 1, "entity": "young man", "attrs": [...]}
            ]
        }
    },
    "frame_archive": {
        "p1_c1_f2": {"prompt_id": 1, "associated_entities": ["1"], "score": 0.87},
        "p1_c2_f1": {"prompt_id": 1, "associated_entities": ["1"], "score": 0.91},
        "p1_c3_f0": {"prompt_id": 1, "associated_entities": ["1"], "score": 0.93},
        "p1_c4_f2": {"prompt_id": 1, "associated_entities": ["1"], "score": 0.89},
        // ... C5-C8 的帧
    },
    "frame_active_memory": ["p1_c3_f0", "p1_c2_f1", "p1_c4_f2"]
}
═══════════════════════════════════════════════════════════════
Prompt 2: 引入第二个角色
═══════════════════════════════════════════════════════════════
Prompt 文本: "The main protagonist in the denim jacket remains seated... Another man, around 30 years old, wearing glasses and a grey sweater..."
Prompt 2, Chunk 1 ⭐ 实体匹配 + 帧检索 + 驱逐
LLM Agent
Step 1: EntityStructExtractor.extract()

[
    EntityStruct(entity="protagonist", attrs=["denim jacket", "seated on bench"], global_id=None),
    EntityStruct(entity="another man", attrs=["30 years old", "glasses", "grey sweater", "coffee cup"], global_id=None)
]
Step 2: GlobalIDManager.assign_ids(is_first_prompt=False) 调用 _match_or_allocate() 进行匹配: 实体 1: "protagonist"
检查 NEW_ENTITY_MARKERS → "protagonist" 不含 "another/other/..."
精确名称匹配 → 未找到
同义词匹配 → "protagonist" 在 SYNONYM_GROUPS[0] 中，"young man" 也在 → 匹配 global_id=1
调用 _update_entity(1, "protagonist", [...], 2)
实体 2: "another man"
检查 NEW_ENTITY_MARKERS → 含 "another" → is_explicitly_new = True
跳过所有匹配，直接 _allocate_new_id() → global_id=2
Step 3: 更新 mapping.json

"global_registry": {
    "1": {
        "name": "man_1",
        "all_entities": ["young man", "protagonist"],  // 新增
        "all_attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook", "seated on bench"],
        "instances": [
            {"prompt_id": 1, ...},
            {"prompt_id": 2, "entity": "protagonist", "attrs": ["denim jacket", "seated on bench"]}  // 新增
        ]
    },
    "2": {
        "name": "man_2",
        "all_entities": ["another man"],
        "all_attrs": ["30 years old", "glasses", "grey sweater", "coffee cup"],
        "instances": [
            {"prompt_id": 2, "entity": "another man", "attrs": [...]}
        ]
    }
}
Memory Bank（帧检索）
当前实体 global_id 列表: [1, 2]
遍历 frame_archive:
p1_c3_f0: associated_entities=["1"] → 与 [1,2] 有交集 ✓
p1_c2_f1: associated_entities=["1"] → 与 [1,2] 有交集 ✓
p1_c4_f2: associated_entities=["1"] → 与 [1,2] 有交集 ✓
按 score 排序取 top 3
初始化 frame_active_memory: ["p1_c3_f0", "p1_c2_f1", "p1_c4_f2"]
Memory Bank（驱逐）
生成 P2_C1 后，驱逐 P1_C9
Query text（用 Prompt 2 的实体）:

query = "protagonist denim jacket seated on bench another man 30 years old glasses grey sweater coffee cup"
选帧 p1_c9_f1，associated_entities: ["1"]（帧中只有 man_1），score = 0.86
0.86 < 0.89（当前最低分），不替换
frame_active_memory: 保持不变
Video Model
KV Cache 组成	内容
Sink	3 帧（P1_C1，保持不变）
Mem	3 帧（从 frame_active_memory 加载）
Local	6 帧（P1_C10 + P2_C1）
Prompt 2, Chunk 2
Memory Bank
驱逐 P1_C10
该帧仍只有 man_1，associated_entities: ["1"]
Video Model
Local: P2_C1 + P2_C2
Prompt 2, Chunk 3+
Memory Bank
驱逐 P2_C1
重要: 该帧有两人同框，associated_entities: ["1", "2"]
高分帧可能进入 frame_active_memory

"p2_c1_f0": {
    "frame_path": "data/p2_c1_f0.pt",
    "prompt_id": 2,
    "associated_entities": ["1", "2"],
    "score": 0.94
}
Prompt 2 结束时 mapping.json 状态

{
    "global_registry": {
        "1": {..., "instances": [P1, P2]},
        "2": {..., "instances": [P2]}
    },
    "frame_archive": {
        // Prompt 1 的帧（只有 man_1）
        "p1_c3_f0": {"associated_entities": ["1"], "score": 0.93},
        // Prompt 2 的帧（有 man_1 + man_2）
        "p2_c1_f0": {"associated_entities": ["1", "2"], "score": 0.94},
        "p2_c5_f1": {"associated_entities": ["1", "2"], "score": 0.93}
    },
    "frame_active_memory": ["p2_c1_f0", "p1_c3_f0", "p2_c5_f1"]
}
═══════════════════════════════════════════════════════════════
Prompt 3: 引入第三个角色
═══════════════════════════════════════════════════════════════
Prompt 文本: "The protagonist and the man in the grey sweater are talking... A young woman in her late 20s, with long hair and wearing a flowing white dress..."
Prompt 3, Chunk 1 ⭐
LLM Agent
Step 1: EntityStructExtractor.extract()

[
    EntityStruct(entity="protagonist", attrs=["denim jacket"], global_id=None),
    EntityStruct(entity="man in grey sweater", attrs=["talking on bench"], global_id=None),
    EntityStruct(entity="young woman", attrs=["late 20s", "long hair", "white dress", "shoulder bag"], global_id=None)
]
Step 2: GlobalIDManager.assign_ids(is_first_prompt=False) 实体 1: "protagonist"
同义词匹配 → global_id=1
实体 2: "man in grey sweater"
精确匹配 → 未找到
同义词匹配 → "man" 在 SYNONYM_GROUPS[0]，但已有 global_id=1
属性匹配 → "grey sweater" 匹配 global_id=2 的 attrs → global_id=2
实体 3: "young woman"
精确匹配 → 未找到
同义词匹配 → "woman" 在 SYNONYM_GROUPS[1]，无已有匹配
属性匹配 → 无匹配
分配新 ID → global_id=3
Step 3: 更新 mapping.json

"global_registry": {
    "1": {..., "all_entities": ["young man", "protagonist"]},
    "2": {..., "all_entities": ["another man", "man in grey sweater"]},
    "3": {
        "name": "woman_1",
        "all_entities": ["young woman"],
        "all_attrs": ["late 20s", "long hair", "white dress", "shoulder bag"],
        "instances": [
            {"prompt_id": 3, "entity": "young woman", "attrs": [...]}
        ]
    }
}
Memory Bank（帧检索）
当前 global_id 列表: [1, 2, 3]
遍历 frame_archive:
global_id=3 是新的，无历史帧
按 global_id=1, 2 检索
优先选同时含 1 和 2 的帧（Prompt 2 的帧）
初始化 frame_active_memory: ["p2_c1_f0", "p1_c3_f0", "p2_c5_f1"]
Memory Bank（驱逐）
驱逐 P2_C9
选帧，associated_entities 仍为 ["1", "2"]
Prompt 3, Chunk 3+
Memory Bank
驱逐 P3_C1
首次出现 global_id=3 的帧:

"p3_c1_f2": {
    "associated_entities": ["1", "2", "3"],
    "score": 0.92
}
后续检索 global_id=3 时可以找到了
Prompt 3, Chunk 10（最后一个 chunk）
最终 mapping.json 状态

{
    "global_registry": {
        "1": {
            "name": "man_1",
            "all_entities": ["young man", "protagonist"],
            "all_attrs": ["late 20s", "messy black hair", "denim jacket", "holding sketchbook", "seated on bench"],
            "instances": [
                {"prompt_id": 1, "entity": "young man", "attrs": [...]},
                {"prompt_id": 2, "entity": "protagonist", "attrs": [...]},
                {"prompt_id": 3, "entity": "protagonist", "attrs": [...]}
            ]
        },
        "2": {
            "name": "man_2",
            "all_entities": ["another man", "man in grey sweater"],
            "all_attrs": ["30 years old", "glasses", "grey sweater", "coffee cup", "talking on bench"],
            "instances": [
                {"prompt_id": 2, ...},
                {"prompt_id": 3, ...}
            ]
        },
        "3": {
            "name": "woman_1",
            "all_entities": ["young woman"],
            "all_attrs": ["late 20s", "long hair", "white dress", "shoulder bag"],
            "instances": [
                {"prompt_id": 3, ...}
            ]
        }
    },
    "frame_archive": {
        // Prompt 1: 单人帧
        "p1_c3_f0": {"prompt_id": 1, "associated_entities": ["1"], "score": 0.93},
        
        // Prompt 2: 双人帧
        "p2_c1_f0": {"prompt_id": 2, "associated_entities": ["1", "2"], "score": 0.94},
        "p2_c5_f1": {"prompt_id": 2, "associated_entities": ["1", "2"], "score": 0.93},
        
        // Prompt 3: 三人帧
        "p3_c4_f0": {"prompt_id": 3, "associated_entities": ["1", "2", "3"], "score": 0.95},
        "p3_c7_f1": {"prompt_id": 3, "associated_entities": ["1", "2", "3"], "score": 0.92}
    },
    "frame_active_memory": ["p3_c4_f0", "p2_c1_f0", "p2_c5_f1"]
}
═══════════════════════════════════════════════════════════════
关键流程总结
═══════════════════════════════════════════════════════════════
时序表
阶段	LLM Agent	Sink	Mem	Local	驱逐
P1 C1	✓ 提取+分配ID	←C1	0	C1	无
P1 C2	-	C1	0	C1+C2	无
P1 C3	-	C1	0→1	C2+C3	C1→选帧
P1 C4	-	C1	1→2	C3+C4	C2→选帧
P1 C5	-	C1	2→3	C4+C5	C3→选帧
P1 C6-10	-	C1	3(更新)	滑动	每chunk选帧
P2 C1	✓ 匹配+新ID	C1	3(检索)	P1C10+P2C1	P1C9→选帧
P2 C2-10	-	C1	3(更新)	滑动	每chunk选帧
P3 C1	✓ 匹配+新ID	C1	3(检索)	P2C10+P3C1	P2C9→选帧
P3 C2-10	-	C1	3(更新)	滑动	每chunk选帧
核心类职责
类	触发时机	职责
EntityStructExtractor	每个 prompt 的 chunk 1	提取 entity + attrs
GlobalIDManager	每个 prompt 的 chunk 1	匹配/分配 global_id
LLMAgent	每个 prompt 的 chunk 1	协调以上两个组件
MemoryBank	每个 chunk	帧检索 + 驱逐选帧 + 更新 top 3
数据流

Prompt 文本
    ↓
EntityStructExtractor.extract()
    ↓
List[EntityStruct(entity, attrs, global_id=None)]
    ↓
GlobalIDManager.assign_ids()
    ↓
List[EntityStruct(entity, attrs, global_id=分配值)]
    ↓
MemoryBank.register_entity()  →  更新 global_registry
    ↓
MemoryBank.retrieve_frames()  →  初始化 frame_active_memory
    ↓
Video Model 生成 chunk
    ↓
MemoryBank.evict_and_select()  →  更新 frame_archive + frame_active_memory