整体架构解释
1. 文件结构

IAM_2/
├── __init__.py              # 模块入口，导出所有类
├── llm_agent.py             # LLM Agent 模块
├── memory_bank.py           # Memory Bank 模块
├── agent_causal_inference.py # Pipeline 整合模块
├── test_iam2.py             # 测试代码
├── plan.md                  # 设计文档
├── pipeline.md              # 流程文档
└── mapping.json             # 数据结构示例
2. 各文件的核心类
llm_agent.py

┌─────────────────────────────────────────────────────────────┐
│ EntityStruct (数据类)                                        │
│   - entity: str          # 实体名称 (young man, protagonist) │
│   - attrs: List[str]     # 属性列表 [late 20s, denim jacket] │
│   - global_id: int|None  # 全局ID (提取时为None)             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LLMWrapper (LLM封装)                                         │
│   - 加载 Qwen3-0.6B                                          │
│   - 支持 MPS/CUDA/CPU                                        │
│   - generate(system_prompt, user_prompt) -> str             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ EntityStructExtractor                                        │
│   - 输入: prompt 文本                                        │
│   - 输出: List[EntityStruct] (global_id=None)               │
│   - 功能: 使用LLM从prompt中提取人物实体和属性                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ GlobalIDManager                                              │
│   - 输入: entities, global_registry, is_first_prompt        │
│   - 输出: 分配了global_id的entities                          │
│   - 功能:                                                    │
│     - 第一个prompt: 直接分配新ID                             │
│     - 后续prompt: LLM匹配或分配新ID                          │
│     - 检测"another/other"等标记 → 强制新ID                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ LLMAgent (协调器)                                            │
│   - 组合 EntityStructExtractor + GlobalIDManager            │
│   - process_prompt(prompt, prompt_id, registry)             │
│   - 返回: (entities, registry_update)                       │
└─────────────────────────────────────────────────────────────┘
memory_bank.py

┌─────────────────────────────────────────────────────────────┐
│ FrameInfo (数据类)                                           │
│   - frame_id: str        # p1_c3_f0 格式                     │
│   - frame_path: str      # 文件路径                          │
│   - prompt_id: int       # 所属prompt                        │
│   - associated_entities: List[str]  # 关联的entity ID        │
│   - score: float         # 分数                              │
│   - kv_cache: Dict       # KV cache数据                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MemoryBank                                                   │
│   核心数据:                                                   │
│   - global_registry: Dict  # 实体注册表                      │
│   - frame_archive: Dict    # 所有帧存档                      │
│   - frame_active_memory: List[str]  # 当前3帧记忆            │
│                                                              │
│   核心方法:                                                   │
│   - register_entities()     # 注册/更新实体                  │
│   - retrieve_initial_frames()  # 检索初始记忆帧              │
│   - select_frame_from_chunk()  # 从驱逐chunk选帧             │
│   - update_active_memory()  # 更新top-3记忆帧                │
│   - get_memory_kv()         # 获取记忆帧KV                   │
│   - save/load_from_json()   # 持久化                         │
└─────────────────────────────────────────────────────────────┘
agent_causal_inference.py

┌─────────────────────────────────────────────────────────────┐
│ AgentCausalInferencePipeline                                 │
│   继承: InteractiveCausalInferencePipeline                   │
│                                                              │
│   新增成员:                                                   │
│   - llm_agent: LLMAgent                                      │
│   - agent_memory_bank: MemoryBank                            │
│   - current_prompt_id, current_chunk_id, current_entities   │
│                                                              │
│   重写方法:                                                   │
│   - inference() # 在关键位置插入Agent逻辑                    │
│                                                              │
│   新增方法:                                                   │
│   - _process_prompt_start()   # prompt开始时调用Agent        │
│   - _process_chunk_eviction() # chunk驱逐时选帧              │
│   - _get_evicted_chunk_kv()   # 获取被驱逐的KV               │
└─────────────────────────────────────────────────────────────┘
3. 数据流向 (按 pipeline.md 流程)
假设 interactive_inference.jsonl 包含 3 个 prompts:

Prompt 1: "A young man in his late 20s, with messy black hair..."
Prompt 2: "The main protagonist in the denim jacket... Another man..."
Prompt 3: "The protagonist and the man in the grey sweater... A young woman..."
Prompt 1, Chunk 1 (首次触发 LLM Agent)

                           ┌─────────────────┐
    Prompt 1 文本 ────────►│EntityStructExtractor│
                           └────────┬────────┘
                                    │ LLM 提取
                                    ▼
                     ┌────────────────────────────┐
                     │ [EntityStruct(             │
                     │   entity="young man",      │
                     │   attrs=[late 20s, ...],   │
                     │   global_id=None)]         │
                     └────────────┬───────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │GlobalIDManager │
                         │is_first_prompt=│
                         │True            │
                         └───────┬────────┘
                                 │ 直接分配 ID=1
                                 ▼
                     ┌────────────────────────────┐
                     │ [EntityStruct(             │
                     │   entity="young man",      │
                     │   global_id=1)]            │
                     └────────────┬───────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  MemoryBank    │
                         │register_entities│
                         └───────┬────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────┐
                 │ global_registry = {             │
                 │   "1": {                        │
                 │     name: "man_1",              │
                 │     all_entities: ["young man"],│
                 │     all_attrs: [...],           │
                 │     instances: [...]            │
                 │   }                             │
                 │ }                               │
                 │ frame_archive = {}              │
                 │ frame_active_memory = []        │
                 └─────────────────────────────────┘
Prompt 1, Chunk 3 (首次驱逐)

                     ┌────────────────────────────┐
                     │ Local KV Cache 满 (6帧)    │
                     │ 驱逐 Chunk 1 (3帧)         │
                     └────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │MemoryBank.select_frame_from_chunk│
                    │                                   │
                    │ 1. 构建查询文本:                  │
                    │    "young man late 20s denim..."  │
                    │                                   │
                    │ 2. 交叉注意力计算每帧分数        │
                    │                                   │
                    │ 3. 选最高分帧 → p1_c1_f2          │
                    │    score=0.87                     │
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────┐
                 │ frame_archive = {               │
                 │   "p1_c1_f2": {                 │
                 │     prompt_id: 1,               │
                 │     associated_entities: ["1"], │
                 │     score: 0.87                 │
                 │   }                             │
                 │ }                               │
                 │ frame_active_memory = ["p1_c1_f2"]│
                 └─────────────────────────────────┘
Prompt 2, Chunk 1 (触发 LLM Agent + 实体匹配)

                           ┌─────────────────┐
    Prompt 2 文本 ────────►│EntityStructExtractor│
                           └────────┬────────┘
                                    │ LLM 提取
                                    ▼
                     ┌────────────────────────────┐
                     │ [EntityStruct(             │
                     │   entity="protagonist",    │
                     │   attrs=[denim jacket],    │
                     │   global_id=None),         │
                     │  EntityStruct(             │
                     │   entity="another man",    │ ← 包含"another"
                     │   attrs=[grey sweater],    │
                     │   global_id=None)]         │
                     └────────────┬───────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │GlobalIDManager │
                         │is_first_prompt=│
                         │False           │
                         └───────┬────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       │                         │                         │
       ▼                         ▼                         ▼
"protagonist"             检查registry              "another man"
       │                         │                         │
       │                         ▼                         │
       │              ┌────────────────┐                   │
       │              │ LLM 匹配:      │                   │
       │              │ protagonist ≈  │                   │
       │              │ young man      │                   │
       │              │ → matched_id=1 │                   │
       │              └───────┬────────┘                   │
       │                      │                            │
       │    ┌─────────────────┘          包含"another" ───┘
       │    │                            强制新ID
       ▼    ▼                                  │
   global_id=1                           global_id=2
       │                                       │
       └───────────────┬───────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  MemoryBank    │
              │retrieve_initial│
              │_frames([1,2])  │
              └───────┬────────┘
                      │ 检索与entity 1,2相关的帧
                      │ 按score排序取top-3
                      ▼
            ┌────────────────────────┐
            │frame_active_memory =   │
            │ [p1_c3_f0, p1_c2_f1,  │
            │  p1_c4_f2]            │ ← 初始记忆帧
            └────────────────────────┘
Prompt 3, Chunk 1 (更复杂的匹配)

                           ┌─────────────────┐
    Prompt 3 文本 ────────►│EntityStructExtractor│
                           └────────┬────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────┐
                     │ [protagonist,                   │
                     │  man in grey sweater,          │
                     │  young woman]                  │
                     └────────────┬───────────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │GlobalIDManager │
                         └───────┬────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       │                         │                         │
       ▼                         ▼                         ▼
"protagonist"        "man in grey sweater"        "young woman"
       │                         │                         │
       │ LLM匹配               LLM匹配                   无匹配
       │ → young man           → another man            新实体
       │ → ID=1                 → ID=2                    │
       │                         │                        ▼
       │                         │                     ID=3
       │                         │                        │
       └─────────────────────────┴────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │ global_registry = {              │
              │   "1": {name: "man_1",           │
              │         all_entities: ["young man", "protagonist"]},│
              │   "2": {name: "man_2",           │
              │         all_entities: ["another man", "man in grey sweater"]},│
              │   "3": {name: "woman_1",         │
              │         all_entities: ["young woman"]}│
              │ }                                │
              └──────────────────────────────────┘
4. 测试代码说明
运行方式
测试代码可以在 Mac 本地运行，因为:
使用 MockLLMWrapper 模拟 LLM 响应，不需要真实模型
不依赖 MemFlow 的视频生成模块

# 在项目根目录运行
cd /Users/acumen7/project/251222

# 运行所有测试
python -m pytest IAM_2/test_iam2.py -v

# 或者直接运行
python IAM_2/test_iam2.py
测试内容
测试类	测试内容	预期效果
TestEntityStruct	数据类基本功能	创建、to_dict、from_dict 正确
TestEntityStructExtractor	实体提取	从 prompt 提取出正确的实体和属性
TestGlobalIDManager	ID 分配和匹配	第一个 prompt 分配新 ID；后续 prompt 正确匹配或分配
TestLLMAgent	Agent 完整流程	提取+分配+registry 更新
TestMemoryBank	帧管理	注册、选帧、更新 active memory、保存/加载
TestIntegration	完整 3 个 prompt 流程	验证 ID 分配符合 pipeline.md 预期
关键测试场景
test_full_pipeline_flow (集成测试):

输入: 3 个 prompts (来自 pipeline.md)

预期结果:
- Prompt 1: young man → ID=1
- Prompt 2: protagonist → ID=1 (匹配), another man → ID=2 (新)
- Prompt 3: protagonist → ID=1, man in grey sweater → ID=2, young woman → ID=3

最终 global_registry 包含 3 个实体
5. 与真实 MemFlow 整合
如果要在服务器上运行完整的视频生成:

from IAM_2.agent_causal_inference import AgentCausalInferencePipeline

# 创建 pipeline
pipeline = AgentCausalInferencePipeline(
    args=config,
    device=torch.device("cuda"),
    llm_model_path="/path/to/Qwen3-0.6B"
)

# 生成视频
video = pipeline.inference(
    noise=sampled_noise,
    text_prompts_list=prompts_list,
    switch_frame_indices=[30, 60],  # 帧切换位置
    save_mapping=True,
    mapping_path="mapping.json"
)
这样在每个 prompt 切换时会自动:
调用 LLM Agent 提取实体
匹配/分配 global ID
检索初始记忆帧
在 chunk 驱逐时选择最佳帧更新 memory
您觉得这个架构和流程是否符合 plan.md 和 pipeline.md 的设计？
