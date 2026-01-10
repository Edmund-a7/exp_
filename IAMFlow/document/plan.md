### 任务：
流式交互的自回归视频生成；每 10s 一个 prompt 
### 工作：
以MemFlow 为基础框架，用 Agent 的思路解决：以 LLM 作为调度中枢，利用一些工具对生成的视频和对应的 prompt提取信息，找到关键信息来做下一段的生成。

就是以 llm 作为中枢调度器，对每个chunk 的生成，每个 prompt 的切换做处理。

### 设计

#### 一、核心架构：三层控制流
##### 控制层 (LLM Agent)：
- 频率：每个 Prompt 触发一次 (每 10s)。
- 职责：解析prompt，提取entity 和attributes，维护 Global ID，打包作为元数据
- 详细解释：
-- 在起始阶段，即第一个 prompt 的前两个 chunk，在 memory bank 中保存entity 和attrs；
--- 从第三个 chunk 开始，local 开始驱逐 chunk；memory bank 将 chunk 和当前 entity 和attrs 拼接的文本 token做交叉注意力，取分数最高的 1 帧，作为当前这些entity 的对应帧，和分数一起存入 memory bank，即建立和当前 entity 和 attrs 的索引；
--- 第四/五个 chunk，memory bank 再次选出一帧，和分数一起存入 memory bank，至此达成当前 prompt 的 3 帧记忆帧
--- 第六个 chunk-第 N 个 chunk，memory bank 选出一帧， 按照分数更新当前 prompt的 3 帧记忆帧
-- 后续阶段，即从第二个 prompt 提取 entity 和attributes，这一步看 memory bank，如果有一致的就分配相同的全局ID，接着从memory bank 中匹配全局 ID，取出全局ID 相同的分数最高的 3 帧，作为该 prompt 的起始 3 帧记忆帧；
--- 更新 memory_bank，将相同全局ID的entity 的 attributes 附加在后面，新增全局ID 的 entity 和 attributes 也加进去，并加上和这几帧记忆帧的索引。
--- 第一个 chunk-第 N 个 chunk，memory bank选出一帧，按照分数更新当前 prompt的 3 帧记忆帧
##### 数据层 (Memory Bank)
- 频率：每个 Chunk 触发一次 (更新) + 每个 Prompt 触发一次 (检索)。
- 职责：维护数据库，根据Agent提供的entity 和attr，计算当前被驱逐 chunk的分数，将最高分的Frame Token保存下来，作为记忆帧
- 详细解释：
-- 在起始阶段，即第一个 prompt 的前两个 chunk，memory bank中保存entity 和attrs；此时记忆帧与分数的索引为空，mem 也为空
--- 从第三个 chunk 开始，local 开始驱逐 chunk；memory bank 将 chunk 和当前 entity 和attrs 拼接的文本 token做交叉注意力，取分数最高的 1 帧，作为当前这些entity 的对应帧，和分数一起存入 memory bank，即建立和当前 entity 和 attrs 的索引；同时，将这帧 token 放到 mem 里备用。
--- 第四/五个 chunk，memory bank 再次选出一帧，和分数一起存入 memory bank，mem 长度达到 3，至此达成当前 prompt 的 3 帧记忆帧
-- 后续阶段，memory bank 接收 agent 的更新/新增操作，每个chunk 依然是计算分数，建立 entity 对记忆帧和分数的索引，存 mem。
##### 执行层 (Video Model)：
- 频率：每个 Chunk 触发。
- 职责：接收 [Sink(3) + Mem(3) + Local(6)] 的 KV Cache，进行 Attention 计算生成视频 token，生成的 token 一方面是保存作为结果，一方面是存入 local，另一方面是从local 中驱逐的 token被 memory bank 用于提取记忆帧。
#### 二、LLM Agent 提取格式与 Memory Bank 保存格式
prompt：IAM/interactive_inference.jsonl
控制层 (LLM Agent)：IAM/llm_agent.json
数据层 (Memory Bank)：IAM/memory_bank.json
根据 prompt_id 和 chunk_id，决定Memory Bank的工作模式：
- prompt_id = 1：保存entity 和attrs 即可，不需要与现有数据进行匹配
- chunk_id = 2-N：需要将当前 prompt 的 entity 和 attrs，与当前被驱逐的 chunk 进行注意力分数计算，取分数最高的 1 帧，和分数一起建立和这些entity 和attrs 的索引。
- prompt_id = 2-N, chunk_id = 1：需要将当前 prompt 的 entity 和 attrs，与 memory bank 中保存的global_id, entity, attrs 进行匹配，优先按照global_id 匹配的比例，选出当前 prompt 使用的初始 3 帧记忆帧；若global_id 不匹配，就找语义相似度最高的prompt，用它的 3帧记忆帧作为当前 prompt 使用的初始 3 帧记忆帧。
