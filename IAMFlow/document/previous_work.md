### LongLive
- Streaming Long Tuning：长视频训练；先生成 5s 短视频，之后基于前一段的KV-Cache继续生成下一段，直到 60s
- Frame Sink：固定首个 chunk 的 3 帧 token，加入 kv_cache
- Prompt Switch：把上一段生成的 token 和新的 prompt 再做一次前向过程，得到新的kv_cache

### MemFlow
- 记忆库： 根据文本 prompt 动态更新的 3 帧token
- 根据文本进行记忆库更新NAM：
-- 三帧 token，每生成一个 chunk 后，保留最新 chunk 的首帧，在旧 kv_cache 中根据prompt的交叉注意力分数选两帧；也就是mem 是选出的 2 帧历史帧和 1 帧新 chunk 的首帧；最终 kv_cache 由sink: 3,mem: 3,local: 6 组成
- 根据视觉进行进一步检索SMA：
-- 把 sink 和 bank 拼接，计算当前的 q 和拼接后各帧的点积，取 top-3；
- 最终 是检索出的top-3 和local拼接，组成 最终9 帧的kv_cache;x = attention(roped_query, k_cat,v_cat)
