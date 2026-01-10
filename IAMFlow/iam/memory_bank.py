"""
IAM_2 Memory Bank 模块

功能:
1. 维护 global_registry (实体注册表)
2. 维护 frame_archive (帧存档)
3. 维护 frame_active_memory (当前3帧记忆)
4. 帧选择和驱逐逻辑

数据结构参考: mapping.json
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

# 导入本地模块 (支持直接运行和作为包导入)
try:
    from .llm_agent import EntityStruct
except ImportError:
    from llm_agent import EntityStruct


@dataclass
class FrameInfo:
    """帧信息数据结构"""
    frame_id: str
    frame_path: str
    prompt_id: int
    associated_entities: List[str]  # global_id列表
    score: float
    # 存储所有 transformer block 的 KV cache
    # List[Dict[str, torch.Tensor]] - 每个 block 一个 {"k": ..., "v": ...}
    kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None

    def to_dict(self) -> Dict:
        return {
            "frame_path": self.frame_path,
            "prompt_id": self.prompt_id,
            "associated_entities": self.associated_entities,
            "score": self.score
        }


class MemoryBank:
    """
    记忆库管理器

    职责:
    1. 维护 global_registry (实体注册表)
    2. 维护 frame_archive (帧存档)
    3. 维护 frame_active_memory (当前3帧记忆)
    4. 帧选择和驱逐
    """

    def __init__(self,
                 text_encoder=None,
                 max_memory_frames: int = 3,
                 frame_seq_length: int = 1560,
                 num_transformer_blocks: int = 30,
                 save_dir: str = "data",
                 save_frames_to_disk: bool = False):
        """
        初始化Memory Bank

        Args:
            text_encoder: WanTextEncoder实例，用于文本编码 (现在不直接使用，改用 crossattn_cache)
            max_memory_frames: 最大记忆帧数量
            frame_seq_length: 每帧的序列长度
            num_transformer_blocks: transformer block 数量 (默认30)
            save_dir: 帧数据保存目录
            save_frames_to_disk: 是否将帧 KV 保存到磁盘 (默认False，仅保存在内存中以提升性能)
        """
        self.text_encoder = text_encoder
        self.max_memory_frames = max_memory_frames
        self.frame_seq_length = frame_seq_length
        self.num_transformer_blocks = num_transformer_blocks
        self.save_dir = save_dir
        self.save_frames_to_disk = save_frames_to_disk

        # 核心数据结构
        self.global_registry: Dict[str, Dict] = {}
        self.frame_archive: Dict[str, FrameInfo] = {}
        self.frame_active_memory: List[str] = []  # frame_id列表

        # KV cache存储 (内存中) - 每个 frame_id 对应 30 个 block 的 KV
        self._frame_kv_store: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

    # ============ 实体注册相关 ============

    def register_entities(self,
                          entities: List[EntityStruct],
                          prompt_id: int,
                          registry_update: Optional[Dict] = None) -> None:
        """
        注册/更新实体到global_registry

        Args:
            entities: 已分配ID的实体列表
            prompt_id: prompt序号
            registry_update: LLMAgent提供的更新信息 (可选)
        """
        if registry_update:
            # 使用LLMAgent提供的更新信息
            for gid, info in registry_update.items():
                if info.get("action") == "create":
                    self.global_registry[gid] = {
                        "name": info["name"],
                        "all_entities": info["all_entities"],
                        "all_attrs": info["all_attrs"],
                        "instances": info["instances"]
                    }
                elif info.get("action") == "update":
                    if gid in self.global_registry:
                        reg = self.global_registry[gid]
                        new_entity = info["new_entity"]
                        new_attrs = info["new_attrs"]

                        if new_entity not in reg["all_entities"]:
                            reg["all_entities"].append(new_entity)
                        for attr in new_attrs:
                            if attr not in reg["all_attrs"]:
                                reg["all_attrs"].append(attr)
                        reg["instances"].append({
                            "prompt_id": info["prompt_id"],
                            "entity": new_entity,
                            "attrs": new_attrs
                        })
        else:
            # 直接从entities注册
            for entity in entities:
                gid = str(entity.global_id)
                if gid not in self.global_registry:
                    entity_type = self._infer_entity_type(entity.entity)
                    type_count = sum(
                        1 for v in self.global_registry.values()
                        if v.get("name", "").startswith(entity_type)
                    )
                    self.global_registry[gid] = {
                        "name": f"{entity_type}_{type_count + 1}",
                        "all_entities": [entity.entity],
                        "all_attrs": entity.attrs.copy(),
                        "instances": [{
                            "prompt_id": prompt_id,
                            "entity": entity.entity,
                            "attrs": entity.attrs.copy()
                        }]
                    }
                else:
                    reg = self.global_registry[gid]
                    if entity.entity not in reg["all_entities"]:
                        reg["all_entities"].append(entity.entity)
                    for attr in entity.attrs:
                        if attr not in reg["all_attrs"]:
                            reg["all_attrs"].append(attr)
                    reg["instances"].append({
                        "prompt_id": prompt_id,
                        "entity": entity.entity,
                        "attrs": entity.attrs.copy()
                    })

    def _infer_entity_type(self, entity_name: str) -> str:
        """推断实体类型"""
        entity_lower = entity_name.lower()
        if any(w in entity_lower for w in ["woman", "girl", "lady", "female", "she"]):
            return "woman"
        elif any(w in entity_lower for w in ["man", "boy", "guy", "male", "he", "protagonist"]):
            return "man"
        else:
            return "person"

    # ============ 帧检索相关 ============

    def retrieve_initial_frames(self, entity_ids: List[int]) -> List[str]:
        """
        根据entity_ids从frame_archive检索初始记忆帧

        策略:
        1. 优先选择包含更多匹配entity的帧
        2. 按score排序取top-k

        Args:
            entity_ids: 当前prompt的实体ID列表

        Returns:
            检索到的frame_id列表
        """
        print(f"[DEBUG] retrieve_initial_frames: entity_ids={entity_ids}, archive_size={len(self.frame_archive)}")

        if not self.frame_archive:
            print(f"[DEBUG] retrieve_initial_frames: No frames in archive")
            return []

        entity_id_strs = [str(eid) for eid in entity_ids]

        # 计算每帧与当前entities的匹配度
        frame_scores = []
        for frame_id, frame_info in self.frame_archive.items():
            # 计算交集数量
            intersection = set(frame_info.associated_entities) & set(entity_id_strs)
            match_count = len(intersection)

            if match_count > 0:
                # 综合考虑匹配数量和原始分数
                combined_score = match_count * 10 + frame_info.score
                frame_scores.append((frame_id, combined_score, frame_info.score, match_count))

        print(f"[DEBUG] retrieve_initial_frames: {len(frame_scores)} frames matched entities")

        if not frame_scores:
            # 没有匹配的帧，返回分数最高的帧
            all_frames = [(fid, fi.score) for fid, fi in self.frame_archive.items()]
            all_frames.sort(key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in all_frames[:self.max_memory_frames]]
            print(f"[DEBUG] retrieve_initial_frames: No entity match, using top-score frames: {selected}")
        else:
            # 按综合分数排序
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in frame_scores[:self.max_memory_frames]]
            print(f"[DEBUG] retrieve_initial_frames: Selected by entity match:")
            for fid, combined, orig, match in frame_scores[:self.max_memory_frames]:
                print(f"  - {fid}: match_count={match}, combined_score={combined:.4f}, orig_score={orig:.4f}")

        # 更新active memory
        self.frame_active_memory = selected.copy()

        return selected

    # ============ 帧选择相关 ============

    def select_frame_from_chunk(self,
                                evicted_chunk_kv: List[Dict[str, torch.Tensor]],
                                crossattn_cache: List[Dict[str, torch.Tensor]],
                                prompt_id: int,
                                chunk_id: int,
                                current_entity_ids: List[int],
                                current_entities: Optional[List['EntityStruct']] = None,
                                prompt_text: Optional[str] = None) -> Tuple[str, float]:
        """
        从驱逐的chunk中选择最佳帧

        方案 A: 使用 crossattn_cache 计算注意力分数，但通过实体信息加权
        - 保持原始特征空间对齐
        - 根据 entity/attrs 在 prompt 中的位置精确加权

        Args:
            evicted_chunk_kv: 驱逐chunk的KV cache (所有30个block)
                              List[{"k": [B, L, H, D], "v": [B, L, H, D]}]
            crossattn_cache: cross-attention cache (所有30个block)
                             List[{"k": [B, 512, H, D], "v": [B, 512, H, D], "is_init": bool}]
            prompt_id: 当前prompt序号
            chunk_id: 当前chunk序号
            current_entity_ids: 当前prompt的实体ID列表
            current_entities: 当前prompt的实体列表 (用于构建实体权重)
            prompt_text: 当前 prompt 文本 (用于精确定位实体位置)

        Returns:
            (frame_id, score) - 选中的帧ID和分数
        """
        # 检查 crossattn_cache 是否已初始化
        if not crossattn_cache[0].get("is_init", False):
            import warnings
            warnings.warn("[MemoryBank] crossattn_cache not initialized, selecting first frame by default")
            frame_scores = torch.ones(evicted_chunk_kv[0]["k"].shape[1] // self.frame_seq_length)
        else:
            # 方案 A: 使用 crossattn_cache + 实体聚焦加权
            frame_scores = self._compute_frame_scores_with_entity_focus(
                evicted_chunk_kv[0],
                crossattn_cache[0],
                current_entities,
                prompt_text
            )

        # 选择最高分帧
        best_frame_idx = frame_scores.argmax().item()
        best_score = frame_scores[best_frame_idx].item()

        print(f"[DEBUG] select_frame_from_chunk: frame_scores={frame_scores.tolist()}")
        print(f"[DEBUG] select_frame_from_chunk: best_frame_idx={best_frame_idx}, best_score={best_score:.4f}")

        # 生成frame_id
        frame_id = f"p{prompt_id}_c{chunk_id}_f{best_frame_idx}"

        # 提取该帧的KV cache (所有30个block)
        frame_kv = self._extract_frame_kv_all_blocks(evicted_chunk_kv, best_frame_idx)

        # 保存帧信息
        frame_info = FrameInfo(
            frame_id=frame_id,
            frame_path=os.path.join(self.save_dir, f"{frame_id}.pt"),
            prompt_id=prompt_id,
            associated_entities=[str(eid) for eid in current_entity_ids],
            score=best_score,
            kv_cache=frame_kv
        )

        self.frame_archive[frame_id] = frame_info
        self._frame_kv_store[frame_id] = frame_kv

        # 保存KV到文件
        self._save_frame_kv(frame_id, frame_kv)

        return frame_id, best_score

    def _compute_frame_scores_with_crossattn(self,
                                              chunk_kv: Dict[str, torch.Tensor],
                                              crossattn_cache_block: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        使用 crossattn_cache["k"] 计算每帧分数

        与 MemFlow 的 compress_kv_bank 方法一致:
        - text_q = crossattn_cache["k"]  # [B, 512, H, D]
        - chunk_k = chunk_kv["k"]        # [B, L, H, D]
        - 计算 attention score

        Args:
            chunk_kv: 单个 block 的 chunk KV, {"k": [B, L, H, D], "v": [B, L, H, D]}
            crossattn_cache_block: 单个 block 的 crossattn cache

        Returns:
            每帧的分数 [num_frames]
        """
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        text_q = crossattn_cache_block["k"]  # [B, 512, H, D]

        B, L, H, D = chunk_k.shape

        # 计算帧数
        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.tensor([1.0], device=chunk_k.device)

        # 按 MemFlow compress_kv_bank 的方式计算注意力分数
        # q_reshaped: [B*H, 512, D]
        # k_reshaped: [B*H, L, D]
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = chunk_k.permute(0, 2, 1, 3).reshape(B * H, L, D)

        # 计算 attention scores: [B*H, 512, L]
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D ** -0.5)

        # 聚合分数:
        # 1. 对 text tokens 平均: [B*H, L]
        # 2. 对 heads 平均: [B, L]
        # 3. 按帧分组平均: [B, num_frames]
        scores_per_token = attn_scores.mean(dim=1)  # [B*H, L]
        scores_per_token = scores_per_token.view(B, H, L).mean(dim=1)  # [B, L]

        # 按帧分组
        frame_scores = []
        for i in range(num_frames):
            start = i * self.frame_seq_length
            end = (i + 1) * self.frame_seq_length
            frame_score = scores_per_token[:, start:end].mean()  # scalar
            frame_scores.append(frame_score)

        return torch.tensor(frame_scores, device=chunk_k.device)

    def _compute_frame_scores_with_entity_focus(self,
                                                  chunk_kv: Dict[str, torch.Tensor],
                                                  crossattn_cache_block: Dict[str, torch.Tensor],
                                                  entities: Optional[List['EntityStruct']] = None,
                                                  prompt_text: Optional[str] = None) -> torch.Tensor:
        """
        方案 A: 使用 crossattn_cache 计算帧分数，但聚焦于实体相关的 token

        原理:
        1. 使用原始 crossattn 方法计算每个 text token 对每帧的注意力分数
        2. 根据 entity/attrs 在 prompt 中的位置构建权重向量
        3. 加权聚合得到最终帧分数

        优势:
        - 保持特征空间对齐（crossattn_cache 与 chunk_kv 在同一空间）
        - 通过精确定位实体位置加权，聚焦于实体相关的 token

        Args:
            chunk_kv: 单个 block 的 chunk KV, {"k": [B, L, H, D], "v": [B, L, H, D]}
            crossattn_cache_block: 单个 block 的 crossattn cache
            entities: 实体列表（用于构建权重，可选）
            prompt_text: 原始 prompt 文本（用于精确定位实体位置）

        Returns:
            每帧的分数 [num_frames]
        """
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        text_q = crossattn_cache_block["k"]  # [B, 512, H, D]

        B, L, H, D = chunk_k.shape
        num_text_tokens = text_q.shape[1]  # 512

        # 计算帧数
        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.tensor([1.0], device=chunk_k.device)

        # Step 1: 计算完整的 token-to-position 注意力矩阵
        # q_reshaped: [B*H, 512, D]
        # k_reshaped: [B*H, L, D]
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = chunk_k.permute(0, 2, 1, 3).reshape(B * H, L, D)

        # attn_scores: [B*H, 512, L]
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D ** -0.5)

        # Step 2: 构建实体权重向量 [512]
        entity_weights = self._build_entity_token_weights(entities, num_text_tokens, prompt_text)
        entity_weights = entity_weights.to(chunk_k.device)  # [512]

        if entities is not None and len(entities) > 0:
            entity_query = self.build_entity_attrs_query(entities)
            print(f"[DEBUG] Entity-focus mode: '{entity_query[:50]}...' (weights applied)")
        else:
            print(f"[DEBUG] No entities, using uniform weights")

        # Step 3: 加权聚合
        # attn_scores: [B*H, 512, L]
        # entity_weights: [512] -> [1, 512, 1]
        weights = entity_weights.view(1, -1, 1)  # [1, 512, 1]

        # 加权注意力分数
        weighted_scores = attn_scores * weights  # [B*H, 512, L]

        # 对 text tokens 加权求和
        # scores_per_position: [B*H, L]
        scores_per_position = weighted_scores.sum(dim=1) / (weights.sum() + 1e-8)

        # 对 heads 平均: [B, L]
        scores_per_position = scores_per_position.view(B, H, L).mean(dim=1)

        # Step 4: 按帧分组
        frame_scores = []
        for i in range(num_frames):
            start = i * self.frame_seq_length
            end = (i + 1) * self.frame_seq_length
            frame_score = scores_per_position[:, start:end].mean()  # scalar
            frame_scores.append(frame_score)

        return torch.tensor(frame_scores, device=chunk_k.device)

    def _build_entity_token_weights(self,
                                     entities: Optional[List['EntityStruct']],
                                     num_tokens: int,
                                     prompt_text: Optional[str] = None) -> torch.Tensor:
        """
        构建实体 token 权重向量

        策略:
        - 无实体信息时，返回均匀权重
        - 有实体信息时，精确定位 entity 和 attrs 在 prompt 中的位置并加权

        Args:
            entities: 实体列表（可选）
            num_tokens: token 总数（通常为 512）
            prompt_text: 原始 prompt 文本（用于精确定位）

        Returns:
            权重向量 [num_tokens]
        """
        weights = torch.ones(num_tokens)

        if entities is None or len(entities) == 0:
            # 无实体信息，返回均匀权重
            return weights

        if prompt_text is None or len(prompt_text) == 0:
            # 无 prompt 文本，使用简单的中间区域加权（fallback）
            entity_start = int(num_tokens * 0.10)
            entity_end = int(num_tokens * 0.85)
            weights[entity_start:entity_end] = 1.5
            return weights

        # 精确定位 entity 和 attrs 在 prompt 中的位置
        prompt_lower = prompt_text.lower()
        prompt_len = len(prompt_text)

        # 收集所有关键词及其在 prompt 中的位置
        keyword_positions = []  # [(start_ratio, end_ratio), ...]

        for entity in entities:
            # 查找 entity 名称
            entity_lower = entity.entity.lower()
            pos = prompt_lower.find(entity_lower)
            if pos != -1:
                start_ratio = pos / prompt_len
                end_ratio = (pos + len(entity_lower)) / prompt_len
                keyword_positions.append((start_ratio, end_ratio))

            # 查找每个属性
            for attr in entity.attrs:
                attr_lower = attr.lower()
                pos = prompt_lower.find(attr_lower)
                if pos != -1:
                    start_ratio = pos / prompt_len
                    end_ratio = (pos + len(attr_lower)) / prompt_len
                    keyword_positions.append((start_ratio, end_ratio))

        if not keyword_positions:
            # 没有找到任何关键词，使用 fallback
            entity_start = int(num_tokens * 0.10)
            entity_end = int(num_tokens * 0.85)
            weights[entity_start:entity_end] = 1.5
            return weights

        # 将字符位置比例映射到 token 位置，并应用权重
        # 假设 token 位置与字符位置大致成比例（简化假设）
        base_weight = 1.0
        entity_weight = 2.5  # 实体相关区域的权重

        for start_ratio, end_ratio in keyword_positions:
            # 扩展一点范围，因为 tokenization 可能跨越边界
            start_token = max(0, int((start_ratio - 0.02) * num_tokens))
            end_token = min(num_tokens, int((end_ratio + 0.02) * num_tokens))

            # 应用权重（取最大值，避免重叠区域被覆盖）
            for i in range(start_token, end_token):
                weights[i] = max(weights[i].item(), entity_weight)

        # 对非实体区域稍微降权
        # 特别是开头的场景描述和结尾的镜头描述
        scene_end = int(num_tokens * 0.08)
        camera_start = int(num_tokens * 0.92)

        for i in range(scene_end):
            if weights[i] == base_weight:  # 只降权未被标记为实体的部分
                weights[i] = 0.7

        for i in range(camera_start, num_tokens):
            if weights[i] == base_weight:
                weights[i] = 0.5

        print(f"[DEBUG] Entity keyword positions: {len(keyword_positions)} keywords found")

        return weights

    def _extract_frame_kv_all_blocks(self,
                                      all_blocks_kv: List[Dict[str, torch.Tensor]],
                                      frame_idx: int) -> List[Dict[str, torch.Tensor]]:
        """
        从所有 block 的 chunk KV 中提取指定帧的 KV

        Args:
            all_blocks_kv: 所有 block 的 chunk KV
            frame_idx: 帧索引

        Returns:
            该帧在所有 block 的 KV cache
        """
        start = frame_idx * self.frame_seq_length
        end = (frame_idx + 1) * self.frame_seq_length

        frame_kv_all_blocks = []
        for block_kv in all_blocks_kv:
            frame_kv_all_blocks.append({
                "k": block_kv["k"][:, start:end].clone(),
                "v": block_kv["v"][:, start:end].clone()
            })

        return frame_kv_all_blocks

    def _save_frame_kv(self, frame_id: str, frame_kv: List[Dict[str, torch.Tensor]]) -> None:
        """保存帧KV到文件 (所有 block)"""
        if not self.save_frames_to_disk:
            return  # 仅在内存中保存，跳过磁盘 I/O

        path = os.path.join(self.save_dir, f"{frame_id}.pt")
        torch.save(frame_kv, path)

    def _load_frame_kv(self, frame_id: str) -> Optional[List[Dict[str, torch.Tensor]]]:
        """加载帧KV (所有 block)"""
        if frame_id in self._frame_kv_store:
            return self._frame_kv_store[frame_id]

        path = os.path.join(self.save_dir, f"{frame_id}.pt")
        if os.path.exists(path):
            kv = torch.load(path, weights_only=False)
            self._frame_kv_store[frame_id] = kv
            return kv
        return None

    # ============ Active Memory管理 ============

    def update_active_memory(self, frame_id: str, score: float) -> None:
        """
        按分数更新top-k记忆帧

        Args:
            frame_id: 新帧ID
            score: 新帧分数
        """
        if frame_id not in self.frame_archive:
            return

        old_memory = self.frame_active_memory.copy()

        if len(self.frame_active_memory) < self.max_memory_frames:
            # 未达上限，直接添加
            if frame_id not in self.frame_active_memory:
                self.frame_active_memory.append(frame_id)
                print(f"[DEBUG] update_active_memory: Added {frame_id} (score={score:.4f}), memory not full")
        else:
            # 已达上限，比较分数
            # 找到当前active memory中分数最低的帧
            min_score = float('inf')
            min_idx = -1
            min_fid = None

            for idx, fid in enumerate(self.frame_active_memory):
                if fid in self.frame_archive:
                    finfo = self.frame_archive[fid]
                    if finfo.score < min_score:
                        min_score = finfo.score
                        min_idx = idx
                        min_fid = fid

            # 如果新帧分数更高，替换最低分帧
            if score > min_score and min_idx >= 0:
                self.frame_active_memory[min_idx] = frame_id
                print(f"[DEBUG] update_active_memory: Replaced {min_fid}(score={min_score:.4f}) with {frame_id}(score={score:.4f})")
            else:
                print(f"[DEBUG] update_active_memory: Kept existing frames, new score {score:.4f} <= min_score {min_score:.4f}")

    def get_memory_kv(self, device: torch.device = None) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        获取当前active memory的KV cache (所有 block)

        Returns:
            List[{"k": [B, L, H, D], "v": [B, L, H, D]}] - 每个 block 的拼接 KV
            如果没有记忆帧则返回None
        """
        if not self.frame_active_memory:
            return None

        # 收集所有帧的 KV
        all_frames_kv = []
        for frame_id in self.frame_active_memory:
            kv = self._load_frame_kv(frame_id)
            if kv is not None:
                all_frames_kv.append(kv)

        if not all_frames_kv:
            return None

        # 拼接每个 block 的 KV
        # all_frames_kv: List[List[Dict]] - [num_frames, num_blocks, {"k", "v"}]
        num_blocks = len(all_frames_kv[0])
        result = []

        for block_idx in range(num_blocks):
            k_list = []
            v_list = []
            for frame_kv in all_frames_kv:
                k = frame_kv[block_idx]["k"]
                v = frame_kv[block_idx]["v"]
                if device is not None:
                    k = k.to(device)
                    v = v.to(device)
                k_list.append(k)
                v_list.append(v)

            result.append({
                "k": torch.cat(k_list, dim=1),
                "v": torch.cat(v_list, dim=1)
            })

        return result

    def get_active_frame_count(self) -> int:
        """获取当前active memory中的帧数"""
        return len(self.frame_active_memory)

    # ============ 持久化相关 ============

    def save_to_json(self, path: str) -> None:
        """
        保存Memory Bank状态到JSON文件

        Args:
            path: 保存路径
        """
        data = {
            "global_registry": self.global_registry,
            "frame_archive": {
                fid: finfo.to_dict()
                for fid, finfo in self.frame_archive.items()
            },
            "frame_active_memory": self.frame_active_memory
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_from_json(self, path: str) -> None:
        """
        从JSON文件加载Memory Bank状态

        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.global_registry = data.get("global_registry", {})
        self.frame_active_memory = data.get("frame_active_memory", [])

        # 加载frame_archive
        self.frame_archive = {}
        for fid, finfo_dict in data.get("frame_archive", {}).items():
            self.frame_archive[fid] = FrameInfo(
                frame_id=fid,
                frame_path=finfo_dict.get("frame_path", ""),
                prompt_id=finfo_dict.get("prompt_id", 0),
                associated_entities=finfo_dict.get("associated_entities", []),
                score=finfo_dict.get("score", 0.0)
            )

    def clear(self) -> None:
        """清空Memory Bank"""
        self.global_registry = {}
        self.frame_archive = {}
        self.frame_active_memory = []
        self._frame_kv_store = {}

    # ============ 辅助方法 ============

    def build_entity_attrs_query(self, entities: List[EntityStruct]) -> str:
        """
        构建用于帧选择的entity-attrs查询文本

        Args:
            entities: 实体列表

        Returns:
            拼接的查询文本
        """
        parts = []
        for entity in entities:
            entity_str = entity.entity
            attrs_str = " ".join(entity.attrs)
            parts.append(f"{entity_str} {attrs_str}")

        return " ".join(parts)

    def get_entity_ids(self, entities: List[EntityStruct]) -> List[int]:
        """获取实体ID列表"""
        return [e.global_id for e in entities if e.global_id is not None]

    def get_registry_summary(self) -> str:
        """获取registry摘要信息"""
        lines = []
        for gid, info in self.global_registry.items():
            entities = info.get("all_entities", [])
            attrs = info.get("all_attrs", [])
            lines.append(f"ID {gid} ({info.get('name', 'unknown')}): entities={entities}")

        return "\n".join(lines)


# ============ 测试代码 ============

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("Testing Memory Bank (Multi-Block Version)")
    print("=" * 60)

    # 配置
    num_blocks = 30  # transformer blocks
    batch_size = 1
    num_frames = 3
    seq_len = num_frames * 1560
    num_heads = 12
    head_dim = 128

    # 创建Memory Bank
    memory_bank = MemoryBank(
        text_encoder=None,
        max_memory_frames=3,
        num_transformer_blocks=num_blocks
    )

    # 测试实体注册
    print("\n1. Testing entity registration...")

    entities_p1 = [
        EntityStruct(entity="young man", attrs=["late 20s", "messy black hair", "denim jacket"], global_id=1)
    ]
    memory_bank.register_entities(entities_p1, prompt_id=1)
    print(f"After P1: {memory_bank.global_registry}")

    entities_p2 = [
        EntityStruct(entity="protagonist", attrs=["denim jacket", "seated on bench"], global_id=1),
        EntityStruct(entity="another man", attrs=["30 years old", "glasses", "grey sweater"], global_id=2)
    ]
    memory_bank.register_entities(entities_p2, prompt_id=2)
    print(f"After P2: {memory_bank.global_registry}")

    # 测试帧选择
    print("\n2. Testing frame selection (with mock crossattn_cache)...")

    # 测试用的 prompt 文本
    test_prompt = "A realistic video of a modern city park. The protagonist in the denim jacket is seated on bench. Another man, 30 years old, wearing glasses and a grey sweater, walks into the frame."

    # 模拟所有 block 的 chunk KV
    mock_chunk_kv_all_blocks = []
    for _ in range(num_blocks):
        mock_chunk_kv_all_blocks.append({
            "k": torch.randn(batch_size, seq_len, num_heads, head_dim),
            "v": torch.randn(batch_size, seq_len, num_heads, head_dim)
        })

    # 模拟 crossattn_cache (已初始化)
    mock_crossattn_cache = []
    for _ in range(num_blocks):
        mock_crossattn_cache.append({
            "k": torch.randn(batch_size, 512, num_heads, head_dim),
            "v": torch.randn(batch_size, 512, num_heads, head_dim),
            "is_init": True
        })

    frame_id, score = memory_bank.select_frame_from_chunk(
        evicted_chunk_kv=mock_chunk_kv_all_blocks,
        crossattn_cache=mock_crossattn_cache,
        prompt_id=1,
        chunk_id=3,
        current_entity_ids=[1, 2],
        current_entities=entities_p2,
        prompt_text=test_prompt  # 测试精确定位实体位置
    )
    print(f"Selected frame: {frame_id}, score: {score:.4f}")
    print(f"Frame KV stored for {num_blocks} blocks")

    # 测试active memory更新
    print("\n3. Testing active memory update (max_memory_frames=3)...")
    memory_bank.update_active_memory(frame_id, score)
    print(f"[Init] Frame {frame_id} (score={score:.4f}) -> ACCEPTED (slot 1/3)")
    print(f"  Active memory: {memory_bank.frame_active_memory}")

    # 添加更多帧，测试top-k保留逻辑
    for chunk_id in range(4, 8):
        mock_kv = []
        for _ in range(num_blocks):
            mock_kv.append({
                "k": torch.randn(batch_size, seq_len, num_heads, head_dim),
                "v": torch.randn(batch_size, seq_len, num_heads, head_dim)
            })

        fid, sc = memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=mock_kv,
            crossattn_cache=mock_crossattn_cache,
            prompt_id=1,
            chunk_id=chunk_id,
            current_entity_ids=[1, 2],
            current_entities=entities_p2,
            prompt_text=test_prompt  # 测试精确定位实体位置
        )

        # 获取更新前的状态
        old_memory = memory_bank.frame_active_memory.copy()
        current_count = len(old_memory)
        old_min_score = min(
            memory_bank.frame_archive[f].score for f in old_memory
        ) if current_count > 0 else None

        memory_bank.update_active_memory(fid, sc)

        # 判断结果
        if fid in memory_bank.frame_active_memory:
            if current_count < memory_bank.max_memory_frames:
                print(f"[Chunk {chunk_id}] Frame {fid} (score={sc:.4f}) -> ACCEPTED (slot {current_count+1}/3)")
            else:
                print(f"[Chunk {chunk_id}] Frame {fid} (score={sc:.4f}) -> ACCEPTED (replaced min={old_min_score:.4f})")
        else:
            print(f"[Chunk {chunk_id}] Frame {fid} (score={sc:.4f}) -> REJECTED (min={old_min_score:.4f} is higher)")

        print(f"  Active memory: {memory_bank.frame_active_memory}")

    # 测试获取记忆 KV
    print("\n4. Testing get_memory_kv...")
    memory_kv = memory_bank.get_memory_kv()
    if memory_kv:
        print(f"Memory KV: {len(memory_kv)} blocks")
        print(f"  Block 0 K shape: {memory_kv[0]['k'].shape}")
        print(f"  Block 0 V shape: {memory_kv[0]['v'].shape}")

    # 测试帧检索
    print("\n5. Testing frame retrieval...")
    retrieved = memory_bank.retrieve_initial_frames([1, 2])
    print(f"Retrieved frames for entities [1, 2]: {retrieved}")

    # 测试保存/加载
    print("\n6. Testing save/load...")
    memory_bank.save_to_json("test_memory_bank.json")
    print("Saved to test_memory_bank.json")

    memory_bank_loaded = MemoryBank(text_encoder=None, num_transformer_blocks=num_blocks)
    memory_bank_loaded.load_from_json("test_memory_bank.json")
    print(f"Loaded global_registry: {memory_bank_loaded.global_registry}")
    print(f"Loaded frame_archive keys: {list(memory_bank_loaded.frame_archive.keys())}")
    print(f"Loaded active_memory: {memory_bank_loaded.frame_active_memory}")

    # 清理测试文件
    if os.path.exists("test_memory_bank.json"):
        os.remove("test_memory_bank.json")
    # 清理 KV 文件
    for fid in memory_bank.frame_archive:
        kv_path = os.path.join(memory_bank.save_dir, f"{fid}.pt")
        if os.path.exists(kv_path):
            os.remove(kv_path)

    print("\nTest completed!")
