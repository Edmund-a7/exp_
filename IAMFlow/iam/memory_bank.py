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
import re
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
    score: float                    # 综合分数 (向后兼容)
    entity_score: float = 0.0      # entity 维度分数
    scene_score: float = 0.0       # scene 维度分数
    scene_texts: List[str] = field(default_factory=list)  # 该帧对应 prompt 的场景文本
    # 存储所有 transformer block 的 KV cache
    # List[Dict[str, torch.Tensor]] - 每个 block 一个 {"k": ..., "v": ...}
    kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None

    def to_dict(self) -> Dict:
        return {
            "frame_path": self.frame_path,
            "prompt_id": self.prompt_id,
            "associated_entities": self.associated_entities,
            "score": self.score,
            "entity_score": self.entity_score,
            "scene_score": self.scene_score,
            "scene_texts": self.scene_texts,
        }


class MemoryBank:
    """
    记忆库管理器

    职责:
    1. 维护 global_registry (实体注册表)
    2. 维护 frame_archive (帧存档)
    3. 维护双层 active memory: id_memory (管"谁") + scene_memory (管"在哪")
    4. 帧选择和驱逐
    """

    def __init__(self,
                 text_encoder=None,
                 max_memory_frames: int = 3,
                 max_id_memory_frames: int = 4,
                 max_scene_memory_frames: int = 2,
                 frame_seq_length: int = 1560,
                 num_transformer_blocks: int = 30,
                 save_dir: str = "data",
                 save_frames_to_disk: bool = False):
        """
        初始化Memory Bank

        Args:
            text_encoder: WanTextEncoder实例，用于文本编码 (现在不直接使用，改用 crossattn_cache)
            max_memory_frames: 最大记忆帧数量 (向后兼容，仅在未启用双层记忆时使用)
            max_id_memory_frames: ID Memory 最大帧数 (entity 驱动，1~4 帧)
            max_scene_memory_frames: Scene Memory 最大帧数 (scene 驱动，1~2 帧)
            frame_seq_length: 每帧的序列长度
            num_transformer_blocks: transformer block 数量 (默认30)
            save_dir: 帧数据保存目录
            save_frames_to_disk: 是否将帧 KV 保存到磁盘 (默认False，仅保存在内存中以提升性能)
        """
        self.text_encoder = text_encoder
        self.max_memory_frames = max_memory_frames
        self.max_id_memory_frames = max_id_memory_frames
        self.max_scene_memory_frames = max_scene_memory_frames
        self.frame_seq_length = frame_seq_length
        self.num_transformer_blocks = num_transformer_blocks
        self.save_dir = save_dir
        self.save_frames_to_disk = save_frames_to_disk

        # 核心数据结构
        self.global_registry: Dict[str, Dict] = {}
        self.frame_archive: Dict[str, FrameInfo] = {}

        # 双层 active memory (Phase 2)
        self.id_memory: List[str] = []       # entity 驱动，按 entity_score top-k
        self.scene_memory: List[str] = []    # scene 驱动，按 scene_score top-k

        # KV cache存储 (内存中) - 每个 frame_id 对应 30 个 block 的 KV
        self._frame_kv_store: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

    @property
    def frame_active_memory(self) -> List[str]:
        """向后兼容属性: 返回 id_memory + scene_memory 的去重合并"""
        return list(dict.fromkeys(self.id_memory + self.scene_memory))

    @frame_active_memory.setter
    def frame_active_memory(self, value: List[str]):
        """向后兼容 setter: 直接赋值时写入 id_memory，清空 scene_memory"""
        self.id_memory = list(value)
        self.scene_memory = []

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

    def _compute_dynamic_id_budget(self, required_entity_ids: List[str]) -> int:
        """
        根据 ID 覆盖度计算 ID Memory 需要的帧数 (贪心集合覆盖)

        策略:
        1. 优先选覆盖所有 ID 的帧 (1帧搞定 → budget=1)
        2. 不够则贪心选覆盖最多未覆盖 ID 的帧
        3. 直到所有 ID 被覆盖，budget = 选出的帧数
        4. 上限 max_id_memory_frames

        Args:
            required_entity_ids: 需要覆盖的实体 ID 列表 (str)

        Returns:
            动态帧预算
        """
        if not required_entity_ids or not self.frame_archive:
            return 0

        uncovered = set(required_entity_ids)
        budget = 0
        used_frames = set()

        while uncovered and budget < self.max_id_memory_frames:
            best_fid = None
            best_cover = 0
            best_score = -float('inf')

            for fid, fi in self.frame_archive.items():
                if fid in used_frames:
                    continue
                cover = len(set(fi.associated_entities) & uncovered)
                if cover > best_cover or (cover == best_cover and fi.entity_score > best_score):
                    best_fid = fid
                    best_cover = cover
                    best_score = fi.entity_score

            if best_fid is None or best_cover == 0:
                break

            used_frames.add(best_fid)
            uncovered -= set(self.frame_archive[best_fid].associated_entities)
            budget += 1

        # 至少 1 帧 (如果有实体需求且有帧可覆盖)
        return budget

    def _greedy_select_id_frames(self, required_entity_ids: List[str], budget: int) -> List[str]:
        """
        贪心选择覆盖最多 ID 的帧集合

        Args:
            required_entity_ids: 需要覆盖的实体 ID 列表 (str)
            budget: 帧预算

        Returns:
            选中的 frame_id 列表
        """
        if not required_entity_ids or not self.frame_archive or budget <= 0:
            return []

        uncovered = set(required_entity_ids)
        selected = []

        while uncovered and len(selected) < budget:
            best_fid = None
            best_cover = 0
            best_score = -float('inf')

            for fid, fi in self.frame_archive.items():
                if fid in selected:
                    continue
                cover = len(set(fi.associated_entities) & uncovered)
                if cover > best_cover or (cover == best_cover and fi.entity_score > best_score):
                    best_fid = fid
                    best_cover = cover
                    best_score = fi.entity_score

            if best_fid is None or best_cover == 0:
                break

            selected.append(best_fid)
            uncovered -= set(self.frame_archive[best_fid].associated_entities)

        # 如果还有预算剩余且 uncovered 为空，用 entity_score top-k 填充
        if len(selected) < budget:
            remaining = [
                (fid, fi.entity_score)
                for fid, fi in self.frame_archive.items()
                if fid not in selected
            ]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for fid, _ in remaining:
                if len(selected) >= budget:
                    break
                selected.append(fid)

        return selected

    def retrieve_initial_frames(self, entity_ids: List[int],
                                scene_texts: Optional[List[str]] = None) -> List[str]:
        """
        双路检索: 分别填充 id_memory 和 scene_memory

        ID Memory: 按 entity_id 匹配 + entity_score 排序
        Scene Memory: 按 scene_score 排序 (如果有 scene_texts)

        Args:
            entity_ids: 当前prompt的实体ID列表
            scene_texts: 当前prompt的场景文本列表

        Returns:
            去重合并后的 frame_id 列表 (= frame_active_memory property)
        """
        print(f"[DEBUG] retrieve_initial_frames: entity_ids={entity_ids}, scene_texts={scene_texts}, archive_size={len(self.frame_archive)}")

        if not self.frame_archive:
            print(f"[DEBUG] retrieve_initial_frames: No frames in archive")
            return []

        # === ID Memory 路径: 动态预算 + 贪心集合覆盖 ===
        entity_id_strs = [str(eid) for eid in entity_ids]
        if entity_id_strs:
            budget = self._compute_dynamic_id_budget(entity_id_strs)
            self.id_memory = self._greedy_select_id_frames(entity_id_strs, budget)
            print(f"[DEBUG] retrieve_initial_frames: dynamic budget={budget}, id_memory={self.id_memory}")
        else:
            # 无实体时不填充 ID Memory，保持双路径独立
            self.id_memory = []
            print(f"[DEBUG] retrieve_initial_frames: no entities, id_memory=[]")

        # === Scene Memory 路径: 按 scene_score ===
        if scene_texts:
            scene_candidates = []
            for fid, fi in self.frame_archive.items():
                relevance = self._compute_scene_relevance(fi, scene_texts)
                scene_candidates.append((fid, relevance, fi.scene_score))

            # 先按当前 scene query 的匹配度排序，再用历史 scene_score 打破平局
            scene_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            self.scene_memory = [f[0] for f in scene_candidates[:self.max_scene_memory_frames]]
            print(f"[DEBUG] retrieve_initial_frames: scene_memory={self.scene_memory}")
        else:
            self.scene_memory = []

        result = self.frame_active_memory  # property: 去重合并
        print(f"[DEBUG] retrieve_initial_frames: combined={result}")
        return result

    # ============ 帧选择相关 ============

    def select_frame_from_chunk(self,
                                evicted_chunk_kv: List[Dict[str, torch.Tensor]],
                                crossattn_cache: List[Dict[str, torch.Tensor]],
                                prompt_id: int,
                                chunk_id: int,
                                current_entity_ids: List[int],
                                current_entities: Optional[List['EntityStruct']] = None,
                                prompt_text: Optional[str] = None,
                                scene_texts: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        从驱逐的chunk中选择最佳帧 (双路打分)

        同时计算 entity_score 和 scene_score，综合分数 = 0.6*entity + 0.4*scene

        Args:
            evicted_chunk_kv: 驱逐chunk的KV cache (所有30个block)
            crossattn_cache: cross-attention cache (所有30个block)
            prompt_id: 当前prompt序号
            chunk_id: 当前chunk序号
            current_entity_ids: 当前prompt的实体ID列表
            current_entities: 当前prompt的实体列表
            prompt_text: 当前 prompt 文本
            scene_texts: 当前 prompt 的场景文本列表

        Returns:
            (frame_id, score) - 选中的帧ID和综合分数
        """
        if not evicted_chunk_kv or not crossattn_cache:
            raise ValueError("evicted_chunk_kv and crossattn_cache must not be empty")

        num_candidate_frames = max(1, evicted_chunk_kv[0]["k"].shape[1] // self.frame_seq_length)
        available_layers = min(len(evicted_chunk_kv), len(crossattn_cache))
        has_initialized_layer = any(
            crossattn_cache[layer_idx].get("is_init", False)
            for layer_idx in range(available_layers)
        )

        device = evicted_chunk_kv[0]["k"].device
        dtype = evicted_chunk_kv[0]["k"].dtype

        if not has_initialized_layer:
            import warnings
            warnings.warn("[MemoryBank] crossattn_cache not initialized, selecting first frame by default")
            entity_scores = torch.ones(num_candidate_frames, device=device, dtype=dtype)
            scene_scores = torch.ones(num_candidate_frames, device=device, dtype=dtype)
        else:
            num_text_tokens = crossattn_cache[0]["k"].shape[1]

            # 1. Entity 路径: entity+attrs token 权重打分
            entity_weights = self._build_entity_token_weights(
                current_entities, num_text_tokens, prompt_text
            )
            entity_scores = self._consensus_score(
                evicted_chunk_kv, crossattn_cache, entity_weights,
                available_layers, num_candidate_frames
            )

            # 2. Scene 路径: scene text token 权重打分
            if scene_texts:
                scene_weights = self._build_scene_token_weights(
                    scene_texts, num_text_tokens, prompt_text, current_entities
                )
                scene_scores = self._consensus_score(
                    evicted_chunk_kv, crossattn_cache, scene_weights,
                    available_layers, num_candidate_frames
                )
            else:
                scene_scores = torch.zeros(num_candidate_frames, device=device, dtype=dtype)

        # 3. 综合分数
        combined_scores = 0.6 * entity_scores + 0.4 * scene_scores

        # 选择最高分帧
        best_frame_idx = combined_scores.argmax().item()
        best_score = combined_scores[best_frame_idx].item()
        best_entity_score = entity_scores[best_frame_idx].item()
        best_scene_score = scene_scores[best_frame_idx].item()

        print(f"[DEBUG] select_frame_from_chunk: entity_scores={entity_scores.tolist()}")
        print(f"[DEBUG] select_frame_from_chunk: scene_scores={scene_scores.tolist()}")
        print(f"[DEBUG] select_frame_from_chunk: combined={combined_scores.tolist()}")
        print(f"[DEBUG] select_frame_from_chunk: best_frame_idx={best_frame_idx}, score={best_score:.4f} (entity={best_entity_score:.4f}, scene={best_scene_score:.4f})")

        # 生成frame_id
        frame_id = f"p{prompt_id}_c{chunk_id}_f{best_frame_idx}"

        # 提取该帧的KV cache (所有30个block)
        frame_kv = self._extract_frame_kv_all_blocks(evicted_chunk_kv, best_frame_idx)

        # 保存帧信息 (含双分数)
        frame_info = FrameInfo(
            frame_id=frame_id,
            frame_path=os.path.join(self.save_dir, f"{frame_id}.pt"),
            prompt_id=prompt_id,
            associated_entities=list(dict.fromkeys(str(eid) for eid in current_entity_ids)),
            score=best_score,
            entity_score=best_entity_score,
            scene_score=best_scene_score,
            scene_texts=(scene_texts or []).copy(),
            kv_cache=frame_kv
        )

        self.frame_archive[frame_id] = frame_info
        self._frame_kv_store[frame_id] = frame_kv

        # 保存KV到文件
        self._save_frame_kv(frame_id, frame_kv)

        return frame_id, best_score

    def _consensus_score(self,
                         evicted_chunk_kv: List[Dict[str, torch.Tensor]],
                         crossattn_cache: List[Dict[str, torch.Tensor]],
                         token_weights: torch.Tensor,
                         available_layers: int,
                         num_candidate_frames: int) -> torch.Tensor:
        """
        多层共识打分通用方法: 用 3 个代表层投票

        Args:
            evicted_chunk_kv: 驱逐chunk的KV cache
            crossattn_cache: cross-attention cache
            token_weights: token 权重向量 [S]
            available_layers: 可用层数
            num_candidate_frames: 候选帧数

        Returns:
            [num_frames] 每帧分数
        """
        device = evicted_chunk_kv[0]["k"].device
        dtype = evicted_chunk_kv[0]["k"].dtype

        valid_layers = []
        valid_weights = []
        for layer_idx, layer_weight in zip(self.CONSENSUS_LAYERS, self.CONSENSUS_WEIGHTS):
            if layer_idx < available_layers and crossattn_cache[layer_idx].get("is_init", False):
                valid_layers.append(layer_idx)
                valid_weights.append(layer_weight)

        if not valid_layers:
            fallback_layer = 0
            for layer_idx in range(available_layers):
                if crossattn_cache[layer_idx].get("is_init", False):
                    fallback_layer = layer_idx
                    break
            valid_layers = [fallback_layer]
            valid_weights = [1.0]

        weight_sum = sum(valid_weights)
        if weight_sum <= 0:
            valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)
        else:
            valid_weights = [w / weight_sum for w in valid_weights]

        frame_scores = None
        for layer_idx, layer_weight in zip(valid_layers, valid_weights):
            layer_scores = self._compute_frame_scores_fast(
                evicted_chunk_kv[layer_idx],
                crossattn_cache[layer_idx],
                token_weights
            )
            std = layer_scores.std()
            if std > 1e-8:
                layer_scores = (layer_scores - layer_scores.mean()) / std
            if frame_scores is None:
                frame_scores = torch.zeros_like(layer_scores)
            frame_scores = frame_scores + layer_scores * layer_weight

        if frame_scores is None:
            frame_scores = torch.ones(num_candidate_frames, device=device, dtype=dtype)

        return frame_scores

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
            return torch.tensor([1.0], device=chunk_k.device, dtype=chunk_k.dtype)

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

        return torch.tensor(frame_scores, device=chunk_k.device, dtype=chunk_k.dtype)

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
            return torch.tensor([1.0], device=chunk_k.device, dtype=chunk_k.dtype)

        # Step 1: 计算完整的 token-to-position 注意力矩阵
        # q_reshaped: [B*H, 512, D]
        # k_reshaped: [B*H, L, D]
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = chunk_k.permute(0, 2, 1, 3).reshape(B * H, L, D)

        # attn_scores: [B*H, 512, L]
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D ** -0.5)

        # Step 2: 构建实体权重向量 [512]
        entity_weights = self._build_entity_token_weights(entities, num_text_tokens, prompt_text)
        entity_weights = entity_weights.to(device=chunk_k.device, dtype=chunk_k.dtype)  # [512]

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

        return torch.tensor(frame_scores, device=chunk_k.device, dtype=chunk_k.dtype)

    # ============ 多层共识快速打分 ============

    # 代表层索引和权重: 浅层(纹理) / 中层(结构) / 深层(语义)
    # layer 0 权重最高：经过验证的 baseline，且浅层 cross-attn K 与视觉 token 的
    # 对齐最直接；深层过于抽象，帧间区分度低，仅作辅助修正
    CONSENSUS_LAYERS = [0, 15, 29]
    CONSENSUS_WEIGHTS = [0.5, 0.3, 0.2]

    def _compute_frame_scores_fast(self,
                                    chunk_kv: Dict[str, torch.Tensor],
                                    crossattn_cache_block: Dict[str, torch.Tensor],
                                    entity_weights: torch.Tensor) -> torch.Tensor:
        """
        等价快速打分：先聚合Q，再与帧均值K点积

        数学等价于 _compute_frame_scores_with_entity_focus，但复杂度从
        O(512 × L × D) 降到 O(512×D + F×D)，加速约 4000x。

        原始: sum_i(w_i * Q_i · K^T) = (sum_i(w_i * Q_i)) · K^T

        Args:
            chunk_kv: 单个 block 的 chunk KV, {"k": [B, L, H, D]}
            crossattn_cache_block: 单个 block 的 crossattn cache, {"k": [B, S, H, D]}
            entity_weights: [S] 实体权重向量 (S=512)

        Returns:
            [num_frames] 每帧分数
        """
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        text_q = crossattn_cache_block["k"]  # [B, S, H, D]
        B, L, H, D = chunk_k.shape

        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.ones(1, device=chunk_k.device, dtype=chunk_k.dtype)

        valid_length = num_frames * self.frame_seq_length
        if valid_length != L:
            chunk_k = chunk_k[:, :valid_length]

        # Step 1: 加权聚合 Q → [B, H, D]
        w = entity_weights.to(device=text_q.device, dtype=text_q.dtype)  # [S]
        w = w / (w.sum() + 1e-8)
        q_agg = torch.einsum('bshd,s->bhd', text_q, w)  # [B, H, D]

        # Step 2: K 按帧分组取均值 → [B, F, H, D]
        k_frames = chunk_k.reshape(B, num_frames, self.frame_seq_length, H, D)
        k_agg = k_frames.mean(dim=2)  # [B, F, H, D]

        # Step 3: 点积 → [B, H, F]
        scores = torch.einsum('bhd,bfhd->bhf', q_agg, k_agg) * (D ** -0.5)

        # Step 4: mean over batch and heads → [F]
        scores = scores.mean(dim=(0, 1))  # [F]
        return scores

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

    def _build_scene_token_weights(self,
                                    scene_texts: Optional[List[str]],
                                    num_tokens: int,
                                    prompt_text: Optional[str] = None,
                                    entities: Optional[List['EntityStruct']] = None) -> torch.Tensor:
        """
        构建 scene token 权重向量 — 与 _build_entity_token_weights 互补

        策略:
        - scene 关键词区域权重高 (2.5x)
        - entity/attrs 区域权重低 (0.5x)
        - 无 scene 信息时返回均匀权重

        Args:
            scene_texts: 场景文本列表 (如 ["snowy forest", "overcast daylight"])
            num_tokens: token 总数 (通常为 512)
            prompt_text: 原始 prompt 文本 (用于精确定位)
            entities: 实体列表 (用于降权 entity 区域)

        Returns:
            权重向量 [num_tokens]
        """
        weights = torch.ones(num_tokens)

        if not scene_texts:
            return weights

        if prompt_text is None or len(prompt_text) == 0:
            # 无 prompt 文本，使用开头区域加权 (场景描述通常在 prompt 开头)
            scene_end = int(num_tokens * 0.30)
            weights[:scene_end] = 2.0
            return weights

        prompt_lower = prompt_text.lower()
        prompt_len = len(prompt_text)

        # Step 1: 定位 scene 关键词并加权
        scene_weight = 2.5
        scene_positions = []

        for scene_text in scene_texts:
            scene_lower = scene_text.lower()
            pos = prompt_lower.find(scene_lower)
            if pos != -1:
                start_ratio = pos / prompt_len
                end_ratio = (pos + len(scene_lower)) / prompt_len
                scene_positions.append((start_ratio, end_ratio))

        for start_ratio, end_ratio in scene_positions:
            start_token = max(0, int((start_ratio - 0.02) * num_tokens))
            end_token = min(num_tokens, int((end_ratio + 0.02) * num_tokens))
            for i in range(start_token, end_token):
                weights[i] = max(weights[i].item(), scene_weight)

        # Step 2: 降权 entity/attrs 区域
        if entities:
            entity_suppress = 0.5
            for entity in entities:
                entity_lower = entity.entity.lower()
                pos = prompt_lower.find(entity_lower)
                if pos != -1:
                    start_ratio = pos / prompt_len
                    end_ratio = (pos + len(entity_lower)) / prompt_len
                    start_token = max(0, int((start_ratio - 0.02) * num_tokens))
                    end_token = min(num_tokens, int((end_ratio + 0.02) * num_tokens))
                    for i in range(start_token, end_token):
                        if weights[i] <= 1.0:  # 不覆盖已标记为 scene 的区域
                            weights[i] = entity_suppress

                for attr in entity.attrs:
                    attr_lower = attr.lower()
                    pos = prompt_lower.find(attr_lower)
                    if pos != -1:
                        start_ratio = pos / prompt_len
                        end_ratio = (pos + len(attr_lower)) / prompt_len
                        start_token = max(0, int((start_ratio - 0.02) * num_tokens))
                        end_token = min(num_tokens, int((end_ratio + 0.02) * num_tokens))
                        for i in range(start_token, end_token):
                            if weights[i] <= 1.0:
                                weights[i] = entity_suppress

        if scene_positions:
            print(f"[DEBUG] Scene keyword positions: {len(scene_positions)} keywords found")
        else:
            print(f"[DEBUG] No scene keywords found in prompt, using uniform weights")

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

    @staticmethod
    def _normalize_scene_texts(scene_texts: Optional[List[str]]) -> Tuple[set, set]:
        """将 scene 文本归一化为短语集合和 token 集合。"""
        if not scene_texts:
            return set(), set()

        phrases = set()
        tokens = set()
        for text in scene_texts:
            if not isinstance(text, str):
                continue
            norm = text.strip().lower()
            if not norm:
                continue
            phrases.add(norm)

            # 仅英文/数字 token
            tokens.update(re.findall(r"[a-z0-9]+", norm))
        return phrases, tokens

    @staticmethod
    def _compute_scene_distance(old_texts: List[str], new_texts: List[str]) -> float:
        """
        计算两组 scene text 的语义距离 (基于 token Jaccard)

        Jaccard 距离: 1 - |A∩B| / |A∪B|

        Args:
            old_texts: 上一个 prompt 的场景文本
            new_texts: 当前 prompt 的场景文本

        Returns:
            距离值 [0, 1]，0 = 完全相同，1 = 完全不同
        """
        _, old_tokens = MemoryBank._normalize_scene_texts(old_texts)
        _, new_tokens = MemoryBank._normalize_scene_texts(new_texts)

        if not old_tokens and not new_tokens:
            return 0.0
        if not old_tokens or not new_tokens:
            return 1.0

        intersection = len(old_tokens & new_tokens)
        union = len(old_tokens | new_tokens)
        return 1.0 - intersection / union

    def _compute_scene_relevance(self, frame_info: FrameInfo, query_scene_texts: List[str]) -> float:
        """计算 frame 与当前 scene query 的相关度。"""
        query_phrases, query_tokens = self._normalize_scene_texts(query_scene_texts)
        if not query_phrases and not query_tokens:
            return frame_info.scene_score

        frame_phrases, frame_tokens = self._normalize_scene_texts(frame_info.scene_texts)
        if not frame_phrases and not frame_tokens:
            return frame_info.scene_score

        phrase_overlap = len(query_phrases & frame_phrases) / max(1, len(query_phrases))
        token_overlap = len(query_tokens & frame_tokens) / max(1, len(query_tokens))

        # 当前 query 匹配度为主，历史 scene_score 作为轻量先验
        return 0.7 * phrase_overlap + 0.2 * token_overlap + 0.1 * max(0.0, frame_info.scene_score)

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

    def update_id_memory(self, frame_id: str, entity_score: float) -> None:
        """
        按 entity_score top-k 更新 ID Memory 槽

        Args:
            frame_id: 新帧ID
            entity_score: entity 维度分数
        """
        if frame_id not in self.frame_archive:
            return

        if len(self.id_memory) < self.max_id_memory_frames:
            if frame_id not in self.id_memory:
                self.id_memory.append(frame_id)
                print(f"[DEBUG] update_id_memory: Added {frame_id} (entity_score={entity_score:.4f})")
        else:
            # 找到 id_memory 中 entity_score 最低的帧
            min_score = float('inf')
            min_idx = -1
            min_fid = None

            for idx, fid in enumerate(self.id_memory):
                if fid in self.frame_archive:
                    finfo = self.frame_archive[fid]
                    if finfo.entity_score < min_score:
                        min_score = finfo.entity_score
                        min_idx = idx
                        min_fid = fid

            if entity_score > min_score and min_idx >= 0:
                self.id_memory[min_idx] = frame_id
                print(f"[DEBUG] update_id_memory: Replaced {min_fid}(score={min_score:.4f}) with {frame_id}(score={entity_score:.4f})")
            else:
                print(f"[DEBUG] update_id_memory: Kept existing, new {entity_score:.4f} <= min {min_score:.4f}")

    def update_scene_memory(self, frame_id: str, scene_score: float) -> None:
        """
        按 scene_score top-k 更新 Scene Memory 槽

        Args:
            frame_id: 新帧ID
            scene_score: scene 维度分数
        """
        if frame_id not in self.frame_archive:
            return

        if len(self.scene_memory) < self.max_scene_memory_frames:
            if frame_id not in self.scene_memory:
                self.scene_memory.append(frame_id)
                print(f"[DEBUG] update_scene_memory: Added {frame_id} (scene_score={scene_score:.4f})")
        else:
            # 找到 scene_memory 中 scene_score 最低的帧
            min_score = float('inf')
            min_idx = -1
            min_fid = None

            for idx, fid in enumerate(self.scene_memory):
                if fid in self.frame_archive:
                    finfo = self.frame_archive[fid]
                    if finfo.scene_score < min_score:
                        min_score = finfo.scene_score
                        min_idx = idx
                        min_fid = fid

            if scene_score > min_score and min_idx >= 0:
                self.scene_memory[min_idx] = frame_id
                print(f"[DEBUG] update_scene_memory: Replaced {min_fid}(score={min_score:.4f}) with {frame_id}(score={scene_score:.4f})")
            else:
                print(f"[DEBUG] update_scene_memory: Kept existing, new {scene_score:.4f} <= min {min_score:.4f}")

    def get_memory_kv(self, device: torch.device = None) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        获取当前active memory的KV cache (所有 block)

        Returns:
            List[{"k": [B, L, H, D], "v": [B, L, H, D]}] - 每个 block 的拼接 KV
            如果没有记忆帧则返回None
        """
        if not self.frame_active_memory:
            return None

        # 按时间顺序排序帧 ID
        frame_ids = sorted(self.frame_active_memory, key=self._frame_sort_key)

        # 收集所有帧的 KV
        all_frames_kv = []
        for frame_id in frame_ids:
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

    @staticmethod
    def _frame_sort_key(frame_id: str) -> Tuple[int, int, int, str]:
        """按 prompt_id, chunk_id, frame_idx 排序帧 ID"""
        match = re.match(r"p(\d+)_c(\d+)_f(\d+)", frame_id)
        if not match:
            return (1 << 30, 1 << 30, 1 << 30, frame_id)
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)), frame_id)

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
            "frame_active_memory": self.frame_active_memory,
            "id_memory": self.id_memory,
            "scene_memory": self.scene_memory,
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
        # 加载双层记忆 (向后兼容: 旧 JSON 只有 frame_active_memory)
        if "id_memory" in data:
            self.id_memory = data["id_memory"]
            self.scene_memory = data.get("scene_memory", [])
        else:
            # 旧格式兼容: frame_active_memory → id_memory
            self.id_memory = data.get("frame_active_memory", [])
            self.scene_memory = []

        # 加载frame_archive
        self.frame_archive = {}
        for fid, finfo_dict in data.get("frame_archive", {}).items():
            self.frame_archive[fid] = FrameInfo(
                frame_id=fid,
                frame_path=finfo_dict.get("frame_path", ""),
                prompt_id=finfo_dict.get("prompt_id", 0),
                associated_entities=finfo_dict.get("associated_entities", []),
                score=finfo_dict.get("score", 0.0),
                entity_score=finfo_dict.get("entity_score", 0.0),
                scene_score=finfo_dict.get("scene_score", 0.0),
                scene_texts=finfo_dict.get("scene_texts", []),
            )

    def clear(self) -> None:
        """清空Memory Bank"""
        self.global_registry = {}
        self.frame_archive = {}
        self.id_memory = []
        self.scene_memory = []
        self._frame_kv_store = {}

    def clear_frame_store(self) -> None:
        """
        只清空帧 KV 存储，释放 GPU 显存。
        保留 registry 和 archive 元数据。
        用于 VAE decode 前释放显存。
        """
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
