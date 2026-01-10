"""
IAM Agent Causal Inference Pipeline

整合 LLM Agent 和 Memory Bank 到 IAM_Flow 的 InteractiveCausalInferencePipeline 中
完全替代 MemFlow 的帧选择逻辑，使用 IAM 的 entity-based 帧选择

架构:
- 继承 InteractiveCausalInferencePipeline
- 禁用 MemFlow 的 NAM 帧选择
- 使用 IAM 的 entity-attrs 帧选择

流程 (与 pipeline.md 对齐):
1. 每个 prompt 的 chunk 1: LLM Agent 提取实体 → ID 分配/匹配 → 检索初始记忆帧
2. chunk 3 开始: 驱逐 chunk → IAM 选帧 → 更新 top-3 记忆帧 → 注入 kv_bank
"""

import os
import sys
from typing import List, Optional, Dict, Tuple, Any

import torch
import torch.distributed as dist

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.debug_option import DEBUG

# IAM 模块
from iam.llm_agent import LLMAgent, EntityStruct
from iam.memory_bank import MemoryBank


class AgentCausalInferencePipeline(InteractiveCausalInferencePipeline):
    """
    整合 LLM Agent 和 Memory Bank 的推理 Pipeline

    继承自 InteractiveCausalInferencePipeline，完全替代 MemFlow 的帧选择逻辑:
    1. 禁用 MemFlow 的 _apply_cache_updates_before (NAM)
    2. 使用 IAM 的 entity-based 帧选择
    3. 将 IAM 选择的帧 KV 注入 kv_bank
    """

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
        llm_model_path: str = "../Qwen3-0.6B",
        max_memory_frames: int = 3,
        save_dir: str = "data/agent_frames",
        save_frames_to_disk: bool = False
    ):
        """
        初始化 Agent Pipeline

        Args:
            args: 配置参数
            device: 计算设备
            generator: 视频生成模型
            text_encoder: 文本编码器
            vae: VAE 模型
            llm_model_path: LLM 模型路径
            max_memory_frames: 最大记忆帧数量
            save_dir: 帧数据保存目录
            save_frames_to_disk: 是否将帧 KV 保存到磁盘 (默认False，仅保存在内存中以提升性能)
        """
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)

        # 初始化 LLM Agent
        self.llm_agent = LLMAgent(model_path=llm_model_path)

        # 初始化 Memory Bank (使用父类的 text_encoder)
        self.agent_memory_bank = MemoryBank(
            text_encoder=self.text_encoder,
            max_memory_frames=max_memory_frames,
            frame_seq_length=self.frame_seq_length,
            num_transformer_blocks=self.num_transformer_blocks,
            save_dir=save_dir,
            save_frames_to_disk=save_frames_to_disk
        )

        # 状态追踪
        self.current_prompt_id = 0
        self.current_chunk_id = 0
        self.current_entities: List[EntityStruct] = []
        self.current_prompt_text: str = ""  # 当前 prompt 文本，用于精确定位实体位置

        # 配置
        self.llm_model_path = llm_model_path
        self.max_memory_frames = max_memory_frames
        self.save_dir = save_dir
        self.save_frames_to_disk = save_frames_to_disk

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # IAM: MemFlow 的 NAM 和 SMA 代码已从 causal_model.py 中删除
        # 不再需要禁用标志

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        save_mapping: bool = True,
        mapping_path: str = "mapping.json",
        profile: bool = False):
        """
        带 Agent 的视频生成推理 (完全替代 MemFlow 帧选择)

        Args:
            noise: 噪声张量, shape = (B, T_out, C, H, W)
            text_prompts_list: prompt 列表
            switch_frame_indices: prompt 切换帧索引
            return_latents: 是否返回 latent
            low_memory: 低内存模式
            save_mapping: 是否保存 mapping.json
            mapping_path: mapping.json 保存路径
            profile: 是否启用性能分析

        Returns:
            生成的视频 (和 latent)
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # ===== Profiling Setup =====
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)

            # IAM specific timers
            agent_start = torch.cuda.Event(enable_timing=True)
            agent_end = torch.cuda.Event(enable_timing=True)
            memory_start = torch.cuda.Event(enable_timing=True)
            memory_end = torch.cuda.Event(enable_timing=True)

            block_times = []
            agent_times = []  # Per-prompt agent processing time
            memory_times = []  # Per-chunk memory bank time

            init_start.record()

        # 重置状态
        self._reset_agent_state()

        # 编码所有 prompts
        if DEBUG:
            print(f"[AgentPipeline] text_prompts_list: {text_prompts_list}")
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # 初始化 caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        kv_bank1_size = self.bank_size * self.frame_seq_length
        self._initialize_kv_bank(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device, kv_bank1_size=kv_bank1_size
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # 时序循环
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        # ===== 处理第一个 prompt =====
        if profile:
            agent_start.record()

        self._process_prompt_start(
            prompt_text=text_prompts_list[0][0],
            prompt_id=1,
            is_first_prompt=True
        )

        if profile:
            agent_end.record()
            torch.cuda.synchronize()
            agent_time = agent_start.elapsed_time(agent_end)
            agent_times.append(("Prompt 1", agent_time))

        for block_idx, current_num_frames in enumerate(all_num_frames):
            # ===== 关键: 始终禁用 MemFlow 的 bank 更新 =====
            # IAM 会自己管理 bank
            update_bank = False

            if profile:
                block_start.record()

            # ===== 1. 检测 prompt 切换 =====
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1

                # ===== 2. Recache (必须在 LLM Agent 处理之前) =====
                # 重置 kv_cache 和 crossattn_cache
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])

                # ===== 3. LLM Agent 处理新 prompt =====
                if profile:
                    agent_start.record()

                # chunk_id 重置在 _process_prompt_start 中统一处理
                self._process_prompt_start(
                    prompt_text=text_prompts_list[segment_idx][0],
                    prompt_id=segment_idx + 1,
                    is_first_prompt=False
                )

                if profile:
                    agent_end.record()
                    torch.cuda.synchronize()
                    agent_time = agent_start.elapsed_time(agent_end)
                    agent_times.append((f"Prompt {segment_idx + 1}", agent_time))

                if DEBUG:
                    print(f"[AgentPipeline] Switched to segment {segment_idx} at frame {current_start_frame}")

                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )

            cond_in_use = cond_list[segment_idx]
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            # ===== 4. 空间去噪循环 =====
            for index, current_timestep in enumerate(self.denoising_step_list):
                # q_bank=True 让模型读取 bank，但 update_bank=False 不让 MemFlow 更新
                if index == 0:
                    q_bank = True
                else:
                    q_bank = False

                timestep = (
                    torch.ones([batch_size, current_num_frames],
                               device=noise.device, dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_bank=self.kv_bank1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        update_bank=False,  # 关键: 始终 False
                        q_bank=q_bank,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_bank=self.kv_bank1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        update_bank=False,  # 关键: 始终 False
                        q_bank=q_bank,
                    )

            # 记录输出
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # ===== 5. IAM 帧选择和 bank 更新 (chunk >= 3 时) =====
            # 关键：必须在 clean context 更新前获取被驱逐的 chunk
            self.current_chunk_id += 1
            if self.current_chunk_id >= 3 and self.current_entities:
                if profile:
                    memory_start.record()

                self._process_chunk_eviction(
                    current_start_frame=current_start_frame,
                    current_num_frames=current_num_frames
                )
                # ===== 6. 将 IAM 的帧 KV 注入 kv_bank =====
                self._inject_iam_memory_to_bank()

                if profile:
                    memory_end.record()
                    torch.cuda.synchronize()
                    memory_time = memory_start.elapsed_time(memory_end)
                    memory_times.append((f"Chunk {block_idx}", memory_time))

            # 使用 clean context 更新 cache (但不更新 bank)
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_bank=self.kv_bank1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                update_bank=False,  # 关键: 始终 False，IAM 自己管理
                q_bank=q_bank,
                update_cache=True,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # 更新帧指针
            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # 解码视频
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)

        self.clear_kv_cache()

        # 保存 mapping
        if save_mapping:
            self.agent_memory_bank.save_to_json(mapping_path)
            if DEBUG:
                print(f"[AgentPipeline] Saved mapping to {mapping_path}")

        # ===== Profiling Results =====
        if profile:
            total_time = init_time + diffusion_time + vae_time
            total_agent_time = sum(t for _, t in agent_times)
            total_memory_time = sum(t for _, t in memory_times)

            print("\n" + "=" * 70)
            print("IAM Agent Pipeline Profiling Results")
            print("=" * 70)

            # Overall breakdown
            print(f"\n[Overall Performance]")
            print(f"  - Initialization time:     {init_time:8.2f} ms ({100 * init_time / total_time:5.2f}%)")
            print(f"  - Diffusion generation:    {diffusion_time:8.2f} ms ({100 * diffusion_time / total_time:5.2f}%)")
            print(f"  - VAE decoding:            {vae_time:8.2f} ms ({100 * vae_time / total_time:5.2f}%)")
            print(f"  - Total time:              {total_time:8.2f} ms")
            print(f"  - Throughput:              {num_output_frames / (total_time / 1000):8.2f} FPS")

            # IAM-specific breakdown
            print(f"\n[IAM Components (within diffusion)]")
            print(f"  - Total LLM Agent time:    {total_agent_time:8.2f} ms ({100 * total_agent_time / diffusion_time:5.2f}% of diffusion)")
            print(f"  - Total Memory Bank time:  {total_memory_time:8.2f} ms ({100 * total_memory_time / diffusion_time:5.2f}% of diffusion)")
            print(f"  - Pure diffusion time:     {diffusion_time - total_agent_time - total_memory_time:8.2f} ms ({100 * (diffusion_time - total_agent_time - total_memory_time) / diffusion_time:5.2f}% of diffusion)")

            # Per-prompt agent time
            if agent_times:
                print(f"\n[LLM Agent - Per Prompt]")
                for prompt_name, time_ms in agent_times:
                    print(f"  - {prompt_name:12s} processing: {time_ms:8.2f} ms")

            # Per-chunk memory bank time (show first 5 and last 5 if many)
            if memory_times:
                print(f"\n[Memory Bank - Per Chunk (Chunk 3+)]")
                if len(memory_times) <= 10:
                    for chunk_name, time_ms in memory_times:
                        print(f"  - {chunk_name:12s} eviction:  {time_ms:8.2f} ms")
                else:
                    # Show first 5
                    for chunk_name, time_ms in memory_times[:5]:
                        print(f"  - {chunk_name:12s} eviction:  {time_ms:8.2f} ms")
                    print(f"  - ... ({len(memory_times) - 10} chunks omitted)")
                    # Show last 5
                    for chunk_name, time_ms in memory_times[-5:]:
                        print(f"  - {chunk_name:12s} eviction:  {time_ms:8.2f} ms")
                if memory_times:
                    avg_memory_time = total_memory_time / len(memory_times)
                    print(f"  - Average per chunk:       {avg_memory_time:8.2f} ms")

            # Per-block diffusion time
            print(f"\n[Diffusion - Per Block]")
            if len(block_times) <= 10:
                for i, block_time in enumerate(block_times):
                    print(f"  - Block {i:3d} generation:   {block_time:8.2f} ms ({100 * block_time / diffusion_time:5.2f}% of diffusion)")
            else:
                # Show first 5
                for i in range(5):
                    print(f"  - Block {i:3d} generation:   {block_times[i]:8.2f} ms ({100 * block_times[i] / diffusion_time:5.2f}% of diffusion)")
                print(f"  - ... ({len(block_times) - 10} blocks omitted)")
                # Show last 5
                for i in range(len(block_times) - 5, len(block_times)):
                    print(f"  - Block {i:3d} generation:   {block_times[i]:8.2f} ms ({100 * block_times[i] / diffusion_time:5.2f}% of diffusion)")
            avg_block_time = diffusion_time / len(block_times)
            print(f"  - Average per block:       {avg_block_time:8.2f} ms")

            print("=" * 70 + "\n")

        if return_latents:
            return video, output
        return video

    # ============ Agent 相关方法 ============

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        """
        重写父类方法，增加 kv_bank 索引重置

        IAM 在 prompt 切换时需要:
        1. 调用父类的 recache (重置 kv_cache 和 crossattn_cache)
        2. 重置 kv_bank 的索引 (IAM 会在 _process_prompt_start 中重新注入)
        """
        # 调用父类的 recache
        super()._recache_after_switch(output, current_start_frame, new_conditional_dict)

        # IAM: 重置 kv_bank 索引，准备注入新的记忆帧
        if self.kv_bank1 is not None:
            for blk in self.kv_bank1:
                blk["local_end_index"].zero_()
                blk["global_end_index"].zero_()
                # 注意: 不需要 zero_() k/v 内容，因为 _inject_iam_memory_to_bank 会覆盖
            if DEBUG:
                print(f"[AgentPipeline] Reset kv_bank indices for prompt switch")

    def _reset_agent_state(self) -> None:
        """重置 Agent 状态"""
        self.current_prompt_id = 0
        self.current_chunk_id = 0
        self.current_entities = []
        self.current_prompt_text = ""
        self.agent_memory_bank.clear()
        # 重置 LLM Agent 的 ID 计数器，确保每次推理 global_registry 从 1 开始
        # 注意: LLMAgent 内部使用 id_manager (GlobalIDManager 实例) 来管理 ID
        self.llm_agent.id_manager._next_id = 1
        print(f"[DEBUG] _reset_agent_state: id_manager._next_id={self.llm_agent.id_manager._next_id}, registry={self.agent_memory_bank.global_registry}")

    def _process_prompt_start(self,
                              prompt_text: str,
                              prompt_id: int,
                              is_first_prompt: bool) -> None:
        """
        处理 prompt 开始时的 Agent 逻辑

        Args:
            prompt_text: prompt 文本
            prompt_id: prompt 序号
            is_first_prompt: 是否为第一个 prompt
        """
        self.current_prompt_id = prompt_id
        self.current_chunk_id = 0
        self.current_prompt_text = prompt_text  # 保存当前 prompt 文本，用于精确定位实体位置

        print(f"\n{'='*60}")
        print(f"[DEBUG] === PROMPT {prompt_id} START ===")
        print(f"[DEBUG] prompt_text: {prompt_text[:100]}...")
        print(f"[DEBUG] is_first_prompt: {is_first_prompt}")
        print(f"{'='*60}")

        # 1. LLM Agent 提取实体并分配 ID
        entities, registry_update = self.llm_agent.process_prompt(
            prompt=prompt_text,
            prompt_id=prompt_id,
            global_registry=self.agent_memory_bank.global_registry
        )

        self.current_entities = entities

        print(f"[DEBUG] Extracted {len(entities)} entities:")
        for e in entities:
            print(f"  - entity='{e.entity}', global_id={e.global_id}, attrs={e.attrs}")

        print(f"[DEBUG] Registry update: {list(registry_update.keys()) if registry_update else 'None'}")

        if DEBUG:
            print(f"[AgentPipeline] Prompt {prompt_id} entities:")
            for e in entities:
                print(f"  - {e.entity} (ID: {e.global_id}): {e.attrs}")

        # 2. 更新 Memory Bank 的 registry
        self.agent_memory_bank.register_entities(entities, prompt_id, registry_update)

        print(f"[DEBUG] Global registry after update: {list(self.agent_memory_bank.global_registry.keys())}")

        # 3. 检索初始记忆帧 (非首个 prompt)
        if not is_first_prompt and entities:
            entity_ids = self.agent_memory_bank.get_entity_ids(entities)
            print(f"[DEBUG] Retrieving initial frames for entity_ids: {entity_ids}")
            retrieved_frames = self.agent_memory_bank.retrieve_initial_frames(entity_ids)
            print(f"[DEBUG] Retrieved initial frames: {retrieved_frames}")

            if DEBUG:
                print(f"[AgentPipeline] Retrieved initial frames: {retrieved_frames}")

            # 注入检索到的帧
            self._inject_iam_memory_to_bank()
            print(f"[DEBUG] Injected memory frames to kv_bank")

    def _process_chunk_eviction(self,
                                current_start_frame: int,
                                current_num_frames: int) -> None:
        """
        处理 chunk 驱逐时的帧选择逻辑 (IAM 版本)

        Args:
            current_start_frame: 当前起始帧
            current_num_frames: 当前块帧数
        """
        if not self.current_entities:
            return

        # 获取被驱逐的 chunk 的 KV (从 kv_cache 中，所有 30 个 block)
        evicted_chunk_kv = self._get_evicted_chunk_kv()

        if evicted_chunk_kv is None:
            return

        print(f"\n[DEBUG] --- CHUNK {self.current_chunk_id} EVICTION (Prompt {self.current_prompt_id}) ---")
        print(f"[DEBUG] current_start_frame={current_start_frame}, current_num_frames={current_num_frames}")
        print(f"[DEBUG] evicted_chunk_kv shape: k={evicted_chunk_kv[0]['k'].shape}")

        # IAM 帧选择 (使用 entity-attr 字符串 + prompt 精确定位)
        entity_ids = self.agent_memory_bank.get_entity_ids(self.current_entities)
        frame_id, score = self.agent_memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=evicted_chunk_kv,
            crossattn_cache=self.crossattn_cache,
            prompt_id=self.current_prompt_id,
            chunk_id=self.current_chunk_id,
            current_entity_ids=entity_ids,
            current_entities=self.current_entities,
            prompt_text=self.current_prompt_text  # 传入 prompt 文本用于精确定位
        )

        print(f"[DEBUG] IAM selected frame: {frame_id}, score={score:.4f}")

        # 更新 active memory
        self.agent_memory_bank.update_active_memory(frame_id, score)

        print(f"[DEBUG] Active memory after update: {self.agent_memory_bank.frame_active_memory}")
        print(f"[DEBUG] Frame archive size: {len(self.agent_memory_bank.frame_archive)}")

        if DEBUG:
            print(f"[AgentPipeline] IAM selected frame {frame_id} with score {score:.4f}")
            print(f"[AgentPipeline] Active memory: {self.agent_memory_bank.frame_active_memory}")

    def _get_evicted_chunk_kv(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        获取即将被驱逐的 chunk 的 KV cache (所有 30 个 block)

        IAM 的帧选择逻辑：
        - 从 Local 窗口中**即将被驱逐**的 chunk 中选择最相关的 1 帧存入 Memory Bank
        - Local 窗口大小 = 2 * chunk_length (6 帧 = 2 chunks)
        - 当生成新 chunk 时，最旧的 chunk 会被驱逐

        关键：必须在 clean context 更新 cache **之前**调用，此时：
        - local_end 指向旧的位置（还没加入新 chunk）
        - 被驱逐的 chunk 在 [local_end - 2*chunk_length, local_end - chunk_length)

        示例（Chunk 3 生成时）:
        - local_end = 6 * 16 = 96 (C1+C2 的末尾)
        - chunk_start = max(0, 96 - 2*48) = 0  (C1 起始)
        - chunk_end = max(0, 96 - 48) = 48     (C1 末尾)
        - 返回 C1 的 KV ✅

        Returns:
            List[{"k": [B, L, H, D], "v": [B, L, H, D]}] 或 None
        """
        if self.kv_cache1 is None or len(self.kv_cache1) == 0:
            return None

        # 从第一个 block 获取位置信息
        cache = self.kv_cache1[0]
        k = cache["k"]

        # 获取当前 cache 的有效长度（clean context 更新前）
        local_end = cache["local_end_index"].item()

        # 计算被驱逐的 chunk 的位置
        # num_frame_per_block = 3 (每个 chunk 3 帧)
        chunk_length = self.num_frame_per_block * self.frame_seq_length

        # 被驱逐的 chunk 在 Local 窗口的最前面
        chunk_start = max(0, local_end - 2 * chunk_length)
        chunk_end = max(0, local_end - chunk_length)

        if chunk_end <= chunk_start or chunk_end > k.shape[1]:
            return None

        # 获取所有 block 的被驱逐 chunk KV
        all_blocks_kv = []
        for block_idx in range(len(self.kv_cache1)):
            block_cache = self.kv_cache1[block_idx]
            all_blocks_kv.append({
                "k": block_cache["k"][:, chunk_start:chunk_end].clone(),
                "v": block_cache["v"][:, chunk_start:chunk_end].clone()
            })

        return all_blocks_kv

    def _inject_iam_memory_to_bank(self) -> None:
        """
        将 IAM 的记忆帧 KV 注入到 kv_bank1 (所有 30 个 block)

        这是 IAM 完全替代 MemFlow NAM 的关键方法
        """
        if self.kv_bank1 is None:
            return

        memory_kv = self.agent_memory_bank.get_memory_kv(
            device=self.kv_bank1[0]["k"].device
        )

        if memory_kv is None:
            print(f"[DEBUG] _inject_iam_memory_to_bank: No memory KV available")
            return

        # memory_kv 是 List[{"k": ..., "v": ...}]，每个 block 一个
        memory_length = memory_kv[0]["k"].shape[1]
        num_frames_in_memory = memory_length // self.frame_seq_length

        print(f"[DEBUG] _inject_iam_memory_to_bank: memory_length={memory_length} tokens ({num_frames_in_memory} frames)")
        print(f"[DEBUG] Active memory frames being injected: {self.agent_memory_bank.frame_active_memory}")

        # 将 IAM 的帧 KV 注入到每个 transformer block 的 bank
        for block_idx in range(len(self.kv_bank1)):
            bank = self.kv_bank1[block_idx]
            bank_size = bank["k"].shape[1]

            # 清空并写入 IAM 的帧
            bank["k"].zero_()
            bank["v"].zero_()

            # 写入 IAM 选择的帧 (最多 bank_size)
            write_length = min(memory_length, bank_size)
            bank["k"][:, :write_length] = memory_kv[block_idx]["k"][:, :write_length]
            bank["v"][:, :write_length] = memory_kv[block_idx]["v"][:, :write_length]

            # 更新索引
            bank["local_end_index"].fill_(write_length)
            bank["global_end_index"].fill_(write_length)

        print(f"[DEBUG] Injected {memory_length} tokens to all {len(self.kv_bank1)} blocks")

        if DEBUG:
            print(f"[AgentPipeline] Injected {memory_length} tokens from IAM to kv_bank (all {len(self.kv_bank1)} blocks)")

    # ============ 辅助方法 ============

    def get_agent_status(self) -> Dict[str, Any]:
        """获取 Agent 状态信息"""
        return {
            "current_prompt_id": self.current_prompt_id,
            "current_chunk_id": self.current_chunk_id,
            "current_entities": [e.to_dict() for e in self.current_entities],
            "global_registry": self.agent_memory_bank.global_registry,
            "frame_archive_count": len(self.agent_memory_bank.frame_archive),
            "active_memory": self.agent_memory_bank.frame_active_memory
        }

    def save_agent_state(self, path: str) -> None:
        """保存 Agent 状态"""
        self.agent_memory_bank.save_to_json(path)

    def load_agent_state(self, path: str) -> None:
        """加载 Agent 状态"""
        self.agent_memory_bank.load_from_json(path)


# ============ 便捷函数 ============

def create_agent_pipeline(
    config,
    device: torch.device,
    llm_model_path: str = "../Qwen3-0.6B",
    max_memory_frames: int = 3,
    save_dir: str = "data/agent_frames"
) -> AgentCausalInferencePipeline:
    """
    创建 Agent Pipeline 的便捷函数

    Args:
        config: 配置对象
        device: 计算设备
        llm_model_path: LLM 模型路径
        max_memory_frames: 最大记忆帧数
        save_dir: 帧保存目录

    Returns:
        AgentCausalInferencePipeline 实例
    """
    pipeline = AgentCausalInferencePipeline(
        args=config,
        device=device,
        llm_model_path=llm_model_path,
        max_memory_frames=max_memory_frames,
        save_dir=save_dir
    )
    return pipeline


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("AgentCausalInferencePipeline Module")
    print("=" * 60)
    print("\nThis module requires IAM_Flow dependencies to run.")
    print("Please use test_iam.py for comprehensive testing.")
    print("\nKey components:")
    print("  - AgentCausalInferencePipeline: Main pipeline class")
    print("  - create_agent_pipeline(): Convenience factory function")
    print("\nUsage:")
    print("  from pipeline.agent_causal_inference import AgentCausalInferencePipeline")
    print("  pipeline = AgentCausalInferencePipeline(args, device)")
    print("  video = pipeline.inference(noise, text_prompts_list=..., switch_frame_indices=...)")
