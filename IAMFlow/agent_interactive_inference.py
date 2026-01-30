# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
IAM Agent Interactive Inference Script

Multi-prompt interactive video generation with prompt switching.
使用 AgentCausalInferencePipeline 进行交互式视频生成推理
整合 LLM Agent (实体提取和 ID 匹配) 和 Memory Bank (帧选择)
"""
import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.agent_causal_inference import AgentCausalInferencePipeline
from utils.dataset import MultiTextDataset


# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("IAM Agent causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
# Path override arguments
parser.add_argument("--data_path", type=str, default=None, help="Override data_path in config")
parser.add_argument("--output_folder", type=str, default=None, help="Override output_folder in config")
parser.add_argument("--generator_ckpt", type=str, default=None, help="Override generator_ckpt in config")
parser.add_argument("--lora_ckpt", type=str, default=None, help="Override lora_ckpt in config")
parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
# IAM-specific arguments
parser.add_argument("--llm_model_path", type=str, default="../Qwen3-0.6B",
                    help="Path to the LLM model for entity extraction")
parser.add_argument("--max_memory_frames", type=int, default=3,
                    help="Maximum number of memory frames to keep")
parser.add_argument("--save_dir", type=str, default="data/agent_frames",
                    help="Directory to save frame data")
parser.add_argument("--mapping_path", type=str, default=None,
                    help="Path to save mapping.json (default: output_folder/mapping.json)")
parser.add_argument("--save_frames_to_disk", action="store_true",
                    help="Save frame KV to disk (default: False, keep in memory only for better performance)")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Override config with command line arguments
if args.data_path: config.data_path = args.data_path
if args.output_folder: config.output_folder = args.output_folder
if args.generator_ckpt: config.generator_ckpt = args.generator_ckpt
if args.lora_ckpt: config.lora_ckpt = args.lora_ckpt
if args.seed is not None: config.seed = args.seed

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    # Set device first
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group with backend and timeout
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )

    set_seed(config.seed)
    print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    print(f"Single GPU mode on device {device}")


low_memory = get_cuda_free_memory_gb(device) < 40 if torch.cuda.is_available() else False
torch.set_grad_enabled(False)

# ----------------------------- Create IAM Pipeline -----------------------------
# Resolve IAM parameters: command line args override config file
llm_model_path = args.llm_model_path if args.llm_model_path != "../Qwen3-0.6B" else getattr(config, "llm_model_path", "../Qwen3-0.6B")
max_memory_frames = args.max_memory_frames if args.max_memory_frames != 3 else getattr(config, "max_memory_frames", 3)
save_dir = args.save_dir if args.save_dir != "data/agent_frames" else getattr(config, "save_dir", "data/agent_frames")
save_frames_to_disk = args.save_frames_to_disk or getattr(config, "save_frames_to_disk", False)
use_vllm = getattr(config, "use_vllm", True)  # 默认使用 vLLM

if local_rank == 0:
    print("=" * 60)
    print("IAM Agent Causal Inference Pipeline")
    print("=" * 60)
    print(f"LLM Model Path: {llm_model_path}")
    print(f"Max Memory Frames: {max_memory_frames}")
    print(f"Save Directory: {save_dir}")
    print(f"Save Frames to Disk: {save_frames_to_disk}")
    print(f"Use vLLM: {use_vllm}")
    print("=" * 60)

pipeline = AgentCausalInferencePipeline(
    config,
    device=device,
    llm_model_path=llm_model_path,
    max_memory_frames=max_memory_frames,
    save_dir=save_dir,
    save_frames_to_disk=save_frames_to_disk,
    use_vllm=use_vllm
)

# ----------------------------- Load base checkpoint -----------------------------
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # After loading base weights, apply LoRA wrapper to the generator's transformer model
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights (if lora_ckpt is provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
print("dtype", pipeline.generator.model.dtype)
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- Build dataset -----------------------------
# Parse switch_frame_indices
switch_frame_indices: List[int] = [int(x) for x in config.switch_frame_indices.split(",") if x.strip()]

# Create dataset
dataset = MultiTextDataset(config.data_path)

# Validate number of segments & switch_frame_indices length
num_segments = len(dataset[0]["prompts_list"])
assert len(switch_frame_indices) == num_segments - 1, (
    "The number of switch_frame_indices should be the number of prompt segments minus 1"
)

print("Number of segments:", num_segments)
print("Switch frame indices:", switch_frame_indices)

num_prompts_total = len(dataset)
print(f"Number of prompt lines: {num_prompts_total}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ----------------------------- Inference loop -----------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    prompts_list: List[str] = batch_data["prompts_list"]  # type: ignore

    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            60,
            104,
        ],
        device=device,
        dtype=torch.bfloat16,
    )

    # Determine mapping path for this sample
    if args.mapping_path:
        mapping_path = args.mapping_path
    else:
        mapping_path = os.path.join(config.output_folder, f"mapping_{idx}.json")

    # IAM Agent inference
    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts_list=prompts_list,
        switch_frame_indices=switch_frame_indices,
        return_latents=False,
        low_memory=low_memory,
        save_mapping=True,
        mapping_path=mapping_path,
        profile=True,  # Enable profiling
    )

    current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "iam_lora"
    elif getattr(config, 'use_ema', False):
        model_type = "iam_ema"
    else:
        model_type = "iam"

    for seed_idx in range(config.num_samples):
        if config.save_with_index:
            output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4")
        else:
            # Use the first prompt segment as the filename prefix to avoid overly long names
            short_name = prompts_list[0][0][:100].replace("/", "_")
            output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")
        write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)

    if local_rank == 0:
        print(f"[IAM] Saved video to {output_path}")
        print(f"[IAM] Saved mapping to {mapping_path}")

        # Print agent status
        status = pipeline.get_agent_status()
        print(f"[IAM] Final registry entities: {len(status['global_registry'])}")
        print(f"[IAM] Frame archive count: {status['frame_archive_count']}")
        print(f"[IAM] Active memory frames: {status['active_memory']}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()

if local_rank == 0:
    print("=" * 60)
    print("IAM Agent Inference Complete!")
    print("=" * 60)
