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
IAM Agent Single Prompt Inference Script

Similar to MemFlow's inference.py but uses AgentCausalInferencePipeline for IAM capabilities.
Generates video from a single text prompt (no interactive multi-prompt switching).
"""
import argparse
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from einops import rearrange

from utils.misc import set_seed
from utils.dataset import TextDataset
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.agent_causal_inference import AgentCausalInferencePipeline


# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("IAM Agent single prompt inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
# IAM-specific arguments (optional, for advanced usage)
parser.add_argument("--llm_model_path", type=str, default=None,
                    help="Path to the LLM model for entity extraction")
parser.add_argument("--max_memory_frames", type=int, default=None,
                    help="Maximum number of memory frames to keep")
parser.add_argument("--save_dir", type=str, default=None,
                    help="Directory to save frame data")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )

    set_seed(config.seed + local_rank)
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    config.distributed = False
    print(f"Single GPU mode on device {device}")

print(f"Free VRAM {get_cuda_free_memory_gb(device)} GB")
low_memory = get_cuda_free_memory_gb(device) < 40 if torch.cuda.is_available() else False

torch.set_grad_enabled(False)

# ----------------------------- Create IAM Pipeline -----------------------------
# Resolve IAM parameters from config or command line
llm_model_path = args.llm_model_path or getattr(config, "llm_model_path", "../Qwen3-0.6B")
max_memory_frames = args.max_memory_frames if args.max_memory_frames is not None else getattr(config, "max_memory_frames", 3)
save_dir = args.save_dir or getattr(config, "save_dir", "data/agent_frames")

if local_rank == 0:
    print("=" * 60)
    print("IAM Agent Single Prompt Inference")
    print("=" * 60)
    print(f"LLM Model Path: {llm_model_path}")
    print(f"Max Memory Frames: {max_memory_frames}")
    print(f"Save Directory: {save_dir}")
    print("=" * 60)

pipeline = AgentCausalInferencePipeline(
    config,
    device=device,
    llm_model_path=llm_model_path,
    max_memory_frames=max_memory_frames,
    save_dir=save_dir
)

# ----------------------------- Load base checkpoint -----------------------------
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

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

    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
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
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- Build dataset -----------------------------
extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

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
    idx = batch_data['idx'].item()

    # Unpack batch data
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    all_video = []
    num_generated_frames = 0

    # Get prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    if local_rank == 0:
        print(f"Processing prompt: {prompts[0][:100]}...")

    # For single prompt inference, we wrap it as a single-segment list
    text_prompts_list = [[p] for p in prompts]  # Each sample gets a list with one prompt

    # Generate video using IAM pipeline (single segment, no switching)
    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts_list=text_prompts_list,
        switch_frame_indices=[],  # No switching for single prompt
        return_latents=False,
        low_memory=low_memory,
        save_mapping=False,  # Don't need mapping for single prompt
        profile=True,  # Enable profiling
    )

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += video.shape[1]

    # Final output video
    video_out = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "iam_lora"
        elif getattr(config, 'use_ema', False):
            model_type = "iam_ema"
        else:
            model_type = "iam"

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video_out[seed_idx].to(torch.uint8), fps=16)

        if local_rank == 0:
            print(f"[IAM] Saved video to {output_path}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()

if local_rank == 0:
    print("=" * 60)
    print("IAM Agent Single Prompt Inference Complete!")
    print("=" * 60)