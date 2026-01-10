#!/bin/bash
# IAM Agent Interactive Inference Script
# Multi-prompt interactive video generation with prompt switching
# Usage: bash agent_interactive_inference.sh

echo "=========================================="
echo "IAM Agent Interactive Inference"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=29502 \
  agent_interactive_inference.py \
  --config_path configs/agent_interactive_inference.yaml \
  --llm_model_path ./Qwen3-4B-Instruct-2507 \
  --max_memory_frames 3 \
  --save_dir data/agent_frames
