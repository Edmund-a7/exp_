#!/bin/bash
# IAM Agent Single Prompt Inference Script
# Similar to MemFlow's inference.sh but uses IAM capabilities
# Usage: bash agent_inference.sh

echo "=========================================="
echo "IAM Agent Single Prompt Inference"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_port=29501 \
  agent_inference.py \
  --config_path configs/agent_inference.yaml\
  --llm_model_path ./Qwen3-4B-Instruct-2507
