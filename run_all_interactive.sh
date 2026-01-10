#!/bin/bash
# =============================================================================
# Unified Interactive Inference for LongLive / MemFlow / IAMFlow
# =============================================================================

# ===================== MODIFY THESE PATHS =====================
PROJECT_ROOT="/root/autodl-tmp/home/joey/exp"
WAN_MODEL_PATH="${PROJECT_ROOT}/IAMFlow/wan_models/Wan2.1-T2V-1.3B"
QWEN_MODEL_PATH="${PROJECT_ROOT}/IAMFlow/Qwen3-4B-Instruct-2507"

# Checkpoints
LONGLIVE_BASE="${PROJECT_ROOT}/LongLive/longlive_models/models/longlive_base.pt"
LONGLIVE_LORA="${PROJECT_ROOT}/LongLive/longlive_models/models/lora.pt"
MEMFLOW_BASE="${PROJECT_ROOT}/IAMFlow/checkpoints/base.pt"
MEMFLOW_LORA="${PROJECT_ROOT}/IAMFlow/checkpoints/lora.pt"
IAMFLOW_BASE="${PROJECT_ROOT}/IAMFlow/checkpoints/base.pt"
IAMFLOW_LORA="${PROJECT_ROOT}/IAMFlow/checkpoints/lora.pt"

# Input / Output
INPUT_PROMPTS="${PROJECT_ROOT}/interactive_1_2.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results"
SEED=42
# ==============================================================

export WAN_MODEL_PATH="${WAN_MODEL_PATH}"

mkdir -p "${OUTPUT_DIR}/longlive" "${OUTPUT_DIR}/memflow" "${OUTPUT_DIR}/iamflow"

echo "=========================================="
echo "Input: ${INPUT_PROMPTS}"
echo "Output: ${OUTPUT_DIR}"
echo "WAN Model: ${WAN_MODEL_PATH}"
echo "=========================================="

# LongLive
echo "[1/3] Running LongLive..."
cd "${PROJECT_ROOT}/LongLive"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
    interactive_inference.py \
    --config_path configs/longlive_interactive_inference.yaml \
    --data_path "${INPUT_PROMPTS}" \
    --output_folder "${OUTPUT_DIR}/longlive" \
    --generator_ckpt "${LONGLIVE_BASE}" \
    --lora_ckpt "${LONGLIVE_LORA}" \
    --seed ${SEED}

# MemFlow
echo "[2/3] Running MemFlow..."
cd "${PROJECT_ROOT}/MemFlow"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
    interactive_inference.py \
    --config_path configs/interactive_inference.yaml \
    --data_path "${INPUT_PROMPTS}" \
    --output_folder "${OUTPUT_DIR}/memflow" \
    --generator_ckpt "${MEMFLOW_BASE}" \
    --lora_ckpt "${MEMFLOW_LORA}" \
    --seed ${SEED}

# IAMFlow
echo "[3/3] Running IAMFlow..."
cd "${PROJECT_ROOT}/IAMFlow"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29502 \
    agent_interactive_inference.py \
    --config_path configs/agent_interactive_inference.yaml \
    --data_path "${INPUT_PROMPTS}" \
    --output_folder "${OUTPUT_DIR}/iamflow" \
    --generator_ckpt "${IAMFLOW_BASE}" \
    --lora_ckpt "${IAMFLOW_LORA}" \
    --llm_model_path "${QWEN_MODEL_PATH}" \
    --seed ${SEED}

echo "=========================================="
echo "Done! Results in: ${OUTPUT_DIR}"
echo "  longlive/ memflow/ iamflow/"
echo "=========================================="
