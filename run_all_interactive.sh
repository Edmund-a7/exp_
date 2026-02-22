#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1
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
IAMFLOW_CONFIG="${PROJECT_ROOT}/IAMFlow/configs/agent_interactive_inference_continuity.yaml"
# 只测 id-memory 时改为:
# IAMFLOW_CONFIG="${PROJECT_ROOT}/IAMFlow/configs/agent_interactive_inference_id_only.yaml"

# Input / Output
# INPUT_PROMPTS="${PROJECT_ROOT}/memflow_benchmark.jsonl"
# OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results/memflow_benchmark"
INPUT_PROMPTS="${PROJECT_ROOT}/id_stress_test_6_batch.jsonl"
# 如果你把 prompt 放在 Benchmark/prompt_expander 下，启用下面这一行：
# INPUT_PROMPTS="${PROJECT_ROOT}/Benchmark/prompt_expander/id_stress_test_6_batch.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results/id_stress_test_batch_6"
# INPUT_PROMPTS="${PROJECT_ROOT}/interactive_1_2.jsonl"
# OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results/interactive_1_2"
SEED=42
USE_LIGHTVAE=false  # 设为 true 启用 LightVAE (75% 通道剪枝)
# ==============================================================

export WAN_MODEL_PATH="${WAN_MODEL_PATH}"

mkdir -p "${OUTPUT_DIR}/longlive" "${OUTPUT_DIR}/memflow" "${OUTPUT_DIR}/iamflow"

# 日志文件（带时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LONGLIVE_LOG="${LOG_DIR}/longlive.log"
MEMFLOW_LOG="${LOG_DIR}/memflow.log"
IAMFLOW_LOG="${LOG_DIR}/iamflow.log"

echo "=========================================="
echo "Input: ${INPUT_PROMPTS}"
echo "Output: ${OUTPUT_DIR}"
echo "WAN Model: ${WAN_MODEL_PATH}"
echo "Logs: ${LOG_DIR}"
echo "=========================================="

# Basic checks
if [ ! -f "${INPUT_PROMPTS}" ]; then
    echo "[ERROR] Input prompt file not found: ${INPUT_PROMPTS}"
    echo "Hint: set INPUT_PROMPTS to one of:"
    echo "  1) \${PROJECT_ROOT}/id_stress_test_6_batch.jsonl"
    echo "  2) \${PROJECT_ROOT}/Benchmark/prompt_expander/id_stress_test_6_batch.jsonl"
    exit 1
fi
if [ ! -f "${IAMFLOW_BASE}" ]; then
    echo "[ERROR] IAMFLOW_BASE not found: ${IAMFLOW_BASE}"
    exit 1
fi
if [ ! -f "${IAMFLOW_LORA}" ]; then
    echo "[ERROR] IAMFLOW_LORA not found: ${IAMFLOW_LORA}"
    exit 1
fi
if [ ! -d "${QWEN_MODEL_PATH}" ]; then
    echo "[ERROR] QWEN_MODEL_PATH not found: ${QWEN_MODEL_PATH}"
    exit 1
fi
if [ ! -f "${IAMFLOW_CONFIG}" ]; then
    echo "[ERROR] IAMFLOW_CONFIG not found: ${IAMFLOW_CONFIG}"
    exit 1
fi

# # LongLive
# echo "[1/3] Running LongLive..."
# echo "Log: ${LONGLIVE_LOG}"
# cd "${PROJECT_ROOT}/LongLive"
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
#     interactive_inference.py \
#     --config_path configs/longlive_interactive_inference.yaml \
#     --data_path "${INPUT_PROMPTS}" \
#     --output_folder "${OUTPUT_DIR}/longlive" \
#     --generator_ckpt "${LONGLIVE_BASE}" \
#     --lora_ckpt "${LONGLIVE_LORA}" \
#     --seed ${SEED} \
#     2>&1 | tee "${LONGLIVE_LOG}"

# # MemFlow
# echo "[2/3] Running MemFlow..."
# echo "Log: ${MEMFLOW_LOG}"
# cd "${PROJECT_ROOT}/MemFlow"
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 \
#     interactive_inference.py \
#     --config_path configs/interactive_inference.yaml \
#     --data_path "${INPUT_PROMPTS}" \
#     --output_folder "${OUTPUT_DIR}/memflow" \
#     --generator_ckpt "${MEMFLOW_BASE}" \
#     --lora_ckpt "${MEMFLOW_LORA}" \
#     --seed ${SEED} \
#     2>&1 | tee "${MEMFLOW_LOG}"

# IAMFlow (使用 python 而非 torchrun，避免与 vLLM 分布式环境冲突)
echo "[3/3] Running IAMFlow..."
echo "Log: ${IAMFLOW_LOG}"
cd "${PROJECT_ROOT}/IAMFlow"
CUDA_VISIBLE_DEVICES=0 python agent_interactive_inference.py \
    --config_path "${IAMFLOW_CONFIG}" \
    --data_path "${INPUT_PROMPTS}" \
    --output_folder "${OUTPUT_DIR}/iamflow" \
    --generator_ckpt "${IAMFLOW_BASE}" \
    --lora_ckpt "${IAMFLOW_LORA}" \
    --llm_model_path "${QWEN_MODEL_PATH}" \
    --seed ${SEED} \
    $( [ "${USE_LIGHTVAE}" = "true" ] && echo "--use_lightvae" ) \
    2>&1 | tee "${IAMFLOW_LOG}"

echo "=========================================="
echo "Done! Results in: ${OUTPUT_DIR}"
echo "  longlive/ memflow/ iamflow/"
echo "=========================================="
