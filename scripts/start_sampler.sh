#!/usr/bin/env bash
# Start one vLLM sampler. The trainer pushes a fresh LoRA adapter to it after
# every step, so VLLM_ALLOW_RUNTIME_LORA_UPDATING must be on: without it the
# push returns 404 and the policy silently falls back to the base model while
# the training log still looks normal.
#
#   ./scripts/start_sampler.sh <GPU> <PORT> [MODEL_PATH] [GPU_MEM_FRACTION]
set -euo pipefail
GPU=${1:?usage: start_sampler.sh <gpu> <port> [model] [mem_frac]}
PORT=${2:?}
MODEL=${3:-${SSC_MODEL:-Qwen/Qwen3-8B}}
FRAC=${4:-0.45}
cd "$(dirname "$0")/.."
mkdir -p logs
CUDA_VISIBLE_DEVICES="$GPU" VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$(basename "$MODEL")" \
    --port "$PORT" --max-model-len "${MAX_LEN:-16384}" \
    --gpu-memory-utilization "$FRAC" \
    --enable-lora --max-lora-rank "${LORA_RANK:-32}" --max-loras "${MAX_LORAS:-2}" \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    > "logs/vllm_${PORT}.log" 2>&1 &
echo "sampler $MODEL on GPU$GPU :$PORT (pid $!) -> logs/vllm_${PORT}.log"
echo "wait for it:  curl -s http://localhost:$PORT/v1/models"
