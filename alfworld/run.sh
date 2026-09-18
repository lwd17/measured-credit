#!/usr/bin/env bash
# Train one arm on ALFWorld.
#
#   ./alfworld/run.sh <arm> [GPU] [SAMPLER_PORT]
#
# Arms: grpo | anchor_cf   (add NONNEG=1 for the gated measured-credit recipe)
#
# 40 steps with checkpoints every 10: in this environment every arm becomes
# unstable past ~40 steps, so the reported checkpoint is chosen from the first
# 40 by the in-loop validation score. Override with STEPS.
#
# Start the sampler first: ./scripts/start_sampler.sh <GPU> <PORT> <MODEL>
set -euo pipefail
ARM=${1:?usage: run.sh <arm> [gpu] [port]}
GPU=${2:-0}; PORT=${3:-8100}
cd "$(dirname "$0")/.."
: "${ALFWORLD_DATA:?export ALFWORLD_DATA=<output of alfworld-download>}"
mkdir -p logs runs

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u alfworld/train.py \
    --arm "$ARM" \
    --steps "${STEPS:-40}" \
    --eval_every "${EVAL_EVERY:-10}" \
    --prompts_per_step "${PROMPTS:-4}" --group "${GROUP:-8}" \
    --max_turns "${MAX_TURNS:-30}" --hard_types \
    --n_train_tasks "${N_TRAIN:-400}" --n_eval_tasks "${N_EVAL:-24}" \
    --token_budget "${TOKEN_BUDGET:-16384}" \
    --collect_workers "${COLLECT_WORKERS:-32}" --annot_workers "${ANNOT_WORKERS:-64}" \
    --seed "${SEED:-1}" --lr "${LR:-3e-5}" \
    --omega "${OMEGA:-1.0}" --alpha "${ALPHA:-0.5}" \
    ${NONNEG:+--nonneg_winners} \
    ${MODEL:+--model_path "$MODEL"} \
    --base_url "http://localhost:${PORT}/v1" \
    --out_dir "runs/alf_${TAG:-$ARM}_s${SEED:-1}" \
    > "logs/alf_${TAG:-$ARM}.log" 2>&1 &
echo "ALFWorld arm=$ARM tag=${TAG:-$ARM} GPU$GPU :$PORT  pid $!"
