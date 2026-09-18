#!/usr/bin/env bash
# Train one arm on WebShop.
#
#   ./webshop/run.sh <arm> [GPU] [SAMPLER_PORT]
#
# Arms: grpo | anchor_cvmax   (anchor_cvmax is the measured-credit recipe)
#
# The training reward is binary; under group standardisation 1/0 is equivalent
# to the environment's 10/0. The measurement, however, must use the graded
# score (--graded_measurement): with a binary measurement, opening a product
# page always scores 0 and the whole item-selection signal disappears.
#
# Start the sampler first: ./scripts/start_sampler.sh <GPU> <PORT> <MODEL>
set -euo pipefail
ARM=${1:?usage: run.sh <arm> [gpu] [port]}
GPU=${2:-0}; PORT=${3:-8100}
cd "$(dirname "$0")/.."
: "${WEBSHOP_PATH:?export WEBSHOP_PATH=<your WebShop checkout>}"
mkdir -p logs runs

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u webshop/train.py \
    --arm "$ARM" \
    --steps "${STEPS:-100}" \
    --eval_every "${EVAL_EVERY:-10}" \
    --prompts_per_step "${PROMPTS:-4}" --group "${GROUP:-8}" \
    --max_turns "${MAX_TURNS:-15}" \
    --n_train_tasks "${N_TRAIN:-400}" --n_eval_tasks "${N_EVAL:-48}" \
    --token_budget "${TOKEN_BUDGET:-16384}" \
    --collect_workers "${COLLECT_WORKERS:-16}" --env_workers "${ENV_WORKERS:-16}" \
    --seed "${SEED:-1}" --lr "${LR:-3e-5}" --max_grad_norm 1.0 \
    --binary_reward --graded_measurement \
    --omega "${OMEGA:-1.0}" --alpha "${ALPHA:-0.5}" \
    --anneal_steps "${ANNEAL:-60}" --anneal_regret \
    --item_credit "${ITEM_CREDIT:-centred}" --regret_k "${REGRET_K:-5}" \
    --regret_lambda "${REGRET_LAMBDA:-2}" \
    ${MODEL:+--model_path "$MODEL"} \
    --base_url "http://localhost:${PORT}/v1" \
    --out_dir "runs/ws_${TAG:-$ARM}_s${SEED:-1}" \
    > "logs/ws_${TAG:-$ARM}.log" 2>&1 &
echo "WebShop arm=$ARM tag=${TAG:-$ARM} GPU$GPU :$PORT  pid $!"
