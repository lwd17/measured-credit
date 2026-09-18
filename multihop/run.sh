#!/usr/bin/env bash
# Train one arm on multi-hop QA (Search-R1 data + a retrieval service).
#
#   ./multihop/run.sh <arm> [GPU] [SAMPLER_PORT]
#
# Arms: grpo | anchor_cf   (add NONNEG=1 for the gated measured-credit recipe)
#
# Defaults are the recipe reported for this environment: half-weight step term
# (omega 0.5), substitution regret as the dominant signal (lambda 4), the answer
# turn treated as unmeasurable so it keeps GRPO's credit, no annealing, and no
# extra zero-variance groups in the batch.
#
# Start the sampler first: ./scripts/start_sampler.sh <GPU> <PORT> <MODEL>
set -euo pipefail
ARM=${1:?usage: run.sh <arm> [gpu] [port]}
GPU=${2:-0}; PORT=${3:-8100}
cd "$(dirname "$0")/.."
: "${SEARCHR1_DIR:?export SEARCHR1_DIR=<directory with train.parquet/test.parquet>}"
mkdir -p logs runs

CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python -u multihop/train.py \
    --arm "$ARM" --env wiki \
    --steps "${STEPS:-120}" \
    --eval_every "${EVAL_EVERY:-20}" \
    --prompts_per_step "${PROMPTS:-8}" --group "${GROUP:-8}" \
    --max_turns "${MAX_TURNS:-4}" --top_k "${TOP_K:-3}" \
    --reward em --measure f1 \
    --n_train_tasks "${N_TRAIN:-4000}" --n_eval_tasks "${N_EVAL:-200}" \
    --token_budget "${TOKEN_BUDGET:-16384}" \
    --seed "${SEED:-1}" --lr "${LR:-3e-5}" --max_grad_norm 1.0 \
    --omega "${OMEGA:-0.5}" --regret_lambda "${REGRET_LAMBDA:-4}" \
    --answer_credit mask --answer_regret --question_candidate \
    --dynamic_sampling --dyn_max_factor 8 --flat_extra_max 0 \
    ${NONNEG:+--nonneg_winners} \
    ${MODEL:+--model_path "$MODEL"} \
    --wiki_url "${WIKI_URL:-http://127.0.0.1:8080}" \
    --base_url "http://localhost:${PORT}/v1" \
    --out_dir "runs/qa_${TAG:-$ARM}_s${SEED:-1}" \
    > "logs/qa_${TAG:-$ARM}.log" 2>&1 &
echo "Multi-hop QA arm=$ARM tag=${TAG:-$ARM} GPU$GPU :$PORT  pid $!"
