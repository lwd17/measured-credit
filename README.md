# Measured Credit

Per-turn credit for multi-turn agents, obtained by **re-running the environment**
instead of estimating it from the observed return.

A trajectory ends with a single reward, and the usual group-relative update
spreads that one reward over every turn of the trajectory. This repository does
something else: it asks the environment a counterfactual question about each turn
and uses the measured answer as the per-turn signal.

Two measurement operators, one for each kind of question:

| operator | question | how |
|---|---|---|
| **ablation** | did this turn contribute? | delete it, replay the rest, compare outcomes |
| **substitution** | was this the right choice here? | replay with an alternative that was available in the same state |

Both feed one combination rule:

```
A(i,t) = A_E(i) + omega * step(i,t)     additive form
A(i,t) = A_E(i) * q(i,t), mean(q) = 1   mass-preserving form
```

`A_E` is the group-standardised episode advantage. Setting `omega = 0`
(additive) or `alpha = 1` (mass-preserving) removes the measured term entirely
and recovers the plain group-relative update bit for bit; `tests/test_alf_arms.py`
asserts it.

Three environments are supported, each with the operator that is defined there:

| environment | measurement | arm |
|---|---|---|
| ALFWorld | ablation: drop one action, replay the episode | `--arm anchor_cf --nonneg_winners` |
| WebShop | ablation on "buy now" value + substitution among the items on the result page | `--arm anchor_cvmax` |
| Multi-hop QA (Search-R1 data) | substitution among sibling queries issued from the same evidence state | `--arm anchor_cf --nonneg_winners --regret_lambda 1` |

Every arm runs through the same loop, the same optimiser and the same rollout
path; `--arm` selects a vector of per-turn advantages and nothing else, so any
per-turn credit rule can be dropped in and compared on equal footing.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env && $EDITOR .env && source .env
```

`requirements.txt` covers training and inference. The three environments have
their own data dependencies; see below.

## Models

Any chat model that vLLM can serve with a LoRA adapter. The results this code was
written for used:

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`

Download once and point `--model_path` at the local directory:

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir ~/models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
```

Training is LoRA (r=32, alpha=64) on the attention and MLP projections; the base
weights stay frozen and are shared with the sampler.

## Sampler

Every run needs one vLLM server with runtime LoRA loading enabled. The trainer
pushes the adapter to it after each step.

```bash
./scripts/start_sampler.sh <GPU> <PORT> <MODEL_PATH>
# e.g. ./scripts/start_sampler.sh 0 8100 ~/models/Qwen3-4B
```

One sampler per arm is strongly recommended. Sharing a sampler between several
arms slows collection down and, for long-horizon evaluation, changes the numbers.

## Datasets

### ALFWorld

```bash
pip install alfworld[full]
alfworld-download -f
export ALFWORLD_DATA=~/.cache/alfworld
```

Training draws from the `train` split, evaluation from `valid_unseen` (134
tasks). `--hard_types` restricts training to the four task types that need long
horizons (`pick_clean_then_place`, `pick_heat_then_place`, `pick_cool_then_place`,
`pick_two_obj_and_place`).

### WebShop

Follow the upstream WebShop setup (https://github.com/princeton-nlp/WebShop) and
make the environment importable, then:

```bash
export WEBSHOP_PATH=/path/to/webshop
```

The split follows the official one: goal indices below 500 are the test set,
everything above is training. `ssc/env/webshop_env.py` enforces this; do not
reuse a split from an older checkout.

### Multi-hop QA

Data is the Search-R1 release (NQ + HotpotQA for training, seven datasets for
test):

```bash
huggingface-cli download PeterJinGo/nq_hotpotqa_train --repo-type dataset \
  --local-dir data/searchr1
export SEARCHR1_DIR=$PWD/data/searchr1
```

Retrieval is a separate HTTP service that must answer `POST /search` with
`{"query": [...]}` and `POST /browse` with `{"url": ["wiki://<doc_id>", ...]}`.
Point the trainer at it with `--wiki_url`. Any index can be used as long as all
arms in a comparison use the same one.

## Running

Each environment has a thin wrapper that fills in the configuration used for the
reported runs:

```bash
./alfworld/run.sh  <arm> [GPU] [PORT]
./webshop/run.sh   <arm> [GPU] [PORT]
./multihop/run.sh  <arm> [GPU] [PORT]
```

Or call the trainer directly. The three commands below are the exact recipes for
our method in each environment.

**ALFWorld** — ablation credit, additive, gated:

```bash
python alfworld/train.py --arm anchor_cf --nonneg_winners \
  --steps 40 --prompts_per_step 4 --group 8 --max_turns 30 --hard_types \
  --n_train_tasks 400 --n_eval_tasks 24 --eval_every 10 \
  --lr 3e-5 --omega 1.0 --gamma 0.95 --lora_r 32 \
  --model_path ~/models/Qwen3-4B --base_url http://localhost:8100/v1 \
  --out_dir runs/alf_ours_s1 --seed 1
```

**WebShop** — mass-preserving weights from graded "buy now" deltas plus
substitution regret over the result page:

```bash
python webshop/train.py --arm anchor_cvmax \
  --steps 100 --prompts_per_step 4 --group 8 --max_turns 15 \
  --n_train_tasks 400 --n_eval_tasks 48 --eval_every 10 \
  --lr 3e-5 --omega 1.0 --gamma 0.95 --binary_reward --graded_measurement \
  --alpha 0.5 --anneal_steps 60 --anneal_regret \
  --item_credit centred --regret_k 5 --regret_lambda 2 \
  --model_path ~/models/Qwen3-4B --base_url http://localhost:8100/v1 \
  --out_dir runs/ws_ours_s1 --seed 1
```

**Multi-hop QA** — substitution credit over sibling queries, additive, gated:

```bash
python multihop/train.py --arm anchor_cf --env wiki --nonneg_winners \
  --steps 120 --prompts_per_step 8 --group 8 --max_turns 4 --top_k 3 \
  --reward em --measure f1 --n_train_tasks 4000 \
  --lr 3e-5 --omega 0.5 --gamma 0.95 \
  --answer_credit mask --regret_lambda 1 --answer_regret --question_candidate \
  --dynamic_sampling --flat_extra_max 0 \
  --model_path ~/models/Qwen3-4B --base_url http://localhost:8100/v1 \
  --wiki_url http://127.0.0.1:8080 --out_dir runs/qa_ours_s1 --seed 1
```

`--arm grpo` runs the same command without the measurement flags, keeping the
steps, group size, learning rate and seed.

## Evaluating

```bash
python alfworld/eval.py  --arms <run-prefix> --steps 30 --n_tasks 134 --repeats 3 \
                         --max_turns 50 --temperature 0.4 --base_url ... --out report.json
python webshop/eval.py   --arms <run-prefix> --steps 80 --n_tasks 500 --repeats 2 \
                         --temperature 0.4 --base_url ... --out report.json
python multihop/eval.py  --runs <run-dir> --steps 120 --split test --n_per_source 300 \
                         --base_url ... --out report.json
```

The evaluation protocols follow each environment's standard setup: ALFWorld on
the 134 `valid_unseen` tasks with a 50-step budget at temperature 0.4, WebShop on
the 500 held-out goals at temperature 0.4, multi-hop QA greedily on the seven
Search-R1 test sets. Run evaluation on a sampler that nothing else is using.

## Three invariants

The switches below exist because violating any one of them cost a large amount of
end-task performance in our runs. They are the practical content of the method.

1. **A turn that could not be measured stays neutral.** It must not be entered
   into the normalisation as a measured zero, which would push it below the turns
   that were measured. `masked_zscore` and `--answer_credit mask` implement this.
2. **A trajectory with positive episode advantage never receives a negative step
   term** (`--nonneg_winners`). A zero-mean step term necessarily assigns negative
   credit to some turns of a successful trajectory, and those turns are often the
   exploration that made the success possible.
3. **The step term must not dominate the episode advantage.** Where the
   measurement is sparse, its z-score is large; `--omega` and `--regret_lambda`
   are the dials, and the additive step term should stay at or below the scale of
   `A_E`.

## Layout

```
alfworld/ webshop/ multihop/   entry points: train.py, eval.py, run.sh
ssc/credit/                    advantage assembly and the measurement operators
  alf_arms.py                    turn_advantages: arms, gates, normalisation
  anchor_cf.py                   ALFWorld ablation replay
  webshop_cf.py                  WebShop commit deltas and item regret
  multihop_cf.py                 QA evidence-value deltas and query regret
  counterfactual.py              contribution weights for the mass-preserving form
  grpo_patch.py                  the clipped policy-gradient loss
ssc/env/                       environments and policies
ssc/train/                     batching, one optimiser step, LoRA push to sampler
tests/                         unit tests, including the degeneracy checks
```

## Tests

```bash
pytest tests/ -q
```

The ALFWorld measurement tests touch the real environment and need
`ALFWORLD_DATA`; the rest run offline.
