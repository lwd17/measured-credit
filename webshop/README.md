# WebShop

Shopping from a text storefront: search, read result pages, open products, pick
options, buy. The reward at the end is a graded match against the requested
attributes; training uses the binary version of it.

## Measurement: ablation on value, substitution on choice

Two measured quantities feed the advantage.

**Commit deltas** (`webshop_commit_deltas`) ask, at each turn, what the graded
score would be if the agent bought right now:

```
V*(t)             = max over prefixes up to t of "buy now" score
credit(t)         = V*(t+1) - V*(t)
```

**Item regret** (`webshop_regret`) is the substitution counterfactual: on a
result page, replay the purchase of each of the top-K alternatives and compare
against the item the agent actually opened. `--item_credit centred` centres that
comparison within the same page across the rollouts of one task, which keeps the
signal from degenerating into a preference for particular product ids.

Both measurements must use the **graded** score (`--graded_measurement`) even
though the reward is binary. Under a binary measurement, opening a product page
scores 0 no matter which product it is, every item regret is zero, and the
selection signal is gone.

## Advantage

```
A(i,t) = A_E(i) * q(i,t) + lambda * regret(i,t),   lambda = 2
q      = contribution_weights(delta; alpha, w_max), mean(q) = 1 over measured turns
```

`q` is applied only to trajectories with positive episode advantage: re-weighting
a failed trajectory would also dampen the penalty on the action it should stop
repeating. Over the first 60 steps `alpha -> 1`, `omega -> 0` and `lambda -> 0`,
so the arm anneals into plain GRPO.

## Run

```bash
export WEBSHOP_PATH=/path/to/webshop
./scripts/start_sampler.sh 0 8100 ~/models/Qwen3-4B

TAG=ours ./webshop/run.sh anchor_cvmax 0 8100
TAG=grpo ./webshop/run.sh grpo         0 8100
```

The split is the official one: goal indices below 500 are test, the rest are
training (`ssc/env/webshop_env.py`). A format failure does not end the episode;
the projection always emits an action, because under a binary reward an early
termination becomes an absorbing state the policy learns to fall into.

## Evaluate

```bash
python webshop/eval.py --arms ws_ours --seed 1 --steps 80 \
  --n_tasks 500 --repeats 2 --temperature 0.4 \
  --base_url http://localhost:8100/v1 --served_name Qwen3-4B \
  --out reports/ws_ours.json
```

500 held-out goals, two runs each, temperature 0.4. The report contains both the
graded score and the exact-match success rate.
