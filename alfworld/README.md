# ALFWorld

Household tasks in a text world. A trajectory ends with a single binary reward,
and the question this setting answers is: which of the thirty turns mattered?

## Measurement: ablation

For every **successful** rollout the trainer deletes one action, replays the
remaining sequence in a pinned copy of the same game, and records

```
delta(i,t) = R(tau_i) - R(tau_i without turn t)
```

This is an exact leave-one-out sweep over that rollout's own actions; two
deletions share a replay only when they produce the same action sequence, which
is the same counterfactual. Failed rollouts measure almost nothing this way
(dropping an action turns a loss into a win on about 3% of their turns), so they
receive a cheap fallback: how much of what remains from turn t was spent in
states that some successful rollout in the group also visited.

`ssc/credit/anchor_cf.py` holds the replay; `anchor_deltas(..., batched=True)` is
the fast path and `batched=False` runs the same sweep one environment at a time
so the two can be checked against each other.

## Advantage

```
A(i,t) = A_E(i) + omega * step(i,t),  omega = 1
step   = centre_within_trajectory(masked_zscore(delta))
gate   : if A_E(i) > 0 then step(i,t) <- max(step(i,t), 0)
```

The gate (`--nonneg_winners`) is the difference between the method working and
not working on smaller models. A zero-mean step term gives negative credit to
the turns of a successful trajectory whose deletion did not change the outcome;
most of those are exploratory navigation, which is exactly what a weaker policy
needs in order to find objects at all.

## Run

```bash
export ALFWORLD_DATA=~/.cache/alfworld            # output of `alfworld-download -f`
./scripts/start_sampler.sh 0 8100 ~/models/Qwen3-4B

NONNEG=1 TAG=ours ./alfworld/run.sh anchor_cf 0 8100     # measured credit
            TAG=grpo ./alfworld/run.sh grpo      0 8100  # no measurement
```

`--hard_types` (set by `run.sh`) restricts training to the four task types that
need long horizons. Training uses a 30-step budget.

## Evaluate

```bash
python alfworld/eval.py --arms alf_ours --arm_seed 1 --steps 30 \
  --n_tasks 134 --repeats 3 --max_turns 50 --temperature 0.4 \
  --base_url http://localhost:8100/v1 --served_name Qwen3-4B \
  --out reports/alf_ours.json
```

134 `valid_unseen` tasks, three runs each, 50-step budget, temperature 0.4. Use
a sampler that nothing else is using: a shared sampler changes long-horizon
results, not just their speed.

## Notes

- The replay must run in the parent process. The batched path needs
  `asynchronous=True`, and the annotation pool's workers are daemonic, so they
  cannot spawn the children it requires.
- Give each run a distinct `--out_dir`; the adapter name pushed to the sampler is
  derived from it, and two runs sharing a name overwrite each other's weights.
