# Multi-hop QA

Search-augmented question answering on the Search-R1 data: the policy issues up
to three searches and then answers. Reward is exact match.

## Measurement: substitution between sibling queries

Deletion is nearly silent here. Evidence accumulates monotonically, so removing
one search usually does not change what the model can answer: in our runs 74% of
search turns have a deletion delta of exactly zero. The measurement that is
informative in this environment is substitution.

**Evidence value.** `V(E_t)` is the score of the greedy answer produced from the
evidence gathered in the first t searches. The per-search credit is
`V(E_{t+1}) - V(E_t)`, which may be negative when a retrieved passage pulls the
answer off course.

**Query regret** (`multihop_regret`). At a given evidence state, the other
rollouts of the same question proposed different queries; replaying each of them
gives a measured value for what was available at that state. The agent's own
query is scored against that pool, centred. The question text itself is added as
a fixed candidate so that almost every turn has something to compare against.

**The answer turn is not measurable** and is marked as such
(`--answer_credit mask`). Entering it as a measured zero pushes it below the
search turns in the normalisation, which stops training the step that actually
produces the answer; when we did that, the policy stopped searching and the score
collapsed.

## Advantage

```
A(i,t) = A_E(i) + omega * step(i,t) + lambda * regret(i,t)
omega  = 0.5      lambda = 4
gate   : if A_E(i) > 0 then step(i,t) <- max(step(i,t), 0)
```

The step term is deliberately at half weight. Where the measurement is sparse its
z-score is large, and a full-weight sparse term costs out-of-domain accuracy: the
training pool is NQ + HotpotQA while five of the seven test sets are something
else.

## Run

```bash
export SEARCHR1_DIR=$PWD/data/searchr1     # train.parquet / test.parquet
export WIKI_URL=http://127.0.0.1:8080      # retrieval service, see below
./scripts/start_sampler.sh 0 8100 ~/models/Qwen3-4B

NONNEG=1 TAG=ours ./multihop/run.sh anchor_cf 0 8100
            TAG=grpo ./multihop/run.sh grpo   0 8100
```

### Retrieval service

The trainer expects an HTTP service with two endpoints:

- `POST /search` with `{"query": ["..."]}` returning ranked snippets with
  `doc_id`;
- `POST /browse` with `{"url": ["wiki://<doc_id>", ...]}` returning full
  passages.

Any index works as long as every arm in a comparison uses the same one; the
retriever changes absolute scores substantially.

## Evaluate

```bash
python multihop/eval.py --runs qa_ours_s1 --steps 120 \
  --split test --n_per_source 300 \
  --base_url http://localhost:8100/v1 --served_name Qwen3-4B \
  --out reports/qa_ours.json
```

Greedy decoding on the seven Search-R1 test sets, sampled evenly per source.

**Checkpoint selection.** The validation pool can only be carved out of the
training file, which holds NQ and HotpotQA alone, so it does not predict the
seven-set test score — in our runs the two were uncorrelated and occasionally
inverted. Report the final checkpoint, the same one for every arm, rather than
selecting on that validation pool.
