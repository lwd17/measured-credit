"""Measured marginal contribution of each retrieval step, by evidence ablation.

ALFWorld and WebShop recompute the outcome themselves from a re-executed action
script, so an ablation there costs zero model calls. Multi-hop QA cannot: the
final answer is a string the policy produced, and replaying a shortened script
reproduces that same string, so the reward never moves and the delta would be
zero everywhere by construction -- a measurement that reads zero because of the
protocol, not because of the trajectory.

So the ablation removes a step's UNIQUE evidence and regenerates the answer:

    delta(t) = F1(answer | all evidence) - F1(answer | evidence without step t)

which costs one generation each. Two things keep that affordable. Regeneration
is greedy, so an identical evidence set gives an identical answer and results
are cached by the evidence itself -- a step whose documents were also retrieved
by another step changes nothing, hits the cache, and correctly measures zero.
And the answer is regenerated alone rather than re-running the whole trajectory,
which is what makes this cheaper than a rollout instead of comparable to one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ssc.env.multihop_env import MultiHopState, exact_match, f1


def _scorer(metric: str):
    return exact_match if metric == "em" else f1


@dataclass
class MultiHopCredit:
    deltas: list[list[float]]
    n_calls: int
    n_steps: int
    # True where an ablation was actually RUN. An answer turn has no evidence to
    # drop, so its zero means "not measured", not "measured and worth nothing" --
    # a distinction the advantage pipeline has to keep, because anything scored
    # as a plain zero gets pushed below the mean and becomes a penalty. Left in,
    # it scored the answer turn 1.324 below the searches around it and the policy
    # stopped answering at all.
    mask: list[list[bool]] = None
    # For the regret / centring term only: per turn, "search" / "answer" / None
    # (not measurable).
    kinds: list[list[str | None]] = None
    # For probes: V(E_0), V(E_1), ... for each rollout (E_0 = closed book).
    values: list[list[float]] = None


ANSWER_SYSTEM = ("Answer the question from the documents. Reply with only the "
                 "answer: a name, a date, or yes/no. No explanation.")

# Used in the open-domain setting: the reader may use its own knowledge. In the
# probe (logs/wq_signal_probe.log), an "answer from the documents" reader got
# questions it could answer closed book wrong as soon as it was handed an
# irrelevant passage (V prefix [1.0, 0.0]). That is an artefact forced by the
# prompt, not the value of retrieval; what V(E) should measure is "how well one
# can answer right now".
ANSWER_SYSTEM_OPEN = ("Answer the question. Use the documents if they help; otherwise "
                      "use your own knowledge. Reply with only the answer: a name, a "
                      "date, a number, or yes/no. No explanation.")


def answer_prompt(question: str, evidence: str, system: str = ANSWER_SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content":
             f"Documents:\n{evidence}\n\nQuestion: {question}\nAnswer:"}]


def multihop_deltas(generate, states: list[MultiHopState],
                    rollouts, cache: dict | None = None,
                    metric: str = "f1") -> MultiHopCredit:
    """`generate(messages) -> str` must be greedy for the cache to be sound.

    One row per rollout, one entry per TURN (answer turns included, at 0.0), so
    the row lines up with `Rollout.turns` the way the other two environments'
    rows do.
    """
    cache = {} if cache is None else cache
    calls = 0

    def ask(question: str, evidence: str) -> str:
        key = (question, evidence)
        if key not in cache:
            nonlocal calls
            cache[key] = generate(answer_prompt(question, evidence))
            calls += 1
        return cache[key]

    score = _scorer(metric)
    out: list[list[float]] = []
    masks: list[list[bool]] = []
    n_steps = 0
    for st, roll in zip(states, rollouts):
        gold = st.task.all_answers()
        base = score(ask(st.task.question, st.evidence()), gold)
        # Turn index -> retrieval index. Answer turns have no evidence to drop.
        search_at = [i for i, t in enumerate(roll.turns)
                     if (t.tool_args or {}).get("kind") == "search"]
        row = [0.0] * len(roll.turns)
        keep = [False] * len(roll.turns)
        for s, turn_i in enumerate(search_at):
            if s >= len(st.retrieved):
                break
            abl = ask(st.task.question, st.evidence(drop_step=s))
            row[turn_i] = base - score(abl, gold)
            keep[turn_i] = True
            n_steps += 1
        out.append(row)
        masks.append(keep)
    return MultiHopCredit(out, calls, n_steps, masks)


def _asker(generate, cache: dict, system: str = ANSWER_SYSTEM):
    """(question, evidence) -> greedy answer, cached; returns (ask, counter)."""
    counter = {"calls": 0}

    def ask(question: str, evidence: str) -> str:
        key = (system, question, evidence)
        if key not in cache:
            cache[key] = generate(answer_prompt(question, evidence, system))
            counter["calls"] += 1
        return cache[key]
    return ask, counter


def _search_turns(roll) -> list[int]:
    return [i for i, t in enumerate(roll.turns)
            if (t.tool_args or {}).get("kind") == "search"]


def multihop_commit_deltas(generate, states: list[MultiHopState], rollouts,
                           cache: dict | None = None, running_max: bool = True,
                           mask_answer: bool = True, metric: str = "f1",
                           answer_credit: str | None = None,
                           repeat_penalty: float = 0.0,
                           answer_system: str = ANSWER_SYSTEM) -> MultiHopCredit:
    """Multi-hop version of WebShop's `webshop_commit_deltas`: credit is the value
    of answering right now.

        V(E_t)   = score of answering (greedily) straight from the evidence
                   gathered by the first t searches; E_0 = closed book
        V*(t)    = max_{s<=t} V(E_s)                       (running_max)
        credit(search t) = V*(t+1) - V*(t) >= 0            did this search raise
                                                           how well we can answer now
    With running_max=False it is V(E_{t+1}) - V(E_t), which may be negative (new
    evidence pulls the greedy answer off course).

    Telescoping decomposition of the return:
    R = V(E_0) + sum_t [V(E_{t+1}) - V(E_t)] + [R - V(E_T)], whose last term is the
    answer turn's "extraction residual" (the sampled answer relative to the greedy
    one) and is not credited to retrieval. The answer turn is masked by default
    (the multiplicative form then uses the mean weight over the measured turns);
    with `mask_answer=False` it takes part in the within-anchor-group comparison
    with 0 (in one evidence state, the V* gain of "search once more" against the 0
    of "answer now" -- the multi-hop version of the buy-too-early pathology that
    WebShop's `unmask_commit` fixes).

    `answer_credit` overrides that switch:
        "mask"            the answer turn is not measurable (the multiplicative
                          form uses the mean weight over the measured turns)
        "zero"            the answer turn takes part with 0 (within-anchor-group form)
        "prior_residual"  the answer turn = V(E_0) + [R_m - V*(T)]: the part the
                          model already knew closed book, plus the extraction residual
    The last option closes the telescoping decomposition: sum_t credit(t) = R_m,
    where R_m is the sampled answer's score under `metric`. 93% of the base model's
    rollouts contain a single search, and in a two-turn [search, answer] rollout,
    measuring only the search necessarily makes the multiplicative weights uniform
    (probe: 0% non-uniform). Once the answer turn is counted according to the
    decomposition, a rollout where the model already knew the answer and retrieval
    did not help moves credit off the irrelevant query and onto the answer turn --
    which is exactly GRPO's misattribution here (an irrelevant query rewarded
    because the outcome happened to be right).
    """
    cache = {} if cache is None else cache
    ask, counter = _asker(generate, cache, answer_system)
    score = _scorer(metric)
    if answer_credit is None:
        answer_credit = "mask" if mask_answer else "zero"
    out, masks, kinds, values = [], [], [], []
    n_steps = 0
    for st, roll in zip(states, rollouts):
        gold = st.task.all_answers()
        q = st.task.question
        # V along the evidence prefixes: E_0 is empty, E_s = the first s searches
        vals = []
        for s in range(len(st.retrieved) + 1):
            ids = [i for step in st.retrieved[:s] for i in step]
            vals.append(score(ask(q, st.evidence_for(ids)), gold))
        values.append(vals)
        row = [0.0] * len(roll.turns)
        keep = [False] * len(roll.turns)
        krow: list[str | None] = [None] * len(roll.turns)
        best = vals[0] if vals else 0.0      # V* under running_max, else current V(E_t)
        s = 0
        for t, turn in enumerate(roll.turns):
            kind = (turn.tool_args or {}).get("kind")
            if kind == "search" and s + 1 < len(vals):
                v0, v1 = vals[s], vals[s + 1]
                # A search that brings back no new document (a repeated query, or
                # results that are all passages already read) burns a turn for
                # nothing. The V increment for it is exactly 0, and zero is not a
                # penalty -- in the 2Wiki cases, issuing the same query twice in a
                # row accounts for part of the questions we lose. `repeat_penalty`
                # gives it an explicit negative credit, sized against the turn
                # budget of the whole trajectory.
                if repeat_penalty and s < len(st.retrieved):
                    prev = {i for step_ids in st.retrieved[:s] for i in step_ids}
                    if prev and not (set(st.retrieved[s]) - prev):
                        row[t] = -repeat_penalty
                        keep[t] = True
                        krow[t] = "search"
                        n_steps += 1
                        s += 1
                        continue
                if running_max:
                    nxt = max(best, v1)
                    row[t] = nxt - best
                    best = nxt
                else:
                    # Evidence only accumulates, so a search that drags in a
                    # distractor passage lowers how well one can answer right now.
                    # That harm is real (probe: V drops on 22% of search steps),
                    # so the negative sign is kept.
                    row[t] = v1 - v0
                    best = v1
                keep[t] = True
                krow[t] = "search"
                n_steps += 1
                s += 1
            elif kind == "answer":
                krow[t] = "answer"
                if answer_credit == "zero":
                    row[t] = 0.0
                    keep[t] = True
                elif answer_credit == "prior_residual":
                    r_m = score(getattr(st, "final", "") or "", gold)
                    row[t] = vals[0] + (r_m - best)
                    keep[t] = True
        out.append(row); masks.append(keep); kinds.append(krow)
    return MultiHopCredit(out, counter["calls"], n_steps, masks, kinds, values)


def multihop_regret(generate, states: list[MultiHopState], rollouts,
                    cache: dict | None = None, mode: str = "centred",
                    include_answer: bool = True, metric: str = "f1",
                    running_max: bool = True, propose_query=None,
                    question_candidate: bool = False,
                    answer_system: str = ANSWER_SYSTEM,
                    answer_regret_metric: str | None = None) -> MultiHopCredit:
    """Measured value of this rollout's action relative to its sibling actions in
    the same evidence state.

    WebShop's `webshop_regret(mode="centred")` treats the top K items on a result
    page as a "measured virtual anchor group". Multi-hop has no ready-made list of
    candidates, but the different queries that the 8 rollouts of one question issue
    from the same evidence state (the same set of already retrieved documents) are
    exactly that -- and the value of each candidate can be measured by replay
    (retrieval is deterministic, greedy answering is deterministic) instead of
    estimated.

        state s = the set of retrieved documents (deduplicated as a set: two
                  queries that fetch the same documents are the same state)
        Q(search q | s) = max(V(E_s), V(E_s ∪ ret(q)))      (running_max)
        Q(answer | s)   = V(E_s)                            answering now cashes in
                                                            exactly V(E_s)
        centred: Q(own) - mean(pool)      regret: min(0, Q(own) - max(pool))

    The candidate pool is deduplicated by "evidence set reached", so a state where
    all 8 rollouts issue the same query has nothing to compare against: credit 0,
    flagged as not measurable (not "measured and worth zero"). A step-level
    baseline that compares discounted returns in the same state drags all the
    downstream sampling noise in with them; what is compared here is the measured
    value of the action itself.

    `propose_query(roll, state) -> str | None`: the **lookahead candidate** for the
    answer turn. The base model almost never searches twice (1.07 searches per
    rollout), so "answer now vs search once more" has no counterpart inside the
    group. Here the current policy is asked, in the context it saw just before
    answering (greedy, and instructed to search once more), to propose a query;
    Q = max(V(E), V(E ∪ ret(q'))) is measured by replay and enters as one candidate
    for that state. This is the multi-hop version of the buy turn's
    "V(now) - V(completion)" in WebShop's `webshop_regret`: the candidate action is
    proposed by the policy and its value is measured by replay, not estimated. An
    answer forced by an exhausted turn budget gets no lookahead (there was no
    alternative).

    `question_candidate`: add one fixed candidate at the initial state, "use the
    question text verbatim as the query", so every group has something to compare
    against on the first step (when all 8 rollouts issue the same query the sibling
    pool holds only 1 candidate).

    `answer_regret_metric` (e.g. "em"): add an **extraction regret** on the answer
    turn
        min(0, score(sampled answer) - score(greedy short answer on the same evidence))
    Cases (logs/wq_cases_40.log): half of the questions L_mxS@40 loses to GRPO are
    answers that are too specific or too long -- "Austin, Texas" against "Austin",
    "Cudjoe Key, Marco Island" against "Florida" -- where the evidence is enough
    and the greedy reader gets it right, but the sampled answer loses EM on
    formatting. This term is negative only in that situation and measures "with
    this evidence the question could have been answered". It is the answer-turn
    version of WebShop's buy regret V(now)-V(completion).
    """
    cache = {} if cache is None else cache
    ask, counter = _asker(generate, cache, answer_system)
    score = _scorer(metric)
    n = len(rollouts)
    # Per rollout and per turn: state key, action-candidate key, action value
    per: list[list[tuple | None]] = []
    v_cache: dict = {}

    def value(st: MultiHopState, ids: tuple) -> float:
        key = (st.task.question, ids)
        if key not in v_cache:
            v_cache[key] = score(ask(st.task.question, st.evidence_for(list(ids))),
                                 st.task.all_answers())
        return v_cache[key]

    extra: dict = {}          # state -> {candidate key: value}; lookahead / question
    n_lookahead = n_lookahead_gain = 0

    def add_candidate(st: MultiHopState, prefix: list, query: str, tag: str) -> float | None:
        q = " ".join((query or "").split())
        if not q:
            return None
        got = st.retriever.search(q)
        nxt = prefix + [i for i in got if i not in prefix]
        here = value(st, tuple(prefix))
        v = value(st, tuple(nxt))
        qv = max(here, v) if running_max else v
        extra.setdefault(frozenset(prefix), {})[(tag, frozenset(nxt))] = qv
        return qv

    for st, roll in zip(states, rollouts):
        rows: list[tuple | None] = [None] * len(roll.turns)
        prefix: list = []
        s = 0
        if question_candidate:
            add_candidate(st, [], st.task.question, "q")
        for t, turn in enumerate(roll.turns):
            args = turn.tool_args or {}
            kind = args.get("kind")
            skey = frozenset(prefix)
            here = value(st, tuple(prefix))
            if kind == "search" and s < len(st.retrieved):
                got = list(st.retrieved[s])
                nxt = prefix + [i for i in got if i not in prefix]
                v = value(st, tuple(nxt))
                q = max(here, v) if running_max else v
                rows[t] = (skey, ("s", frozenset(nxt)), q, "search")
                prefix = nxt
                s += 1
            elif kind == "answer" and include_answer:
                rows[t] = (skey, ("a",), here, "answer")
                budget = args.get("step_budget")
                if propose_query is not None and (budget is None or t + 1 < int(budget)):
                    try:
                        q2 = propose_query(roll, st)
                    except Exception:  # noqa: BLE001
                        q2 = None
                    if q2:
                        qv = add_candidate(st, prefix, q2, "la")
                        n_lookahead += 1
                        if qv is not None and qv > here + 1e-9:
                            n_lookahead_gain += 1
        per.append(rows)

    # state -> {candidate key: value}
    pools: dict = {}
    for rows in per:
        for r in rows:
            if r is None:
                continue
            skey, akey, q, _ = r
            pools.setdefault(skey, {})[akey] = q
    for skey, cands in extra.items():
        pools.setdefault(skey, {}).update(cands)

    out, masks, kinds = [], [], []
    n_steps = 0
    a_score = _scorer(answer_regret_metric) if answer_regret_metric else None
    for st, roll, rows in zip(states, rollouts, per):
        row = [0.0] * len(rows)
        keep = [False] * len(rows)
        krow: list[str | None] = [None] * len(rows)
        for t, r in enumerate(rows):
            if r is None:
                continue
            skey, akey, q, kind = r
            krow[t] = kind
            pool = list(pools[skey].values())
            if len(pool) >= 2:                # nothing to compare: not measurable, not zero
                if mode == "centred":
                    row[t] = q - sum(pool) / len(pool)
                else:
                    row[t] = min(0.0, q - max(pool))
                keep[t] = True
                n_steps += 1
            if kind == "answer" and a_score is not None:
                golds = st.task.all_answers()
                ids = [i for step in st.retrieved for i in step]
                reader = ask(st.task.question, st.evidence_for(ids))
                a_reg = min(0.0, a_score(getattr(st, "final", "") or "", golds) - a_score(reader, golds))
                if a_reg < 0:
                    row[t] += a_reg
                    keep[t] = True
        out.append(row); masks.append(keep); kinds.append(krow)
    cr = MultiHopCredit(out, counter["calls"], n_steps, masks, kinds)
    cr.n_lookahead, cr.n_lookahead_gain = n_lookahead, n_lookahead_gain
    return cr
