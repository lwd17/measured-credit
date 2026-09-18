"""Anchor counterfactual credit on WebShop.

Same measurement as `anchor_cf` on ALFWorld -- delete one action, replay the
rest, read the outcome -- with three differences the environment forces:

  the reward is GRADED   WebShop scores attribute match in [0, 1], so a delta
                         is a continuous quantity rather than the 0/1 ALFWorld
                         produces. Measured on 213 ablations of the untrained
                         policy, 64% moved the outcome (ALFWorld: 58%), and the
                         values that move spread across the range instead of
                         landing on a single step.

  replay runs OUT OF PROCESS   WebShop pins 2022 dependencies and its own
                         Python, so ablations go through the worker rather than
                         a batched env. There is no batch to share a reset
                         across, which is where ALFWorld's 9.5x came from; what
                         remains is the pooling.

  the state key is the page   an observation string identifies where the agent
                         is well enough to pool on: the same search results
                         page reached twice is the same page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STEP_RE = re.compile(r"Step \d+/\d+")
ASIN_RE = re.compile(r"\b(b0[0-9a-z]{8})\b")   # WebShop product id, in page order


def page_key(observation: str) -> str:
    """What ablations pool on. The step counter is stripped for the reason it
    is stripped on ALFWorld: leaving it in makes every page unique and pooling
    degenerates into full leave-one-out at full price."""
    o = STEP_RE.sub("", observation or "").strip().lower()
    return " ".join(o.split())[:600]


@dataclass
class WebShopCredit:
    deltas: list[list[float]]
    n_replays: int
    n_turns: int
    # True where the ablation carries usable placement information. A turn whose
    # removal drove a positive return to exactly zero was structurally required:
    # its delta IS the episode return, which A_E already supplies. Marking it
    # rather than zeroing it is the difference between "no claim" and a penalty.
    mask: list[list[bool]] = None
    kinds: list[list[str | None]] = None   # regret term only: "item" / "buy" / None per turn


def webshop_deltas(worker, sessions: list[int], action_seqs: list[list[str]],
                   observations: list[list[str]],
                   fill_offpath: bool = True,
                   drop_required: bool = True,
                   pool: bool = False,
                   binary: bool = False,
                   mask_structural: bool = False) -> WebShopCredit:
    """Measured marginal contribution of every turn in one group.

    One replay per distinct (page, action) pair, plus one base replay per
    rollout. The base is replayed rather than taken from the record for the
    same reason as on ALFWorld: a mismatch between the recorded reward and a
    replay would otherwise hide inside every delta as a uniform shift.
    """
    n = len(action_seqs)
    if n == 0:
        return WebShopCredit([], 0, 0)

    # Pooling was designed for ALFWorld, where a replay costs a 1.6 s environment
    # reset and one measurement per (state, action) is the difference between an
    # affordable annotation and an unaffordable one. WebShop's replays are far
    # cheaper, and measured there the approximation costs more than it saves:
    # against full leave-one-out it correlates at rho = 0.719 and hands 14% of
    # turns an outright different value, while saving 2.1% of a training step.
    # Off by default here for that reason; ALFWorld keeps it.
    if not pool:
        rep = {(i, t): (i, t) for i, acts in enumerate(action_seqs)
               for t in range(len(acts))}
        keyer = lambda i, t, a: (i, t)  # noqa: E731
    else:
        rep = {}
        keyer = lambda i, t, a: (page_key(  # noqa: E731
            observations[i][t] if t < len(observations[i]) else ""), a)
        for i, acts in enumerate(action_seqs):
            for t, a in enumerate(acts):
                rep.setdefault(keyer(i, t, a), (i, t))
    if not rep:
        return WebShopCredit([[0.0] * len(a) for a in action_seqs], 0, 0)

    # The ablation must be scored the same way the episode is, or the step term
    # and the episode term measure different quantities.
    def score(session, actions):
        r = worker.replay(session, actions)
        return (1.0 if r >= 1.0 else 0.0) if binary else r

    bases = [score(sessions[i], action_seqs[i]) for i in range(n)]
    measured: dict[tuple[str, str], float] = {}
    for k, (i, t) in rep.items():
        measured[k] = score(sessions[i],
                            action_seqs[i][:t] + action_seqs[i][t + 1:])

    # The delta is pooled, so its base must be the rollout the ablation ran on.
    # Subtracting the representative's ablated outcome from a DIFFERENT
    # rollout's base compares two unrelated runs -- on ALFWorld that error cost
    # 0.26 of correlation with planner progress before it was found.
    delta = {k: bases[rep[k][0]] - measured[k] for k in measured}
    required: set = set()
    if drop_required:
        # A turn whose removal returns EXACTLY zero was structurally required:
        # the script can no longer terminate, so its delta IS the episode return
        # and says nothing about where inside the trajectory the credit belongs.
        # A_E already carries that quantity.
        #
        # On WebShop these are `click[buy now]` and the opening `search`, one of
        # each in almost every trajectory and always at the same two positions.
        # Left in, the step term puts its largest positive value on the last turn
        # of every rollout -- a turn-position prior wearing a measurement's
        # clothes, and position alone tracks causal progress at rho +0.464, which
        # is why this codebase keeps a `position` control at all. Trained on, it
        # taught the policy to buy whatever it saw first: mean episode length
        # fell 7.4 -> 3.0 turns while GRPO held at 7.3, and held-out success went
        # 25.0% -> 4.2% while GRPO climbed to 35.4%.
        #
        # The criterion is measured, not a hand-listed action name: it fires only
        # where the ablation drove a positive return to exactly zero. It bites
        # only where the reward is graded enough for partial deltas to exist,
        # which is what leaves the remaining signal about placement rather than
        # about necessity.
        required = {k for k in delta
                    if bases[rep[k][0]] > 0 and abs(measured[k]) < 1e-9}

    # Mask by the ROLE of the action rather than by the measured result.
    # Deleting click[buy now] or the opening search necessarily leaves the
    # script with no reward, so their delta is identically the return of the
    # whole trajectory -- the quantity A_E already carries, and one that says
    # nothing about which turn inside the trajectory the credit belongs to.
    # Measured, these two kinds of turn are 11% of all turns but 18% of the
    # |delta| mass. `commit_deltas` masks the commit turn and `multihop_deltas`
    # masks the answer turn; the same idea was simply missing here.
    #
    # Note this is not a replacement for `drop_required`: measured, masking the
    # first and last turns moved the correlation between the step term and turn
    # position from +0.309 to +0.341 (up, not down), and the fraction of
    # successful trajectories being pushed down from 54% to 50%. The position
    # prior is an intrinsic property of the deletion operator on WebShop, not
    # something these two anchor actions cause. What this switch removes is
    # double counting, not the position prior.
    structural: set[tuple[int, int]] = set()
    if mask_structural:
        for i, acts in enumerate(action_seqs):
            for t, a in enumerate(acts):
                al = a.strip().lower()
                if al == "click[buy now]" or (t == 0 and al.startswith("search")):
                    structural.add((i, t))

    out, masks = [], []
    for i, acts in enumerate(action_seqs):
        row = []
        for t, a in enumerate(acts):
            k = keyer(i, t, a)
            row.append(None if (k in required or (i, t) in structural)
                       else delta.get(k, 0.0))
        # A dropped turn must come out NEUTRAL, and zero is not neutral: the
        # step term is centred within the trajectory, so a zero among positive
        # deltas lands BELOW the row mean and becomes a penalty. Setting the
        # first version of this to zero taught the policy the opposite lesson --
        # it stopped buying, episodes ran 12.7 turns against 7.4 and held-out
        # score fell 0.544 -> 0.258. The row's own mean is the value that
        # centres to exactly zero, which is what "no information" should mean.
        out.append([0.0 if v is None else v for v in row])
        masks.append([v is not None for v in row])
    if fill_offpath:
        filled = _fill_offpath(out, observations, bases)
        for i in filled:
            masks[i] = [True] * len(masks[i])
    return WebShopCredit(out, len(measured) + n,
                         sum(len(a) for a in action_seqs), masks)


def _fill_offpath(out, observations, bases) -> list[int]:
    """Give a rollout that failed and measured nothing SOMETHING, for free.

    `delta = R(tau) - R(tau minus a_t)` is 0 - 0 on a failure: there is no
    reward to lose. Measured on 24 WebShop rollouts, deletion put a non-zero
    value on 0% of the turns of failing rollouts -- not a few, none -- and a row
    of all zeros survives the group z-score as one constant and is then erased
    entirely by within-trajectory centring. Those rollouts are the majority on
    the hard tasks, so the method was silent exactly where the headroom is.

    The port of the ALFWorld version, which measured rho +0.172 against planner
    progress at no extra replay: a page one of the group's SUCCESSFUL rollouts
    passed through is a page from which the goal demonstrably was reachable, so

        how much of what remains from turn t was spent on pages a winner
        also visited

    grades a failed rollout's turns -- high while it is still on a route that
    worked, falling once it has wandered off. Only rows that measured NOTHING
    are filled; a failure that did produce a counterfactual keeps it, because
    that is a real measurement and this is not.
    """
    winners = [i for i, b in enumerate(bases) if b > 0]
    if not winners:
        return []
    filled = []
    on_path = {page_key(o) for j in winners for o in observations[j]}
    for i, row in enumerate(out):
        if bases[i] > 0 or any(abs(v) > 1e-9 for v in row):
            continue
        L = len(row)
        obs_i = observations[i]
        on = [1.0 if t < len(obs_i) and page_key(obs_i[t]) in on_path else 0.0
              for t in range(L)]
        for t in range(L):
            row[t] = sum(on[t:]) / max(L - t, 1)
        filled.append(i)

    return filled

COMMIT_ACTION = "click[buy now]"


def webshop_commit_deltas(worker, sessions: list[int], action_seqs: list[list[str]],
                          observations: list[list[str]],
                          commit: str = COMMIT_ACTION,
                          running_max: bool = False,
                          lookahead_k: int = 0,
                          mask_commit: bool = True,
                          binary: bool = False) -> WebShopCredit:
    """Credit by CASHING OUT, not by deleting.

    Deleting one action asks "was this turn necessary". On a browsing task the
    honest answer for almost every navigation click is no -- another click
    reaches the same product -- so the measurement is silent exactly where the
    agent spends its turns: 173 navigation steps carried a non-zero deletion
    delta on 13% of them, against 91% for the attribute clicks that decide the
    purchase.

    This asks a different question at the SAME single-turn granularity:

        V(t) = reward of replaying the first t actions and then buying
        credit(t) = V(t+1) - V(t)

    which is what the step actually bought. Opening a better item earns credit,
    backing out of it costs credit -- the first negative signal any operator
    here has produced -- and a click that changes nothing scores zero because it
    changed nothing, not because nothing could be measured. On the same
    trajectories it moved 29% of turns against deletion's 7%, and 7% of the
    turns of FAILED rollouts, where deletion is structurally silent (0 - 0).

    It is the cheapest possible estimate of the state's value: no policy call,
    one replay per prefix, where VinePPO-style forward sampling would need k
    generations per state. The price is that the environment must offer a way to
    commit and collect partial credit -- WebShop and multi-hop QA do, ALFWorld
    does not, which is why this is an operator the environment declares rather
    than a change to the method.

    The commit turn itself is masked by default: its own value is not a
    difference of two states, and under trajectory centring a zero would be
    pushed below the row mean and read as a penalty.

    `mask_commit=False` is for the combination forms that centre WITHIN an
    anchor group. Signal-level case (eval 6342, the 8 rollouts of L_vsg@30): on
    a product page the V* increment of "click an option" was +0.14, while
    "buy now", once masked, landed in a single-member group under its own unique
    key with a step term identically 0 -- so the comparison anchor grouping is
    supposed to make (on one product page: click one more option versus buy now)
    never happened, and buying early was never compared against anything.
    Unmasked, the commit turn takes its measured value: under run-to-optimum
    that is max(best, V(t)) - best, identically 0 (buying only cashes out V(t)),
    and centring it together with the option clicks of the same group makes it
    negative -- which is exactly the sign the pathology in the traces deserves.
    Forms that centre within a trajectory should still mask it.

    `lookahead_k` covers the blind spot V* has on SEARCH turns. On a results
    page `buy now` does nothing, so V is identically 0 and the V* increment of a
    search turn is 0 however good or bad the query was: the largest group in the
    anchor grouping -- the search page every rollout shares -- is mute for us,
    whereas a step-level baseline that estimates credit from observed returns
    compares different queries there through discounted returns, and measured it
    learned to rewrite the query (pasting the instruction verbatim 23% of the
    time against our 37%), which is where the gap in full marks on the easy
    tasks comes from. The lookahead defines the value of a search turn as "the
    best value reachable by a single buy among the first k items of the results
    page":

        LA(t+1) = max( V(t+1), max_{asin in top-k of results page} V(prefix + click[asin]) )
        credit(search) = max(0, LA(t+1) - V*(t))

    This is still a replay measurement, not an estimate; it changes the credit
    of the search turn only, and `best` does not swallow the lookahead value, so
    the later "which item to open" is still scored by the value actually opened.
    Hand-written rewrites are separable (eval 5988: pasting verbatim gives the
    first 5 items [0.64, 0.64, 0.73, 0.0, 0.45], after a rewrite
    [0.64, 0.64, 0.73, 0.64, 0.45], and a bad rewrite gives all zeros).

    The queries the POLICY itself proposes are not separable: at temperature 1.0
    the 8 rollouts of a task contain 2.7 distinct queries for the base model and
    3.0 for the step-level baseline at step 100, yet their first 5 items are
    identical (the within-group spread of the lookahead credit is non-zero on
    0 of 6 tasks, and 0 of 6 for that baseline too). WebShop retrieval is blunt
    to rewrites of this kind, so the switch is inert on WebShop: kept, off by
    default, and not a fix.
    """
    n = len(action_seqs)
    if n == 0:
        return WebShopCredit([], 0, 0, [])

    cache: dict[tuple[int, tuple[str, ...]], float] = {}
    n_replays = 0

    def value(session: int, prefix: tuple[str, ...]) -> float:
        key = (session, prefix)
        if key not in cache:
            nonlocal n_replays
            # The measurement has to be in the same currency as the training
            # reward. Under `--binary_reward` A_E is the z-score of a binary
            # {0,1} outcome, while this always took the graded score, so we were
            # weighting by "how much more partial credit is available" while
            # optimising "can it succeed exactly". webshop_deltas had this
            # parameter all along; commit_deltas and regret were missing it.
            r = worker.replay(session, list(prefix) + [commit])
            cache[key] = (1.0 if r >= 1.0 else 0.0) if binary else r
            n_replays += 1
        return cache[key]

    def lookahead(session: int, prefix: tuple[str, ...], page: str) -> float:
        best_here = value(session, prefix)
        for asin in ASIN_RE.findall((page or "").lower())[:lookahead_k]:
            best_here = max(best_here, value(session, prefix + (f"click[{asin}]",)))
        return best_here

    # `running_max` replaces the myopia of V. V(t) is "what buying right now
    # would score", so standing on a decent product page and paging on to look
    # for something better makes V(t+1)-V(t) zero or negative -- the operator
    # penalises "give up the acceptable thing in front of you and go find a
    # better one", which is exactly the exploration an exact match requires.
    # Trained on, trajectories collapsed from 8.5 turns to 3.8, held-out success
    # went 24.2% -> 7.9%, and the score was propped up by partial credit: the
    # policy had learned to buy the first thing that looked close enough.
    #
    #     V*(t) = max_{s <= t} V(s)          best thing buyable so far
    #     credit(t) = V*(t+1) - V*(t) >= 0   did this step improve the best option
    #
    # Leaving a good product costs nothing any more (the best option is still
    # there); only finding something better earns credit. The price is that the
    # operator is completely silent on navigation: measured over 231 turns the
    # navigation column averages exactly +0.000, all of the signal lands on
    # "open a product" (+0.449) and "pick an attribute" (+0.122), and the
    # non-zero rate falls from V's 50% to 32%. That is the right sparsity --
    # the real decisions are which item to open and which options to tick -- but
    # it means this operator only fits a sign-preserving multiplicative
    # combination: under an additive form, centring within the trajectory turns
    # those zeros into negative penalties, which trades the myopia bias for a
    # redundancy penalty and lands back where we started.
    out, masks = [], []
    for i, acts in enumerate(action_seqs):
        sess = sessions[i]
        row, keep = [], []
        best = None
        for t, a in enumerate(acts):
            if a.strip().lower() == commit:
                if mask_commit or not running_max:
                    row.append(0.0); keep.append(False)
                    continue
                # Buying cashes out V(t): v1 := v0, so under run-to-optimum
                # the increment is identically 0, but the turn DOES take part
                # in the anchor group.
                v0 = value(sess, tuple(acts[:t]))
                if best is None:
                    best = v0
                row.append(max(best, v0) - best); keep.append(True)
                best = max(best, v0)
                continue
            v0 = value(sess, tuple(acts[:t]))
            v1 = value(sess, tuple(acts[:t + 1]))
            if running_max:
                if best is None:
                    best = v0
                nxt = max(best, v1)
                credit = nxt - best
                is_search = a.strip().lower().startswith("search[")
                page = observations[i][t + 1] if i < len(observations) and t + 1 < len(observations[i]) else None
                if lookahead_k > 0 and is_search and page:
                    credit = max(credit, lookahead(sess, tuple(acts[:t + 1]), page) - best)
                row.append(credit); keep.append(True)
                best = nxt
            else:
                row.append(v1 - v0); keep.append(True)
        out.append(row); masks.append(keep)
    return WebShopCredit(out, n_replays, sum(len(a) for a in action_seqs), masks)


def webshop_regret(worker, sessions: list[int], action_seqs: list[list[str]],
                   observations: list[list[str]], k: int = 5,
                   commit: str = COMMIT_ACTION, max_options: int = 16,
                   completion_beam: int = 4,
                   item_value: str = "completion",
                   mode: str = "regret",
                   binary: bool = False) -> WebShopCredit:
    """Measured regret against the best available alternative.

    Non-positive, per turn, and it needs no variation inside the group.

    The L_vsg@60 case: of the 9 mid-band tasks, 6 bought the same "universal
    shirt" `b09qqp3356` (which reliably scores 0.57 with one size option). If
    all 8 rollouts open it there is no variation inside the anchor group, the
    step term centres to 0 and A_E is about 0 as well -- a within-group
    comparison needs someone in the group to have taken the right path, which is
    an exploration problem credit assignment cannot reach. But we have a
    simulator, so the paths NOT taken can be measured:

        open product a from a results page:  regret = V(prefix + a) - max_{j <= k} V(prefix + click[item_j])
        buy now on a product page:           regret = V(prefix)     - V(prefix + greedy option completion)
        any other turn:                      0

    It is a counterfactual over unsampled actions, so all 8 rollouts opening the
    wrong item each still receive negative credit, which is why it must **not**
    be centred within the group afterwards (a constant would simply be
    subtracted away); it is added as an uncentred term:
    A = A_E + omega * (within-group step term) + lambda * regret. A step-level
    baseline that estimates credit from observed returns only has the
    trajectories it sampled and cannot produce this.

    `mode`: what the turn that opens a product from a results page receives.
      "regret"  = min(0, Q(a) - max_j Q(j)): penalise only "opened something
                  worse than the best item on the page"; non-positive, uncentred.
      "centred" = Q(a) - mean_j Q(j) (j over the first k items on the page plus
                  the one opened): **use the measurement to treat the candidates
                  on the page as a virtual anchor group**. Opening a product
                  above the page mean scores positive and below it negative, and
                  the zero mean does not require the 8 rollouts to open
                  different products -- which is the heart of the
                  item-preference pathology: when all 8 open the same item the
                  within-group contrast of a step-level baseline is empty and
                  regret only hands out a negative sign, while centred both
                  penalises that item and rewards the relative standing of a
                  better one on the page. The buy-now turn uses completion
                  regret in both modes.
    `item_value`: what counts as a product's value when products are compared.
      "buy"        = what buying immediately after opening it would score.
                     **Biased**: the right product often needs options to reach
                     full score, so buying at once may give only 0.57, while a
                     wrong product that needs no options gives 0.75 at once --
                     which would wrongly penalise opening the right product.
      "completion" = what a greedy option completion after opening it would
                     score (default). That is the value of the action "open this
                     one". Price: at most max_options + completion_beam^2
                     replays per product, cached by (session, prefix), so a
                     state that all 8 trajectories pass through is measured once.

    Cost: at most k replays per results page and at most
    max_options + completion_beam^2 per product page, cached by
    (session, prefix); a state that all 8 trajectories pass through is measured
    once.
    """
    n = len(action_seqs)
    if n == 0:
        return WebShopCredit([], 0, 0, [])
    cache: dict[tuple[int, tuple[str, ...]], float] = {}
    n_replays = 0

    def value(session: int, prefix: tuple[str, ...]) -> float:
        key = (session, prefix)
        if key not in cache:
            nonlocal n_replays
            # The measurement has to be in the same currency as the training
            # reward. Under `--binary_reward` A_E is the z-score of a binary
            # {0,1} outcome, while this always took the graded score, so we were
            # weighting by "how much more partial credit is available" while
            # optimising "can it succeed exactly". webshop_deltas had this
            # parameter all along; commit_deltas and regret were missing it.
            r = worker.replay(session, list(prefix) + [commit])
            cache[key] = (1.0 if r >= 1.0 else 0.0) if binary else r
            n_replays += 1
        return cache[key]

    NAV = {"buy now", "back to search", "< prev", "next >", "description",
           "features", "reviews", "attributes"}
    OPTION_RE = re.compile(r"\[([^\[\]]{1,40})\]")

    def item_val(session: int, prefix: tuple[str, ...]) -> float:
        """Value of the state reached after opening a given product."""
        if item_value == "buy":
            return value(session, prefix)
        st = page_after(session, prefix)
        page = st.get("obs", "")
        return completion(session, prefix, page, st.get("clickables")) if page else value(session, prefix)

    page_cache: dict[tuple[int, tuple[str, ...]], dict] = {}

    def page_after(session: int, prefix: tuple[str, ...]) -> dict:
        key = (session, prefix)
        if key not in page_cache:
            nonlocal n_replays
            try:
                st = worker.observe(session, list(prefix))
            except AttributeError:
                st = {}
            page_cache[key] = st if isinstance(st, dict) else {"obs": str(st), "clickables": []}
            n_replays += 1
        return page_cache[key]

    def completion(session: int, prefix: tuple[str, ...], page: str,
                   clickables=None) -> float:
        # Clickable options on a product page: prefer the clickables the
        # environment reports, otherwise parse short bracketed strings out of
        # the observation.
        raw = list(clickables) if clickables else OPTION_RE.findall(page or "")
        opts = []
        for o in raw:
            ol = str(o).strip().lower()
            if ol in NAV or ASIN_RE.fullmatch(ol) or ol in opts:
                continue
            opts.append(ol)
        opts = opts[:max_options]
        best = value(session, prefix)
        single = {o: value(session, prefix + (f"click[{o}]",)) for o in opts}
        pool = sorted(opts, key=lambda o: -single[o])[:completion_beam]
        chosen: tuple[str, ...] = ()
        while pool:
            cand = {o: value(session, prefix + tuple(f"click[{c}]" for c in chosen) + (f"click[{o}]",))
                    for o in pool}
            o, v = max(cand.items(), key=lambda kv: kv[1])
            if v <= best + 1e-9:
                break
            chosen += (o,); best = v; pool.remove(o)
        return best

    out, masks, kinds = [], [], []
    for i, acts in enumerate(action_seqs):
        sess = sessions[i]
        obs = observations[i] if i < len(observations) else []
        row, keep, krow = [], [], []
        for t, a in enumerate(acts):
            al = a.strip().lower()
            page = obs[t] if t < len(obs) else ""
            reg = 0.0
            kind = None
            m = re.fullmatch(r"click\[(b0[0-9a-z]{8})\]", al)
            if m and page:
                asins = ASIN_RE.findall(page.lower())[:k]
                if asins:
                    v_open = item_val(sess, tuple(acts[:t + 1]))
                    vals = [item_val(sess, tuple(acts[:t]) + (f"click[{s_}]",)) for s_ in asins]
                    if mode == "centred":
                        pool = vals + ([v_open] if m.group(1) not in asins else [])
                        reg = v_open - sum(pool) / len(pool)
                    else:
                        reg = min(0.0, v_open - max(vals))
                    kind = "item"
            elif al == commit and page and "buy now" in page.lower():
                v_now = value(sess, tuple(acts[:t]))
                st = page_after(sess, tuple(acts[:t]))
                reg = min(0.0, v_now - completion(sess, tuple(acts[:t]), page, st.get("clickables")))
                kind = "buy"
            row.append(reg); keep.append(True); krow.append(kind)
        out.append(row); masks.append(keep); kinds.append(krow)
    return WebShopCredit(out, n_replays, sum(len(a) for a in action_seqs), masks, kinds)


def center_item_regret(deltas: list[list[float]], kinds: list[list[str | None]],
                       anchor_keys: list[list[str]]) -> list[list[float]]:
    """Make the "open a product" regret relative across the 8 rollouts of one task.

    The "buy now" regret stays absolute.

    Why: absolute regret is a cross-task quantity. In 42 of the 70 shirt tasks
    of the training split the best item is the same asin, and absolute regret
    says "every other item is worse than this one" in every one of them, so the
    policy memorises the product id: over greedy decoding on 120 evaluation
    tasks the base model opened it 28 times, ours 35 times, and a step-level
    baseline that estimates credit from observed returns 11 times. That
    baseline's step term only ever compares rollouts of the same task in the
    same state, which is why it did not memorise the id. Here the
    product-opening regret is placed in the anchor group of the same state and
    zero-meaned: all 8 open the same item -> all zeros (that task no longer
    teaches any id preference); they open different items -> the good ones
    positive, the bad ones negative. The buy-now regret knows nothing about ids
    (it only says "you could have completed the options"), so it keeps its
    absolute form.
    """
    from ssc.credit.alf_arms import anchor_step_advantage
    keys, vals, buy = [], [], []
    for i, (drow, krow, arow) in enumerate(zip(deltas, kinds, anchor_keys)):
        kr, vr, br = [], [], []
        for t, (d, k) in enumerate(zip(drow, krow)):
            a = arow[t] if t < len(arow) else f"__na_{i}_{t}"
            if k == "item":
                kr.append(a); vr.append(float(d)); br.append(0.0)
            else:
                kr.append(f"__nonitem_{i}_{t}"); vr.append(0.0); br.append(float(d) if k == "buy" else 0.0)
        keys.append(kr); vals.append(vr); buy.append(br)
    centred = anchor_step_advantage(keys, vals, divide_by_std=False)
    return [[c + b for c, b in zip(cr, br)] for cr, br in zip(centred, buy)]
