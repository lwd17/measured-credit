"""Per-turn advantages for measured-credit training.

Every arm shares one combination rule and differs only in what fills the
step-level slot:

    A(i,t) = A_E(i) + omega * step(i,t)          additive arms
    A(i,t) = A_E(i) * q(i,t)                     mass-preserving arm

A_E is the group-standardised episode advantage, identical to GRPO.

    grpo         step = 0, i.e. plain GRPO
    anchor_cf    step = the measured counterfactual, z-scored over the
                 measured turns of the group and then centred within each
                 trajectory
    anchor_mass  q built from the measured counterfactual so that the mean
                 weight over measured turns is 1, applied only where A_E > 0

Two switches guard the step term (see `turn_advantages`):
`nonneg_winners` keeps a positive-advantage trajectory from ever receiving a
negative step term, and `zero_uniform_groups` silences groups whose outcomes
are all equal.
"""


from __future__ import annotations

import math
import random

EPS = 1e-6


def rloo_advantage(rewards: list[float]) -> list[float]:
    """REINFORCE Leave-One-Out: each rollout is scored against the mean of the
    OTHERS, not against a mean that includes itself.

    GRPO's z-score puts the sample in its own baseline, which biases the
    advantage toward zero by a factor of (n-1)/n and, more importantly, makes
    the baseline depend on the very rollout being scored. RLOO is the standard
    correction and is a baseline any group-relative method is expected to be
    compared against. It costs nothing extra -- the same rollouts, a different
    denominator -- so leaving it out of the table would be a gap with no excuse.
    """
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def group_advantage(rewards: list[float]) -> list[float]:
    """Standard GRPO episode advantage: z-score the return within the group."""
    n = len(rewards)
    if n == 0:
        return []
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    sd = math.sqrt(var)
    return [(r - mu) / (sd + EPS) for r in rewards]


def signed_scale(values: list[float], centred: bool) -> list[float]:
    """Put a step signal on a unit scale, centring it only when that is sound.

    Z-scoring assumes the group contains both better and worse trajectories: it
    forces zero mean, so roughly half the turns come out NEGATIVE whatever the
    values were. That assumption breaks on a group whose rollouts all succeeded,
    and it breaks in the direction that costs the most.

    In an all-success group the episode advantage is exactly zero -- no outcome
    variance to normalise -- so GRPO makes no update at all and the tasks the
    policy already solves are left alone. A step-level baseline that estimates
    credit from observed returns is also silent there. A measured
    counterfactual is not zero (68.4% of its deltas are non-zero, and it tracks
    causal progress at rho = +0.982), so this arm is the only one still moving.
    Centred, it pushes DOWN on every turn below the group mean -- on
    trajectories that all reached the goal, where no turn deserves a negative
    advantage. Measured on the 158 evaluation tasks the base model already
    solves, that cost 3.6 points, while the same arm gained 6-7 points on the
    tasks it did not.

    So centring is applied only where its premise holds. Where every rollout
    ended alike, the signal is scaled into [0, 1] instead: necessary turns are
    reinforced relative to redundant ones, and nothing is suppressed.
    """
    n = len(values)
    if n == 0:
        return []
    if centred:
        return zscore(values)
    hi = max(values)
    lo = min(values)
    if hi - lo < EPS:
        return [0.0] * n
    return [(v - lo) / (hi - lo) for v in values]


def zscore(values: list[float]) -> list[float]:
    """Z-score a step-level signal across everything it is defined over.

    Every step signal is put on one scale before it enters the sum, so `omega`
    means the same thing whichever arm supplies it. Without this a comparison
    between arms would partly be a comparison of magnitudes: a raw measured
    signal lives in [0, 0.05] while a z-scored one is of order 1, and at a
    shared omega the raw arm would simply carry a much smaller step-level term
    rather than a different one.
    """
    n = len(values)
    if n == 0:
        return []
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd < EPS:
        return [0.0] * n
    return [(v - mu) / sd for v in values]



def masked_zscore(values: list[float], keep: list[bool],
                  divide_by_std: bool = True) -> list[float]:
    """Z-score over the entries that carry a measurement; the rest stay 0.

    Plain `zscore` subtracts the mean of EVERYTHING, so a turn with no
    counterfactual to run -- scored 0 because nothing was measured, not because
    a measurement came back zero -- is pushed to `-mean/sd` and reads as a
    penalty. That is how the multi-hop policy learned to stop answering: the
    answer turn is the one turn whose evidence cannot be dropped, and it came
    out 1.324 below the searches around it, against +0.176 for a permuted
    control built from the same numbers. The WebShop commit action failed the
    same way.
    """
    seen = [v for v, k in zip(values, keep) if k]
    if not seen:
        return [0.0] * len(values)
    mu = sum(seen) / len(seen)
    var = sum((v - mu) ** 2 for v in seen) / len(seen)
    sd = var ** 0.5
    if divide_by_std and sd <= EPS:
        return [0.0] * len(values)
    # The step-level literature reports this normalisation factor as
    # task-dependent rather than universally helpful: on harder tasks, dividing
    # by the group standard deviation "exaggerates gradients from overly
    # difficult samples or highly imbalanced groups", and fixing the factor at 1
    # scores higher. Our groups on WebShop are more imbalanced -- 7 distinct
    # non-zero measurements across 279 turns -- so the divisor is set by a
    # handful of values and blows the survivors up. Centring alone is the same
    # signal without that amplification; the batch-level rescale that follows
    # still puts omega on a stable scale.
    d = sd if divide_by_std else 1.0
    return [(v - mu) / d if k else 0.0 for v, k in zip(values, keep)]


def centre_within_trajectory(step: list[list[float]],
                             mask: list[list[bool]] | None = None
                             ) -> list[list[float]]:
    """Strip the trajectory-level component out of a step signal.

    A per-turn term has two parts. How its per-trajectory MEANS differ is
    measured at rho = +0.838 for anchor_cf against the episode advantage on
    mixed-outcome groups, and at +0.940 and +0.926 for two return-based step
    signals over the same groups -- it is very nearly a second copy of the
    outcome, and adding it to A_E amplifies the outcome signal rather than
    saying anything about turns. How values vary WITHIN a trajectory is the only part that
    answers "which turn".

    That split may be the whole story behind the step-level literature: two step
    signals of wildly different fidelity -- one tracking planner progress at
    rho = +0.146, the other at rho = 1.0 -- trained to +4.2 and +4.1,
    indistinguishable, which is what one expects if the shared work is
    amplifying A_E rather than placing credit.

    So the two jobs are separated here: A_E decides how much a trajectory gets,
    and the step term only decides how that is distributed inside it. Each row
    is centred to zero mean, then the whole is rescaled to unit variance so that
    `omega` keeps the meaning it has in the uncentred arms.

    This may cost the gain rather than sharpen it. If the arms that use it fall
    back to grpo, the amplification WAS the mechanism, and that is worth knowing.
    """
    centred = []
    for i, row in enumerate(step):
        if not row:
            centred.append([])
            continue
        keep = mask[i] if mask is not None else None
        if keep is not None:
            # Centre over the turns that CARRY a measurement, and leave the rest
            # at exactly zero. Dividing by the row length instead makes every
            # unmeasured turn negative: a WebShop trajectory of 13 turns with 2
            # non-zero deltas centres to [+0.169, -0.031 x 11, +0.169], so each
            # browsing click is pushed down every step. But "deleting this turn
            # changed nothing" is evidence that the turn was REDUNDANT, not that
            # it was harmful -- and in a search task the browsing whose single
            # deletion changes nothing is exactly the information gathering the
            # task requires. Trained on, that penalty shortened episodes from
            # 10.9 turns to 5.1 while GRPO held at 7.3, and held-out success
            # fell instead of rising.
            #
            # ALFWorld hid this because there the same pressure is good advice:
            # its successful routes run 15-16 turns against a 30-turn budget and
            # going straight to the goal is correct. The rule states what the
            # measurement actually licenses, so it applies to both.
            seen = [v for v, k in zip(row, keep) if k]
            mu = sum(seen) / len(seen) if seen else 0.0
            centred.append([v - mu if k else 0.0 for v, k in zip(row, keep)])
        else:
            mu = sum(row) / len(row)
            centred.append([v - mu for v in row])
    if mask is None:
        flat = zscore([v for row in centred for v in row])
        out, i = [], 0
        for row in centred:
            out.append(flat[i:i + len(row)])
            i += len(row)
        return out
    # Rescale over the measured entries only, and leave the rest at exactly
    # zero. Passing the zeros through `zscore` subtracts the global mean from
    # them, which is how an unmeasurable turn became a PENALTY twice: the
    # WebShop policy stopped buying and the multi-hop policy stopped answering,
    # both after the terminal action -- the one turn with no counterfactual to
    # run -- was scored 1.3 below the searches around it.
    seen = [v for row, keep in zip(centred, mask) for v, k in zip(row, keep) if k]
    sd = (sum(v * v for v in seen) / len(seen)) ** 0.5 if seen else 0.0
    scale = 1.0 / sd if sd > EPS else 0.0
    return [[v * scale if k else 0.0 for v, k in zip(row, keep)]
            for row, keep in zip(centred, mask)]


def similarity_clusters(keys: list[str], threshold: float = 0.9) -> dict[str, int]:
    """Approximate same-state grouping for FREE-TEXT states, e.g. multi-hop QA.

    Same-state grouping normally needs no similarity at all: where two
    trajectories can reach a byte-identical state, hashing the state text is
    enough, and it is what ALFWorld and WebShop use. Measured here, exact keys
    put 96.2% of ALFWorld turns in a group of two or more, and collapse ~79
    WebShop turns per group into ~30 distinct keys (mean group size 2.6).

    Similarity clustering exists for the case where states are free text and
    never repeat byte-for-byte, which is why the default threshold is the 0.9
    longest-matching-subsequence ratio used in search-augmented QA. Multi-hop QA
    is the one environment in this codebase that should pass
    `sim_threshold=0.9`.

    Coarser grouping is not free. `reports/alf_audit_robustness.json` sweeps key
    definitions and the correlation with planner progress drifts with coverage:
    -0.050 at 0.973, -0.072 at 0.962, and +0.356 for a deliberately degenerate
    turn-index key at 0.984. Similarity clustering sits at 0.984 -- the coarse
    end -- so on a repeating-state environment it moves the grouped signal
    toward the turn-position prior rather than away from it.

    Exact duplicates are collapsed first, so the quadratic part runs over the
    DISTINCT strings only. `quick_ratio` is an upper bound on `ratio`, so it
    rejects most pairs without the expensive comparison. Clustering is greedy
    against the first member of each cluster and iterates in first-seen order,
    so the assignment is deterministic.
    """
    import difflib

    order, seen = [], set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)

    reps: list[str] = []
    cluster: dict[str, int] = {}
    for k in order:
        m = difflib.SequenceMatcher(None, k, "")
        m.set_seq1(k)
        for i, r in enumerate(reps):
            m.set_seq2(r)
            if m.real_quick_ratio() < threshold or m.quick_ratio() < threshold:
                continue
            if m.ratio() >= threshold:
                cluster[k] = i
                break
        else:
            cluster[k] = len(reps)
            reps.append(k)
    return cluster


def anchor_step_advantage(anchor_keys: list[list[str]],
                          returns: list[list[float]],
                          divide_by_std: bool = True,
                          sim_threshold: float | None = None) -> list[list[float]]:
    """A_S: z-score the discounted return within each same-state group.

    Same-state grouping utility, used by the WebShop item-credit path to pool
    turns that share a state key.

    A state visited by only one rollout in the group has nothing to compare
    against and gets 0. So does a group whose rollouts all ended alike -- zero
    variance, zero signal -- and that case is common enough to matter: at group
    size 2 it left A_S identically zero on every turn of a probe run while 94%
    of turns sat in a group of 2 or more.
    """
    if sim_threshold is not None:
        flat = [k for keys in anchor_keys for k in keys]
        cl = similarity_clusters(flat, sim_threshold)
        anchor_keys = [[f"c{cl[k]}" for k in keys] for keys in anchor_keys]

    groups: dict[str, list[tuple[int, int]]] = {}
    for i, keys in enumerate(anchor_keys):
        for t, k in enumerate(keys):
            groups.setdefault(k, []).append((i, t))

    out = [[0.0] * len(k) for k in anchor_keys]
    for _, members in groups.items():
        if len(members) < 2:
            continue
        vals = [returns[i][t] for i, t in members]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(var)
        for (i, t), v in zip(members, vals):
            out[i][t] = (v - mu) / (sd + EPS) if divide_by_std else (v - mu)
    return out


def anneal_dose(omega: float, alpha: float, step: int,
                anneal_steps: int | None) -> tuple[float, float]:
    """Use the step term early and withdraw it late: omega decays linearly to
    0 and alpha rises linearly to 1, after which the arm is plain GRPO.

    The motivation is a shape every step-level arm shares on WebShop: they peak
    at 0.62-0.64 around @40-50 and then flatten or decline, while GRPO, which
    has no step term, rises monotonically from @10 to @100 and reaches 0.644.
    The step term is worth +0.07 early (at @40, same number of steps) and a drag
    after that -- more and more groups succeed outright, A_E goes to zero, and a
    unit-variance step term is then the only thing left, pushing a position
    prior and hedging behaviour. Annealing takes the good half of each regime.

    omega=0 and alpha=1 both reduce exactly to GRPO (asserted in
    tests/test_alf_arms.py), so the end point of the anneal is the baseline
    itself and nothing new is introduced. With anneal_steps=None the inputs are
    returned unchanged, leaving existing arms bit-identical.
    """
    if not anneal_steps or anneal_steps <= 0:
        return omega, alpha
    frac = min(1.0, max(0.0, step / anneal_steps))
    return omega * (1.0 - frac), alpha + (1.0 - alpha) * frac


def _mask_for(a_star, mask):
    """Default: every turn carries a measurement.

    ALFWorld deletes any action and re-runs, so no turn is unmeasurable there
    and the default keeps its behaviour unchanged. WebShop and multi-hop both
    have a terminal action whose ablation is not defined, and they pass a mask.
    """
    if mask is not None:
        return mask
    return [[True] * len(row) for row in a_star]


def turn_advantages(arm: str, rewards: list[float], lengths: list[int],  # noqa: C901
                    anchor_keys: list[list[str]] | None = None,
                    a_star: list[list[float]] | None = None,
                    gamma: float = 0.95, omega: float = 1.0
                    ,
                    a_star_mask: list[list[bool]] | None = None,
                    step_std: bool = True,
                    alpha: float = 0.5, w_max: float = 3.0,
                    dose: float = 1.0, fallback: bool = False,
                    fallback_omega: float | None = None,
                    neutral_unmeasured: bool = False,
                    nonneg_winners: bool = False,
                    zero_uniform_groups: bool = False) -> list[list[float]]:
    """Per-turn advantage for every rollout in one group.

    Returns a list aligned with `rewards`, each entry a list of length
    `lengths[i]`. The GRPO arm returns the episode advantage repeated, which is
    exactly what GRPO does and keeps the three arms on one code path.
    """
    episode = group_advantage(rewards)
    if arm == "grpo":
        return [[a] * T for a, T in zip(episode, lengths)]

    if arm == "anchor_cf":
        if a_star is None:
            raise ValueError("the anchor_cf arm needs measured deltas")
        # `anchor_cf` z-scores, so its step term is zero-mean like every other
        # step signal here. Keeping the scaling identical across arms is what
        # makes them differ in ONE thing -- where the per-turn signal comes
        # from -- which is the question the paper asks. An earlier version
        # differed in two, and no result could be attributed.
        #
        # What it used to do, kept as `anchor_cf_uncentred`: skip centring when
        # the group's outcomes did not vary, mapping the deltas to [0, 1]. The
        # rationale (see `signed_scale`) was that centring invents negative
        # advantages for turns on trajectories that all reached the goal. The
        # cost was not noticed: [0, 1] is not zero-mean, so on every homogeneous
        # group the arm added a positive constant to every sampled turn --
        # +0.36 on a worked example -- and a constant in the advantage is a
        # uniform push on whatever was sampled, including the turns of
        # trajectories that all FAILED. Measured over 27 steps against grpo it
        # ran -1.9 points and fell -0.43 points per step (t = -2.01), while an
        # otherwise identical centred arm rose. Every arm here is zero-mean.
        vals = [v for row in a_star for v in row]
        keep = [k for row in _mask_for(a_star, a_star_mask) for k in row]
        flat = masked_zscore(vals, keep, divide_by_std=step_std)
        step, i = [], 0
        for row in a_star:
            step.append(flat[i:i + len(row)])
            i += len(row)
        step = centre_within_trajectory(step, _mask_for(a_star, a_star_mask))
    elif arm == "anchor_mass":
        # The original form from sections 7/53/54 of the runbook. It is
        # structurally different from the other arms:
        #
        #     A(i,t) = A_E(i) * q(i,t),   q >= 0,   mean(q) = 1 per trajectory
        #
        # The additive form `A_E + omega * A_step` centres the step term to zero
        # mean, so some turns of every trajectory necessarily receive a negative
        # step term: on WebShop, 26-54% of the turns of a successful trajectory
        # ended with a negative advantage, where plain GRPO would flip none of
        # them. Placed correctly that is where the gain comes from (0.76 against
        # planner ground truth on ALFWorld); placed wrongly it dismantles
        # successful behaviour the policy has already learned.
        #
        # The multiplicative form cannot do that: q is non-negative, so every
        # turn of a successful trajectory only gets a little more or a little
        # less, with the sign following A_E. mean(q)=1 also keeps the total
        # advantage mass of each trajectory exactly equal to GRPO's, so the step
        # term cannot inflate the gradient norm (under the additive form the
        # clip trigger rate rose from GRPO's 15% to 75%, correlating with the
        # score at rho = -0.921).
        #
        # alpha is the dose knob: at alpha=1 every q is 1 and this reduces
        # exactly to GRPO.
        # apply_to="positive": trajectories with A_E <= 0 stay uniform.
        # Reweighting is done only on positive advantage, otherwise it would
        # also wipe out the penalty on the action a failed trajectory retried
        # over and over -- which is precisely the action the policy should learn
        # not to repeat (the same-named parameter of `ssc_token_advantage` has
        # the same reason).
        if a_star is None:
            raise ValueError("the anchor_mass arm needs measured deltas")
        from ssc.credit.counterfactual import contribution_weights

        keep_rows = _mask_for(a_star, a_star_mask)
        out = []
        for row_a, keep, e, T in zip(a_star, keep_rows, episode, lengths):
            if e <= 0:
                out.append([e] * T)
                continue
            seen = [max(v, 0.0) for v, k in zip(row_a, keep) if k]
            if neutral_unmeasured:
                # 09-12 WebShop probe: the operator is structurally silent on
                # search and navigation steps (buying straight off the results
                # page is identically 0), so a delta of 0 on those steps means
                # "not measurable", not "contributed nothing". Normalising them
                # together with the measurable steps (mean(q)=1) pushes those
                # zeros below 1: inside successful trajectories, search fell
                # from GRPO's +0.66 to +0.53 and navigation from +0.74 to +0.62,
                # while a return-based step baseline gives +1.27/+1.78 on those
                # same two kinds of step. So an unmeasurable step keeps weight 1
                # -- exactly GRPO's share -- and mass is conserved only among
                # the measurable ones. ALFWorld's masked_zscore is the additive
                # version of the same principle.
                keep_T = list(keep) + [False] * (T - len(keep))
                q_meas = contribution_weights(seen, alpha=alpha, w_max=w_max) if seen else []
                it = iter(q_meas)
                q = [next(it) if k else 1.0 for k in keep_T[:T]]
                out.append([e * w for w in q])
                continue
            # An unmeasured turn takes the mean of the measured ones: it
            # corresponds to "no information", and under the multiplicative form
            # the neutral value for "no information" is a weight of 1, i.e. the
            # average bucket.
            fill = (sum(seen) / len(seen)) if seen else 0.0
            d = [(max(v, 0.0) if k else fill) for v, k in zip(row_a, keep)]
            d = list(d) + [fill] * (T - len(d))
            q = contribution_weights(d[:T], alpha=alpha, w_max=w_max)
            out.append([e * w for w in q])
        return out
    else:
        raise ValueError(f"unknown arm {arm!r}")

    # 09-13 ALFWorld 4B gating (the three gated_* cases in
    # tests/test_alf_arms.py). The probe (results/_probes/alf4b_dose_probe)
    # measured two things on real 4B/8B batches:
    #   1. Exploration steps inside successful trajectories whose deletion does
    #      not change the outcome (mostly `go to`) are pushed negative by the
    #      centred step term: on 4B, 54% of `go to` turns received a negative
    #      advantage, mean -0.60, while a return-based step baseline gives +0.60
    #      on the same turns (GRPO +0.38). In the failure cases the 4B policy
    #      had stopped walking around to look for objects and looped on
    #      look/examine in place. Exploration that is "redundant" in hindsight
    #      was necessary information gathering at decision time, and deletion
    #      carries exactly that hindsight bias.
    #   2. On groups whose outcomes are all equal (A_E=0, zero GRPO gradient)
    #      the step term still emits a position-shaped signal of |A| ~ 0.9
    #      (negative early, positive late), twice as strong on 4B as on 8B and
    #      unrelated to the outcome.
    # nonneg_winners: clamp the step term to >=0 on trajectories with positive
    #   advantage (under a binary reward, the successful ones). Necessary steps
    #   are still raised; redundant ones simply stop earning extra, fall back to
    #   A_E, and are no longer penalised.
    # zero_uniform_groups: zero the step term on groups whose outcomes are all
    #   equal, reducing exactly to GRPO.
    # With both switches off the result is bit-identical to before.
    if zero_uniform_groups and len({round(r, 9) for r in rewards}) == 1:
        step = [[0.0] * len(row) for row in step]
    if nonneg_winners:
        step = [[max(v, 0.0) for v in row] if a > 0 else list(row)
                for row, a in zip(step, episode)]
    out = []
    for a, s_row, T in zip(episode, step, lengths):
        row = [a + omega * (s_row[t] if t < len(s_row) else 0.0) for t in range(T)]
        out.append(row)
    return out
