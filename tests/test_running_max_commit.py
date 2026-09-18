"""The running-max operator: what V* changes relative to V is that leaving a
good item is no longer penalised.

V's myopia is measured, not conjectured: training `anchor_cv`, trajectories
collapsed from 8.5 turns to 3.8 and held-out success fell 24.2% -> 7.9%, with
the score held up by partial credit -- the policy learned to buy the first
passable item it saw. The cause is that V(t) is "what ordering right now would
be worth", and paging on to look for something better makes it drop.

A fake worker pins the replay to a known order-value curve, so the tests can
assert that the two operators give different credit on that same curve and that
V*'s credit is never negative.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssc.credit.webshop_cf import COMMIT_ACTION, webshop_commit_deltas

# A complete "find a decent one -> leave it -> find a better one" trajectory.
# CURVE[i] = the reward from taking the first i actions and then ordering directly.
#   action 0  search[x]          0.0 -> 0.0   not standing on any item yet
#   action 1  click[b...1]       0.0 -> 0.6   found a decent one
#   action 2  click[next >]      0.6 -> 0.2   paged away from it <- V penalises, V* neutral
#   action 3  click[b...2]       0.2 -> 0.9   found a better one <- V* credits only the +0.3
#   action 4  click[buy now]                  the commit turn, masked out
CURVE = [0.0, 0.0, 0.6, 0.2, 0.9]
ACTS = [["search[x]", "click[b000000001]", "click[next >]",
         "click[b000000002]", COMMIT_ACTION]]


class FakeWorker:
    def replay(self, session, actions):
        assert actions[-1] == COMMIT_ACTION
        return CURVE[len(actions) - 1]


def _deltas(running_max):
    return webshop_commit_deltas(FakeWorker(), [0], ACTS, [[]],
                                 running_max=running_max).deltas[0]


def test_V_penalises_leaving_a_good_item_and_V_star_does_not():
    v, vs = _deltas(False), _deltas(True)
    assert v[2] < 0, f"V should penalise leaving a good item, got {v[2]:+.3f}"
    assert abs(vs[2]) < 1e-9, f"V* should give a neutral zero, got {vs[2]:+.3f}"


def test_V_star_is_never_negative():
    """The best option only ever improves, so V* is non-negative on every turn
    -- finding nothing better is a zero, not a penalty.

    This is the precondition for pairing it with the sign-preserving
    multiplicative form: under the additive form, centring within a trajectory
    pushes those zeros below the row mean and turns them into penalties, which
    only trades V's myopia for the deletion operator's redundancy penalty and
    lands back where we started.
    """
    assert all(x >= -1e-12 for x in _deltas(True)), _deltas(True)


def test_V_star_credits_only_the_improvement():
    vs = _deltas(True)
    assert abs(vs[1] - 0.6) < 1e-9, vs      # first buyable item found: the full 0.6
    assert abs(vs[2]) < 1e-9, vs            # leaving it: zero
    assert abs(vs[3] - 0.3) < 1e-9, vs      # a better one: only the increment 0.9-0.6
    assert abs(sum(vs) - 0.9) < 1e-9, vs    # the sum is the value of the final best option


def test_commit_turn_is_masked_under_both():
    for rm in (False, True):
        cr = webshop_commit_deltas(FakeWorker(), [0], ACTS, [[]], running_max=rm)
        assert cr.mask[0][-1] is False, cr.mask
        assert all(cr.mask[0][:-1]), cr.mask


# ---- one-step lookahead for the search turn ---------------------------------
# `buy now` is inert on a results page (V=0), so a search turn's V* increment is
# identically 0 and the largest anchor group of all -- the search page -- is mute
# for us. Lookahead defines a search turn's value as the best one-click purchase
# among the first k items on the results page.
PAGE = "Page 1 (Total results: 50) [SEP] Next > [SEP] B000000001 [SEP] x [SEP] B000000002 [SEP] y [SEP] B000000003 [SEP] z"
ITEM_VALUE = {"b000000001": 0.5, "b000000002": 0.9, "b000000003": 0.2}


class LookaheadWorker:
    def replay(self, session, actions):
        assert actions[-1] == COMMIT_ACTION
        body = actions[:-1]
        if not body or body == ["search[x]"]:
            return 0.0                          # ordering directly on the search/results page: 0
        if body[0] == "search[x]" and len(body) == 2 and body[1].startswith("click[b"):
            return ITEM_VALUE.get(body[1][6:-1], 0.0)
        return 0.0


def _la(k, acts, obs):
    return webshop_commit_deltas(LookaheadWorker(), [0], [acts], [obs],
                                 running_max=True, lookahead_k=k).deltas[0]


def test_lookahead_off_keeps_search_step_silent():
    d = _la(0, ["search[x]", "click[b000000001]", COMMIT_ACTION], ["", PAGE, "item"])
    assert abs(d[0]) < 1e-9 and abs(d[1] - 0.5) < 1e-9, d


def test_lookahead_credits_search_with_best_reachable_value_and_keeps_item_credit():
    d = _la(3, ["search[x]", "click[b000000001]", COMMIT_ACTION], ["", PAGE, "item"])
    assert abs(d[0] - 0.9) < 1e-9, d      # best of the first 3 items is 0.9: the query's "potential"
    assert abs(d[1] - 0.5) < 1e-9, d      # the item actually opened is still scored on its own value
                                          # (lookahead did not swallow `best`)


def test_lookahead_respects_k():
    d = _la(1, ["search[x]", "click[b000000002]", COMMIT_ACTION], ["", PAGE, "item"])
    assert abs(d[0] - 0.5) < 1e-9, d      # only the first item is considered
    assert abs(d[1] - 0.9) < 1e-9, d


def test_lookahead_is_zero_when_search_is_the_last_turn_or_page_missing():
    assert abs(_la(3, ["search[x]"], [""])[0]) < 1e-9
    assert abs(_la(3, ["search[x]", COMMIT_ACTION], ["", ""])[0]) < 1e-9


# ---- the commit turn taking part in the anchor group ------------------------
def test_unmasked_commit_turn_gets_zero_delta_and_is_kept():
    cr = webshop_commit_deltas(FakeWorker(), [0], ACTS, [[]], running_max=True, mask_commit=False)
    assert cr.mask[0][-1] is True
    assert abs(cr.deltas[0][-1]) < 1e-9          # committing only cashes in V(t), so under
                                                 # running-max the increment is 0
    assert cr.deltas[0][:-1] == webshop_commit_deltas(FakeWorker(), [0], ACTS, [[]], running_max=True).deltas[0][:-1]


from ssc.credit.webshop_cf import webshop_regret

RESULTS_PAGE = "Instruction: [SEP] x [SEP] Page 1 [SEP] Next > [SEP] B000000001 [SEP] a [SEP] B000000002 [SEP] b [SEP] B000000003 [SEP] c"
PRODUCT_PAGE = "Instruction: [SEP] x [SEP] [navy] [SEP] [black] [SEP] [large] [SEP] [Buy Now] [SEP] [Description]"
ITEM = {"b000000001": 0.57, "b000000002": 1.0, "b000000003": 0.2}
OPT = {("navy",): 0.9, ("large",): 0.8, ("navy", "large"): 1.0, ("large", "navy"): 1.0}


class RegretWorker:
    def observe(self, session, actions):
        if len(actions) >= 2 and actions[1].startswith("click[b"):
            return {"obs": PRODUCT_PAGE, "clickables": ["navy", "black", "large", "Buy Now", "Description"]}
        return {"obs": RESULTS_PAGE if actions else "", "clickables": []}

    def replay(self, session, actions):
        body = [a for a in actions[:-1]]
        if not body or body == ["search[x]"]:
            return 0.0
        item = body[1][6:-1] if len(body) >= 2 and body[1].startswith("click[b") else None
        if item is None:
            return 0.0
        base = ITEM.get(item, 0.0)
        opts = tuple(a[6:-1] for a in body[2:] if a.startswith("click[") and a[6:-1] in ("navy", "black", "large"))
        if item == "b000000001" and opts:
            return OPT.get(opts, base)
        return base


def _regret(acts, obs):
    return webshop_regret(RegretWorker(), [0], [acts], [obs], k=5).deltas[0]


def test_regret_is_zero_when_the_best_item_was_opened():
    r = _regret(["search[x]", "click[b000000002]", COMMIT_ACTION], ["", RESULTS_PAGE, "no options here"])
    assert abs(r[1]) < 1e-9 and abs(r[0]) < 1e-9


def test_regret_charges_opening_the_hedge_item_even_when_every_rollout_does_it():
    """The b09qqp3356 shape: the rollout opened an item worth 0.57 on an
    immediate buy while the page held one worth 1.0 on an immediate buy. Under
    the "buy-now value" variant the regret is 0.57 - 1.0, which does not depend
    on some sibling in the group having opened the right item."""
    r = webshop_regret(RegretWorker(), [0], [["search[x]", "click[b000000001]", COMMIT_ACTION]],
                       [["", RESULTS_PAGE, "no options here"]], k=5, item_value="buy").deltas[0]
    assert abs(r[1] - (0.57 - 1.0)) < 1e-9, r
    assert all(x <= 1e-12 for x in r)


def test_completion_value_does_not_charge_an_item_that_completes_to_full_marks():
    """The bias of "buy-now value": b1 is worth 0.57 bought immediately but 1.0
    once navy+large are selected -- it IS the right item. The completion-value
    variant gives it a regret of 0 (the page's best is also 1.0), while the
    buy-now variant mischarges it -0.43."""
    r = _regret(["search[x]", "click[b000000001]", COMMIT_ACTION], ["", RESULTS_PAGE, PRODUCT_PAGE])
    assert abs(r[1]) < 1e-9, r


def test_regret_charges_buying_before_completing_options():
    r = _regret(["search[x]", "click[b000000001]", COMMIT_ACTION], ["", RESULTS_PAGE, PRODUCT_PAGE])
    assert abs(r[2] - (0.57 - 1.0)) < 1e-9, r      # buying now is 0.57; completing navy+large reaches 1.0


def test_regret_is_zero_after_completing_options():
    r = _regret(["search[x]", "click[b000000001]", "click[navy]", "click[large]", COMMIT_ACTION],
                ["", RESULTS_PAGE, PRODUCT_PAGE, PRODUCT_PAGE, PRODUCT_PAGE])
    assert abs(r[4]) < 1e-9, r


def test_centred_item_credit_is_zero_mean_over_the_page_and_rewards_above_average_items():
    """The item-preference failure: when all 8 rollouts open the same item there
    is no contrast inside the group. `centred` mode uses the measurements of the
    first k items on the page as a virtual group: opening b2 (completes to 1.0)
    scores positive, opening b3 (0.2) scores negative, and b1 (completes to 1.0,
    tied best with b2) also scores positive -- `regret` mode gives b1 a 0."""
    RESULTS = "Page 1 (Total results: 50) [SEP] Next > [SEP] B000000001 [SEP] x [SEP] B000000002 [SEP] y [SEP] B000000003 [SEP] z"
    def credit(item, mode):
        return webshop_regret(RegretWorker(), [0], [["search[x]", f"click[{item}]", COMMIT_ACTION]],
                              [["", RESULTS, PRODUCT_PAGE]], k=5, mode=mode).deltas[0][1]
    # RegretWorker: b1 completes to 1.0 (navy+large), b2 buys now at 1.0, b3 is 0.2
    # -> page mean (1.0+1.0+0.2)/3 = 0.733
    assert abs(credit("b000000001", "regret")) < 1e-9
    assert credit("b000000001", "centred") > 0.2
    assert credit("b000000003", "centred") < -0.4
    assert abs(credit("b000000002", "centred") - credit("b000000001", "centred")) < 1e-9


# ---- centring item-opening regret within the sibling group ------------------
def test_item_regret_centred_within_sibling_group_kills_shared_id_preference():
    """All 8 rollouts open the same (wrong) item: the absolute regret gives each
    of them -0.4, which teaches across tasks that "other items are worse than
    this one". After zero-meaning within the group they are all 0."""
    from ssc.credit.webshop_cf import center_item_regret
    deltas=[[0.0,-0.4,0.0],[0.0,-0.4,0.0]]; kinds=[[None,"item","buy"],[None,"item","buy"]]
    keys=[["S","R","P"],["S","R","P"]]
    out=center_item_regret(deltas,kinds,keys)
    assert all(abs(out[i][1])<1e-9 for i in range(2)), out


def test_item_regret_centred_rewards_the_better_sibling_and_keeps_buy_regret_absolute():
    """Same results page, one rollout opens the right item (regret 0) and one
    the wrong one (-0.6): after centring they become +0.3 / -0.3, while the
    commit regret of -0.2 is kept as it is."""
    from ssc.credit.webshop_cf import center_item_regret
    deltas=[[0.0,0.0,-0.2],[0.0,-0.6,0.0]]; kinds=[[None,"item","buy"],[None,"item","buy"]]
    keys=[["S","R","P1"],["S","R","P2"]]
    out=center_item_regret(deltas,kinds,keys)
    assert abs(out[0][1]-0.3)<1e-9 and abs(out[1][1]+0.3)<1e-9, out
    assert abs(out[0][2]+0.2)<1e-9 and abs(out[1][2])<1e-9, out
    assert abs(out[0][0])<1e-9 and abs(out[1][0])<1e-9


def test_webshop_regret_reports_kinds():
    cr=webshop_regret(RegretWorker(),[0],[["search[x]","click[b000000001]",COMMIT_ACTION]],[["",RESULTS_PAGE,PRODUCT_PAGE]],k=5)
    assert cr.kinds[0]==[None,"item","buy"], cr.kinds
