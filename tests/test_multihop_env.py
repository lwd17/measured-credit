"""The multi-hop environment's guarantees, pinned before a trainer uses them.

The ablation regenerates an answer from reduced evidence, so the two properties
that make the measurement meaningful are that retrieval is a pure function of
the query and that dropping a step drops exactly that step's contribution.
"""

import pytest

from ssc.env.multihop_env import (MultiHopState, MultiHopTask, Retriever,
                                  exact_match, f1, normalise)


def _task():
    return MultiHopTask(
        task_id="t", question="Who directed Sinister and what year?",
        answer="Scott Derrickson 2012",
        titles=["Sinister", "Scott Derrickson", "Ed Wood", "Kansas"],
        paragraphs=["Sinister is a 2012 horror film directed by Scott Derrickson.",
                    "Scott Derrickson is an American director born in 1966.",
                    "Ed Wood was an American filmmaker.",
                    "Kansas is a state."],
        gold_titles=["Sinister", "Scott Derrickson"])


def test_retrieval_is_a_pure_function_of_the_query():
    """Every delta is a difference of two replays; a retriever that reorders
    between calls would put that jitter inside the measurement."""
    r = Retriever(_task())
    assert r.search("Sinister horror film") == r.search("Sinister horror film")
    assert r.search("") == []


def test_ties_break_by_paragraph_order():
    t = _task()
    t.titles = ["doc"] * 4
    t.paragraphs = ["same words here"] * 4
    ids = Retriever(t, k=4).search("same words here")
    assert ids == sorted(ids), ids


def test_bm25_does_not_punish_a_long_supporting_paragraph():
    """Jaccard divides by document length, so the long gold paragraph lost to a
    short distractor that happened to share one term."""
    t = _task()
    t.titles = ["gold", "short"]
    t.paragraphs = ["scott derrickson directed sinister " + "filler word " * 60,
                    "derrickson"]
    assert Retriever(t, k=1).search("scott derrickson sinister") == [0]


def test_gold_index_matches_titles_case_insensitively():
    t = _task()
    t.gold_titles = ["sinister", "SCOTT DERRICKSON"]
    assert t.gold_index() == {0, 1}


def test_dropping_a_step_removes_only_that_step():
    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[0], [1], [3]]
    full = st.evidence()
    for title in ("Sinister", "Scott Derrickson", "Kansas"):
        assert f"[{title}]" in full
    without = st.evidence(drop_step=1)
    assert "[Scott Derrickson]" not in without
    assert "[Sinister]" in without and "[Kansas]" in without


def test_a_paragraph_two_steps_retrieved_survives_dropping_one():
    """Evidence is de-duplicated, so a step is only worth what it UNIQUELY
    brought -- dropping a redundant retrieval must leave the text in place, and
    its delta must therefore come out near zero rather than crediting it twice."""
    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[0, 1], [1]]
    # paragraph 1 was brought by BOTH steps, so neither ablation can remove it
    # and both deltas will be near zero -- the credit belongs to neither alone.
    assert "[Scott Derrickson]" in st.evidence(drop_step=0)
    assert "[Scott Derrickson]" in st.evidence(drop_step=1)
    # paragraph 0 was unique to step 0, so only that ablation removes it
    assert "[Sinister]" not in st.evidence(drop_step=0)
    assert "[Sinister]" in st.evidence(drop_step=1)


def test_scoring_ignores_articles_case_and_punctuation():
    assert exact_match("The United States", "united states") == 1.0
    assert normalise("Yes.") == "yes"
    assert f1("Scott Derrickson", "Derrickson Scott") == pytest.approx(1.0)
    assert 0.0 < f1("United States", "The United States of America") < 1.0
    assert f1("Kansas", "Ohio") == 0.0


def test_empty_evidence_is_still_a_valid_prompt():
    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[0]]
    assert st.evidence(drop_step=0) == "(no documents retrieved)"


def test_deltas_credit_only_what_a_step_uniquely_brought():
    """Two steps that fetched the same paragraph each measure ~0: neither was
    necessary on its own. The cache makes that free as well as correct."""
    from ssc.credit.multihop_cf import multihop_deltas
    from ssc.detector.rollout import Rollout, Turn

    t = _task()

    def generate(messages):
        # answers correctly only when the Derrickson paragraph is present
        return ("Scott Derrickson 2012"
                if "[Scott Derrickson]" in messages[1]["content"] else "unknown")

    def roll_with(kinds):
        r = Rollout(task_id="t", question=t.question)
        for i, k in enumerate(kinds):
            r.turns.append(Turn(index=i, text="x", tool_name="multihop",
                                tool_args={"kind": k, "arg": "q"}))
        return r

    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[1], [1]]            # both steps fetched the same paragraph
    cr = multihop_deltas(generate, [st], [roll_with(["search", "search", "answer"])])
    assert cr.deltas[0] == [0.0, 0.0, 0.0], cr.deltas

    st2 = MultiHopState(task=t, retriever=Retriever(t))
    st2.retrieved = [[1], [3]]           # only step 0 is load-bearing
    cr2 = multihop_deltas(generate, [st2], [roll_with(["search", "search", "answer"])])
    assert cr2.deltas[0][0] > 0.9 and cr2.deltas[0][1] == 0.0, cr2.deltas
    assert cr2.deltas[0][2] == 0.0, "an answer turn has no evidence to drop"


def test_multihop_context_is_replayed_verbatim():
    """The reconstruction must equal the messages the policy actually saw."""
    from ssc.detector.rollout import Rollout, Turn
    from ssc.train.batch import _multihop_history

    shown = ["Question: who?", "Results:\n[A] alpha", "Results:\n[B] beta"]
    r = Rollout(task_id="t", question="who?")
    for i, sh in enumerate(shown):
        r.turns.append(Turn(index=i, text=f"Action: search[q{i}]",
                            tool_name="multihop",
                            tool_args={"kind": "search", "arg": f"q{i}",
                                       "shown": sh}))
    for upto in range(len(shown)):
        msgs = _multihop_history(r, upto, "sys")
        assert msgs[0] == {"role": "system", "content": "sys"}
        users = [m["content"] for m in msgs if m["role"] == "user"]
        assert users == shown[:upto + 1], (upto, users)
        assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(assistants) == upto
        assert msgs[-1]["role"] == "user", "generation follows the user turn"


def test_commit_deltas_credit_the_move_not_the_necessity():
    """V(t+1) - V(t): what the step bought, including when it lost ground."""
    from ssc.credit.webshop_cf import webshop_commit_deltas

    # prefix -> what buying right there is worth
    worth = {(): 0.0,
             ("search[x]",): 0.0,
             ("search[x]", "click[a]"): 0.4,
             ("search[x]", "click[a]", "click[< prev]"): 0.0,
             ("search[x]", "click[a]", "click[< prev]", "click[b]"): 0.2}

    class Worker:
        def replay(self, session, actions):
            assert actions[-1] == "click[buy now]"
            return worth[tuple(actions[:-1])]

    acts = [["search[x]", "click[a]", "click[< prev]", "click[b]", "click[buy now]"]]
    cr = webshop_commit_deltas(Worker(), [0], acts, [[""] * 5])
    assert cr.deltas[0] == [0.0, 0.4, -0.4, 0.2, 0.0], cr.deltas[0]
    assert cr.mask[0] == [True, True, True, True, False], cr.mask[0]
    # opening the good item is +0.4 and backing out of it is -0.4: deletion
    # cannot produce that second number at all.
    assert cr.n_replays == 5, cr.n_replays


def test_unpooled_measures_every_turn_on_its_own_trajectory():
    """Pooling gives one (page, action) measurement to every matching turn.

    On ALFWorld that trade is worth it -- a replay costs a 1.6 s reset. On
    WebShop it correlates with full leave-one-out at only rho = 0.719 and hands
    14% of turns a value measured on a different rollout, to save 2.1% of a
    training step. So the default here measures each turn on its own trajectory.
    """
    from ssc.credit.webshop_cf import webshop_deltas

    # Same page text and same action in both rollouts, but deleting it costs
    # something only in the first -- the case pooling cannot represent.
    acts = [["click[x]", "click[buy now]"], ["click[x]", "click[buy now]"]]
    obs = [["Page P", "Page P"], ["Page P", "Page P"]]

    class Worker:
        def replay(self, session, actions):
            full = len(actions) == 2
            if session == 0:
                return 0.8 if full else 0.2
            return 0.5 if full else 0.5      # rollout 1: the click is redundant

    unpooled = webshop_deltas(Worker(), [0, 1], acts, obs, fill_offpath=False,
                              drop_required=False, pool=False)
    pooled = webshop_deltas(Worker(), [0, 1], acts, obs, fill_offpath=False,
                            drop_required=False, pool=True)
    assert unpooled.deltas[0][0] == pytest.approx(0.6)
    assert unpooled.deltas[1][0] == pytest.approx(0.0), "rollout 1 measures itself"
    # pooling measures rollout 0 once and gives 0.6 to rollout 1 as well
    assert pooled.deltas[1][0] == pytest.approx(0.6), pooled.deltas
    assert unpooled.n_replays > pooled.n_replays


def test_binary_reward_plus_drop_required_empties_the_signal():
    """The interaction that made a whole WebShop arm run as plain GRPO.

    With a binary reward every pivotal turn scores 1 -> 0 (dropped as
    "structurally required") and every redundant one scores 1 -> 1 (delta zero),
    so the two cases are exhaustive and nothing survives. ALFWorld has a binary
    reward and applies no such filter, which is the setting where the method
    works; WebShop under a binary reward must match it.
    """
    from ssc.credit.webshop_cf import webshop_deltas

    acts = [["search[x]", "click[a]", "click[blue]", "click[buy now]"]]
    obs = [["Search", "Results", "Item", "Item"]]

    class Worker:
        """Full script matches perfectly; dropping the option or the commit
        loses the match; dropping the results click does not."""

        def replay(self, session, actions):
            if "click[blue]" not in actions or "click[buy now]" not in actions:
                return 0.4
            return 1.0

    drop = webshop_deltas(Worker(), [0], acts, obs, fill_offpath=False,
                          binary=True, drop_required=True)
    keep = webshop_deltas(Worker(), [0], acts, obs, fill_offpath=False,
                          binary=True, drop_required=False)
    assert all(abs(v) < 1e-9 for v in drop.deltas[0]), drop.deltas
    assert sum(1 for v in keep.deltas[0] if abs(v) > 1e-9) >= 2, keep.deltas


# ---------------------------------------------------------------------------
# Measurement operators for open-domain QA (the Search-R1 setting): V*, the value
# of answering now, and the value of a query relative to its siblings in the group
# ---------------------------------------------------------------------------

def _roll(kinds):
    from ssc.detector.rollout import Rollout, Turn
    r = Rollout(task_id="t", question="q")
    for i, k in enumerate(kinds):
        r.turns.append(Turn(index=i, text="x", tool_name="multihop",
                            tool_args={"kind": k, "arg": "q"}))
    return r


def _gen_needs_both():
    """Answers correctly only when both the Sinister and the Derrickson
    paragraph are present; one paragraph alone gets half the answer."""
    def generate(messages):
        ev = messages[1]["content"]
        has0, has1 = "[Sinister]" in ev, "[Scott Derrickson]" in ev
        if has0 and has1:
            return "Scott Derrickson 2012"
        if has1:
            return "Scott Derrickson"
        if has0:
            return "2012"
        return "unknown"
    return generate


def test_commit_deltas_follow_the_running_best_answer_value():
    from ssc.credit.multihop_cf import multihop_commit_deltas

    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[1], [3], [0]]          # half right -> distractor -> fully right
    cr = multihop_commit_deltas(_gen_needs_both(), [st],
                                [_roll(["search", "search", "search", "answer"])])
    v = cr.values[0]
    assert v[0] == 0.0 and v[-1] == pytest.approx(1.0)
    assert cr.deltas[0][0] == pytest.approx(v[1])            # the first search moves V from 0 to half right
    assert cr.deltas[0][1] == 0.0, "the distractor did not raise the best answer"
    assert cr.deltas[0][2] == pytest.approx(1.0 - v[1])     # the second paragraph completes it
    assert cr.mask[0] == [True, True, True, False], "the answer turn is masked by default"
    assert cr.kinds[0] == ["search", "search", "search", "answer"]
    # unmasked, the answer turn takes part in the comparison with a 0
    cr2 = multihop_commit_deltas(_gen_needs_both(), [st],
                                 [_roll(["search", "search", "search", "answer"])],
                                 mask_answer=False)
    assert cr2.mask[0][-1] is True and cr2.deltas[0][-1] == 0.0


def test_commit_deltas_without_running_max_can_go_negative():
    from ssc.credit.multihop_cf import multihop_commit_deltas

    t = _task()

    def generate(messages):
        ev = messages[1]["content"]
        # as soon as the Kansas distractor shows up the answer is wrong
        if "[Kansas]" in ev:
            return "Kansas"
        return "Scott Derrickson 2012" if "[Scott Derrickson]" in ev else "unknown"

    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[1], [3]]
    cr = multihop_commit_deltas(generate, [st], [_roll(["search", "search", "answer"])],
                                running_max=False)
    assert cr.deltas[0][0] == pytest.approx(1.0) and cr.deltas[0][1] == pytest.approx(-1.0)
    cr2 = multihop_commit_deltas(generate, [st], [_roll(["search", "search", "answer"])])
    assert cr2.deltas[0][1] == 0.0, "under running-max, leaving a good answer is not penalised"


def test_regret_compares_sibling_queries_at_the_same_evidence_state():
    """Three rollouts each issue one query from the closed-book state: the one
    that fetches the gold paragraph scores positive, the one that fetches the
    distractor scores negative, and answering directly (scoring 0) is negative
    too. The second state differs in every rollout, so there is nothing left to
    compare against."""
    from ssc.credit.multihop_cf import multihop_regret

    t = _task()
    gen = _gen_needs_both()
    sts = []
    for ids in ([[1]], [[3]], []):
        s = MultiHopState(task=t, retriever=Retriever(t)); s.retrieved = ids; sts.append(s)
    rolls = [_roll(["search", "answer"]), _roll(["search", "answer"]), _roll(["answer"])]
    cr = multihop_regret(gen, sts, rolls, mode="centred")
    q = [0.5 + 0.0, 0.0, 0.0]        # Q(search->half right)=f1("Scott Derrickson")~=0.667, Q(distractor)=0, Q(answer)=0
    a, b, c = cr.deltas[0][0], cr.deltas[1][0], cr.deltas[2][0]
    assert a > 0 and b < 0 and c < 0, (a, b, c)
    assert b == pytest.approx(c), "fetching the distractor and answering directly are worth the same here"
    assert cr.mask[0] == [True, False], "the second state has only itself, so it is unmeasurable"
    assert cr.kinds[2] == ["answer"]
    reg = multihop_regret(gen, sts, rolls, mode="regret")
    assert reg.deltas[0][0] == 0.0 and reg.deltas[1][0] < 0, \
        "the regret term only charges what falls below the best"


def test_regret_needs_a_second_distinct_candidate():
    from ssc.credit.multihop_cf import multihop_regret

    t = _task()
    sts = []
    for _ in range(3):
        s = MultiHopState(task=t, retriever=Retriever(t)); s.retrieved = [[1]]; sts.append(s)
    rolls = [_roll(["search", "answer"])] * 3
    cr = multihop_regret(_gen_needs_both(), sts, rolls)
    assert all(v == 0.0 for row in cr.deltas for v in row)
    assert not any(cr.mask[0][:1]), "identical queries leave nothing to compare against: unmeasurable, not zero"
    assert cr.n_calls <= 3, "cache: the same evidence is generated only once"


def test_scores_take_the_best_of_several_gold_answers():
    assert exact_match("political leader", ["politician", "political leader"]) == 1.0
    assert f1("politician", ["x", "politician"]) == pytest.approx(1.0)


def test_prior_residual_answer_credit_closes_the_telescoping_sum():
    """A search turn gets the V* increment, the answer turn gets the closed-book
    value plus the extraction residual, and the total is the score of the
    sampled answer."""
    from ssc.credit.multihop_cf import multihop_commit_deltas

    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[1], [0]]
    st.final = "Scott Derrickson"          # the sampled answer is only half right (F1~=0.667)
                                           # although the evidence supported a full answer
    cr = multihop_commit_deltas(_gen_needs_both(), [st], [_roll(["search", "search", "answer"])],
                                answer_credit="prior_residual")
    total = sum(cr.deltas[0])
    assert total == pytest.approx(f1("Scott Derrickson", t.answer))
    assert cr.deltas[0][2] < 0, \
        "the evidence supported a full answer but none was given: the extraction residual is negative"
    assert cr.mask[0] == [True, True, True]

    # the model already knew it closed-book and retrieval did not help: the
    # credit lands on the answer turn
    def knows(messages):
        return "Scott Derrickson 2012"
    st2 = MultiHopState(task=t, retriever=Retriever(t)); st2.retrieved = [[3]]; st2.final = "Scott Derrickson 2012"
    cr2 = multihop_commit_deltas(knows, [st2], [_roll(["search", "answer"])],
                                 answer_credit="prior_residual")
    assert cr2.deltas[0] == pytest.approx([0.0, 1.0])


def test_lookahead_candidate_penalises_answering_when_one_more_search_would_help():
    from ssc.credit.multihop_cf import multihop_regret

    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t)); st.retrieved = [[1]]
    r = _roll(["search", "answer"])
    for x in r.turns:
        x.tool_args["step_budget"] = 4
    calls = []

    def propose(roll, state):
        calls.append(1)
        return "Sinister 2012 horror film"      # fetches paragraph 0 and completes the second hop
    cr = multihop_regret(_gen_needs_both(), [st], [r], mode="regret", propose_query=propose)
    assert calls, "an answer turn with budget left should run the lookahead"
    assert cr.deltas[0][1] < 0 and cr.mask[0][1], \
        "one more search would have completed the answer: answering now is penalised"
    assert cr.n_lookahead == 1 and cr.n_lookahead_gain == 1
    # no lookahead for an answer turn whose budget is exhausted
    r2 = _roll(["search", "search", "search", "answer"])
    for x in r2.turns:
        x.tool_args["step_budget"] = 4
    st2 = MultiHopState(task=t, retriever=Retriever(t)); st2.retrieved = [[1], [3], [2]]
    calls.clear()
    cr2 = multihop_regret(_gen_needs_both(), [st2], [r2], mode="regret", propose_query=propose)
    assert not calls and cr2.deltas[0][-1] == 0.0


def test_question_candidate_gives_every_group_a_comparison_at_the_start():
    from ssc.credit.multihop_cf import multihop_regret

    t = _task()
    sts = []
    for _ in range(2):
        s = MultiHopState(task=t, retriever=Retriever(t)); s.retrieved = [[3]]; sts.append(s)
    rolls = [_roll(["search", "answer"])] * 2
    cr = multihop_regret(_gen_needs_both(), sts, rolls, mode="centred")
    assert not cr.mask[0][0], "two identical queries: nothing to compare against"
    cr2 = multihop_regret(_gen_needs_both(), sts, rolls, mode="centred", question_candidate=True)
    assert cr2.mask[0][0] and cr2.deltas[0][0] < 0, \
        "the question text as a query fetches the gold paragraph, so the distracting query loses"


def test_raw_marginal_value_keeps_the_harm_of_a_distracting_search():
    """The wiki setting: evidence accumulates, a distractor lowers how well the
    question can be answered right now, the negative sign is kept, and the
    decomposition still closes."""
    from ssc.credit.multihop_cf import multihop_commit_deltas

    t = _task()

    def generate(messages):
        ev = messages[1]["content"]
        if "[Kansas]" in ev:
            return "Kansas"
        return "Scott Derrickson 2012"        # known closed-book

    st = MultiHopState(task=t, retriever=Retriever(t)); st.retrieved = [[3]]; st.final = "Kansas"
    cr = multihop_commit_deltas(generate, [st], [_roll(["search", "answer"])],
                                running_max=False, answer_credit="prior_residual")
    assert cr.deltas[0][0] == pytest.approx(-1.0), "a distracting search: -1"
    assert sum(cr.deltas[0]) == pytest.approx(f1("Kansas", t.answer))
    assert cr.values[0] == [1.0, 0.0]


def test_answer_regret_fires_only_when_the_reader_would_have_scored_higher():
    from ssc.credit.multihop_cf import multihop_regret

    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t)); st.retrieved = [[0, 1]]
    st.final = "Scott Derrickson in 2012"     # evidence is sufficient and the greedy reader answers
                                              # correctly; the sampled answer has one extra word and loses EM
    gen = _gen_needs_both()
    cr = multihop_regret(gen, [st], [_roll(["search", "answer"])], mode="centred",
                         answer_regret_metric="em")
    assert cr.deltas[0][1] == pytest.approx(-1.0) and cr.mask[0][1]
    st.final = "Scott Derrickson 2012"
    cr2 = multihop_regret(gen, [st], [_roll(["search", "answer"])], mode="centred",
                          answer_regret_metric="em")
    assert cr2.deltas[0][1] == 0.0


def test_repeat_search_gets_a_negative_credit_not_a_zero():
    """The 2Wiki case: the same query is searched twice in a row and the second
    search brings back no new document. V's increment is exactly 0, and a zero
    is not a penalty -- the policy cannot learn "do not repeat" from it.
    `repeat_penalty` gives it an explicit negative credit."""
    from ssc.credit.multihop_cf import multihop_commit_deltas

    t = _task()
    st = MultiHopState(task=t, retriever=Retriever(t))
    st.retrieved = [[1], [1], [0]]        # the second search is an exact repeat
    st.final = "Scott Derrickson 2012"
    r = _roll(["search", "search", "search", "answer"])
    cr = multihop_commit_deltas(_gen_needs_both(), [st], [r], running_max=False,
                                answer_credit="prior_residual", repeat_penalty=0.25)
    assert cr.deltas[0][1] == pytest.approx(-0.25), cr.deltas[0]
    assert cr.mask[0][1] is True
    # with the penalty off it is still zero (the previous behaviour is unchanged)
    cr0 = multihop_commit_deltas(_gen_needs_both(), [st], [r], running_max=False,
                                 answer_credit="prior_residual")
    assert cr0.deltas[0][1] == 0.0
    # a search that brings back a new document is unaffected
    assert cr.deltas[0][2] > 0
