"""ALFWorld environment adapter (replaces ScienceWorld as the primary env).

Why this environment: ScienceWorld was chosen for its unknown-substance task
families and then measured, and the measurement disqualified it. Its reward is
a cumulative subgoal score that moves a median of four times per trajectory, so
per-turn credit was already recoverable by differencing the score and a
credit-assignment method had nothing left to contribute. The premise the whole
method rests on -- that the terminal return is the only signal, so where credit
lands is the sole lever -- was false there.

ALFWorld was checked against that failure before anything was built on it
(`scripts/probe_alfworld.py`):

    SPARSE      the expert trajectory earns 0.0 on 43 of 44 steps and 1.0 on
                the last, so return-to-go really is constant across turns;
    LONG        44 expert steps against ScienceWorld's 25-31;
    REPLAYABLE  a pinned game file replays to identical rewards, so leave-one-out
                counterfactuals cost environment steps rather than model calls.

The third property is what the instrument needs and it is the one that is easy
to certify by accident: `env.reset()` advances to the NEXT game, so replaying
twice without pinning compares two different games, and both traces open with
the same TextWorld banner. `pinned_env` exists so that a replay is a replay.

One ACTION TURN = one policy generation = one environment action, matching the
ScienceWorld and search-QA adapters so the detector, credit-mass accounting and
counterfactual sweep work unchanged.

Reward: binary. `info["won"]` is the success flag and the environment pays 1.0
on the winning step, nothing before it. Unlike ScienceWorld there is no partial
score, which is the entire point.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from dataclasses import dataclass

from ssc.detector.rollout import Outcome, Rollout, Turn

ALFWORLD_DATA = os.environ.setdefault(
    "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

# Cheaper than rebuilding the list: `AlfredTWEnv.__init__` walks 8810 task
# directories and opens a JSON in each to check solvability, which costs ~5 s.
# That is invisible once per process and ruinous across a 16-worker
# counterfactual pool rebuilt every training step, so the walk happens once and
# the result is reused. See `_collect_game_files`.
GAME_LIST_CACHE = os.path.join(os.path.dirname(__file__), "_alfworld_games.json")

ACTION_RE = re.compile(r"^\s*action\s*:\s*(.+)$", re.I | re.M)

# Building a TextWorld env is NOT thread-safe. Gym registration mutates a global
# registry and the PDDL/grammar loader keeps parser state on the side, so
# concurrent construction corrupts both: collecting 8 rollouts across 8 threads
# crashed all 8, with `IndexError: pop from empty list` from the registry and
# `FailedToken: expecting 'template'` from the grammar parser. Rollouts are run
# concurrently to keep the samplers busy, so construction is serialised here
# rather than by making callers remember. Stepping is left unlocked -- the
# envs are independent once built, and holding the lock through a rollout would
# serialise the very thing the threads exist to overlap.
_BUILD_LOCK = threading.Lock()

# ALFWorld's six task types, as they appear in the game directory names.
TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light",
              "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_two_obj_and_place")

# The four whose optimal plans are long enough for per-turn credit to have
# anything to assign. Measured untrained on 192 rollouts with the admissible
# list shown: these four come back 48% over 21-25 turns, while
# pick_and_place_simple and look_at_obj_in_light come back 83% over 14 turns and
# include 4-5 turn solutions. They are 34% of a spread pool and can only dilute
# a per-turn effect, so training and evaluation can be restricted to these.
HARD_TYPES = ("pick_two_obj_and_place", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep")

SYSTEM_PROMPT = """You are an agent in a text-based household environment.

You are given a household task. You must explore the rooms, find the objects you
need, and manipulate them to complete the task.

Each turn, look at the observation and issue exactly ONE action, on its own line:
Action: <your action>

{action_source}

Guidance:
- Receptacles must be opened before you can see inside them. 'go to <recep>'
  moves you there; if it is closed, 'open <recep>' before taking anything.
- You can carry only ONE object at a time. Take it, then go to the destination,
  then put it there.
- 'put <obj> in/on <recep>' requires you to be holding <obj> and to be at
  <recep>.
- Cleaning uses a sinkbasin, heating a microwave, cooling a fridge. The pattern
  is: hold the object, go to the appliance, then use the clean/heat/cool action.
- 'examine <obj>' and 'look' inspect without consuming the object.
- If you cannot find an object, systematically visit the receptacles that would
  plausibly hold it rather than repeating the same one.

RESPONSE FORMAT -- this is strict:
Your reply must END with a line of exactly the form
Action: <one action>
Nothing may follow that line, and it must be a command, not prose.

Correct:
  Action: go to cabinet 4
  Action: take mug 1 from countertop 1
  Action: clean mug 1 with sinkbasin 1

Wrong (these are spent as wasted steps):
  Alternatively I could check the drawer
  I think the mug is in the cabinet
  Action: let me look around first"""

# Two openings for one prompt. Telling a policy it will be shown a list and then
# not showing one is a false instruction, and it costs more than politeness: the
# hidden-list setting is exactly where the policy has to generate ALFWorld's
# object-numbering syntax itself, so the prompt has to say that is the job.
WITH_LIST = """You will be shown the list of admissible actions each turn. Your \
action MUST be one of them, copied exactly. Anything else is rejected and wastes \
the step."""

WITHOUT_LIST = """No list of legal actions is given. You must write the command \
yourself, using the exact object names from the observation -- ALFWorld numbers \
them, so it is "take mug 1 from countertop 1", never "take the mug". A command \
the environment does not recognise is answered with "Nothing happens" and the \
step is wasted."""

OBJECTS_ONLY = """You will be shown the OBJECTS present each turn, but not the \
commands. Compose the command yourself from the verbs above and those exact \
names -- ALFWorld numbers them, so it is "take mug 1 from countertop 1", never \
"take the mug". A command the environment does not recognise is answered with \
"Nothing happens" and the step is wasted."""

# Three settings, not two booleans, because two booleans admit a state that
# means nothing ("hide the list but show it"). Measured base rates for an
# untrained Qwen3-8B:
#
#   full     65%   the list is the task; 20 GRPO steps buy 2.3 points and every
#                  arm lands inside a 2-point band
#   none      0%   0/24, every rollout burning all 30 turns. Not harder -- the
#                  policy cannot tell `cabinet 7` from `cabinet 17`, so 22% of
#                  its commands are refused and it never finds anything. With no
#                  successes there are no positive examples, group advantage is
#                  zero everywhere, and no arm can move.
#   objects        the middle: names are given, composition is not
ACTION_HINTS = {"full": WITH_LIST, "objects": OBJECTS_ONLY, "none": WITHOUT_LIST}


def system_prompt(action_hint: str = "full") -> str:
    if action_hint not in ACTION_HINTS:
        raise ValueError(f"action_hint must be one of {sorted(ACTION_HINTS)}, "
                         f"got {action_hint!r}")
    return SYSTEM_PROMPT.format(action_source=ACTION_HINTS[action_hint])


OBJECT_RE = re.compile(r"\b([a-z]+(?:basin|top)?) (\d+)\b")


def objects_in(admissible: list[str]) -> list[str]:
    """Distinct numbered objects mentioned by the admissible commands.

    Taken from the command list rather than parsed out of the observation text:
    the commands are what the environment will actually accept, so this gives
    the policy the right vocabulary without giving it the answer. What it must
    still work out is which verb applies and in what order.
    """
    seen = {}
    for cmd in admissible:
        for m in OBJECT_RE.finditer(cmd.lower()):
            seen.setdefault(f"{m.group(1)} {m.group(2)}", None)
    return list(seen)


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------
def _config(max_steps: int) -> dict:
    """ALFWorld wants its full YAML schema present even when most of it is dead.

    Only a handful of keys matter here -- the data paths, the logic files, the
    expert type and the step limit -- but `AlfredTWEnv` reads the rest during
    `init_env`, so the unused branches are filled with the package defaults
    rather than trimmed.
    """
    D = "$ALFWORLD_DATA"
    return {
        "dataset": {"data_path": f"{D}/json_2.1.1/train",
                    "eval_id_data_path": f"{D}/json_2.1.1/valid_seen",
                    "eval_ood_data_path": f"{D}/json_2.1.1/valid_unseen",
                    "num_train_games": -1, "num_eval_games": -1},
        "logic": {"domain": f"{D}/logic/alfred.pddl",
                  "grammar": f"{D}/logic/alfred.twl2"},
        "env": {"type": "AlfredTWEnv", "regen_game_files": False,
                "domain_randomization": False, "task_types": [1, 2, 3, 4, 5, 6],
                "expert_timeout_steps": 150, "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
                "hybrid": {"start_eps": 100000, "thor_prob": 0.5,
                           "eval_mode": "tw", "num_eval_games": 100},
                "thor": {"screen_width": 300, "screen_height": 300,
                         "smooth_nav": False, "save_frames_to_disk": False,
                         "save_frames_path": "./videos/"}},
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "mask_rcnn": {"pretrained_model_path": f"{D}/detectors/mrcnn.pth"},
        "general": {"random_seed": 42, "use_cuda": False, "visdom": False,
                    "task": "alfred", "training_method": "dagger",
                    "save_path": "./training/", "observation_pool_capacity": 3,
                    "hide_init_receptacles": False,
                    "training": {"batch_size": 1, "max_episode": 1,
                                 "smoothing_eps": 0.1,
                                 "optimizer": {"learning_rate": 0.001,
                                               "clip_grad_norm": 5}},
                    "evaluate": {"run_eval": False, "batch_size": 1,
                                 "env": {"type": "AlfredTWEnv"}},
                    "checkpoint": {"report_frequency": 1, "experiment_tag": "ssc",
                                   "load_pretrained": False,
                                   "load_from_tag": "not loading",
                                   "load_graph_generation_model_from_tag": "not loading",
                                   "save_frequency": 1000},
                    "model": {"encoder_layers": 1, "decoder_layers": 1,
                              "encoder_conv_num": 5, "block_hidden_dim": 64,
                              "n_heads": 1, "dropout": 0.1, "block_dropout": 0.1,
                              "recurrent": True}},
        "rl": {"action_space": "admissible", "max_target_length": 20,
               "beam_width": 10, "generate_top_k": 3,
               "training": {"max_nb_steps_per_episode": max_steps,
                            "learn_start_from_this_episode": 0,
                            "target_net_update_frequency": 500},
               "replay": {"accumulate_reward_from_final": True,
                          "count_reward_lambda": 0.0,
                          "novel_object_reward_lambda": 0.0,
                          "discount_gamma_game_reward": 1.0,
                          "discount_gamma_count_reward": 0.0,
                          "discount_gamma_novel_object_reward": 0.0,
                          "replay_memory_capacity": 500000,
                          "replay_memory_priority_fraction": 0.5,
                          "update_per_k_game_steps": 5, "replay_batch_size": 64,
                          "multi_step": 3, "replay_sample_history_length": 4,
                          "replay_sample_update_from": 2},
               "epsilon_greedy": {"noisy_net": False, "epsilon_anneal_episodes": 1000,
                                  "epsilon_anneal_from": 0.3, "epsilon_anneal_to": 0.1}},
        "dagger": {"action_space": "generation", "max_target_length": 20,
                   "beam_width": 10, "generate_top_k": 5,
                   "unstick_by_beam_search": False,
                   "training": {"max_nb_steps_per_episode": max_steps},
                   "fraction_assist": {"fraction_assist_anneal_episodes": 50000,
                                       "fraction_assist_anneal_from": 1.0,
                                       "fraction_assist_anneal_to": 0.01},
                   "fraction_random": {"fraction_random_anneal_episodes": 0,
                                       "fraction_random_anneal_from": 0.0,
                                       "fraction_random_anneal_to": 0.0},
                   "replay": {"replay_memory_capacity": 500000,
                              "update_per_k_game_steps": 5,
                              "replay_batch_size": 64,
                              "replay_sample_history_length": 4,
                              "replay_sample_update_from": 2}},
        "vision_dagger": {"model_type": "resnet", "resnet_fc_dim": 64,
                          "maskrcnn_top_X_detections": 10,
                          "use_exploration_frame_feats": False,
                          "sequence_aggregation_method": "average"},
    }


def _collect_game_files(split: str = "train") -> list[str]:
    """Game files for a split, cached across processes.

    The package's own collector re-derives this by walking every task directory
    and opening the traj JSON inside it. The list is static -- the games ship as
    files and nothing regenerates them here -- so it is walked once and cached.
    """
    cache = GAME_LIST_CACHE.replace(".json", f"_{split}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            files = json.load(f)
        if files:
            return files
    root = {"train": "json_2.1.1/train",
            "valid_seen": "json_2.1.1/valid_seen",
            "valid_unseen": "json_2.1.1/valid_unseen"}[split]
    pattern = os.path.join(ALFWORLD_DATA, root, "**", "game.tw-pddl")
    files = sorted(glob.glob(pattern, recursive=True))
    with open(cache, "w") as f:
        json.dump(files, f)
    return files


def make_env(game_files: list[str], batch_size: int = 1, max_steps: int = 50,
             with_expert: bool = False, with_plan: bool = False):
    """A TextWorld batch env over exactly `game_files`.

    Registration happens here rather than through `AlfredTWEnv.init_env` for two
    reasons. `AlfredTWEnv.__init__` walks 8810 task directories and opens a JSON
    in each, which costs ~5 s -- invisible once, ruinous across a worker pool.
    And `init_env` hard-codes its `EnvInfos`, so `policy_commands` cannot be
    requested through it: `AlfredExpert.load()` sets that flag, but it runs
    AFTER the gym env was built from the infos, so the plan comes back empty.
    That silent empty list is what `with_plan` exists to avoid -- the optimal
    distance-to-goal is the whole of the stronger measurement, and reading it as
    `len([])` would score every state as already solved.

    `with_expert` adds ALFWorld's handcoded expert. It is off for policy
    rollouts, and worth avoiding generally: on `look_at_obj_in_light` it takes
    11 steps where the planner's optimal plan is 3, so it is a demonstration
    policy, not a reference for how much a turn was worth.
    """
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler, AlfredExpert, AlfredInfos)

    infos = textworld.EnvInfos(won=True, admissible_commands=True,
                               policy_commands=with_plan, extras=["gamefile"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    if with_expert:
        infos.extras.append("expert_plan")
        wrappers.append(AlfredExpert(expert_type="handcoded"))

    # Always synchronous. Asynchronous registration spawns one worker process
    # per batch slot, and the annotation pool builds these envs INSIDE
    # multiprocessing workers, which are daemonic and cannot have children --
    # the batch env did not raise there, it HUNG, and eight annotation workers
    # sat blocked indefinitely with no output and no error.
    #
    # Nothing is lost. The batched counterfactual sweep is 18x faster than the
    # sequential one because it resets ONCE instead of T times, not because the
    # slots step in parallel; a reset costs ~1.6 s against ~0.08 s for a step.
    with _BUILD_LOCK:
        env_id = textworld.gym.register_games(
            list(game_files), infos, batch_size=batch_size, asynchronous=False,
            max_episode_steps=max_steps, wrappers=wrappers)
        return textworld.gym.make(env_id)


def pinned_env(game_file: str, max_steps: int = 50, with_expert: bool = False,
               with_plan: bool = False):
    """An env holding ONE game, so `reset()` replays rather than advances.

    This is the precondition for every counterfactual in this codebase. Without
    it `reset()` walks to the next game and a leave-one-out sweep silently
    compares actions against a different house.
    """
    return make_env([game_file], batch_size=1, max_steps=max_steps,
                    with_expert=with_expert, with_plan=with_plan)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
@dataclass
class ALFTask:
    """One ALFWorld game, shaped like `SWTask` / `ssc.env.data.Task`."""

    task_id: str
    question: str            # "put a clean mug in desk" -- filled on first reset
    answer: str = ""         # unused; interface parity
    dataset: str = "alfworld"
    game_file: str = ""
    task_type: str = ""
    split: str = "train"


def _task_type_of(game_file: str) -> str:
    for t in TASK_TYPES:
        if t in game_file:
            return t
    return "unknown"


def load_alfworld(n: int, split: str = "train", stride: int = 1,
                  task_types: tuple[str, ...] | None = None,
                  spread: bool = False) -> list[ALFTask]:
    """A deterministic slice of one split.

    Game paths begin with the task type and the list is sorted, so `[:n]` is a
    single-type sample dressed up as a diverse one -- the first 400 training
    games are all `look_at_obj_in_light`. `stride` walks the list instead.

    A fixed stride is still wrong whenever the pool size is not known in
    advance: stride 3 over the 134 unseen games, filtered to the four harder
    task types, returned 11 `pick_clean` and 1 `pick_cool`. `spread=True`
    ignores `stride` and takes evenly spaced indices across the pool that
    survives filtering, so the sample covers it however large it turns out to be.

    Deriving a stride and slicing -- `files[::len(files)//n][:n]` -- was not
    enough, and failed silently. Over the 92 unseen games of the four harder
    types it gave stride 3, and `files[::3]` yields 31 picks of which `[:24]`
    keeps the first 24, reaching only `files[69]`; the last 22 files are the
    whole `pick_two_obj_and_place` block, because the list is sorted and that
    type sorts last. The eval pool came back 11/7/6/0 while the training pool
    drew 126 of the type that was never scored. Asking for MORE was worse: at
    n=48 the derived stride is 1, the `stride > 1` branch drops out, and the
    result is the bare prefix `files[:48]` -- two types, 0 of the other two.
    """
    files = _collect_game_files(split)
    if task_types:
        files = [f for f in files if _task_type_of(f) in task_types]
    if spread and n > 0:
        # Evenly spaced across the WHOLE list, for any n, with no tail to drop.
        picked = (list(files) if n >= len(files)
                  else [files[(i * len(files)) // n] for i in range(n)])
    else:
        picked = files[::stride][:n] if stride > 1 else files[:n]
    out = []
    for f in picked:
        trial = os.path.basename(os.path.dirname(f))
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.append(ALFTask(task_id=f"alf_{name}_{trial[-6:]}",
                           question="",                 # read from the env
                           game_file=f, task_type=_task_type_of(f), split=split))
    return out


def build_user_message(observation: str, admissible: list[str], step: int,
                       max_turns: int, action_hint: str = "full") -> str:
    """The user message a turn was generated from.

    Module level, and used by both the rollout loop and the training encoder, so
    that the context trained on is the context the policy conditioned on. When
    these were written twice the training side silently dropped the admissible
    list -- present in every prompt at rollout time -- and every turn would have
    been trained against a prompt that never existed.
    """
    body = f"Observation:\n{observation}"
    if admissible:
        if action_hint == "full":
            body += "\n\nAdmissible actions:\n" + "\n".join(
                f"  {a}" for a in admissible)
        elif action_hint == "objects":
            names = objects_in(admissible)
            if names:
                body += "\n\nObjects here: " + ", ".join(names)
    return body + f"\n(step {step}/{max_turns})"


def read_task_description(obs: str) -> str:
    """ALFWorld states the goal in the opening observation, after a banner."""
    if "Your task is to:" in obs:
        return obs.split("Your task is to:")[-1].strip().split("\n")[0].strip()
    return obs.strip().split("\n")[-1].strip()


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
@dataclass
class ALFWorldPolicyConfig:
    model: str
    # 50 is the ALFWorld convention and the budget the published results use
    # against, so the numbers here are comparable to theirs. The handcoded
    # expert needs 44 steps on the two-object tasks, which leaves a competent
    # agent very little slack -- that tightness is the point.
    max_turns: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens_per_turn: int = 2048
    request_timeout: float = 600.0
    max_transient_retries: int = 3
    # A missing `Action:` line is a PARSING failure, not a policy decision;
    # spending an environment step on it conflates "the policy cannot do the
    # task" with "the adapter could not read the reply". Carried over from the
    # ScienceWorld adapter, where that conflation was measured at 64.5% of steps.
    max_format_retries: int = 2
    # Whether to show the admissible-command list each turn. ALFWorld agents in
    # the literature are given it, and an 8B policy cannot guess the exact
    # object-numbering grammar without it.
    action_hint: str = "full"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ALFWorldPolicy:
    """Runs one episode, returning the same Rollout type as the other adapters."""

    def __init__(self, cfg: ALFWorldPolicyConfig, base_url: str):
        """`base_url` may be a comma-separated list of sampler endpoints.

        Rollout collection is the bottleneck, so rollouts are round-robined
        across every endpoint; a trajectory stays on one server so every turn
        sees the same adapter.
        """
        from openai import OpenAI

        self.cfg = cfg
        urls = [u.strip() for u in str(base_url).split(",") if u.strip()]
        self.clients = [OpenAI(base_url=u, api_key="EMPTY",
                               timeout=cfg.request_timeout) for u in urls]
        self.client = self.clients[0]

    @staticmethod
    def extract_action(content: str, admissible: list[str] | None = None
                       ) -> tuple[str, bool]:
        """Return (action, format_ok).

        A missing `Action:` line must not fall back to the whole generation,
        which would send a paragraph to the parser and record a format miss as
        the policy failing. When the admissible list is available the fallback
        looks for one of its entries verbatim in the reply -- a policy that
        named a legal action but dropped the header is a format miss, not a
        wasted step, and `format_ok=False` keeps the miss visible either way.
        """
        body = content.split("</think>")[-1]
        m = None
        for m in ACTION_RE.finditer(body):
            pass
        if m:
            return m.group(1).strip(), True
        low = body.lower()
        if admissible:
            hits = [a for a in admissible if a.lower() in low]
            if hits:
                return max(hits, key=lambda a: low.rfind(a.lower())), False
        lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
        return (lines[-1] if lines else ""), False

    def rollout(self, task: ALFTask, rollout_index: int = 0, seed: int = 0) -> Rollout:
        t0 = time.time()
        env = pinned_env(task.game_file, max_steps=self.cfg.max_turns)
        obs, info = env.reset()
        obs = obs[0]
        goal = read_task_description(obs)
        opening_obs = obs

        client = self.clients[rollout_index % len(self.clients)]
        roll = Rollout(task_id=task.task_id, question=goal or task.question,
                       rollout_index=rollout_index, policy_model=self.cfg.model)

        def admissible_of(i) -> list[str]:
            return list(i.get("admissible_commands", [[]])[0])

        adm = admissible_of(info)

        def user_msg(observation: str, actions: list[str], step: int) -> str:
            return build_user_message(observation, actions, step,
                                      self.cfg.max_turns,
                                      action_hint=self.cfg.action_hint)

        messages = [{"role": "system",
                     "content": system_prompt(self.cfg.action_hint)},
                    {"role": "user",
                     "content": f"Task: {goal}\n\n" + user_msg(obs, adm, 0)}]

        won, done, attempt, n_format_retries = False, False, 0, 0
        while len(roll.turns) < self.cfg.max_turns and not done:
            fmt_try = 0
            try:
                r = client.chat.completions.create(
                    model=self.cfg.model, messages=messages,
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens_per_turn, seed=seed + attempt,
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(k in msg for k in ("context length", "maximum context",
                                          "reduce the length")):
                    roll.stop_reason = "context_limit"
                    break
                if attempt < self.cfg.max_transient_retries:
                    attempt += 1
                    continue
                roll.stop_reason = f"api_error:{type(e).__name__}"
                break
            attempt = 0

            m = r.choices[0].message
            reasoning = getattr(m, "reasoning_content", None) or ""
            content = m.content or ""
            action, fmt_ok = self.extract_action(content, adm)
            # A truncated reply never reached its Action line; parsing its tail
            # would turn a budget problem into a policy error.
            if r.choices[0].finish_reason == "length":
                fmt_ok = False

            while not fmt_ok and fmt_try < self.cfg.max_format_retries:
                fmt_try += 1
                n_format_retries += 1
                nudge = [{"role": "assistant", "content": content},
                         {"role": "user",
                          "content": "That reply had no valid Action line. Reply "
                                     "with exactly one line: Action: <command>, "
                                     "copied from the admissible actions."}]
                try:
                    r = client.chat.completions.create(
                        model=self.cfg.model, messages=messages + nudge,
                        temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                        max_tokens=self.cfg.max_tokens_per_turn,
                        seed=seed + 100 * fmt_try,
                    )
                except Exception:  # noqa: BLE001
                    break
                m = r.choices[0].message
                reasoning = getattr(m, "reasoning_content", None) or ""
                content = m.content or ""
                action, fmt_ok = self.extract_action(content, adm)
                if r.choices[0].finish_reason == "length":
                    fmt_ok = False

            action_text = (f"<think>{reasoning}</think>\n{content}"
                           if reasoning else content)
            roll.format_ok = roll.format_ok and fmt_ok

            # The admissible list the policy SAW while generating this turn,
            # captured before the step overwrites it. Training rebuilds the
            # context from the record, and without this the rebuilt prompt would
            # omit the list that was in front of the policy at generation time --
            # a train/inference mismatch on every single turn.
            #
            # Empty when the list was not shown. Recording it unconditionally
            # would put it back into the rebuilt prompt for a run configured to
            # hide it, which is the same mismatch in the other direction and
            # would be invisible: the arm would train against prompts richer
            # than the ones it acted on.
            shown_admissible = list(adm) if self.cfg.action_hint != "none" else []

            obs_l, sc, done_l, info = env.step([action])
            obs, done = obs_l[0], bool(done_l[0])
            won = bool(info.get("won", [False])[0])
            adm = admissible_of(info)

            roll.turns.append(Turn(
                index=len(roll.turns), text=action_text,
                tool_name="alf_action",
                tool_args={"action": action, "admissible": shown_admissible,
                           "step_shown": len(roll.turns),
                           "step_budget": self.cfg.max_turns,
                           # Turn 0's `observation` is the RESULT of turn 0, so
                           # the room description the first turn was generated
                           # from is otherwise unrecoverable and training would
                           # rebuild the opening prompt without it.
                           **({"opening_obs": opening_obs}
                              if not roll.turns else {})},
                observation=obs, n_generated_tokens=(r.usage.completion_tokens
                                                     if r.usage else 0)))

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",
                             "content": user_msg(obs, adm, len(roll.turns))})
            # Keep the prompt bounded without dropping the task statement. The
            # admissible list makes each turn bulky, so the window is shorter
            # than ScienceWorld's.
            if len(messages) > 17:
                messages = messages[:2] + messages[-14:]

        if not roll.stop_reason:
            roll.stop_reason = "episode_done" if done else "max_turns"

        roll.final_answer = f"won={int(won)}"
        roll.outcome = Outcome(
            success=int(won), reward=1.0 if won else 0.0,
            gold_answer=None,          # no answer string exists
            judge_verdict="alfworld_terminal_success")
        roll.wall_time = round(time.time() - t0, 1)
        roll.n_format_retries = n_format_retries
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        return roll
