"""WebShop adapter: the second environment for replay-measured credit.

ALFWorld alone cannot carry the claim, and the prerequisite the method trades a
planner for -- a deterministic, resettable environment that ENFORCES action
preconditions -- has to be shown to hold somewhere else. Measured here before
anything was built on top:

    replaying one action sequence three times   identical, on 3 sessions
    reset(session=s)                            returns to the same start
    click a product without searching first     refused

That third line is what makes the counterfactual self-checking. Deleting
`search[...]` leaves the following `click[...]` with nothing to click, so the
outcome moves -- the environment does the attribution, exactly as ALFWorld's
physical preconditions do. Multi-hop QA has no such check, which is why the
method would need a different formulation there.

WebShop runs in its own Python 3.8 environment (pinned 2022 dependencies), so
this adapter talks to it through a subprocess worker rather than importing it.
Retrieval uses BM25 instead of the paper's Lucene index -- pyserini 0.17 does
not build here. That changes retrieval QUALITY, not retrieval SEMANTICS: a
replayed query returns the identical ranking. Absolute scores are therefore not
comparable to published WebShop numbers, only across our own arms.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass

WEBSHOP_DIR = os.environ.get("WEBSHOP_PATH", os.path.expanduser("~/webshop"))
WEBSHOP_PY = os.environ.get("WEBSHOP_PYTHON", sys.executable)

SYSTEM_PROMPT = """You are shopping on a text-based web store.

You are given an instruction describing what to buy. Search for it, open a
product, choose the options the instruction asks for, and buy it.

Actions look like:
  search[<query>]        only on the search page
  click[<text>]          the exact text of a clickable item shown below

You will be shown the clickable items each turn. Your action MUST be one of the
listed forms, copied exactly.

Think briefly, then end your reply with exactly one line:

Action: <action>

Nothing may follow that line."""


def build_user_message(observation: str, clickables: list[str],
                       has_search: bool, step: int, max_turns: int) -> str:
    lines = [f"Step {step + 1}/{max_turns}", "", observation.strip(), ""]
    if has_search:
        lines.append("You may search: search[<your query>]")
    if clickables:
        lines.append("Clickable: " + ", ".join(f"click[{c}]" for c in clickables))
    lines.append("")
    lines.append("Your action:")
    return "\n".join(lines)


@dataclass
class WebShopTask:
    task_id: str
    session: int
    question: str = ""
    split: str = "train"


def load_webshop(n: int, split: str = "train", n_products: int = 1000
                 ) -> list[WebShopTask]:
    """Sessions are indices into WebShop's shuffled goal list.

    The split is a deterministic partition of the index range rather than a
    random draw, so train and eval never overlap and both are reproducible from
    the split name alone.
    """
    total = 6910                      # goals loaded with num_products=1000
    # Official split (verl-agent webshop/envs.py): test = the first 500 goal
    # indices, train = the rest. Before 09-13 this was the first 80% for
    # training and the last 20% for evaluation, which put the official test
    # goals into the training pool; no WebShop number produced before that can
    # go into the paper. "eval" is kept as an alias for "test".
    if split == "train":
        lo, hi = 500, total
    elif split in ("eval", "test"):
        lo, hi = 0, 500
    else:
        raise ValueError(f"unknown split {split!r}")
    span = hi - lo
    step = max(span // max(n, 1), 1)
    return [WebShopTask(task_id=f"ws_{split}_{lo + i * step}",
                        session=lo + i * step, split=split)
            for i in range(min(n, span))]


_WORKER = r'''
import json, sys, warnings, contextlib, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "%s")

# The store prints load progress and a tqdm bar to stdout, and stdout is the
# protocol channel -- one stray line and every reply after it fails to parse.
# Everything before READY is captured and discarded.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    import gym
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text",
                   num_products=1000)
sys.stderr.write("READY\n"); sys.stderr.flush()

def state():
    a = env.get_available_actions()
    return {"obs": str(env.observation), "clickables": a.get("clickables", []),
            "has_search": bool(a.get("has_search_bar"))}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    _noise = io.StringIO()
    try:
      with contextlib.redirect_stdout(_noise):
          if req["op"] == "reset":
              env.reset(session=req["session"])
              out = {"ok": True, "reward": 0.0, "done": False, **state()}
          elif req["op"] == "step":
              _, r, d, _ = env.step(req["action"])
              out = {"ok": True, "reward": float(r), "done": bool(d), **state()}
          elif req["op"] == "observe":
              # Reset and run the prefix inside one request, then return the
              # observation and clickables of that state (atomic).
              env.reset(session=req["session"])
              d = False
              for a in req["actions"]:
                  _, r, d, _ = env.step(a)
                  if d:
                      break
              out = {"ok": True, "reward": 0.0, "done": bool(d), **state()}
          elif req["op"] == "replay":
              # Reset, execute a fixed script, report the final reward. This is
              # the ablation primitive: no policy call, so a counterfactual costs
              # environment steps only.
              env.reset(session=req["session"])
              r, d = 0.0, False
              for a in req["actions"]:
                  try:
                      _, r, d, _ = env.step(a)
                  except Exception:
                      break
                  if d:
                      break
              out = {"ok": True, "reward": float(r), "done": bool(d)}
          else:
              out = {"ok": False, "error": "bad op"}
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush()
''' % WEBSHOP_DIR


class WebShopWorker:
    """One long-lived WebShop process; the store loads once, not per episode."""

    def __init__(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)      # the 3.8 env must not see our packages
        self.p = subprocess.Popen(
            [WEBSHOP_PY, "-u", "-c", _WORKER],
            cwd=WEBSHOP_DIR, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        while True:
            line = self.p.stderr.readline()
            if not line:
                raise RuntimeError("webshop worker died before READY")
            if line.strip() == "READY":
                break

    def _call(self, req: dict) -> dict:
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("webshop worker closed")
        return json.loads(line)

    def reset(self, session: int) -> dict:
        return self._call({"op": "reset", "session": session})

    def step(self, action: str) -> dict:
        return self._call({"op": "step", "action": action})

    def replay(self, session: int, actions: list[str]) -> float:
        """The ablation primitive: fixed script, no policy, final reward."""
        out = self._call({"op": "replay", "session": session,
                          "actions": list(actions)})
        return float(out.get("reward", 0.0)) if out.get("ok") else 0.0

    def observe(self, session: int, actions: list[str]) -> dict:
        """The state reached after executing a fixed prefix.

        Returns {"obs": page text, "clickables": clickable items}. The reset and
        the steps happen inside one request, so the call is atomic. Used by
        counterfactuals such as "complete the options" that have to read the
        clickables of a page.
        """
        out = self._call({"op": "observe", "session": session,
                          "actions": list(actions)})
        return out if out.get("ok") else {"obs": "", "clickables": []}

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.p.kill()
