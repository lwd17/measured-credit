"""Every filesystem path and model id this repo needs, resolved in one place.

The experiments were developed against one machine's layout, and that layout
leaked into a dozen argparse defaults. Anyone running this elsewhere would have
had to find and edit each one, and a missed edit is silent: a trainer pointed at
a model that does not exist fails loudly, but a probe pointed at the *wrong*
model runs fine and reports numbers for something else.

So the resolution order is the same everywhere:

    explicit CLI argument   >   environment variable   >   documented default

and anything without a usable default raises with the variable name to set,
rather than falling back to a path that happens to exist on one machine.

Set the model once and every entry point picks it up:

    export SSC_MODEL=Qwen/Qwen3-8B          # a hub id
    export SSC_MODEL=/scratch/models/qwen3  # or a local directory

`--model_path` on any script still overrides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Policy model
# --------------------------------------------------------------------------
# A hub id works as-is; a local directory is used verbatim. Both the trainer
# (transformers) and the sampler (vLLM) are given this same string, which is
# what keeps the behaviour policy and the trained policy the same weights --
# a mismatch there is the one failure this project pays the most for, because
# the loss assumes the sampler is serving the model being trained.
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def model_path(explicit: str | None = None) -> str:
    return explicit or os.environ.get("SSC_MODEL") or DEFAULT_MODEL


# --------------------------------------------------------------------------
# Environment data
# --------------------------------------------------------------------------
def _require(var: str, what: str, how: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(
            f"{what} needs ${var}. {how}\n"
            f"  export {var}=/path/to/it")
    return val


def alfworld_data() -> str:
    """ALFWorld's game files. `alfworld-download` puts them somewhere; point here."""
    val = os.environ.get("ALFWORLD_DATA")
    if not val:
        raise RuntimeError(
            "ALFWorld needs $ALFWORLD_DATA -- the directory `alfworld-download` "
            "wrote its game files to.\n  export ALFWORLD_DATA=/path/to/alfworld_data")
    os.environ.setdefault("ALFWORLD_DATA", val)
    return val


def webshop_dir() -> str:
    """A checkout of princeton-nlp/WebShop with its product index built."""
    return _require("WEBSHOP_DIR", "WebShop",
                    "Clone princeton-nlp/WebShop and build its index.")


def webshop_python() -> str:
    """The interpreter that has WebShop's own (2022-era) dependencies.

    WebShop pins versions that conflict with a current training stack, so the
    environment runs OUT OF PROCESS under its own interpreter. Defaults to the
    current one, which is right only if you installed WebShop into this env.
    """
    return os.environ.get("WEBSHOP_PYTHON", sys.executable)


# --------------------------------------------------------------------------
# Where runs and reports land
# --------------------------------------------------------------------------
ROOT = Path(os.environ.get("SSC_ROOT", Path(__file__).resolve().parents[1]))
RUNS = Path(os.environ.get("SSC_RUNS", ROOT / "runs"))
REPORTS = Path(os.environ.get("SSC_REPORTS", ROOT / "reports"))
LOGS = Path(os.environ.get("SSC_LOGS", ROOT / "logs"))


def describe() -> str:
    """One-line summary of what is resolved, for a run's log header."""
    def _try(f):
        try:
            return f()
        except RuntimeError:
            return "(unset)"
    return (f"model={model_path()}  alfworld={_try(alfworld_data)}  "
            f"webshop={_try(webshop_dir)}  runs={RUNS}")
