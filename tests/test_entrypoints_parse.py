"""The entry points must at least parse and print --help.

The test suite does not import these scripts: their main() starts environments
and talks to a sampler. So a missing bracket can pass every other test and still
break every run. This pins "it parses" and "its --help works".
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = ["webshop/train.py", "webshop/eval.py",
               "alfworld/train.py", "alfworld/eval.py",
               "multihop/train.py", "multihop/eval.py"]


def test_every_entrypoint_parses():
    for rel in ENTRYPOINTS:
        src = (ROOT / rel).read_text()
        ast.parse(src, filename=rel)


def test_train_help_lists_the_shipped_arms():
    """Only the measured-credit arms and the GRPO baseline are shipped."""
    for rel, arms in (("webshop/train.py", ("grpo", "anchor_cvmax")),
                      ("alfworld/train.py", ("grpo", "anchor_cf")),
                      ("multihop/train.py", ("grpo", "anchor_cf"))):
        out = subprocess.run([sys.executable, str(ROOT / rel), "--help"],
                             capture_output=True, text=True, timeout=180, cwd=ROOT)
        assert out.returncode == 0, (rel, out.stderr[-500:])
        for arm in arms:
            assert arm in out.stdout, (rel, arm)
