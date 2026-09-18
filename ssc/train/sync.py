"""Push an updated LoRA adapter into the vLLM sampler between training steps.

Without this the rollout policy never changes and the whole training loop is a
no-op that still produces plausible logs -- the most dangerous failure mode in
this project, and the reason it is checked rather than assumed. `verify_active`
sends a probe request naming the adapter and fails loudly if the server does
not serve it.

Requires the server to have been started with

    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 ... --enable-lora --max-lora-rank R

Adapters are given a fresh name per step because vLLM caches by name: reusing
one name after overwriting the files on disk can serve stale weights.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LoRASync:
    """Pushes an adapter to EVERY sampler the policy may draw from.

    `base_url` may be a comma-separated list. This must match the endpoint list
    the rollout policy uses: rollouts are round-robined across samplers for
    throughput, so an adapter pushed to only one of them leaves the others
    serving the BASE model. Nothing errors -- the run simply mixes updated and
    un-updated policies, and the logs look normal. `verify_active` therefore
    checks every endpoint, not just the first.
    """

    base_url: str
    timeout: float = 300.0

    @property
    def urls(self) -> list[str]:
        """Server roots, with any trailing `/v1` removed.

        The docstring above requires `base_url` to match the endpoint list the
        rollout policy uses, and that list ends in `/v1` because it is an
        OpenAI-compatible client base. Every request here appends its own
        `/v1/...`, so keeping the suffix produced `/v1/v1/load_lora_adapter` and
        a 404 -- from a route that IS present in the server's OpenAPI document,
        which makes the failure read like a disabled feature rather than a
        malformed URL.
        """
        out = []
        for u in str(self.base_url).split(","):
            u = u.strip().rstrip("/")
            if not u:
                continue
            if u.endswith("/v1"):
                u = u[: -len("/v1")]
            out.append(u)
        return out

    def _post_one(self, url: str, path: str, payload: dict) -> tuple[int, str]:
        req = urllib.request.Request(
            f"{url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read().decode()[:400]
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()[:400]
        except Exception as e:  # noqa: BLE001
            return -1, f"{type(e).__name__}: {e}"

    def _post(self, path: str, payload: dict) -> tuple[int, str]:
        """First endpoint only -- used by helpers that are per-endpoint."""
        return self._post_one(self.urls[0], path, payload)

    def load(self, name: str, path: str) -> None:
        """Load, replacing any adapter already registered under this name.

        vLLM rejects a repeated name with 400 "has already been loaded", which
        a rerun of the same step hits immediately. Since the files on disk have
        changed, the registered copy is STALE -- unloading and reloading is the
        only correct response; treating the 400 as success would serve old
        weights while the log claimed the update landed.
        """
        payload = {"lora_name": name, "lora_path": str(path)}
        for url in self.urls:
            code, body = self._post_one(url, "/v1/load_lora_adapter", payload)
            if code in (200, 201):
                continue
            # Retry on a name collision. This used to try once and raise on the
            # race where a previous run under the same name has just died and
            # its registration has not been cleaned up yet -- wq_L_uni died here
            # twice in the early hours of 09-09, once three steps after startup
            # and once 18 minutes in, each time wasting half a day of GPU.
            # Now it unloads and retries three times with backoff before raising.
            if code == 400 and "already been loaded" in body:
                import time as _t
                for attempt in range(3):
                    self._post_one(url, "/v1/unload_lora_adapter", {"lora_name": name})
                    _t.sleep(1.0 + attempt)
                    code, body = self._post_one(url, "/v1/load_lora_adapter", payload)
                    if code in (200, 201):
                        break
                if code in (200, 201):
                    continue
            raise RuntimeError(f"load_lora_adapter failed on {url} [{code}]: {body}")

    def unload(self, name: str) -> None:
        # Best effort on every endpoint: a name missing from one is not an error
        # worth aborting a training step for.
        for url in self.urls:
            self._post_one(url, "/v1/unload_lora_adapter", {"lora_name": name})

    def list_models(self, url: str | None = None) -> list[str]:
        try:
            with urllib.request.urlopen(f"{url or self.urls[0]}/v1/models",
                                        timeout=30) as r:
                return [m["id"] for m in json.loads(r.read()).get("data", [])]
        except Exception:  # noqa: BLE001
            return []

    def verify_active(self, name: str) -> None:
        """Fail loudly if the sampler is not actually serving `name`.

        A silent fallback to the base model would make every arm identical while
        still logging normally.
        """
        missing = [u for u in self.urls if name not in self.list_models(u)]
        if missing:
            raise RuntimeError(
                f"adapter {name!r} is not served by {missing}. Rollouts are "
                "round-robined across all endpoints, so a partial push would "
                "silently mix updated and base-model policies. Were all servers "
                "started with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1?")

    def swap(self, new_name: str, path: str, old_name: str | None = None) -> str:
        """Load the new adapter, verify it, then drop the previous one."""
        self.load(new_name, path)
        self.verify_active(new_name)
        if old_name and old_name != new_name:
            self.unload(old_name)
        return new_name
