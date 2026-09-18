"""Search-R1-style multi-turn search environment (runbook sections 16, 27).

Section 16 chooses read-only search QA as the primary environment precisely
because a wrong hypothesis can be abandoned without irreversibly changing the
world -- which is the assumption SSC rests on (contrast section 17's ALFWorld,
where semantic invalidation != physical irrelevance).

Retrieval is the frozen Wikipedia BM25 index already built for this project:
6,407,814 articles / 20,476,167 passages from the pinned dump
wikimedia/wikipedia@b04c8d1c (20231101.en), served over HTTP so pyserini's JVM
stays out of the training process. Section 27 requires the retrieval backend to
be IDENTICAL across methods, so the manifest is recorded with every run.
"""

from __future__ import annotations

import json
import urllib.request

# Search-R1's action is a SINGLE query per turn (<search>query</search>).
# Allowing parallel queries collapses a multi-hop question into one action
# turn: measured on HotpotQA with 5 queries/call, median depth was 2 turns and
# 0/12 rollouts could host a section 12 REPLACE event, which needs
# adoption -> governed action -> invalidation -> switch. That is a property of
# the tool I gave the policy, not of the policy.
MAX_QUERIES_PER_CALL = 1
TOP_K = 5


class BM25SearchEnv:
    """Read-only retrieval. No cross-call cache: repeated queries re-execute."""

    name = "frozen_wikipedia_bm25"

    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 120.0,
                 top_k: int = TOP_K):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.top_k = top_k
        self._manifest: dict | None = None

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            return {"error": f"search backend unreachable: {type(e).__name__}: {e}"}

    def search(self, queries: list[str] | str) -> dict:
        qs = [queries] if isinstance(queries, str) else list(queries)
        qs = [q for q in qs if str(q).strip()][:MAX_QUERIES_PER_CALL]
        if not qs:
            return {"error": "search requires at least one non-empty query"}
        return self._post("/search", {"query": qs})

    def format_observation(self, result: dict) -> str:
        """Render retrieval output as the observation text the policy reads.

        Kept deliberately plain: the detector reads these observations too
        (section 11 input 3), and any evaluative wording here would be a
        subtle channel for correctness information.
        """
        if "error" in result:
            return f"Search error: {result['error']}"
        lines: list[str] = []
        for q, hits in (result.get("results") or {}).items():
            lines.append(f'Results for "{q}":')
            if not hits:
                lines.append("  (no results)")
            for i, h in enumerate(hits[: self.top_k], 1):
                lines.append(f"  [{i}] {h.get('title', '?')}: {h.get('snippet', '')}")
        return "\n".join(lines) if lines else "(no results)"

    def manifest(self) -> dict:
        """Section 27: pinned retrieval provenance, recorded per run."""
        if self._manifest is None:
            m = self._post("/manifest", {})
            self._manifest = {
                "env": self.name,
                "retriever": "BM25",
                "top_k": self.top_k,
                "dump_date": m.get("wikipedia_dump_date"),
                "dump_revision": m.get("wikipedia_dump_revision"),
                "num_articles": m.get("index_articles"),
                "num_passages": m.get("index_documents"),
                "index_sha256": m.get("index_listing_sha256"),
                "bm25_k1": m.get("bm25_k1"), "bm25_b": m.get("bm25_b"),
            }
        return self._manifest


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": ("Search a Wikipedia index with ONE keyword query and "
                        "receive ranked snippets. Issue further searches in "
                        "later turns as you learn more."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "A single keyword query."},
            },
            "required": ["query"],
        },
    },
}

# Section 16 is a QA setting; the policy answers in its own words and the
# reward is verified against the reference by the TRAINER, never by the
# detector (section 11).
SYSTEM_PROMPT = """You are a research assistant answering a question using a Wikipedia search tool.

Search whenever you are unsure. You may search several times, refining or changing
your approach based on what you find.

When you are ready, give your final answer on its own line in exactly this form:
Answer: <your short answer>"""

# Search-R1's own instruction, used as a prompt-faithfulness control. The
# difference that matters is the explicit reason-after-every-observation loop
# and the explicit permission to search repeatedly; neither tells the policy to
# reconsider or abandon a hypothesis, so it cannot manufacture REPLACE events.
SEARCH_R1_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> \
first every time you get new information. After reasoning, if you find you lack some knowledge, \
you can call a search engine by using the search tool, and it will return the top searched results. \
You can search as many times as you want. If you find no further external knowledge needed, you \
can directly provide the answer, on its own line in exactly this form:
Answer: <your short answer>"""

PROMPTS = {"default": SYSTEM_PROMPT, "search_r1": SEARCH_R1_PROMPT}
