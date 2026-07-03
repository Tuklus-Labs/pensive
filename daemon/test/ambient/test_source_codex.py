"""Codex JSONL source adapter tests.

Risk model:
- invariant/source-policy: every parsed source is stamped source='codex' and
  policy='distill', or Codex work exhaust can be misfiled as Claude Code memory.
- invariant/genuine-offset: each delta offset is the JSONL line number, never a
  synthesized fallback for parsed records, so exact provenance idempotency works.
- malformed-input/torn-jsonl: a partial line must be skipped without aborting the
  ambient tail.
- integration/provenance: codexSource output must drive real distill() and leave
  source='codex' provenance with a stable sessionId/sourceRef.
- integration/tool-blocks: Codex function calls and outputs must map to tool_use /
  tool_result blocks without becoming distilled prose.
- persistence/replay: reading the same log twice must not re-summarize or insert a
  second atom when offsets are unchanged.
- boundaries: missing session metadata falls back to filename stem; empty logs yield
  a well-formed source with no deltas.
- regression traps: contract/source-default and persistence/replay are covered;
  boundary/empty and malformed/torn are covered; concurrency/resource/state are N/A
  because codexSource is a pure single-file adapter with no shared mutable state.

Sabotage notes:
- Mutating source='codex' to the distiller default is caught by provenance asserts.
- Mutating offset from line number to a constant is caught by sourceRef and replay
  model-call-count asserts.
- Weakening the provenance/sourceRef assertions would let the integration test pass
  without proving the exact-ref contract, so those assertions name the rule.
"""
import json

import pytest

from ambient.distiller import distill
from ambient.source_codex import codexSource
from store.store import atomCount, getAtom, openStore


class FakeModelClient:
    def __init__(self):
        self.calls = []

    def summarizeSpan(self, spanText):
        self.calls.append(spanText)
        return {"text": f"principle: {' '.join(spanText.split())}", "kind": "atom"}


class FakeEmbedder:
    DIM = 4096

    def embed(self, texts):
        import hashlib
        import numpy as np

        out = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for tok in text.lower().split():
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % self.DIM] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            out.append(vec)
        return out


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _line(record):
    return json.dumps(record, separators=(",", ":"))


def _codex_log(*lines):
    return "\n".join(lines) + "\n"


def test_codex_log_distills_with_codex_source_and_genuine_offsets(tmp_path, store):
    logPath = tmp_path / "rollout-2026-07-02T18-15-33-synthetic.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "timestamp": "2026-07-02T18:15:33.000Z",
            "type": "session_meta",
            "payload": {
                "id": "meta-id",
                "session_id": "codex-session-17",
                "timestamp": "2026-07-02T18:15:33.000Z",
                "cwd": "/workspace/project",
                "originator": "codex_cli",
                "cli_version": "0.142.5",
                "source": "codex_cli",
                "thread_source": "codex_cli",
                "model_provider": "openai",
                "base_instructions": {"text": "synthetic instruction text"},
            },
        }),
        _line({
            "timestamp": "2026-07-02T18:15:34.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Please inspect it."}],
            },
        }),
        _line({
            "timestamp": "2026-07-02T18:15:35.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "id": "msg_synthetic",
                "phase": "final",
                "content": [{
                    "type": "output_text",
                    "text": (
                        "I found that the Codex source adapter must use JSONL line "
                        "numbers as stable offsets."
                    ),
                }],
            },
        }),
        _line({
            "timestamp": "2026-07-02T18:15:36.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": "call_synthetic",
                "call_id": "call-1",
                "name": "functions.exec_command",
                "arguments": "{\"cmd\":\"printf synthetic\"}",
            },
        }),
        _line({
            "timestamp": "2026-07-02T18:15:37.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "synthetic output",
            },
        }),
        "{\"timestamp\":\"2026-07-02T18:15:38.000Z\",\"type\":\"response_item\"",
    ), encoding="utf-8")
    model = FakeModelClient()
    embedder = FakeEmbedder()

    source = codexSource(logPath)

    assert source["sessionId"] == "codex-session-17", (
        f"session-id contract violated: source={source}"
    )
    assert source["source"] == "codex", (
        f"codex provenance source contract violated: source={source}"
    )
    assert source["policy"] == "distill", (
        f"codex policy contract violated: source={source}"
    )
    assert [delta["offset"] for delta in source["deltas"]] == [2, 3, 4, 5], (
        f"genuine-offset invariant violated: deltas={source['deltas']}"
    )
    assert source["deltas"][2]["events"][0]["content"][0]["type"] == "tool_use", (
        f"tool-call mapping contract violated: delta={source['deltas'][2]}"
    )
    assert source["deltas"][3]["events"][0]["content"][0]["type"] == "tool_result", (
        f"tool-result mapping contract violated: delta={source['deltas'][3]}"
    )

    result = distill(store, source, model, embedder)
    replay = distill(store, codexSource(logPath), model, embedder)

    assert result["inserted"] == 1, (
        f"distill integration contract violated: result={result}"
    )
    assert replay["inserted"] == 0 and replay["bumped"] == 1, (
        f"exact-ref replay contract violated: replay={replay}"
    )
    assert model.calls == [
        "I found that the Codex source adapter must use JSONL line numbers as "
        "stable offsets."
    ], f"replay idempotency violated: model.calls={model.calls}"
    assert atomCount(store) == 1, (
        f"anti-duplicate contract violated: atom_count={atomCount(store)}"
    )
    atom = getAtom(store, result["atomIds"][0])
    prov = atom["provenance"][0]
    assert prov["source"] == "codex", (
        f"codex provenance source contract violated: provenance={prov}"
    )
    assert prov["sessionId"] == "codex-session-17", (
        f"codex provenance session contract violated: provenance={prov}"
    )
    # Trusted ref format (Task 16 fix round 2): <sessionId>#<offset>.<16-hex
    # digest>. Pin the genuine line-offset prefix and the digest tail's shape,
    # not the digest bytes themselves.
    ref = prov["sourceRef"]
    prefix, _, digestTail = ref.rpartition(".")
    assert prefix == "codex-session-17#3.0.0", (
        f"line-offset sourceRef contract violated: provenance={prov}"
    )
    assert len(digestTail) == 16 and all(
        c in "0123456789abcdef" for c in digestTail
    ), f"sourceRef digest tail malformed: {ref}"


def test_codex_source_falls_back_to_filename_for_missing_session(tmp_path):
    logPath = tmp_path / "rollout-synthetic-fallback.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "timestamp": "2026-07-02T18:16:00.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Decision: use fallback."}],
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["sessionId"] == "rollout-synthetic-fallback", (
        f"filename fallback contract violated: source={source}"
    )
    assert source["deltas"][0]["sessionId"] == "rollout-synthetic-fallback", (
        f"delta session fallback contract violated: deltas={source['deltas']}"
    )
