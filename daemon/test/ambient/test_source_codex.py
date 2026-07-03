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


def test_response_item_messages_are_canonical_when_event_messages_duplicate(tmp_path):
    logPath = tmp_path / "rollout-duplicate-assistant-families.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Use the canonical answer."}],
            },
        }),
        _line({
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Use the canonical answer.",
            },
        }),
        _line({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Use the second answer."}],
            },
        }),
        _line({
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Use the second answer.",
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert [delta["offset"] for delta in source["deltas"]] == [1, 3], (
        f"response_item canonical family violated: deltas={source['deltas']}"
    )
    assert [
        delta["events"][0]["content"][0]["text"]
        for delta in source["deltas"]
    ] == ["Use the canonical answer.", "Use the second answer."]


def test_agent_message_events_are_fallback_when_response_item_messages_absent(tmp_path):
    logPath = tmp_path / "rollout-agent-message-fallback.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Fallback assistant text.",
            },
        }),
        _line({
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Second fallback assistant text.",
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert [delta["offset"] for delta in source["deltas"]] == [1, 2]
    assert [
        delta["events"][0]["content"][0]["text"]
        for delta in source["deltas"]
    ] == ["Fallback assistant text.", "Second fallback assistant text."]


def test_event_user_message_maps_to_user_text_event(tmp_path):
    logPath = tmp_path / "rollout-event-user-message.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "User-side event text.",
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["deltas"] == [{
        "sessionId": "rollout-event-user-message",
        "offset": 1,
        "events": [{
            "role": "user",
            "content": [{"type": "text", "text": "User-side event text."}],
        }],
    }]


def test_session_meta_id_is_used_when_session_id_is_absent(tmp_path):
    logPath = tmp_path / "rollout-id-only-meta.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "session_meta",
            "payload": {"id": "id-only-session"},
        }),
        _line({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Meta id fallback."}],
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["sessionId"] == "id-only-session"
    assert source["deltas"][0]["sessionId"] == "id-only-session"


def test_empty_codex_log_yields_empty_deltas(tmp_path):
    logPath = tmp_path / "rollout-empty.jsonl"
    logPath.write_text("", encoding="utf-8")

    source = codexSource(logPath)

    assert source == {
        "sessionId": "rollout-empty",
        "source": "codex",
        "policy": "distill",
        "deltas": [],
    }


def test_custom_tool_call_and_web_search_call_map_to_tool_use(tmp_path):
    logPath = tmp_path / "rollout-tool-branches.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "id": "custom-1",
                "name": "custom-tool",
                "input": {"needle": "synthetic"},
            },
        }),
        _line({
            "type": "response_item",
            "payload": {
                "type": "web_search_call",
                "id": "search-1",
                "action": "search",
                "status": "completed",
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["deltas"] == [
        {
            "sessionId": "rollout-tool-branches",
            "offset": 1,
            "events": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "custom-tool",
                    "input": {"needle": "synthetic"},
                    "id": "custom-1",
                }],
            }],
        },
        {
            "sessionId": "rollout-tool-branches",
            "offset": 2,
            "events": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "name": "search",
                    "input": {"action": "search", "status": "completed"},
                    "id": "search-1",
                }],
            }],
        },
    ]


@pytest.mark.parametrize("role", ["developer", "system"])
def test_developer_and_system_messages_map_to_user_role(tmp_path, role):
    logPath = tmp_path / f"rollout-{role}-role.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": f"{role} instruction."}],
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["deltas"][0]["events"][0] == {
        "role": "user",
        "content": [{"type": "text", "text": f"{role} instruction."}],
    }


def test_reasoning_record_with_empty_summary_is_excluded(tmp_path):
    logPath = tmp_path / "rollout-empty-reasoning.jsonl"
    logPath.write_text(_codex_log(
        _line({
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "id": "reasoning-1",
                "encrypted_content": "synthetic-encrypted-content",
                "summary": [],
            },
        }),
    ), encoding="utf-8")

    source = codexSource(logPath)

    assert source["deltas"] == []


def test_torn_multibyte_line_is_skipped_without_losing_valid_lines(tmp_path):
    logPath = tmp_path / "rollout-torn-multibyte.jsonl"
    validBefore = _line({
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Before torn bytes."}],
        },
    }).encode("utf-8")
    torn = (
        b'{"type":"response_item","payload":{"type":"message","role":"assistant",'
        b'"content":[{"type":"output_text","text":"torn \xe2\x82'
    )
    validAfter = _line({
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "After torn bytes."}],
        },
    }).encode("utf-8")
    logPath.write_bytes(validBefore + b"\n" + torn + b"\n" + validAfter + b"\n")

    source = codexSource(logPath)

    assert [delta["offset"] for delta in source["deltas"]] == [1, 3]
    assert [
        delta["events"][0]["content"][0]["text"]
        for delta in source["deltas"]
    ] == ["Before torn bytes.", "After torn bytes."]
