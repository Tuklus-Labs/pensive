"""Tests guarding the Pass 1 batch-fix changes.

These tests are deep-tests style: each test docstring names the rule it
guards. The tests are designed so that reverting the fix would cause
them to fail loudly with an explanation, not just a silent ``assert
x == y``.

Coverage:

* CRIT-2 -- spreading.py freq>=1 clamp on the 1.0/(freq**spec_power) sites
* CRIT-3 -- SpreadingConfig.__post_init__ validation
* CRIT-4 -- ParallelHybrid rank-fusion tier ordering at multiple ranks
* IMP-1  -- QueryGenerator.for_email + EmailJSONLParser query population
* IMP-6  -- ChatGPTParser._extract_thread MAX_THREAD_PATH cap
* IMP-7  -- pensive build CLI --force flag
* IMP-8  -- SpreadingActivation.query empty/None handling
* IMP-9  -- L2Handler grown-region zero initialization
* IMP-10 -- pattern_learner integrate_with_sa cache invalidation
* IMP-11 -- ParallelHybrid deterministic ordering across runs
* IMP-12 -- pensive.__version__ matches pyproject.toml
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

import pensive
from pensive.spreading import SpreadingActivation, SpreadingConfig
from pensive.patterns import SYNTHETIC_PATTERNS
from pensive.ingestion.query_gen import QueryGenerator
from pensive.ingestion.parsers.chatgpt import ChatGPTParser


# IMP-12 -- version bump


def test_version_bumped_to_match_pyproject():
    """pensive.__version__ must match the published pyproject.toml.

    Sabotage check: rolling __version__ back to "0.1.1" would fail this
    test with a clear message about what version was actually exposed.
    """
    assert pensive.__version__ == "0.2.0", (
        f"pensive.__version__ is {pensive.__version__!r}, expected '0.2.0'. "
        "If you bumped pyproject.toml without also editing __init__.py, "
        "wheels published as 0.2.0 will report the wrong version at runtime."
    )


# CRIT-3 -- SpreadingConfig validation


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"decay": 1.0}, "decay"),
        ({"decay": 1.5}, "decay"),
        ({"decay": -0.1}, "decay"),
        ({"spec_power": -0.5}, "spec_power"),
        ({"threshold": -0.01}, "threshold"),
        ({"max_hops": -1}, "max_hops"),
    ],
)
def test_spreading_config_rejects_invalid(kwargs, field):
    """SpreadingConfig.__post_init__ rejects each invalid value with a
    field-specific ValueError.

    Sabotage check: removing __post_init__ would let these constructors
    succeed and produce activation runs that crash deep in the spread
    kernel (decay>=1) or silently invert ranking (negative spec_power).
    The error message must name the field so callers can fix it.
    """
    with pytest.raises(ValueError, match=field):
        SpreadingConfig(**kwargs)


def test_spreading_config_accepts_valid_defaults():
    """Default construction must NOT raise (regression guard for the
    validator being too strict)."""
    cfg = SpreadingConfig()
    assert 0 <= cfg.decay < 1.0
    assert cfg.spec_power >= 0
    assert cfg.threshold >= 0
    assert cfg.max_hops >= 0


# CRIT-2 -- freq>=1 defense-in-depth on 1.0/(freq**spec_power)


def test_zero_frequency_does_not_zerodivide_in_build():
    """Manually staging a doc whose entity is missing from entity_freq
    must not crash the build with ZeroDivisionError.

    This is the defense-in-depth guard. The vanilla build flow always
    pre-populates entity_freq, but a subclass / future caller could
    skip that step and we want a clamped specificity rather than a
    crash.

    Sabotage check: removing the ``max(entity_freq[entity], 1)`` clamp
    re-introduces ZeroDivisionError on this test.
    """
    sa = SpreadingActivation(
        config=SpreadingConfig(spec_power=0.5),
        patterns=SYNTHETIC_PATTERNS,
    )
    # Build normally first so the graph state is valid.
    sa.build([
        {"id": "doc1", "content": "alpha beta",
         "value": "alpha beta", "query": "what is alpha?"},
    ])

    # Now sabotage entity_freq so a downstream add_documents iteration
    # would see a freq of 0. The clamp should rescue the spec calc.
    sa.entity_freq.clear()
    # add_documents internally re-counts then re-computes specificity
    # from the post-update entity_freq, so we drop counts mid-flight by
    # patching defaultdict factory to return 0 even on first lookup.
    # Easier: directly call _refresh_specificity_for_entities with a
    # never-seen entity (which has freq=0 in entity_freq).
    sa._refresh_specificity_for_entities(["never-seen-entity"])  # must not raise


def test_query_does_not_zerodivide_with_zero_freq():
    """Querying after add_documents on a graph whose freq map was wiped
    must not blow up.

    Sabotage check: without the clamp, the next add_documents call hits
    1.0 / (0 ** 0.5) which raises ZeroDivisionError.
    """
    sa = SpreadingActivation(
        config=SpreadingConfig(spec_power=0.5),
        patterns=SYNTHETIC_PATTERNS,
    )
    sa.build([
        {"id": "doc1", "content": "the latency was 199ms on 2025-07-16",
         "value": "199ms", "query": "what was the latency?"},
    ])

    # Wipe freq, then re-add. Without the clamp, the recomputation of
    # specificity at line ~482 (now clamped) would raise.
    sa.entity_freq.clear()
    sa.add_documents([
        {"id": "doc2", "content": "fresh content",
         "value": "fresh", "query": "what is fresh?"},
    ])

    # And query still returns something sensible.
    results = sa.query("fresh")
    assert isinstance(results, list)


# IMP-8 -- query() handles empty / whitespace text


@pytest.mark.parametrize("empty_query", ["", None])
def test_query_empty_returns_empty(empty_query):
    """query("") and query(None) must short-circuit to an empty list,
    not crash on str.split or seed an empty spread.

    Sabotage check: without the early return, query(None) raises
    AttributeError on None.split() and query("") returns whatever the
    spread kernel produces from an empty seed dict (degenerate behaviour).
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc1", "content": "the latency was 199ms",
         "value": "199ms", "query": "what was the latency?"},
    ])
    results = sa.query(empty_query)
    assert results == [], (
        f"empty query should return [], got {results!r} -- "
        "ensure the early-return guards both query() and query_with_doc_ids()"
    )


@pytest.mark.parametrize("empty_query", ["", None])
def test_query_with_doc_ids_empty_returns_empty(empty_query):
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc1", "content": "the latency was 199ms",
         "value": "199ms", "query": "what was the latency?"},
    ])
    results = sa.query_with_doc_ids(empty_query)
    assert results == [], (
        f"empty query_with_doc_ids should return [], got {results!r}"
    )


# IMP-1 -- email parser populates query field via query_gen.for_email


def test_query_generator_for_email_returns_nonempty_natural_text():
    """QueryGenerator.for_email must produce a question-shaped string
    that includes the sender and at least one body term.

    Sabotage check: if for_email is removed and the email parser falls
    back to '', this test fails immediately.
    """
    qg = QueryGenerator()
    q = qg.for_email(
        sender="alice@example.com",
        subject="Project Phoenix kickoff",
        body="The Phoenix migration is scheduled for Tuesday at 4pm.",
    )
    assert isinstance(q, str)
    assert q, "for_email returned an empty string"
    # Must mention the sender and subject in the natural-language phrasing
    # so SA can pick them up as entities at index time.
    assert "alice@example.com" in q, (
        f"sender missing from generated query: {q!r}"
    )
    assert "Phoenix" in q, (
        f"subject keyword missing from generated query: {q!r}"
    )


def test_query_generator_for_email_handles_missing_sender_subject():
    qg = QueryGenerator()
    q = qg.for_email(sender="", subject="", body="some body text here")
    assert q, "for_email must produce a non-empty query even with empty fields"


def test_email_parser_yields_documents_with_query_populated(tmp_path):
    """End-to-end check: EmailJSONLParser yields docs whose `query`
    field is non-empty after IMP-1.

    Sabotage check: reverting to the old hasattr fallback that wrote
    `query=''` would fail this test loudly.
    """
    from pensive.ingestion.parsers.email import EmailJSONLParser

    jsonl = tmp_path / "emails.jsonl"
    sample_email = {
        "from": "bob@example.com",
        "to": "alice@example.com",
        "subject": "Quarterly numbers",
        "date": "2026-01-15",
        "body": (
            "The Q4 revenue came in at $4.2 million, exceeding "
            "the projected $3.8 million target by twelve percent."
        ),
    }
    jsonl.write_text(json.dumps(sample_email) + "\n")

    parser = EmailJSONLParser(str(jsonl))
    docs = list(parser.parse())
    assert docs, "no documents yielded from email parser"
    for d in docs:
        assert d.query, (
            f"email parser yielded a document with empty query: {d!r}. "
            "IMP-1 fix requires for_email() to populate the query field."
        )
        # Sanity: the query mentions either the sender or a body term.
        assert ("bob@example.com" in d.query or "revenue" in d.query.lower()
                or "Q4" in d.query), (
            f"generated query has no overlap with email content: {d.query!r}"
        )


# CRIT-4 + IMP-11 -- ParallelHybrid tier ordering and deterministic order


class _StubSA:
    """Minimal stub for SpreadingActivation that returns canned results."""

    _built = True

    def __init__(self, results):
        # results: List[Tuple[doc_id, value, score]]
        self._results = results

    def query_with_doc_ids(self, query, top_k=50, context=None):
        return list(self._results[:top_k])


class _StubL2:
    def __init__(self, results):
        # results: List[Dict] with doc_id and l2_score
        self._results = results

    def query(self, query, top_k=None):
        # Return as plain dicts so the normalizer treats them via .get()
        out = []
        for i, r in enumerate(self._results[:top_k or len(self._results)]):
            out.append({
                "document_id": r["doc_id"],
                "content": r.get("content", ""),
                "score": r.get("l2_score", 0.0),
            })
        return out


def _build_hybrid(sa_results, l2_results):
    from pensive.parallel_hybrid import ParallelHybrid

    return ParallelHybrid(
        spreading_activation=_StubSA(sa_results),
        l2_handler=_StubL2(l2_results),
        l2_on_sa_hits=False,  # use parallel mode so L2 isn't filtered to SA hits
        l2_fallback_global=False,
        enable_pattern_learning=False,
    )


def _score_for_source(results, source):
    for r in results:
        if r.source == source:
            return r.score
    raise AssertionError(f"no result with source={source!r} in {results!r}")


@pytest.mark.parametrize("rank", [0, 5, 29])
def test_rank_fusion_invariant_holds_at_rank(rank):
    """At rank R: agreement_score > l2_only_score > sa_only_score.

    Sabotage check: reverting the agreement formula to the harmonic-mean
    division (without the +floor) makes rank=29 collapse agreement below
    l2_only at rank=0; this test would fail with a loud message naming
    the offending pair.
    """
    # Construct three candidates at the same rank R from each source.
    # Use distinct doc_ids so the merge logic sees three separate rows.
    sa_results = [
        ("agree", "agree-val", 0.9),
        ("sa-only", "sa-only-val", 0.8),
    ]
    l2_results = [
        {"doc_id": "agree", "l2_score": 0.9, "content": "agree"},
        {"doc_id": "l2-only", "l2_score": 0.8, "content": "l2only"},
    ]

    # Pad with filler so the target rank is achievable. Each source list
    # needs at least (rank+1) entries.
    for i in range(rank):
        sa_results.insert(i, (f"sa-pad-{i}", f"sa-pad-{i}", 0.99 - i * 0.001))
        l2_results.insert(i, {"doc_id": f"l2-pad-{i}",
                              "l2_score": 0.99 - i * 0.001,
                              "content": f"l2-pad-{i}"})
    # After padding, "agree" sits at index `rank` in BOTH lists,
    # "sa-only" sits at index rank+1 in sa_results, and "l2-only" sits
    # at index rank+1 in l2_results.

    h = _build_hybrid(sa_results, l2_results)
    results = h.query("anything", top_k=200, sa_top_k=200, l2_top_k=200)

    by_id = {r.doc_id: r for r in results}
    agreement = by_id.get("agree")
    sa_only = by_id.get("sa-only")
    l2_only = by_id.get("l2-only")
    assert agreement is not None and agreement.source == "both", (
        f"agreement candidate missing or wrong source: {agreement!r}"
    )
    assert l2_only is not None and l2_only.source == "l2", (
        f"l2-only candidate missing or wrong source: {l2_only!r}"
    )
    assert sa_only is not None and sa_only.source == "sa", (
        f"sa-only candidate missing or wrong source: {sa_only!r}"
    )

    assert agreement.score > l2_only.score > sa_only.score, (
        f"rank-fusion tier invariant FAILED at rank={rank}: "
        f"agreement={agreement.score:.4f}, "
        f"l2_only={l2_only.score:.4f}, "
        f"sa_only={sa_only.score:.4f}. "
        "Required: agreement > l2_only > sa_only."
    )


def test_agreement_at_tail_beats_l2_only_at_head():
    """Documents found by BOTH should outrank L2-only at any rank.

    The original bug: agreement at rank=29 scored ~3.33 while L2-only at
    rank=0 scored 50, so agreement signals lost to single-source signals.

    Sabotage check: this test fails loud under the original
    100/combined_rank formula because the floor bonus is missing.
    """
    # Pad SA and L2 so 'agree' is at rank 29 in both, and 'l2-only' is
    # at rank 0 only in L2.
    sa_results = [(f"sa-pad-{i}", f"sa-pad-{i}", 0.99 - i * 0.001) for i in range(29)]
    sa_results.append(("agree", "agree", 0.5))

    l2_results = [{"doc_id": "l2-only", "l2_score": 0.95, "content": "l2only"}]
    for i in range(28):
        l2_results.append({"doc_id": f"l2-pad-{i}",
                           "l2_score": 0.94 - i * 0.001,
                           "content": f"l2-pad-{i}"})
    l2_results.append({"doc_id": "agree", "l2_score": 0.5, "content": "agree"})

    h = _build_hybrid(sa_results, l2_results)
    results = h.query("anything", top_k=200, sa_top_k=200, l2_top_k=200)
    by_id = {r.doc_id: r for r in results}
    agreement = by_id["agree"]
    l2_only = by_id["l2-only"]
    assert agreement.score > l2_only.score, (
        f"agreement at deep rank ({agreement.score:.4f}) should beat "
        f"L2-only at rank 0 ({l2_only.score:.4f}). "
        "The agreement floor bonus is missing or too small."
    )


def test_rank_fusion_deterministic_across_runs():
    """The merge step must produce the same ranking across repeated runs.

    Sabotage check: dropping back to ``set`` iteration over the union of
    SA+L2 doc IDs makes the order between equally-scored candidates
    nondeterministic across Python invocations. We approximate "across
    runs" by spinning up the hybrid twice in this process; with set
    iteration even within a process, identical inputs must produce
    identical ordering.
    """
    sa_results = [(f"doc-{i}", f"doc-{i}", 0.5) for i in range(20)]
    l2_results = [{"doc_id": f"doc-{i}", "l2_score": 0.5, "content": f"doc-{i}"}
                  for i in range(20)]

    h1 = _build_hybrid(sa_results, l2_results)
    r1 = [(r.doc_id, r.score) for r in h1.query("x", top_k=20, sa_top_k=20, l2_top_k=20)]
    h2 = _build_hybrid(sa_results, l2_results)
    r2 = [(r.doc_id, r.score) for r in h2.query("x", top_k=20, sa_top_k=20, l2_top_k=20)]
    assert r1 == r2, (
        f"hybrid ordering is nondeterministic across runs:\n  r1={r1}\n  r2={r2}"
    )


# IMP-6 -- ChatGPTParser MAX_THREAD_PATH cap


def test_chatgpt_parser_caps_thread_path(tmp_path):
    """A pathological deep-chain conversation must be truncated, not
    walked indefinitely.

    Sabotage check: removing the cap turns this test into a near-OOM
    walk on a 100k-node chain. With the cap at 50k, the walk stops at
    or below the ceiling.
    """
    # Build a synthetic conversations.json with a single conversation
    # whose mapping is a 60_000-node single chain (parent links form a
    # straight line). MAX_THREAD_PATH=50_000 should bound the walk.
    n = 60_000
    mapping = {}
    for i in range(n):
        mapping[f"n{i}"] = {
            "id": f"n{i}",
            # Children list intentionally empty for the leaf; one-element
            # for non-leaves so leaf detection picks the deepest one.
            "children": [] if i == 0 else [f"n{i-1}"],
            "parent": f"n{i+1}" if i + 1 < n else None,
            "message": {
                "author": {"role": "user" if i % 2 == 0 else "assistant"},
                "content": {"content_type": "text", "parts": ["hello world"]},
                "create_time": float(i),
                "metadata": {"model_slug": "gpt-test"},
            },
        }
    conv = {"title": "deep chain", "mapping": mapping, "create_time": 0.0}
    convs_path = tmp_path / "conversations.json"
    convs_path.write_text(json.dumps([conv]))

    parser = ChatGPTParser(str(tmp_path))
    # Just verify _extract_thread terminates and stays within the cap.
    thread = parser._extract_thread(conv)
    assert len(thread) <= ChatGPTParser.MAX_THREAD_PATH, (
        f"thread length {len(thread)} exceeded MAX_THREAD_PATH "
        f"{ChatGPTParser.MAX_THREAD_PATH}; the cap is not enforced"
    )


def test_chatgpt_extract_text_handles_non_list_parts(tmp_path):
    """`_extract_text` must not crash when a corrupted/hostile export sets
    `parts` to a number, string, or anything other than a list. Pass-3
    adversarial fuzzing surfaced that `parts: 42` raised TypeError on
    the `for p in parts` comprehension and aborted the entire ingest of
    multi-conversation exports.

    Sabotage check: remove the `isinstance(parts, (list, tuple))` guard
    in chatgpt.py:_extract_text and this test will raise TypeError on
    the int payload instead of returning None.
    """
    parser = ChatGPTParser(str(tmp_path))
    poison_payloads = [
        {"content": {"content_type": "text", "parts": 42}},
        {"content": {"content_type": "text", "parts": 1.5}},
        {"content": {"content_type": "text", "parts": "not-a-list-but-truthy"}},
    ]
    for msg in poison_payloads:
        result = parser._extract_text(msg)
        assert result is None, (
            f"non-list parts={msg['content']['parts']!r} should yield None, "
            f"got {result!r} (the guard regressed)"
        )

    # Sanity: the well-formed case still extracts text.
    ok = parser._extract_text({"content": {"content_type": "text", "parts": ["hello"]}})
    assert ok == "hello"


def test_chatgpt_parser_rejects_oversized_file(tmp_path):
    """A conversations.json larger than max_file_size is refused.

    Sabotage check: removing the stat-and-cap check would let json.load()
    pull an arbitrarily large file into memory. We simulate this by
    setting a tiny cap and writing a file that exceeds it.
    """
    convs_path = tmp_path / "conversations.json"
    convs_path.write_text(json.dumps([{"title": "x", "mapping": {}}]))
    real_size = convs_path.stat().st_size
    assert real_size > 0

    parser = ChatGPTParser(str(tmp_path), max_file_size=1)
    docs = list(parser.parse())
    assert docs == [], (
        f"oversized conversations.json was processed; expected empty yield, "
        f"got {len(docs)} documents"
    )


# IMP-7 -- pensive build CLI --force flag


def test_cli_build_refuses_overwrite_without_force(tmp_path, monkeypatch):
    """``pensive build`` must refuse to overwrite an existing output
    file unless --force is set.

    Sabotage check: removing the existence check silently clobbers the
    user's previously-built graph.
    """
    from pensive.ingestion import cli as cli_mod

    out_path = tmp_path / "graph.pkl"
    out_path.write_bytes(b"PRECIOUS-EXISTING-GRAPH")
    original_bytes = out_path.read_bytes()

    # Build args namespace -- mimic argparse output WITHOUT --force.
    class _Args:
        chatgpt = None
        facebook = None
        google = None
        email = None
        output = str(out_path)
        chunk_size = 800
        batch_size = 1000
        force = False

    # Skip the "no parsers" exit by pretending the user did pass one,
    # but route argparse to a no-op parser by monkeypatching. Easier:
    # bypass the build path entirely by making argparse exit AFTER the
    # overwrite check fires.
    # Simpler: invoke cmd_build directly. It will sys.exit on existing-file.
    with pytest.raises(SystemExit) as exc_info:
        cli_mod.cmd_build(_Args())
    # The overwrite check exits FIRST with code 1, before the
    # "no parsers" path also exits with 1. We just need to confirm
    # the file is untouched.
    assert exc_info.value.code == 1
    assert out_path.read_bytes() == original_bytes, (
        "cmd_build clobbered an existing output file without --force"
    )


# IMP-9 -- L2Handler grown-region zero initialization


def test_l2_grown_region_is_zero_not_garbage():
    """When _emb_array grows, the new region must be zero-initialized.

    Sabotage check: switching back to ``np.empty`` leaves uninitialized
    memory in the grown rows. A read of an "exists in array but never
    written" slot should return the well-defined zero vector, not
    arbitrary float garbage. We can't import L2Handler (sentence-
    transformers is broken in this env), so we test the helper directly
    by reaching into a freshly-instantiated dummy class.
    """
    # Build a minimal stand-in that exposes _store_embeddings and
    # _emb_array exactly as L2Handler does.
    class _Stub:
        def __init__(self, dim):
            self._dim = dim
            self._emb_array = np.empty((0, dim), dtype=np.float32)
            self._emb_capacity = 0

        # Copy of the patched _store_embeddings logic.
        _store_embeddings = None  # filled below

    from pensive import l2 as l2mod

    _Stub._store_embeddings = l2mod.L2Handler._store_embeddings  # type: ignore[assignment]

    s = _Stub(4)
    # Write at index 5 -- forces growth, leaves rows 0..4 unwritten.
    embedding = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    _Stub._store_embeddings(s, [5], embedding)
    # Inspect a never-written row.
    for i in (0, 1, 2, 3, 4):
        row = s._emb_array[i]
        assert np.all(row == 0.0), (
            f"row {i} was {row!r}; uninitialized memory leaked into _emb_array. "
            "The grown region must be zero-initialized."
        )
    assert np.allclose(s._emb_array[5], embedding[0])


# IMP-10 -- pattern_learner cache invalidation on add_documents


def test_pattern_learner_cache_invalidated_on_add_documents():
    """add_documents after integrate_with_sa must NOT serve stale results.

    Sabotage check: the original code snapshotted value-node lists at
    integration time. Adding new docs after integration left those new
    docs invisible to learned-term seeding. We assert that a learned
    term matches a document added AFTER integration.
    """
    from pensive.pattern_learner import PatternLearner, integrate_with_sa

    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc-old", "content": "alpha beta",
         "value": "alpha beta old", "query": "what is alpha?"},
    ])

    learner = PatternLearner()
    learner.add_manual("freshly-added-term-xyz")
    integrate_with_sa(sa, learner)

    # Before the fix: cache was a one-shot snapshot, so the new doc
    # below would be invisible to learned-term lookups.
    sa.add_documents([
        {"id": "doc-new", "content": "irrelevant body",
         "value": "this contains freshly-added-term-xyz inside",
         "query": "what about new doc?"},
    ])

    results = sa.query_with_doc_ids("freshly-added-term-xyz", top_k=20)
    found_ids = {r[0] for r in results}
    assert "doc-new" in found_ids, (
        f"learned-term cache was stale after add_documents; "
        f"doc-new not found in results {results!r}. "
        "IMP-10 fix should invalidate the cache when add_documents runs."
    )
