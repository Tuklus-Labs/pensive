"""Gate for what a token budget does to a ranked answer that will not fit.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    A result the ranker found is never silently dropped from the payload. When
    its full body will not fit the budget, it degrades to a Tier-0 handle line,
    so the caller can still see that the memory exists and fetch it by id.

Why. `assemblePayload` fit whole Tier-1 entries top-down and dropped the tail.
Measured 2026-08-13 over 19 probes, comparing what the ranker RETURNED against
what survived into the rendered payload:

    tier  budget   ranked   delivered
    L2      1500    0.789       0.579
    L2      4000    0.789       0.789
    L3      1500    0.789       0.684
    L3      4000    0.789       0.895

At the default budget, four of fifteen ranked answers never reached the caller.
Retrieval was fine; delivery was lossy. This is the same render-versus-rank
defect as the gist (fixed the previous day), one layer further down: the ranker
was blamed for a presentation decision.

Dropping is the wrong degradation because the two failures are indistinguishable
downstream. An absent handle reads exactly like "no such memory", which is the
answer that sends an agent off to re-derive something it already knows. A handle
line costs about 80 characters and says "this exists, here is its id" -- and
since the L1 identity route landed, that id is directly fetchable, so a named
result is a complete answer rather than a tease.

The briefer already had this right for pins: over budget, every pin degrades to
a handle and a warning line is emitted, never a silent omission.
"""
import pytest

from recall.payload import assemblePayload, HANDLE_SCHEME
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text):
    return putAtom(store, {
        "text": text, "kind": "atom", "project": "p", "importance": 0.0,
        "provenance": {"source": "explicit-emit"},
    })


def _results(ids):
    return [{"atomId": i, "score": 1.0 - n * 0.01, "confidence": 0.9,
             "shouldTrust": True, "why": "test"} for n, i in enumerate(ids)]


def test_a_ranked_result_that_does_not_fit_is_named_not_dropped(store):
    """The load-bearing case: the payload NAMES more results than it carries
    bodies for.

    Stated as a relationship rather than an absolute, because naming every
    result is not always arithmetically possible: one of these bodies is 348
    tokens on its own, so a 400-token budget can hold exactly one body and one
    handle. The claim that matters is that the tail degrades instead of
    vanishing, and the discriminator is named > bodied.
    """
    ids = [_put(store, f"body number {i} " + ("filler " * 120)) for i in range(6)]
    out, _tokens, _low = assemblePayload(store, _results(ids), tokenBudget=400)
    named = [i for i in ids if i in out]
    # A Tier-1 entry is the only thing that carries a provenance line, so
    # counting those counts BODIES. Matching on body text does not work: the
    # Tier-0 handle carries a gist, which is the opening of that same body.
    bodied = out.count("source explicit-emit")
    assert len(named) > bodied, (
        f"named {len(named)} but bodied {bodied}: the tail was dropped "
        f"rather than degraded, and an absent handle is indistinguishable from "
        f"'no such memory'"
    )


def test_a_realistic_budget_names_every_ranked_result(store):
    """At the daemon's actual default budget nothing ranked should be lost."""
    ids = [_put(store, f"body number {i} " + ("filler " * 120)) for i in range(6)]
    out, _tokens, _low = assemblePayload(store, _results(ids), tokenBudget=1500)
    missing = [i for i in ids if i not in out]
    assert not missing, f"ranked atoms absent at the default budget: {missing}"


def test_the_top_result_still_gets_its_whole_body(store):
    """Degrading the tail must not degrade the head: the best answer is the one
    the caller most needs in full."""
    top = _put(store, "THE TOP BODY that must appear in full")
    rest = [_put(store, f"tail body {i} " + ("filler " * 120)) for i in range(5)]
    out, _tokens, _low = assemblePayload(store, _results([top] + rest), tokenBudget=400)
    assert "THE TOP BODY that must appear in full" in out


def test_a_generous_budget_still_returns_whole_bodies(store):
    """Regression guard: degradation is for pressure, not the normal case."""
    ids = [_put(store, f"body number {i}") for i in range(4)]
    out, _tokens, _low = assemblePayload(store, _results(ids), tokenBudget=8000)
    for i in range(4):
        assert f"body number {i}" in out


def test_every_degraded_entry_is_a_parseable_handle(store):
    """A named result is only useful if the id can be read back out and handed
    to the L1 identity route."""
    ids = [_put(store, f"body number {i} " + ("filler " * 120)) for i in range(6)]
    out, _tokens, _low = assemblePayload(store, _results(ids), tokenBudget=1500)
    for i in ids:
        assert f"{HANDLE_SCHEME}{i}" in out


def test_payload_still_respects_the_budget(store):
    """Naming the tail must not become a way to blow the budget instead."""
    from recall.payload import estimateTokens
    ids = [_put(store, f"body number {i} " + ("filler " * 120)) for i in range(40)]
    out, _tokens, _low = assemblePayload(store, _results(ids), tokenBudget=400)
    assert estimateTokens(out) <= 400
