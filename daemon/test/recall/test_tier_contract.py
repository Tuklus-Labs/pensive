"""Gate for the L2/L3 tier contract.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    L2 and L3 differ in WHICH CORPUS they read, and an unknown tier is refused
    rather than silently served something else.

Why this file exists. A clean-pass audit found that NOTHING tested this. Zero
tests passed a `tier` argument to `recall`, zero referenced `tierDefaults`, and
the L3 branch plus the unknown-tier `raise` never executed. The tier structure is
the central deliverable of this campaign, `Aegis/CLAUDE.md:113` says "Do not
break the tier structure", and `engine.py`'s own docstring calls the corpus
boundary "the difference between the tiers". It was verified only by `tiergate`,
which needs a live daemon and a quiet box, so a refactor that broke tier routing
would have passed the entire unit suite.

Worse, the one existing assertion that mentions a tier pins the string
"mcp.recall.L2" for the tier that is ALSO the hardcoded default, so hardcoding
the literal would have passed it.
"""
import pytest

from recall.engine import tierDefaults, recall, MEMORY_KINDS, TIERS, DEFAULT_TIER


# --------------------------------------------------------------------------- #
# tierDefaults: the routing table itself                                        #
# --------------------------------------------------------------------------- #


def test_l2_reads_memory_kinds_only():
    d = tierDefaults("L2")
    assert set(d["kinds"]) == set(MEMORY_KINDS), (
        "L2 must restrict to memory kinds; it is the tier that keeps the 283k "
        f"document_chunk corpus out of an agent's default retrieve. got {d['kinds']}")


def test_l3_reads_every_kind():
    d = tierDefaults("L3")
    assert d["kinds"] is None, (
        "L3 must place NO kind restriction: some answers exist only in chunk "
        f"form, so this is a boundary and never a deletion. got {d['kinds']}")


def test_neither_tier_runs_the_cross_encoder():
    """House rule (Gary 2026-08-13): batch encodes on the GPU, stream encodes on
    the CPU. A cross-encoder pass is a stream encode and measured 3164ms for 50
    real pairs on CPU, which is 25x L3's entire budget."""
    for tier in TIERS:
        assert tierDefaults(tier)["rerank"] is False, (
            f"tier {tier} enabled the cross-encoder on the serve path; it "
            "belongs out of band, never blocking first token")


@pytest.mark.parametrize("bad", ["L1", "L4", "l2", "", "L2 ", "memory", None, 2])
def test_an_unknown_tier_is_refused_not_defaulted(bad):
    """`tierDefaults` raises rather than falling back, deliberately: a caller
    naming a tier this engine does not serve holds a wrong belief about the
    contract, and silently serving something else is how that belief survives.

    L1 is included because it is a REAL tier that this function must still
    refuse: identity retrieval has no ranking stage and lives on the lean HTTP
    route, because the MCP tools/call envelope alone measured 2.486ms P95
    against a 1ms budget."""
    with pytest.raises(ValueError):
        tierDefaults(bad)


def test_the_default_tier_is_one_this_engine_serves():
    assert DEFAULT_TIER in TIERS
    tierDefaults(DEFAULT_TIER)


# --------------------------------------------------------------------------- #
# the boundary, end to end through recall()                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def embedderFixture():
    """Session-shared like the other recall tests: the model is ~130MB and
    loading it per test dominates the run."""
    from recall.embedder import Embedder

    return Embedder("BAAI/bge-small-en-v1.5")


@pytest.fixture
def mixedStore(tmp_path):
    """A store holding BOTH a memory atom and a document_chunk whose text
    answers the same query, so the tiers can actually differ."""
    from store.store import openStore, putAtom

    s = openStore(tmp_path / "tier.db")
    ids = {}
    ids["atom"] = putAtom(s, {
        "text": "the survey line spacing decision and why it was reduced",
        "kind": "atom", "project": "aegis", "importance": 0.0,
        "provenance": {"source": "explicit-emit"}})
    ids["chunk"] = putAtom(s, {
        "text": "def surveyLineSpacing(): return 40  # the survey line spacing constant",
        "kind": "document_chunk", "project": "aegis", "importance": 0.0,
        "provenance": {"source": "bulk-import"}})
    try:
        yield s, ids
    finally:
        s.close()


def test_l3_can_return_a_chunk_and_l2_cannot(mixedStore, embedderFixture):
    """THE tier contract, end to end. This is the assertion the campaign built
    the tier structure for and never wrote.

    Asserts INCLUSION before exclusion: L3 must actually return the chunk, or
    the L2 assertion proves nothing (the chunk could be absent from both for an
    unrelated reason)."""
    from recall.vector_index import buildClassIndexes

    store, ids = mixedStore
    indexes = buildClassIndexes(store, embedderFixture.modelId)
    query = "survey line spacing"

    l3 = recall(store, indexes, embedderFixture, query, k=10, tier="L3")
    l3ids = {r["atomId"] for r in l3["results"]}
    assert ids["chunk"] in l3ids, (
        "THIS TEST HAS NO POWER: L3 did not return the chunk at all, so the L2 "
        f"assertion below cannot fail. L3 returned {l3ids}")

    l2 = recall(store, indexes, embedderFixture, query, k=10, tier="L2")
    l2ids = {r["atomId"] for r in l2["results"]}
    assert ids["chunk"] not in l2ids, (
        "L2 returned a document_chunk. L2 exists to keep the bulk corpus out of "
        f"an agent's default retrieve; the boundary is gone. L2 returned {l2ids}")
