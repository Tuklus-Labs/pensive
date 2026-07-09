"""Recall orchestrator + tiered no-slop payload (Task 10).

Risk model (what could silently break, and the test that catches it):

- **Pipeline mis-wiring.** A stage skipped or reordered; the time window applied
  twice (facet + priors); ``now`` sampled more than once; signalHits built wrong
  so trust's evidence is off. Caught by the full-pipeline tests over real models
  (100-atom ordering, timeScope-restricts, entity-facet-threads-into-why,
  superseded-after-index-build) which exercise every stage end to end.
- **Payload slop.** Markdown furniture (headers, bullets, bold) or emoji leaking
  into the lines WE emit; a body cut mid-sentence to fit budget; a Tier-0 gist
  keeping newlines and breaking the one-line handle grammar. Caught by the
  clean-corpus furniture check, the adversarial verbatim/gist check, and the
  atomic-drop budget check.
- **Budget arithmetic.** Off-by-one letting the payload exceed budget, or the
  degrade path (handle-only, then sentinel) not firing. Caught by the tiny-budget
  and top-entry-too-big tests, asserting the module's own heuristic.
- **Sabotage / degenerate inputs.** Empty/whitespace query, a project that
  matches nothing, a kinds filter that excludes everything, a superseded atom in
  results, a Tier-2 edge to a dead atom. Each has a dedicated test asserting the
  low-confidence sentinel (never a crash, never weak padding) or the correct drop.

Loudness: assertions pin exact bodies present/absent (not just lengths), exact
sentinel strings, the exact token bound, and reranked ordering -- a silent
regression in any stage flips a concrete assertion, not a fuzzy one.

Real components end to end: the model loads via the same session-fixture pattern
as test_signals / test_rerank (one embedder + one reranker for the whole session,
a fresh SQLite store per test). Payload-only tests need no model and construct
result dicts by hand over a live store.
"""
import re

import pytest

import recall.engine as engine
from recall.engine import recall, FACET_BOOST, RERANK_HEAD
from recall.payload import (
    assemblePayload,
    assembleTier2,
    tier0Handle,
    tier1Entry,
    estimateTokens,
    SENTINEL_LOW_CONFIDENCE,
    SENTINEL_BUDGET_TOO_SMALL,
    GIST_CHARS,
)
from recall.embedder import Embedder, embedMissing
from recall.vector_index import FlatIndex, buildClassIndexes
from store.store import openStore, putAtom, addEdge, supersede, addFacet

MODEL_ID = "BAAI/bge-small-en-v1.5"

# Fixed reference time so occurred_at-derived dates are deterministic. 2_000_000_000
# unix seconds is 2033-05-18 UTC.
NOW = 2_000_000_000
NOW_DATE = "2033-05-18"
DAY = 86_400

# The GPU stack emits two SwigPy DeprecationWarnings on first import under CPython
# 3.14 (generated SWIG glue, not our code); filter exactly those, mirroring the
# other recall tests, so any real warning still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)

# Emoji ranges we forbid in the FURNITURE we generate (bodies render verbatim and
# may legitimately contain emoji, so this is only ever applied to our own lines).
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U00002B00-\U00002BFF"
    "️"
    "]"
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture(scope="session")
def _rerankerWarm():
    # Page in the ~1.1GB cross-encoder once per session (it shares VRAM with the
    # embedder and this box's llama-servers); every reranking test depends on it.
    from recall.rerank import _getReranker

    _getReranker()


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def indexes(store, embedder):
    embedMissing(store, embedder)
    return buildClassIndexes(store, MODEL_ID)


def _put(store, text, project="aegis", kind="atom", occurredAt=None,
         importance=0.0, source="claude-code", agent=None, sessionId=None):
    prov = {"source": source}
    if agent is not None:
        prov["agent"] = agent
    if sessionId is not None:
        prov["sessionId"] = sessionId
    atomInput = {
        "text": text,
        "kind": kind,
        "project": project,
        "importance": importance,
        "provenance": prov,
    }
    if occurredAt is not None:
        atomInput["occurredAt"] = occurredAt
    return putAtom(store, atomInput)


def _result(atomId, confidence, shouldTrust, score=1.0, supersededBy=None, why="why"):
    row = {
        "atomId": atomId,
        "score": score,
        "confidence": confidence,
        "shouldTrust": shouldTrust,
        "why": why,
    }
    if supersededBy is not None:
        row["supersededBy"] = supersededBy
    return row


def _index(store, embedder):
    embedMissing(store, embedder)
    return FlatIndex().build(store, MODEL_ID)


def _indexes(store, embedder):
    embedMissing(store, embedder)
    return buildClassIndexes(store, MODEL_ID)


# Opaque per-class sentinels for stub tests: dense is monkeypatched and ignores
# its index arg, so these only need to make indexes.get(name) truthy so the
# stubbed dense is still exercised once per class.
_SENTINEL = object()
_STUB_INDEXES = {"memory": _SENTINEL, "code": _SENTINEL}


class _BombIndex:
    """Every method explodes: proves a code path never touched the index."""

    def search(self, vec, k):
        raise AssertionError("index.search must not be called")

    def build(self, store, modelId, kinds=None):
        raise AssertionError("index.build must not be called")


class _BombEmbedder:
    def embed(self, texts):
        raise AssertionError("embedder.embed must not be called")


_BASE_SENTENCES = [
    "the pressure hull rates to {i} meters of depth",
    "acoustic modems trade range for data rate at station {i}",
    "LiFePO4 batteries tolerate cold better than NMC in pack {i}",
    "a doppler velocity log gives bottom-lock speed on run {i}",
    "USBL beacons provide an absolute position fix number {i}",
    "multibeam sonar sweeps a wide bathymetric swath on line {i}",
    "the thruster controller runs a PID loop at {i} hertz",
    "biofouling attaches to the titanium hull within week {i}",
    "the surface relay buoy forwards a telemetry packet {i}",
    "the extended kalman filter fuses INS and DVL at tick {i}",
]


def _hundred_texts():
    return [_BASE_SENTENCES[i % len(_BASE_SENTENCES)].format(i=i) for i in range(100)]


def _stub_recall_dependencies(monkeypatch, fused, reranked):
    seen = {}

    monkeypatch.setattr(engine, "facetSignal", lambda store, hints: {
        "boostSet": set(),
        "filterSet": None,
    })
    monkeypatch.setattr(engine, "bm25", lambda store, query, k, kinds=None: [])
    monkeypatch.setattr(engine, "dense", lambda index, embedder, query, k: [])
    monkeypatch.setattr(engine, "rrf", lambda hits: list(fused))
    monkeypatch.setattr(engine, "applyPriors", lambda fusedPairs, store, hints: fusedPairs)
    # These tail-append tests use synthetic fused ids that are not real atoms, so
    # the real splitByClass (a store SELECT) would drop them all. Stub it to a
    # single populated bucket in fused order; the real interleave then builds the
    # head as that list's prefix, exactly the old fused[:RERANK_HEAD]. Cross-class
    # interleaving is covered by test_rerank_head_is_interleaved_across_classes.
    monkeypatch.setattr(engine, "splitByClass", lambda fusedPairs, store, classNames: {
        name: (list(fusedPairs) if i == 0 else [])
        for i, name in enumerate(classNames)
    })

    def rerankSpy(query, candidates, store):
        seen["rerank_candidates"] = list(candidates)
        return list(reranked)

    def assessSpy(rerankedPairs, signalHits, store, now):
        seen["assessed_pairs"] = list(rerankedPairs)
        return [
            _result(atomId, 0.9, True, score=score)
            for atomId, score in rerankedPairs
        ]

    monkeypatch.setattr(engine, "rerank", rerankSpy)
    monkeypatch.setattr(engine, "assessTrust", assessSpy)
    monkeypatch.setattr(
        engine,
        "assemblePayload",
        lambda store, results, tokenBudget: ("payload", 1, False),
    )
    return seen


# --------------------------------------------------------------------------- #
# Stratified recall: both classes reach the reranker; the head interleaves      #
# --------------------------------------------------------------------------- #


def test_memory_intent_query_surfaces_memory_over_code(
        store, embedder, indexes, _rerankerWarm):
    # A store flooded with code chunks (the real-world 20:1 shape, scaled down)
    # plus a few reasoning-memory atoms. A memory-shaped query must return a
    # memory-kind atom at the top, not a code chunk, because both classes reached
    # the reranker and the reranker sorted on relevance.
    for i in range(40):
        _put(store, f"def handler_{i}(req): return route(req, table_{i})",
             kind="document_chunk")
    good = _put(store,
                "we decided to migrate the ingest pipeline to stratified recall "
                "next sprint because the code corpus was starving memory",
                kind="atom")
    idx = buildClassIndexes(store, MODEL_ID)  # rebuild after inserts
    out = recall(store, idx, embedder,
                 "what did we decide about the ingest pipeline migration", k=5)
    topId = out["results"][0]["atomId"]
    assert topId == good


def test_code_intent_query_surfaces_the_right_chunk(
        store, embedder, _rerankerWarm):
    for i in range(20):
        _put(store, f"notes on sprint planning meeting number {i}", kind="atom")
    target = _put(store,
                  "def _compile(self): acquire build_lock then publish the csr "
                  "matrix only if the graph generation has not moved",
                  kind="document_chunk")
    idx = buildClassIndexes(store, MODEL_ID)
    out = recall(store, idx, embedder,
                 "csr matrix compile build lock generation publish", k=5)
    topId = out["results"][0]["atomId"]
    assert topId == target


def test_rerank_head_is_interleaved_across_classes(store, monkeypatch):
    # White-box: with equal-length per-class fused lists, the head handed to
    # rerank must alternate memory, code, memory, code ... (choke point 2 fixed).
    mem = [_put(store, f"memory {i}", kind="atom") for i in range(5)]
    code = [_put(store, f"code {i}", kind="document_chunk") for i in range(5)]
    # fused order: all memory first, then all code. Naive fused[:RERANK_HEAD] would
    # put every memory ahead of every code; interleave must alternate them.
    fused = [(a, 100 - i) for i, a in enumerate(mem)] + \
            [(a, 50 - i) for i, a in enumerate(code)]

    seen = {}
    monkeypatch.setattr(engine, "facetSignal",
                        lambda store, hints: {"boostSet": set(), "filterSet": None})
    monkeypatch.setattr(engine, "bm25", lambda store, query, k, kinds=None: [])
    monkeypatch.setattr(engine, "dense", lambda index, embedder, query, k: [])
    monkeypatch.setattr(engine, "rrf", lambda hits: list(fused))
    monkeypatch.setattr(engine, "applyPriors",
                        lambda fusedPairs, store, hints: fusedPairs)

    def rerankSpy(query, candidates, store):
        seen["head"] = [a for a, _ in candidates]
        return list(candidates)

    monkeypatch.setattr(engine, "rerank", rerankSpy)
    monkeypatch.setattr(engine, "assessTrust",
                        lambda pairs, hits, store, now: [
                            _result(a, 0.9, True, score=s) for a, s in pairs])
    monkeypatch.setattr(engine, "assemblePayload",
                        lambda store, results, tokenBudget: ("payload", 1, False))

    recall(store, {"memory": None, "code": None}, _BombEmbedder(),
           "irrelevant", k=10)
    head = seen["head"]
    # First four head slots alternate memory, code, memory, code.
    assert head[0] in mem and head[1] in code
    assert head[2] in mem and head[3] in code


# --------------------------------------------------------------------------- #
# Step 1: full pipeline over a seeded 100-atom store                          #
# --------------------------------------------------------------------------- #


def test_recall_100_atoms_ordered_with_confidence_within_budget(
    store, embedder, _rerankerWarm
):
    # Brief Step 1: recall over 100 seeded atoms returns results in reranked
    # (relevance) order -- the order assessTrust preserves -- each carrying a
    # confidence in [0,1] and a why, and the assembled payload's token count by the
    # module's own heuristic is <= tokenBudget.
    for text in _hundred_texts():
        _put(store, text)
    indexes = _indexes(store, embedder)

    out = recall(
        store, indexes, embedder,
        "acoustic modems trade range for data rate",
        k=10, tokenBudget=1500,
    )

    results = out["results"]
    assert results                       # something came back
    assert len(results) <= 10
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)   # reranked order preserved
    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["shouldTrust"], bool)
        assert r["why"]
    # budget honored by the module's own accounting, and tokensUsed IS that count
    assert out["tokensUsed"] <= 1500
    assert out["tokensUsed"] == estimateTokens(out["payload"])
    assert isinstance(out["lowConfidence"], bool)
    # the strongest hit for this exact-phrase query is a "station" atom
    assert "range for data rate" in out["payload"]


def test_recall_reranks_only_head_and_appends_tail_in_rrf_order(
    store, monkeypatch
):
    poolSize = RERANK_HEAD + 5
    fused = [(f"a{i}", float(poolSize - i)) for i in range(poolSize)]
    rerankedHead = list(reversed(fused[:RERANK_HEAD]))
    seen = _stub_recall_dependencies(monkeypatch, fused, rerankedHead)

    out = recall(store, _STUB_INDEXES, _BombEmbedder(), "query", k=poolSize)

    expected = rerankedHead + fused[RERANK_HEAD:]
    assert seen["rerank_candidates"] == fused[:RERANK_HEAD]
    assert seen["assessed_pairs"] == expected
    assert [r["atomId"] for r in out["results"]] == [atomId for atomId, _ in expected]
    assert len(out["results"]) == len(fused)


def test_recall_reranks_whole_pool_when_pool_is_below_head_cap(
    store, monkeypatch
):
    fused = [(f"a{i}", float(10 - i)) for i in range(10)]
    reranked = list(reversed(fused))
    seen = _stub_recall_dependencies(monkeypatch, fused, reranked)

    out = recall(store, _STUB_INDEXES, _BombEmbedder(), "query", k=len(fused))

    assert seen["rerank_candidates"] == fused
    assert seen["assessed_pairs"] == reranked
    assert [r["atomId"] for r in out["results"]] == [atomId for atomId, _ in reranked]


# --------------------------------------------------------------------------- #
# Wire-through: timeScope restricts to the window, applied exactly once        #
# --------------------------------------------------------------------------- #


def test_timescope_restricts_results_to_the_window(store, embedder, _rerankerWarm):
    # timeScope is applied ONCE, in the priors stage (never also as a facet
    # timeRange). Seed topically-identical atoms in and out of the window; only the
    # in-window atoms may come back.
    windowStart, windowEnd = 1_000_000_000, 1_100_000_000
    mid = 1_050_000_000
    outTime = 1_500_000_000
    inIds = {
        _put(store, f"sonar bathymetry survey grid line {i}", occurredAt=mid)
        for i in range(4)
    }
    outIds = {
        _put(store, f"sonar bathymetry survey grid line out {i}", occurredAt=outTime)
        for i in range(4)
    }
    indexes = _indexes(store, embedder)

    out = recall(
        store, indexes, embedder, "sonar bathymetry survey grid",
        timeScope=(windowStart, windowEnd), k=10,
    )

    got = {r["atomId"] for r in out["results"]}
    assert got                                  # some in-window atoms returned
    assert got <= inIds                         # ONLY in-window atoms
    assert got.isdisjoint(outIds)


# --------------------------------------------------------------------------- #
# Wire-through: an entity facet threads into signalHits -> trust "why"         #
# --------------------------------------------------------------------------- #


def test_entity_facet_threads_into_the_why(store, embedder, _rerankerWarm):
    # A query naming an entity ("pensive") boosts the atom carrying that entity
    # facet, and that atom's result must report the facet as evidence in its why --
    # proving boostSet -> signalHits("facet") -> assessTrust wiring end to end.
    hit = _put(store, "notes about the pensive memory engine and its recall path")
    addFacet(store, hit, "entity", "pensive")
    _put(store, "unrelated acoustic modem range note")
    indexes = _indexes(store, embedder)

    out = recall(store, indexes, embedder, "what did we learn about pensive recall", k=10)

    byId = {r["atomId"]: r for r in out["results"]}
    assert hit in byId
    assert "facet match" in byId[hit]["why"]


# --------------------------------------------------------------------------- #
# Realistic superseded surfacing: a stale index returns a since-superseded atom #
# --------------------------------------------------------------------------- #


def test_superseded_after_index_build_surfaces_chained_provenance(
    store, embedder, _rerankerWarm
):
    # The index is built while both atoms are live; the old atom is superseded
    # AFTER, so the (stale) dense index still returns it while bm25's query-time
    # live filter does not. The trust layer must surface it chained to its live
    # successor, and the payload must render that "superseded by" provenance line.
    old = _put(store, "the pressure hull rates to 488 meters of depth")
    new = _put(store, "the pressure hull now rates to 500 meters of depth")
    indexes = _indexes(store, embedder)      # both live, both indexed
    supersede(store, old, new, {"source": "claude-code"})  # index now stale for old

    out = recall(store, indexes, embedder, "pressure hull depth rating", k=10)

    byId = {r["atomId"]: r for r in out["results"]}
    assert old in byId
    assert byId[old]["supersededBy"] == new
    assert byId[old]["shouldTrust"] is False
    assert new in byId
    assert byId[new]["shouldTrust"] is True       # a live successor is trustworthy
    # the chained provenance line renders in the Tier-1 payload
    assert f"superseded by p3://{new}" in out["payload"]


# --------------------------------------------------------------------------- #
# Sabotage: degenerate filters short-circuit to the sentinel, never crash       #
# --------------------------------------------------------------------------- #


def test_project_matching_nothing_returns_sentinel_without_touching_models(store):
    # A project with no live atoms yields an empty filterSet -> the low-confidence
    # sentinel, BEFORE any signal runs. Bomb index/embedder prove no model work.
    _put(store, "an aegis note", project="aegis")

    out = recall(store, {"memory": _BombIndex(), "code": _BombIndex()},
                 _BombEmbedder(), "note", project="ghost", k=10)

    assert out["results"] == []
    assert out["payload"] == SENTINEL_LOW_CONFIDENCE
    assert out["lowConfidence"] is True
    assert out["tokensUsed"] == estimateTokens(SENTINEL_LOW_CONFIDENCE)


def test_kinds_filter_excluding_everything_returns_sentinel(store, embedder):
    # A kinds filter naming no known kind yields no classes, so recall short-
    # circuits to the sentinel before any signal runs (neither dense nor the
    # reranker execute). The embedder is only used to build the fixture indexes.
    for i in range(5):
        _put(store, f"acoustic modem note {i}", kind="atom")
    indexes = _indexes(store, embedder)

    out = recall(store, indexes, embedder, "acoustic modem", kinds=["nonexistent_kind"], k=10)

    assert out["results"] == []
    assert out["payload"] == SENTINEL_LOW_CONFIDENCE
    assert out["lowConfidence"] is True


def test_whitespace_query_returns_sentinel_without_touching_models(store):
    # Empty/whitespace query short-circuits to the sentinel. dense() would embed
    # whitespace to a valid unit vector and return the whole index, so the guard is
    # load-bearing -- bomb models prove the guard fires before any signal.
    _put(store, "some content")

    for q in ["", "   ", "\t\n  "]:
        out = recall(store, {"memory": _BombIndex(), "code": _BombIndex()},
                     _BombEmbedder(), q, k=10)
        assert out["results"] == []
        assert out["payload"] == SENTINEL_LOW_CONFIDENCE
        assert out["lowConfidence"] is True


# --------------------------------------------------------------------------- #
# Payload: token heuristic + handle/provenance format                         #
# --------------------------------------------------------------------------- #


def test_estimate_tokens_is_ceil_len_over_three():
    assert estimateTokens("") == 0
    assert estimateTokens("abc") == 1
    assert estimateTokens("abcd") == 2          # ceil(4/3)
    assert estimateTokens("a" * 3000) == 1000


def test_tier0_handle_format(store):
    aid = _put(store, "a short body about sonar", occurredAt=NOW)
    handle = tier0Handle(store, _result(aid, 0.73, True))

    assert "\n" not in handle
    assert handle.startswith(f"p3://{aid} | ")
    parts = handle.split(" | ")
    assert len(parts) == 4
    assert parts[0] == f"p3://{aid}"
    assert parts[1] == NOW_DATE
    assert parts[2] == "0.73"                   # confidence to 2 decimals
    assert "a short body about sonar" in parts[3]


def test_provenance_line_prefers_agent_then_session_then_unknown(store):
    withAgent = _put(store, "body a", agent="heph")
    entryA = tier1Entry(store, _result(withAgent, 0.5, True))
    assert "source claude-code, heph, recorded " in entryA

    withSession = _put(store, "body b", source="codex", sessionId="sess-1")
    entryB = tier1Entry(store, _result(withSession, 0.5, True))
    assert "source codex, sess-1, recorded " in entryB

    neither = _put(store, "body c", source="bulk-import")
    entryC = tier1Entry(store, _result(neither, 0.5, True))
    assert "source bulk-import, unknown, recorded " in entryC


def test_tier1_entry_is_handle_body_provenance(store):
    aid = _put(store, "the full stored body text", occurredAt=NOW)
    entry = tier1Entry(store, _result(aid, 0.60, True))
    lines = entry.split("\n")
    assert lines[0].startswith(f"p3://{aid} | ")
    assert lines[1] == "the full stored body text"     # body verbatim, its own line
    assert lines[2].startswith("source ")


# --------------------------------------------------------------------------- #
# Payload: budget enforcement is atomic (drop the tail, never cut a body)      #
# --------------------------------------------------------------------------- #


def test_tiny_budget_drops_tail_never_cuts_a_body(store):
    # Brief Step 5: a budget that fits exactly two Tier-1 entries drops the third
    # WHOLE -- the third body (and even its handle) is absent entirely, and the
    # payload equals the two-entry payload byte for byte.
    b1 = "AAAA body one distinctive alpha content here " * 2
    b2 = "BBBB body two distinctive bravo content here " * 2
    b3 = "CCCC body three distinctive charlie content here " * 2
    i1, i2, i3 = _put(store, b1), _put(store, b2), _put(store, b3)
    r1 = _result(i1, 0.9, True, score=5.0)
    r2 = _result(i2, 0.8, True, score=4.0)
    r3 = _result(i3, 0.7, True, score=3.0)

    twoPayload, twoTokens, _ = assemblePayload(store, [r1, r2], 10 ** 9)
    payload, tokens, low = assemblePayload(store, [r1, r2, r3], twoTokens)

    assert low is False
    assert payload == twoPayload            # third entry dropped whole
    assert b1 in payload and b2 in payload
    assert b3 not in payload                # not even a fragment of the body
    assert i3 not in payload                # nor its handle
    assert tokens <= twoTokens


def test_top_entry_too_big_degrades_to_handle_then_sentinel(store):
    big = "x" * 600
    aid = _put(store, big)
    r = _result(aid, 0.9, True)
    entry = tier1Entry(store, r)
    handle = tier0Handle(store, r)
    et, ht = estimateTokens(entry), estimateTokens(handle)
    assert ht < et                          # the handle is strictly smaller

    payload, tokens, low = assemblePayload(store, [r], ht)   # fits handle, not entry
    assert payload == handle
    assert low is False
    assert tokens <= ht

    payload2, _tokens2, _low2 = assemblePayload(store, [r], ht - 1)  # fits neither
    assert payload2 == SENTINEL_BUDGET_TOO_SMALL


# --------------------------------------------------------------------------- #
# Payload: low confidence emits the sentinel + Tier-0 only, never weak bodies   #
# --------------------------------------------------------------------------- #


def test_none_trusted_emits_sentinel_and_tier0_not_weak_bodies(store):
    # Zero trusted results -> the sentinel plus up to 3 Tier-0 handles of the best
    # untrusted hits, and NEVER a Tier-1 body. Bodies run past the gist so their
    # tail marker can only appear if a full body were (wrongly) rendered.
    b1 = "x" * (GIST_CHARS + 5) + "TAILMARKERONE"
    b2 = "y" * (GIST_CHARS + 5) + "TAILMARKERTWO"
    i1, i2 = _put(store, b1), _put(store, b2)
    r1 = _result(i1, 0.5, False, score=5.0)
    r2 = _result(i2, 0.4, False, score=4.0)

    payload, tokens, low = assemblePayload(store, [r1, r2], 10 ** 9)

    assert low is True
    assert payload.split("\n")[0] == SENTINEL_LOW_CONFIDENCE
    assert f"p3://{i1}" in payload and f"p3://{i2}" in payload   # Tier-0 handles present
    assert "TAILMARKERONE" not in payload                       # no Tier-1 body
    assert "TAILMARKERTWO" not in payload
    assert tokens == estimateTokens(payload)


def test_low_confidence_handles_capped_at_three(store):
    ids = [_put(store, f"weak body {i} about sonar depth and modem") for i in range(5)]
    results = [_result(a, 0.3, False, score=5.0 - n) for n, a in enumerate(ids)]

    payload, _tokens, low = assemblePayload(store, results, 10 ** 9)

    assert low is True
    handleLines = [ln for ln in payload.split("\n") if ln.startswith("p3://")]
    assert len(handleLines) == 3            # at most 3, even with 5 untrusted hits


def test_empty_results_is_the_bare_sentinel(store):
    payload, tokens, low = assemblePayload(store, [], 1500)
    assert payload == SENTINEL_LOW_CONFIDENCE
    assert low is True
    assert tokens == estimateTokens(SENTINEL_LOW_CONFIDENCE)


# --------------------------------------------------------------------------- #
# Payload: no-slop furniture, verbatim bodies, sanitized gists                 #
# --------------------------------------------------------------------------- #


def test_no_markdown_furniture_over_a_clean_corpus(store):
    # The furniture WE add carries no markdown and no emoji. With clean bodies the
    # whole Tier-1 payload is furniture + clean text, so the check covers it all.
    ids = [_put(store, f"clean body number {i} about sonar and depth") for i in range(5)]
    results = [_result(a, 0.85, True, score=5.0 - n) for n, a in enumerate(ids)]

    payload, _tokens, low = assemblePayload(store, results, 10 ** 9)

    assert low is False
    for line in payload.split("\n"):
        assert not line.startswith("#")         # no markdown header
        assert not line.startswith("- ")        # no markdown bullet
        assert not line.startswith("* ")
    assert "**" not in payload                  # no bold furniture
    assert not _EMOJI_RE.search(payload)        # no emoji furniture


# Adversarial atom text: headers, bullets, bold, emoji, embedded newlines, and a
# tail long enough that the gist must truncate. Stored text may contain anything.
_ADVERSARIAL = "# Heading line\n- a bullet\n**bold** and 🔥 emoji then " + "z" * 100


def test_adversarial_body_renders_verbatim_and_gist_is_newline_sanitized(store):
    aid = _put(store, _ADVERSARIAL)
    r = _result(aid, 0.9, True)

    entry = tier1Entry(store, r)
    assert _ADVERSARIAL in entry                # body verbatim: markdown/emoji intact
    assert "**bold**" in entry
    assert _EMOJI_RE.search(entry)              # the emoji survives in the body

    handle = tier0Handle(store, r)
    assert "\n" not in handle                   # gist collapsed the newlines out
    assert "# Heading line - a bullet" in handle  # collapsed to a single line


# --------------------------------------------------------------------------- #
# Tier 2: neighborhood renders live outgoing edges only                       #
# --------------------------------------------------------------------------- #


def test_tier2_renders_live_edges_only(store):
    hub = _put(store, "hub atom body text")
    liveNbr = _put(store, "live neighbor body text")
    supNbr = _put(store, "superseded neighbor body")
    supSucc = _put(store, "the successor of the superseded neighbor")
    supersede(store, supNbr, supSucc, {"source": "claude-code"})   # supNbr not live
    tombNbr = _put(store, "tombstoned neighbor body")
    store._conn.execute("UPDATE atoms SET status='tombstone' WHERE id=?", (tombNbr,))
    store._conn.commit()

    addEdge(store, {"src": hub, "dst": liveNbr, "type": "relates"})
    addEdge(store, {"src": hub, "dst": supNbr, "type": "relates"})
    addEdge(store, {"src": hub, "dst": tombNbr, "type": "causes"})

    block = assembleTier2(store, hub)

    assert "hub atom body text" in block                 # Tier-1 portion present
    assert " | 0.00 | " in block                         # default confidence rendered
    assert f"relates -> p3://{liveNbr} live neighbor body text" in block
    assert f"p3://{supNbr}" not in block                 # superseded dst dropped
    assert f"p3://{tombNbr}" not in block                # tombstoned dst dropped
    # supSucc has no edge from hub, so it must not appear as a neighbor line
    assert f"p3://{supSucc}" not in block


def test_facet_boost_constant_is_a_gentle_nudge():
    # The boost is a small multiplicative lift, not an override -- pin the range so
    # a future edit that turns it into a dominating factor trips here.
    assert 1.0 < FACET_BOOST < 1.5
