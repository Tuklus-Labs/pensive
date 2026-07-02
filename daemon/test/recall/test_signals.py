"""Signals: FTS5 BM25, dense delegation, and facet/temporal hints.

Real store + real FTS5 + real embedding model (session fixture). The three
signal sources are exercised against a live SQLite store so the live-only
parity rule (bm25 must agree with FlatIndex on the candidate universe) and the
FTS5 query-sanitization contract are tested end to end, not mocked.
"""
import numpy as np
import pytest

from recall.signals import bm25, dense, facetSignal
from recall.embedder import Embedder, embedMissing
from recall.vector_index import FlatIndex
from store.store import openStore, putAtom, addFacet, supersede, getAtom

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384

# Same two SWIG DeprecationWarnings the embedder test filters: emitted by the
# generated sentence-transformers native glue the first time it imports under
# CPython 3.14, not by our code. Filter exactly those so real warnings surface.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)

SENTENCES = [
    "the pressure hull rates to 488 meters of depth",
    "acoustic modems trade range for data rate",
    "LiFePO4 batteries tolerate cold better than NMC",
    "a doppler velocity log gives bottom-lock speed",
    "USBL beacons provide an absolute position fix",
    "multibeam sonar sweeps a wide bathymetric swath",
]


@pytest.fixture(scope="session")
def embedder():
    # One real model for the whole session (see test_embedder for the rationale).
    return Embedder(MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, project="aegis", occurredAt=None):
    atomInput = {
        "text": text,
        "kind": "atom",
        "project": project,
        "provenance": {"source": "claude-code"},
    }
    if occurredAt is not None:
        atomInput["occurredAt"] = occurredAt
    return putAtom(store, atomInput)


# --------------------------------------------------------------------------- #
# bm25                                                                        #
# --------------------------------------------------------------------------- #

def test_bm25_ranks_containing_atom_above_non_containing(store):
    # Brief Step 1: a distinctive word returns the atom containing it, ranked
    # above one that does not (here the non-containing atom does not match at
    # all, so it is absent -- the strongest form of "ranked above").
    a1 = _put(store, "the biofouling attaches to the titanium hull")
    a2 = _put(store, "acoustic modems trade range for data rate")

    hits = bm25(store, "biofouling", 200)
    ids = [atomId for atomId, _ in hits]

    assert a1 in ids
    assert a2 not in ids
    assert ids[0] == a1


def test_bm25_higher_score_is_better(store):
    # Sign convention: bm25() is a cost (lower is better) in SQLite; we negate
    # it so HIGHER = BETTER. An atom matching more query terms must outrank one
    # matching fewer, and the returned list is best-first.
    a1 = _put(store, "alpha beta gamma delta")   # matches three query terms
    a2 = _put(store, "alpha zeta omicron")       # matches one query term

    hits = bm25(store, "alpha beta gamma", 200)
    ids = [atomId for atomId, _ in hits]
    scores = [score for _, score in hits]

    assert ids[0] == a1
    assert scores == sorted(scores, reverse=True)
    byId = dict(hits)
    assert byId[a1] >= byId[a2]


def test_bm25_excludes_atom_superseded_after_indexing(store):
    # Live-parity / sabotage: an atom indexed while live, then superseded, must
    # not come back. The live filter is at query time, not index time.
    old = _put(store, "xenophobia zeppelin quixotic jamboree")
    new = _put(store, "ordinary replacement text without the rare words")
    supersede(store, old, new, {"source": "claude-code"})

    ids = [atomId for atomId, _ in bm25(store, "xenophobia", 200)]
    assert old not in ids


def test_bm25_empty_and_whitespace_return_empty(store):
    _put(store, "some indexed content")
    assert bm25(store, "", 200) == []
    assert bm25(store, "   \t\n  ", 200) == []


def test_bm25_nonpositive_k_returns_empty(store):
    _put(store, "the biofouling attaches to the hull")
    assert bm25(store, "biofouling", 0) == []
    assert bm25(store, "biofouling", -5) == []


def test_bm25_never_throws_on_nasty_input(store):
    # The recall engine takes raw text straight into FTS5 MATCH. None of these
    # may raise; FTS5 operator/grammar chars must be neutralized by quoting.
    _put(store, "foo bar baz qux")
    _put(store, "hyphenated word here")
    _put(store, "café résumé naïve notes")

    nasty = [
        '"',                    # lone double quote
        'unbalanced "quote',    # unbalanced quote
        'foo"bar',              # embedded quote
        'foo AND bar',          # boolean-looking
        'NOT foo',
        'x AND (y OR z)',
        'hyphenated-word',      # hyphen (FTS5 column-filter / NOT)
        '(parens)',
        'NEAR/2',
        'a*',                   # lone asterisk / prefix token
        'OR',
        'café',                 # unicode
        '🔥🔥🔥',                # emoji only
        '‮',               # bidi control, no word chars
    ]
    for q in nasty:
        hits = bm25(store, q, 200)
        assert isinstance(hits, list)


def test_bm25_boolean_and_hyphen_tokens_are_literal_not_operators(store):
    # Proof the sanitizer neutralizes operators: 'foo AND bar' must behave as a
    # bag of words (foo/and/bar), matching the atom that contains foo and bar --
    # not as an FTS5 AND expression, and not a syntax error.
    a = _put(store, "foo bar baz qux")
    hyphen = _put(store, "hyphenated word here")

    andIds = [i for i, _ in bm25(store, "foo AND bar", 200)]
    assert a in andIds

    hyphenIds = [i for i, _ in bm25(store, "hyphenated-word", 200)]
    assert hyphen in hyphenIds


def test_bm25_caps_huge_query_and_still_matches_early_tokens(store):
    # Task 18's drift watcher posts conversation TAILS into recall as raw
    # queries, so a multi-thousand-token input is designed-in, not hypothetical.
    # The sanitizer caps at the first _MAX_QUERY_TOKENS tokens: the query must
    # return without error and the early (leading) tokens must still match.
    doc = _put(store, "biofouling titanium hull acoustic modem")
    early = "biofouling titanium hull"
    filler = " ".join(f"filler{i}" for i in range(5000))
    query = early + " " + filler

    hits = bm25(store, query, 200)     # 5000+ tokens must not raise
    ids = [atomId for atomId, _ in hits]
    assert doc in ids


# --------------------------------------------------------------------------- #
# dense                                                                       #
# --------------------------------------------------------------------------- #

def test_dense_embeds_query_once_and_delegates(embedder):
    # dense must embed the query exactly once and hand the vector to
    # index.search(vec, k), returning its result verbatim.
    calls = {"embed": 0, "search": None}

    class SpyEmbedder:
        def __init__(self, inner):
            self._inner = inner

        def embed(self, texts):
            calls["embed"] += 1
            return self._inner.embed(texts)

    class SpyIndex:
        def search(self, vec, k):
            calls["search"] = (np.asarray(vec, dtype=np.float32).copy(), k)
            return [("ATOM_A", 0.91), ("ATOM_B", 0.42)]

    out = dense(SpyIndex(), SpyEmbedder(embedder), "acoustic modem range", k=7)

    assert calls["embed"] == 1
    assert calls["search"] is not None
    passedVec, passedK = calls["search"]
    assert passedK == 7
    expected = embedder.embed(["acoustic modem range"])[0]
    assert np.allclose(passedVec, expected, atol=1e-6)
    assert out == [("ATOM_A", 0.91), ("ATOM_B", 0.42)]


def test_dense_end_to_end_returns_self_match_first(embedder, store):
    # Real embedder + real FlatIndex: dense(query) ranks the atom whose text is
    # the query as the top hit.
    ids = [_put(store, s) for s in SENTENCES]
    assert embedMissing(store, embedder) == len(SENTENCES)
    index = FlatIndex().build(store, MODEL_ID)

    for atomId, text in zip(ids, SENTENCES):
        hits = dense(index, embedder, text, k=5)
        assert hits[0][0] == atomId


# --------------------------------------------------------------------------- #
# facetSignal                                                                 #
# --------------------------------------------------------------------------- #

def test_facet_project_hint_filters_to_that_projects_live_atoms(store):
    a1 = _put(store, "aegis note one", project="aegis")
    a2 = _put(store, "aegis note two", project="aegis")
    other = _put(store, "sentinel note", project="sentinel")

    res = facetSignal(store, {"project": "aegis"})
    assert res["filterSet"] == {a1, a2}
    assert other not in res["filterSet"]
    assert res["boostSet"] == set()


def test_facet_project_filter_excludes_superseded(store):
    old = _put(store, "aegis original", project="aegis")
    new = _put(store, "aegis successor", project="aegis")
    supersede(store, old, new, {"source": "claude-code"})

    res = facetSignal(store, {"project": "aegis"})
    assert old not in res["filterSet"]      # superseded -> not live -> excluded
    assert new in res["filterSet"]


def test_facet_time_range_excludes_out_of_range(store):
    # occurred_at drives the window; the out-of-range atom is excluded.
    inRange = _put(store, "happened in 2001", occurredAt=1_000_000_000)
    outRange = _put(store, "happened in 2033", occurredAt=2_000_000_000)

    res = facetSignal(store, {"timeRange": (900_000_000, 1_100_000_000)})
    assert res["filterSet"] == {inRange}
    assert outRange not in res["filterSet"]


def test_facet_time_range_coalesces_to_created_at_when_no_occurred_at(store):
    # occurred_at NULL -> the window is tested against created_at (stamped now).
    noOccurred = _put(store, "no explicit occurrence time")
    created = getAtom(store, noOccurred)["createdAt"]
    hasOld = _put(store, "old occurrence", occurredAt=1_000_000_000)

    res = facetSignal(store, {"timeRange": (created - 5, created + 5)})
    assert noOccurred in res["filterSet"]
    assert hasOld not in res["filterSet"]


def test_facet_time_range_occurred_at_takes_precedence_over_created_at(store):
    # Sharp COALESCE test: an atom created NOW but with occurred_at back in 2001
    # must be judged by occurred_at, so a window around NOW excludes it.
    backdated = _put(store, "recorded now, happened long ago", occurredAt=1_000_000_000)
    plainNow = _put(store, "recorded now, no occurrence stamp")
    created = getAtom(store, plainNow)["createdAt"]

    res = facetSignal(store, {"timeRange": (created - 5, created + 5)})
    assert plainNow in res["filterSet"]
    assert backdated not in res["filterSet"]


def test_facet_project_and_time_range_intersect(store):
    both = _put(store, "aegis in window", project="aegis", occurredAt=1_000_000_000)
    projOnly = _put(store, "aegis out of window", project="aegis", occurredAt=2_000_000_000)
    timeOnly = _put(store, "sentinel in window", project="sentinel", occurredAt=1_000_000_000)

    res = facetSignal(
        store, {"project": "aegis", "timeRange": (900_000_000, 1_100_000_000)}
    )
    assert res["filterSet"] == {both}
    assert projOnly not in res["filterSet"]
    assert timeOnly not in res["filterSet"]


def test_facet_entity_hint_boosts_atoms_with_matching_entity_facet(store):
    # Entities are extracted from the query text via MegaExtractor; the boost
    # set is the live atoms carrying a matching (key='entity', value=<label>)
    # facet -- value is the lowercase surface form extract() returns.
    hit = _put(store, "notes about the memory engine")
    addFacet(store, hit, "entity", "pensive")
    miss = _put(store, "unrelated content")

    res = facetSignal(store, {"query": "what did we learn about pensive"})
    assert hit in res["boostSet"]
    assert miss not in res["boostSet"]
    assert res["filterSet"] is None      # no project/time hint -> no filter


def test_facet_entity_boost_ignores_divergent_value_format(store):
    # Pin the entity convention negatively: value is the lowercase surface form
    # extract() returns. A writer that stores a type-prefixed value or the wrong
    # case must NOT be silently boosted -- this turns a future writer/reader
    # format drift into a loud test failure instead of silently dead boosts.
    canonical = _put(store, "atom with canonical entity facet")
    addFacet(store, canonical, "entity", "pensive")
    wrongCase = _put(store, "atom with uppercase entity facet")
    addFacet(store, wrongCase, "entity", "PENSIVE")
    typePrefixed = _put(store, "atom with type-prefixed entity facet")
    addFacet(store, typePrefixed, "entity", "project:pensive")

    res = facetSignal(store, {"query": "tell me about pensive"})
    assert canonical in res["boostSet"]      # matches the documented format
    assert wrongCase not in res["boostSet"]  # divergent case -> no boost
    assert typePrefixed not in res["boostSet"]  # divergent format -> no boost


def test_facet_entity_boost_excludes_superseded(store):
    old = _put(store, "old pensive note")
    addFacet(store, old, "entity", "pensive")
    new = _put(store, "new pensive note")
    supersede(store, old, new, {"source": "claude-code"})

    res = facetSignal(store, {"query": "pensive recall"})
    assert old not in res["boostSet"]    # superseded -> not live -> not boosted


def test_facet_query_with_no_extractable_entity_boosts_nothing(store):
    # The v2 lesson: ~83% of real queries have no extractable entity. That must
    # be a no-op, never an error.
    hit = _put(store, "some atom")
    addFacet(store, hit, "entity", "pensive")

    res = facetSignal(store, {"query": "what did we talk about yesterday"})
    assert res["boostSet"] == set()
    assert res["filterSet"] is None


def test_facet_empty_hints_returns_no_filter_and_empty_boost(store):
    _put(store, "content")
    res = facetSignal(store, {})
    assert res["boostSet"] == set()
    assert res["filterSet"] is None


def test_facet_degenerate_time_range_start_after_end_is_empty_filter(store):
    _put(store, "content", occurredAt=1_000_000_000)
    res = facetSignal(store, {"timeRange": (5000, 1000)})
    # An impossible window matches nothing: an empty filterSet (not None, which
    # would mean "no filter") -- decided and documented in facetSignal.
    assert res["filterSet"] == set()
    assert res["boostSet"] == set()
