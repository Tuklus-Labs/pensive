"""Cross-encoder rerank of the fused top-50 (Task 8).

The one signal that reads the query and a document TOGETHER. Fusion (Task 7)
looks only at rank positions across signals and never at the query text against a
candidate; this stage scores every ``(query, atom.text)`` pair jointly with a
BAAI/bge-reranker-base cross-encoder. The load-bearing property pinned here: a
paraphrase of the query -- little vocabulary overlap, so fusion buries it -- is
pulled to #1 on meaning. Plus the cost discipline (one batched predict, one
SELECT, the 50-cap) and the desync/empty/truncation edges.

These tests use the REAL model and a REAL store, mirroring test_embedder.py: one
session-scoped load of the ~1.1GB reranker, a fresh SQLite store per test.
"""
import time

import pytest

from recall.rerank import rerank
from store.store import openStore, putAtom

# Same SwigPy DeprecationWarning filter as the embedder tests: the GPU stack
# (torch / sentence-transformers, via a SWIG-wrapped native dep) emits two of them
# on first import under CPython 3.14, from generated SWIG glue, not our code.
# Filter exactly those two by name so any real warning still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)

QUERY = "What runtime should the pensive daemon use?"
# A paraphrase of QUERY: same meaning, largely disjoint surface vocabulary, so the
# lexical (and even dense) signals would rank it BELOW the keyword-sharing
# distractors. The cross-encoder scores meaning, not overlap, so it is what pulls
# the paraphrase to the top.
PARAPHRASE = "Which runtime ought the pensive memory daemon adopt for production?"
DISTRACTORS = [
    "The pressure hull of the AUV rates to 488 meters of depth.",
    "LiFePO4 batteries tolerate cold water better than NMC cells.",
]


@pytest.fixture(scope="session")
def _rerankerWarm():
    # Page in the ~1.1GB cross-encoder ONCE for the whole session -- it shares
    # VRAM with the embedder and this box's llama-servers, so a per-test reload
    # would be wasteful and contend for the card. Every test that reranks depends
    # on this fixture, so the model loads exactly once.
    from recall.rerank import _getReranker

    _getReranker()


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, project="pensive"):
    return putAtom(
        store,
        {
            "text": text,
            "kind": "atom",
            "project": project,
            "provenance": {"source": "claude-code"},
        },
    )


def test_paraphrase_reranks_to_first_even_when_handed_in_last(store, _rerankerWarm):
    # Brief Step 1: three candidates, one a paraphrase of the query, handed in
    # LAST (fusion ranked it last -- the distractors share query keywords, the
    # paraphrase does not). The cross-encoder scores the (query, doc) meaning, so
    # it must pull the paraphrase to #1 from the bottom of the list.
    d1 = _put(store, DISTRACTORS[0])
    d2 = _put(store, DISTRACTORS[1])
    para = _put(store, PARAPHRASE)

    # Fused order with the paraphrase deliberately last. The fused scores are
    # arbitrary -- rerank discards them and assigns its own.
    candidates = [(d1, 3.0), (d2, 2.0), (para, 1.0)]
    reranked = rerank(QUERY, candidates, store)

    ids = [atomId for atomId, _ in reranked]
    assert ids[0] == para                          # paraphrase pulled to the top
    assert set(ids) == {d1, d2, para}              # all three still present, none dropped
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True)  # best-first
    assert scores[0] > scores[1]                   # paraphrase strictly wins


def test_rerank_of_fifty_pairs_under_100ms_warm(store, _rerankerWarm, capsys):
    # Brief Step 5. Fifty candidates, warm model. One warmup rerank OUTSIDE the
    # timer (pays lazy-init / first-batch kernel costs), then time the full
    # rerank(): SELECT + one batched predict + sort. The Phase 0 skeptic measured
    # 4.09 ms/pair CONTENDED for SEQUENTIAL scoring (~205 ms for 50); this is ONE
    # batched forward pass, so the 100 ms warm budget has real headroom.
    ids = [
        _put(store, f"candidate atom {i}: sonar bathymetry navigation depth reading {i}")
        for i in range(50)
    ]
    candidates = [(atomId, float(50 - i)) for i, atomId in enumerate(ids)]

    rerank(QUERY, candidates, store)  # warmup, OUTSIDE the timer

    t0 = time.perf_counter()
    out = rerank(QUERY, candidates, store)
    elapsedMs = (time.perf_counter() - t0) * 1000.0

    assert len(out) == 50
    # Record the measured latency in the test output regardless of pass/fail
    # (capsys.disabled writes to the real stdout, so no -s flag is needed).
    with capsys.disabled():
        print(f"\n[rerank latency] 50 pairs, warm: {elapsedMs:.1f} ms")
    # The bound stays at 100 ms even under GPU contention: this box runs
    # concurrent GPU loads (film encode, embedder, llama-servers), and the phase
    # gate adjudicates a contention flake rather than the bound being loosened
    # here silently.
    assert elapsedMs < 100.0, (
        f"warm 50-pair rerank took {elapsedMs:.1f} ms (>100 ms; GPU contention?)"
    )


def test_cap_scores_only_fifty_and_51st_never_reaches_the_model(
    store, _rerankerWarm, monkeypatch
):
    # Sabotage: hand in 51 candidates. The cap is part of the contract, so exactly
    # 50 are scored and the 51st (last, weakest fused) must NEVER reach the model.
    # A spy on predict captures what actually got scored -- a length assertion on
    # the output alone would not prove the 51st never touched the GPU.
    from recall import rerank as rerankMod

    ids = [
        _put(store, f"atom {i}: navigation obstacle avoidance and depth control {i}")
        for i in range(51)
    ]
    candidates = [(atomId, float(51 - i)) for i, atomId in enumerate(ids)]
    extraId = ids[50]  # the 51st, ranked last
    extraText = store._conn.execute(
        "SELECT text FROM atoms WHERE id = ?", (extraId,)
    ).fetchone()[0]

    model = rerankMod._getReranker()
    seen = []
    realPredict = model.predict

    def spyPredict(inputs, **kwargs):
        seen.append(list(inputs))
        return realPredict(inputs, **kwargs)

    monkeypatch.setattr(model, "predict", spyPredict)

    out = rerank(QUERY, candidates, store)

    assert len(seen) == 1                       # exactly ONE predict call, not a loop
    assert len(seen[0]) == 50                   # exactly 50 pairs scored
    scoredDocs = [doc for _query, doc in seen[0]]
    assert extraText not in scoredDocs          # the 51st never reached the model
    assert len(out) == 50
    assert extraId not in {atomId for atomId, _ in out}  # nor the output


def test_empty_candidates_returns_empty_without_touching_the_model(store, monkeypatch):
    # Sabotage: empty candidates -> [] WITHOUT loading/using the model (VRAM
    # etiquette: no reason to page in 1.1GB to rerank nothing). Replace the loader
    # with a bomb; if rerank touches it on empty input, the test explodes.
    from recall import rerank as rerankMod

    def boom():
        raise AssertionError("model must not be touched for empty candidates")

    monkeypatch.setattr(rerankMod, "_getReranker", boom)
    assert rerank(QUERY, [], store) == []


def test_unknown_candidate_id_raises_valueerror_naming_it(store, monkeypatch):
    # Sabotage: a candidate id absent from the store means index/store desync. The
    # decades rule says surface it loudly -- ValueError naming the id, never a
    # silent drop (parity with applyPriors). And it must fire BEFORE any model
    # work: guard the loader so a regression that scores first is caught.
    from recall import rerank as rerankMod

    real = _put(store, PARAPHRASE)

    def boom():
        raise AssertionError("desync must be detected before the model is touched")

    monkeypatch.setattr(rerankMod, "_getReranker", boom)
    with pytest.raises(ValueError) as exc:
        rerank(QUERY, [(real, 1.0), ("GHOST_NOT_IN_STORE", 0.5)], store)
    assert "GHOST_NOT_IN_STORE" in str(exc.value)


def test_very_long_text_truncates_cleanly_and_still_scores(store, _rerankerWarm):
    # Sabotage: an atom whose text far exceeds the model's 512-token window (Task
    # 18 posts long conversation tails into recall). Explicit max_length=512
    # truncates deliberately; the pair must still SCORE, not raise a length error.
    longText = "sonar navigation depth acoustic modem bathymetry survey " * 500  # ~3500 tokens
    longId = _put(store, longText)
    shortId = _put(store, DISTRACTORS[0])

    out = rerank(QUERY, [(shortId, 2.0), (longId, 1.0)], store)  # must not raise

    assert len(out) == 2
    assert {atomId for atomId, _ in out} == {shortId, longId}
    for _atomId, score in out:
        assert isinstance(score, float)
