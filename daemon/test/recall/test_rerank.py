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
# The adversarial construction that actually pins semantic-over-lexical (the
# property this test exists to protect). The paraphrase is lexically DIVERGENT --
# it shares almost no content word with the query ("runtime" -> "execution
# environment", "pensive daemon" -> "memory service") -- yet means the same thing.
# The distractors are lexically SIMILAR -- each repeats "pensive daemon runtime"
# verbatim -- yet are semantically irrelevant (a logo, a mascot). A pure
# lexical-overlap ranker would therefore rank the DISTRACTORS above the paraphrase;
# only a model that scores meaning promotes the paraphrase. Verified on the real
# model (bge-reranker-base): paraphrase 0.108 vs distractors 0.002 / 0.004 -- a
# ~25x margin the correct way round.
PARAPHRASE = "Which execution environment ought the memory service adopt for production?"
DISTRACTORS = [
    "The pensive daemon runtime logo was repainted last Tuesday.",
    "Someone doodled the pensive daemon runtime mascot on a napkin.",
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
    # Brief Step 1, as a TRUE semantic-over-lexical test. The paraphrase is handed
    # in LAST and shares almost no vocabulary with the query; the two distractors
    # sit above it AND repeat the query's keywords ("pensive daemon runtime"). A
    # lexical ranker would leave the paraphrase last -- the cross-encoder scores
    # meaning, so it must pull it to #1 from the bottom. (See the DISTRACTORS /
    # PARAPHRASE construction above for why this is not a lexical-overlap win.)
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


def _gpuBusyPercent():
    """AMD ROCm GPU utilization from sysfs, or None if it cannot be read.

    Reads ``/sys/class/drm/card*/device/gpu_busy_percent`` -- the kernel's amdgpu
    utilization counter -- taking the maximum across readable cards (the display
    adapter may be card0 while the 7900 XTX is card1, so first-readable can miss
    the busy card). ``torch.cuda.utilization()`` is NVIDIA-only and
    meaningless on ROCm, so it is deliberately NOT used. Returns None on any
    failure (path absent, permission, non-integer), which the caller reads as
    "contention cannot be proven".
    """
    import glob

    readings = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        try:
            with open(path) as fh:
                readings.append(int(fh.read().strip()))
        except (OSError, ValueError):
            continue
    return max(readings) if readings else None


def test_rerank_of_fifty_pairs_under_100ms_warm(store, _rerankerWarm, capsys):
    # Brief Step 5, made contention-aware. Fifty candidates, warm model: one warmup
    # rerank OUTSIDE the timer, then time the full rerank() (SELECT + one batched
    # predict + sort). This box shares the card with llama-servers and, during a
    # build, a Mandelbrot film render, so a single-shot timing can measure the
    # co-tenant instead of the reranker. Sample amdgpu utilization immediately
    # before the timed call and ASSERT the hard 100 ms bound only when the GPU is
    # quiet enough for the number to mean something; when it is busy, SKIP with the
    # number recorded. "Assert when meaningful, record always." The task plan
    # sanctions skip-marking the latency assertion when the runtime is the slow
    # path.
    ids = [
        _put(store, f"candidate atom {i}: sonar bathymetry navigation depth reading {i}")
        for i in range(50)
    ]
    candidates = [(atomId, float(50 - i)) for i, atomId in enumerate(ids)]

    rerank(QUERY, candidates, store)  # warmup, OUTSIDE the timer

    busy = _gpuBusyPercent()  # sampled immediately before the timed call
    t0 = time.perf_counter()
    out = rerank(QUERY, candidates, store)
    elapsedMs = (time.perf_counter() - t0) * 1000.0

    assert len(out) == 50
    # Record the measured latency (and the GPU state that qualifies it) on every
    # run -- capsys.disabled writes to the real stdout, so no -s flag is needed.
    with capsys.disabled():
        print(f"\n[rerank latency] 50 pairs, warm: {elapsedMs:.1f} ms (gpu_busy={busy})")

    if busy is not None and busy > 40:
        # The card is under real load; this timing reflects the co-tenant, not the
        # reranker, so asserting the bound would be meaningless. Skip -- the number
        # is already printed and goes in the skip reason too.
        pytest.skip(
            f"GPU busy {busy}% (>40%): warm 50-pair latency {elapsedMs:.1f} ms "
            "measures the co-tenant load, not the reranker"
        )
    # GPU quiet, OR utilization unreadable (busy is None) so contention cannot be
    # proven -- either way the number reflects the reranker itself, so hold the
    # hard bound.
    assert elapsedMs < 100.0, (
        f"warm 50-pair rerank took {elapsedMs:.1f} ms (>100 ms; gpu_busy={busy})"
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

    # Prove the truncation is REAL, not the text merely happening to fit: tokenizing
    # the (query, longText) pair with the module's own max_length caps the input at
    # exactly 512 tokens (the long text is ~3500 tokens, so it MUST be cut).
    from recall import rerank as rerankMod

    model = rerankMod._getReranker()
    enc = model.tokenizer(
        QUERY, longText, truncation=True, max_length=rerankMod._MAX_LENGTH
    )
    assert len(enc["input_ids"]) == rerankMod._MAX_LENGTH == 512
