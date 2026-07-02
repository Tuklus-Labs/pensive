"""Task 15: the usearch-HNSW index behind the ``VectorIndex`` seam.

The centerpiece is Step 1: an approximate graph index must AGREE with the exact
``FlatIndex`` on the top-5 for >= 95% of 100 probe queries over a 5k-atom store.
That is the whole justification for swapping an approximate index in behind the
same two methods -- if it disagreed with the exact reference, the swap would
silently change recall. Everything else pins the contract the ABC promises
(empty/zero/k<=0 sabotage, build-time live-only parity) and the engine's
size-threshold switch that chooses flat below scale and HNSW at/above it.

The 5k embeddings are the expensive part (~10-30s of GPU work), so they are
seeded ONCE in a session-scoped fixture that prints its wall-clock and shared by
every test that needs scale; the sabotage/parity tests use small function-scoped
stores instead.
"""
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from recall.embedder import Embedder, embedMissing
from recall.vector_index import FlatIndex, VectorIndex, selectIndex, HNSW_THRESHOLD
from recall.hnsw_index import HnswIndex
from store.store import openStore, putAtom, supersede

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384
N_ATOMS = 5000
N_PROBES = 100

# Same GPU-stack SWIG DeprecationWarnings as test_embedder; filter exactly those
# two by name so any real warning still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)

# A small, varied domain vocabulary. Composed into distinct sentences so 5k
# embeddings spread out (no exact-tie clusters that would make flat and HNSW pick
# different-but-equal neighbours and depress agreement for a non-approximation
# reason). Flavour matches the AUV corpus the rest of the suite uses.
_SUBJECTS = [
    "the pressure hull", "the acoustic modem", "the doppler log", "the sonar array",
    "the battery pack", "the titanium frame", "the thruster", "the buoyancy engine",
    "the survey drone", "the relay buoy", "the kalman filter", "the depth sensor",
    "the camera gimbal", "the sample arm", "the beacon", "the ballast weight",
]
_VERBS = [
    "rates to", "descends past", "surveys", "maps", "tolerates", "refracts through",
    "station-keeps at", "ascends from", "transmits across", "calibrates against",
    "fuses data at", "profiles", "inspects", "avoids obstacles near", "logs telemetry over",
]
_OBJECTS = [
    "the thermocline", "the anoxic basin", "the glacial wall", "the survey grid",
    "the deep channel", "the silt plain", "the debris field", "the shoreline shelf",
    "cold bottom water", "the rockfall zone", "the sediment layer", "the methane seep",
    "the mooring line", "the recovery net", "the charging dock", "the dam structure",
]
_ADJ = [
    "cold", "turbid", "stratified", "pressurized", "freshwater", "anoxic", "glacial",
    "sunlit", "murky", "stagnant", "oxygenated", "brackish",
]


def _sentence(rng):
    return (f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
            f"in {rng.choice(_ADJ)} conditions")


def _corpusTexts(n, seed):
    # One RNG stream for reproducibility; a trailing unique marker guarantees
    # textual distinctness so no two atoms embed to the exact same vector.
    rng = random.Random(seed)
    return [f"{_sentence(rng)} number {i}" for i in range(n)]


def _probeTexts(n):
    # Genuine nearest-neighbour queries: same vocabulary, NO marker, one RNG per
    # probe so the set is reproducible and independent of corpus draws. Un-marked
    # probes land among the corpus points and exercise real ANN recall rather than
    # trivial self-matches.
    return [_sentence(random.Random(1000 + i)) for i in range(n)]


_SMALL_TEXTS = [
    "the pressure hull rates to 488 meters of depth",
    "acoustic modems trade range for data rate",
    "LiFePO4 batteries tolerate cold better than NMC",
    "a doppler velocity log gives bottom-lock speed",
    "USBL beacons provide an absolute position fix",
    "multibeam sonar sweeps a wide bathymetric swath",
]


@pytest.fixture(scope="session")
def embedder():
    # One resident model for the whole file (~130MB, shares VRAM with the box's
    # llama-servers). Session scope keeps exactly one copy.
    return Embedder(MODEL_ID)


@pytest.fixture(scope="session")
def bigStore(tmp_path_factory, embedder):
    """A 5k-atom store, seeded and embedded ONCE, shared across the scale tests."""
    path = tmp_path_factory.mktemp("hnsw") / "mem.db"
    store = openStore(path)
    t0 = time.time()
    ids = [
        putAtom(store, {
            "text": text, "kind": "atom", "project": "aegis-auv",
            "provenance": {"source": "claude-code"},
        })
        for text in _corpusTexts(N_ATOMS, seed=42)
    ]
    embedded = embedMissing(store, embedder)
    print(f"\n[hnsw fixture] seeded {len(ids)} atoms and embedded {embedded} "
          f"in {time.time() - t0:.1f}s")
    assert embedded == N_ATOMS
    try:
        yield store, ids
    finally:
        store.close()


@pytest.fixture(scope="session")
def flatBig(bigStore):
    store, _ = bigStore
    return FlatIndex().build(store, MODEL_ID)


@pytest.fixture(scope="session")
def hnswBig(bigStore):
    store, _ = bigStore
    return HnswIndex().build(store, MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, project="aegis-auv"):
    return putAtom(store, {
        "text": text, "kind": "atom", "project": project,
        "provenance": {"source": "claude-code"},
    })


# --------------------------------------------------------------------------- #
# Step 1: the centerpiece -- HNSW agrees with flat on top-5 for >= 95%          #
# --------------------------------------------------------------------------- #


def test_hnsw_agrees_with_flat_on_top5_over_5k(flatBig, hnswBig, embedder):
    """The approximate index must reproduce the exact index's top-5 for >= 95%
    of 100 probes, measured as mean |top5_flat ∩ top5_hnsw| / 5."""
    probes = embedder.embed(_probeTexts(N_PROBES))
    assert len(probes) == N_PROBES

    total = 0.0
    for q in probes:
        flatTop = {atomId for atomId, _ in flatBig.search(q, 5)}
        hnswTop = {atomId for atomId, _ in hnswBig.search(q, 5)}
        assert len(flatTop) == 5 and len(hnswTop) == 5
        total += len(flatTop & hnswTop) / 5.0
    agreement = total / len(probes)
    print(f"\n[hnsw] mean top-5 agreement over {N_PROBES} probes = "
          f"{agreement * 100:.2f}%")
    assert agreement >= 0.95


def test_hnsw_scores_are_descending_cosine_similarity(hnswBig, embedder):
    """Scores are cosine similarity (higher = better) and come back best-first."""
    q = embedder.embed([_probeTexts(1)[0]])[0]
    hits = hnswBig.search(q, 8)
    assert len(hits) == 8
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    # Cosine similarity is bounded by 1.0 (fp32 epsilon), never inflated.
    assert scores[0] <= 1.0 + 1e-6


def test_hnsw_scores_match_flat_scale_for_common_atoms(flatBig, hnswBig, embedder):
    """Pin the score SCALE to flat, not just the ranking.

    The top-5 agreement test compares SETS, which is invariant under any monotonic
    transform of the score -- so a future metric or distance->similarity change
    that preserved ranking but shifted the scale would pass every set test while
    silently distorting RRF fusion downstream (fusion consumes the raw similarity).
    This guards that: for every atom in BOTH indexes' top-10, hnsw's cosine
    similarity must equal flat's score within 1e-3 (measured max delta ~2e-7)."""
    probes = embedder.embed(_probeTexts(20))
    pairs = 0
    worst = 0.0
    for q in probes:
        flatScore = dict(flatBig.search(q, 10))
        hnswSim = dict(hnswBig.search(q, 10))
        for atomId in flatScore.keys() & hnswSim.keys():
            delta = abs(flatScore[atomId] - hnswSim[atomId])
            worst = max(worst, delta)
            assert delta <= 1e-3
            pairs += 1
    # Sanity: the comparison actually ran over a meaningful number of pairs, so a
    # degenerate empty-intersection run cannot vacuously pass.
    assert pairs >= 20
    print(f"\n[hnsw] score scale vs flat: {pairs} common pairs, "
          f"max |delta| = {worst:.2e}")


# --------------------------------------------------------------------------- #
# Score convention on a controlled small store                                 #
# --------------------------------------------------------------------------- #


def test_hnsw_ranks_each_atom_first_for_its_own_text(embedder, store):
    """Self-match: every atom is its own nearest neighbour at ~1.0 cosine."""
    ids = [_put(store, s) for s in _SMALL_TEXTS]
    assert embedMissing(store, embedder) == len(_SMALL_TEXTS)
    index = HnswIndex().build(store, MODEL_ID)

    queries = embedder.embed(_SMALL_TEXTS)
    for atomId, qvec in zip(ids, queries):
        hits = index.search(qvec, k=3)
        assert hits[0][0] == atomId
        assert math.isclose(hits[0][1], 1.0, rel_tol=0, abs_tol=1e-4)


def test_hnsw_normalizes_non_unit_query(embedder, store):
    """Query magnitude must not change ranking and must not inflate scores past a
    true cosine -- the defensive normalization mirrors FlatIndex."""
    for s in _SMALL_TEXTS:
        _put(store, s)
    embedMissing(store, embedder)
    index = HnswIndex().build(store, MODEL_ID)

    unit = embedder.embed([_SMALL_TEXTS[0]])[0]
    scaled = unit * 4.0

    unitHits = index.search(unit, k=6)
    scaledHits = index.search(scaled, k=6)
    assert [a for a, _ in scaledHits] == [a for a, _ in unitHits]
    for (a1, s1), (a2, s2) in zip(unitHits, scaledHits):
        assert a1 == a2
        assert s2 <= 1.0 + 1e-6
        assert math.isclose(s1, s2, rel_tol=0, abs_tol=1e-5)


# --------------------------------------------------------------------------- #
# Sabotage: empty index, zero query, non-positive k, k > n                      #
# --------------------------------------------------------------------------- #


def test_hnsw_search_on_empty_index_returns_empty(embedder, store):
    """An empty store yields an empty index; a never-built index is empty too."""
    built = HnswIndex().build(store, MODEL_ID)  # store has no embeddings
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    assert built.search(qvec, k=5) == []
    assert HnswIndex().search(qvec, k=5) == []


def test_hnsw_search_with_nonpositive_k_returns_empty(embedder, store):
    _put(store, _SMALL_TEXTS[0])
    embedMissing(store, embedder)
    index = HnswIndex().build(store, MODEL_ID)
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    assert index.search(qvec, k=0) == []
    assert index.search(qvec, k=-3) == []


def test_hnsw_zero_vector_query_returns_empty(embedder, store):
    """A zero query has no direction; normalizing would divide by zero, so the
    guard returns [] rather than raising or emitting NaNs."""
    _put(store, _SMALL_TEXTS[0])
    embedMissing(store, embedder)
    index = HnswIndex().build(store, MODEL_ID)
    zero = np.zeros(DIM, dtype=np.float32)
    assert index.search(zero, k=5) == []


def test_hnsw_k_greater_than_n_returns_all(embedder, store):
    """k > n returns every atom (min(k, n)), not an error or padding."""
    ids = [_put(store, s) for s in _SMALL_TEXTS[:3]]
    embedMissing(store, embedder)
    index = HnswIndex().build(store, MODEL_ID)
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    hits = index.search(qvec, k=50)
    assert len(hits) == 3
    assert {a for a, _ in hits} == set(ids)


# --------------------------------------------------------------------------- #
# Build-time live-only parity with FlatIndex                                    #
# --------------------------------------------------------------------------- #


def test_hnsw_build_excludes_superseded_atoms(embedder, store):
    """An atom embedded while live, then superseded, must not surface in a REBUILT
    index -- build filters to live atoms, exactly like FlatIndex. (Query-time
    staleness is reconciled downstream per the Task 6 ledger note; HNSW, like
    Flat, is build-time live-only.)"""
    old = _put(store, _SMALL_TEXTS[0])
    new = _put(store, _SMALL_TEXTS[1])
    embedMissing(store, embedder)  # both live -> both embedded

    supersede(store, old, new, {"source": "claude-code"})
    index = HnswIndex().build(store, MODEL_ID)  # rebuilt after supersession
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    returned = {atomId for atomId, _ in index.search(qvec, k=5)}
    assert old not in returned
    assert new in returned


def test_hnsw_build_excludes_tombstone_atoms(embedder, store):
    """Tombstone atoms are absent from the built index, mirroring FlatIndex."""
    live = _put(store, _SMALL_TEXTS[0])
    dead = _put(store, _SMALL_TEXTS[1])
    embedMissing(store, embedder)
    store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (dead,))
    store._conn.commit()

    index = HnswIndex().build(store, MODEL_ID)
    qvec = embedder.embed([_SMALL_TEXTS[1]])[0]
    returned = {atomId for atomId, _ in index.search(qvec, k=5)}
    assert dead not in returned
    assert live in returned


def test_hnsw_and_flat_share_the_same_candidate_universe(embedder, store):
    """Both indexes load the identical LIVE-atom set at build; over a small store
    every returned atom id is common to both (the swap can't change WHO is
    recallable, only the approximate ordering)."""
    old = _put(store, _SMALL_TEXTS[0])
    new = _put(store, _SMALL_TEXTS[1])
    for s in _SMALL_TEXTS[2:]:
        _put(store, s)
    embedMissing(store, embedder)
    supersede(store, old, new, {"source": "claude-code"})

    flat = FlatIndex().build(store, MODEL_ID)
    hnsw = HnswIndex().build(store, MODEL_ID)
    qvec = embedder.embed([_SMALL_TEXTS[2]])[0]
    flatIds = {a for a, _ in flat.search(qvec, k=10)}
    hnswIds = {a for a, _ in hnsw.search(qvec, k=10)}
    assert flatIds == hnswIds
    assert old not in hnswIds


# --------------------------------------------------------------------------- #
# Interface: HnswIndex is a VectorIndex implementation                          #
# --------------------------------------------------------------------------- #


def test_hnsw_is_a_vector_index():
    assert issubclass(HnswIndex, VectorIndex)


# --------------------------------------------------------------------------- #
# Step 5: the engine's size-threshold switch (selectIndex)                      #
# --------------------------------------------------------------------------- #


def test_hnsw_threshold_default_is_200k():
    """The plan's default, pinned as a named constant so a drift is loud."""
    assert HNSW_THRESHOLD == 200_000


def test_select_index_picks_flat_below_threshold(embedder, store):
    """At shadow scale (a handful of atoms, default 200k threshold) the factory
    returns the exact FlatIndex -- serving behavior is unchanged."""
    for s in _SMALL_TEXTS:
        _put(store, s)
    embedMissing(store, embedder)
    index = selectIndex(store, MODEL_ID)
    assert isinstance(index, FlatIndex)
    # And it is BUILT and usable, not a bare instance.
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    assert index.search(qvec, k=1)[0][0] is not None


def test_select_index_picks_hnsw_at_or_above_threshold(embedder, store, monkeypatch):
    """With the threshold monkeypatched low (no 200k store in a test), a store at
    or above it selects the HNSW index. Tests the exact boundary: count == N.
    threshold -> HNSW (the switch is >=)."""
    ids = [_put(store, s) for s in _SMALL_TEXTS]  # 6 live embedded atoms
    embedMissing(store, embedder)

    # Exactly at threshold -> HNSW (the >= boundary).
    monkeypatch.setattr("recall.vector_index.HNSW_THRESHOLD", len(ids))
    index = selectIndex(store, MODEL_ID)
    assert isinstance(index, HnswIndex)
    qvec = embedder.embed([_SMALL_TEXTS[0]])[0]
    assert index.search(qvec, k=1)[0][0] == ids[0]


def test_select_index_boundary_below_threshold_is_flat(embedder, store, monkeypatch):
    """One below the threshold selects flat -- confirms the switch is strictly
    < / >= around N, not off by one."""
    ids = [_put(store, s) for s in _SMALL_TEXTS]  # 6 atoms
    embedMissing(store, embedder)
    monkeypatch.setattr("recall.vector_index.HNSW_THRESHOLD", len(ids) + 1)
    assert isinstance(selectIndex(store, MODEL_ID), FlatIndex)


def test_select_index_counts_only_live_embedded_atoms(embedder, store, monkeypatch):
    """The count that drives the switch mirrors the build filter: superseded atoms
    do not count toward the threshold (else a store could pick HNSW for atoms that
    are not even in the index)."""
    old = _put(store, _SMALL_TEXTS[0])
    new = _put(store, _SMALL_TEXTS[1])
    for s in _SMALL_TEXTS[2:]:
        _put(store, s)
    embedMissing(store, embedder)  # 6 embedded, all live
    supersede(store, old, new, {"source": "claude-code"})  # now 5 live embedded

    # Threshold at 6: the raw embeddings row count is 6 but only 5 are LIVE, so a
    # correct live-count switch stays on flat.
    monkeypatch.setattr("recall.vector_index.HNSW_THRESHOLD", 6)
    assert isinstance(selectIndex(store, MODEL_ID), FlatIndex)
    # Drop the threshold to the live count and it flips to HNSW.
    monkeypatch.setattr("recall.vector_index.HNSW_THRESHOLD", 5)
    assert isinstance(selectIndex(store, MODEL_ID), HnswIndex)


def test_daemon_startup_uses_factory_and_selects_flat_at_shadow_scale(embedder, store):
    """The daemon builds its resident index through ServeContext, which now goes
    through selectIndex. At shadow scale that still returns a FlatIndex, so the
    serving path is byte-for-byte the pre-Task-15 behavior."""
    from serve.mcp import ServeContext

    for s in _SMALL_TEXTS:
        _put(store, s)
    ctx = ServeContext(store, embedder, MODEL_ID)
    assert isinstance(ctx.index, FlatIndex)


# A clean-room child that seeds embeddings directly (no real embedder needed),
# selects below the default threshold, and asserts usearch stayed unimported.
# argv[1] = daemon/src on sys.path, argv[2] = a fresh sqlite path.
_LAZY_IMPORT_PROBE = r'''
import sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from store.store import openStore, putAtom
from recall.embedder import vecToBlob
from recall.vector_index import selectIndex, FlatIndex

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
store = openStore(sys.argv[2])
rng = np.random.default_rng(0)
for i in range(5):
    aid = putAtom(store, {"text": f"atom {i}", "kind": "atom", "project": "p",
                          "provenance": {"source": "claude-code"}})
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        (aid, MODEL, vecToBlob(v), int(time.time())),
    )
store._conn.commit()

index = selectIndex(store, MODEL)          # default 200k threshold -> FlatIndex
assert isinstance(index, FlatIndex), type(index).__name__
# The binding constraint: a below-threshold selection must NOT pull usearch in.
assert "usearch" not in sys.modules, "selectIndex imported usearch below threshold"
print("LAZY_OK")
'''


def test_selectindex_below_threshold_does_not_import_usearch(tmp_path):
    """The lazy-import constraint is binding, not cosmetic: at shadow scale the
    factory must keep usearch out of the process (so a Flat-only deploy never needs
    the dependency). Checked in a SUBPROCESS on purpose -- this very test module
    imports HnswIndex at the top, which pulls usearch into the pytest process, so an
    in-process ``'usearch' not in sys.modules`` assert would ALWAYS fail (a false
    red) or, worse, be quietly deleted to make it pass (a false green). The
    clean-room child imports only the flat path and seeds embeddings directly."""
    src = str(Path(__file__).resolve().parents[2] / "src")
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_PROBE, src, str(tmp_path / "lazy.db")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "LAZY_OK" in proc.stdout
