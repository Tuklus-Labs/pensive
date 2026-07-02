import math

import numpy as np
import pytest

from recall.embedder import Embedder, embedMissing, vecToBlob, blobToVec
from recall.vector_index import VectorIndex, FlatIndex
from store.store import openStore, putAtom, supersede

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384

# The GPU stack (torch / sentence-transformers, via a SWIG-wrapped native dep)
# emits two DeprecationWarnings the first time it imports under CPython 3.14:
# "builtin type SwigPyPacked has no __module__ attribute" and the same for
# SwigPyObject. They come from generated SWIG glue, not our code, and are
# harmless. Filter exactly those two by name so any real warning still surfaces.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)

# Twenty distinct sentences: each should be its own nearest neighbour, so the
# flat index must rank every atom first for its own text.
SENTENCES = [
    "the pressure hull rates to 488 meters of depth",
    "acoustic modems trade range for data rate",
    "LiFePO4 batteries tolerate cold better than NMC",
    "a doppler velocity log gives bottom-lock speed",
    "USBL beacons provide an absolute position fix",
    "titanium grade five has the best strength to weight",
    "the thermocline refracts acoustic ray paths",
    "multibeam sonar sweeps a wide bathymetric swath",
    "biofouling attaches to the hull within weeks",
    "an extended Kalman filter fuses INS and DVL",
    "methane can pool in the deep anoxic zone",
    "the surface relay buoy carries a VHF radio",
    "brushless thrusters give six degrees of freedom",
    "hydrostatic testing validates the hull at depth",
    "a lawnmower grid pattern covers the survey area",
    "the emergency ascent drops a steel weight",
    "wet-mate connectors seal the charging contacts",
    "seiche oscillations shift the lake surface slowly",
    "forward-looking sonar handles obstacle avoidance",
    "the CTD profiles conductivity temperature and depth",
]


@pytest.fixture(scope="session")
def embedder():
    # Load the real model once for the whole session. It is ~130MB and shares
    # VRAM with the box's llama-servers, so a per-test reload would be wasteful
    # and could contend for the GPU. Session scope keeps exactly one copy.
    return Embedder(MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, project="aegis-auv"):
    return putAtom(
        store,
        {
            "text": text,
            "kind": "atom",
            "project": project,
            "provenance": {"source": "claude-code"},
        },
    )


def _embeddingRows(store, modelId=None):
    if modelId is None:
        return store._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", (modelId,)
    ).fetchone()[0]


def test_embed_returns_unit_normalized_vectors_of_model_dim(embedder):
    texts = SENTENCES[:3]
    vecs = embedder.embed(texts)

    assert len(vecs) == 3
    for v in vecs:
        assert v.dtype == np.float32
        assert v.shape == (DIM,)
        # Unit-normalized within float32 tolerance.
        assert math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=0, abs_tol=1e-5)


def test_embed_empty_list_returns_empty(embedder):
    assert embedder.embed([]) == []


def test_flat_index_ranks_each_atom_first_for_its_own_text(embedder, store):
    # Brief Step 5, strengthened: not just one atom -- every one of the 20 must
    # rank itself #1 when queried with its own text.
    ids = [_put(store, s) for s in SENTENCES]
    assert embedMissing(store, embedder) == len(SENTENCES)

    index = FlatIndex().build(store, MODEL_ID)
    queries = embedder.embed(SENTENCES)
    for atomId, qvec in zip(ids, queries):
        hits = index.search(qvec, k=5)
        assert hits[0][0] == atomId
        # Self-match of a unit vector is ~1.0 cosine.
        assert math.isclose(hits[0][1], 1.0, rel_tol=0, abs_tol=1e-4)


def test_search_respects_k_and_orders_by_descending_score(embedder, store):
    for s in SENTENCES:
        _put(store, s)
    embedMissing(store, embedder)
    index = FlatIndex().build(store, MODEL_ID)

    qvec = embedder.embed([SENTENCES[0]])[0]
    hits = index.search(qvec, k=3)
    assert len(hits) == 3
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_embed_missing_is_idempotent(embedder, store):
    for s in SENTENCES[:5]:
        _put(store, s)

    first = embedMissing(store, embedder)
    assert first == 5
    assert _embeddingRows(store, MODEL_ID) == 5

    # Second call has nothing new to embed.
    second = embedMissing(store, embedder)
    assert second == 0
    assert _embeddingRows(store, MODEL_ID) == 5


def test_embed_missing_is_model_keyed_additive(embedder, store):
    # A second model must ADD rows, never replace this model's. Simulate the
    # second model by writing rows under a fake model id directly (no second
    # model download) -- the point under test is the (atom_id, model_id) keying.
    ids = [_put(store, s) for s in SENTENCES[:4]]
    embedMissing(store, embedder)
    assert _embeddingRows(store, MODEL_ID) == 4

    other = "sim-model-v0"
    now = 1_600_000_000
    store._conn.executemany(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        [(aid, other, vecToBlob(np.zeros(DIM, dtype=np.float32)), now) for aid in ids],
    )
    store._conn.commit()

    # Both models' rows coexist; the real model's rows were not overwritten.
    assert _embeddingRows(store, MODEL_ID) == 4
    assert _embeddingRows(store, other) == 4
    assert _embeddingRows(store) == 8
    # One atom now carries two vectors, one per model.
    twoModels = store._conn.execute(
        "SELECT COUNT(DISTINCT model_id) FROM embeddings WHERE atom_id = ?", (ids[0],)
    ).fetchone()[0]
    assert twoModels == 2
    # And embedMissing under the real model still sees nothing to do.
    assert embedMissing(store, embedder) == 0


def test_embed_missing_skips_superseded_atoms(embedder, store):
    old = _put(store, SENTENCES[0])
    new = _put(store, SENTENCES[1])
    supersede(store, old, new, {"source": "claude-code"})  # old -> 'superseded'

    embedded = embedMissing(store, embedder)
    assert embedded == 1  # only the live successor
    assert _embeddingRows(store, MODEL_ID) == 1
    # The superseded atom got no embedding row.
    got = store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE atom_id = ?", (old,)
    ).fetchone()[0]
    assert got == 0


def test_embed_missing_skips_tombstone_atoms(embedder, store):
    live = _put(store, SENTENCES[0])
    dead = _put(store, SENTENCES[1])
    # No store API tombstones directly yet; set the status to exercise the filter
    # against the other non-live state schema.sql defines.
    store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (dead,))
    store._conn.commit()

    embedded = embedMissing(store, embedder)
    assert embedded == 1
    got = store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE atom_id = ?", (dead,)
    ).fetchone()[0]
    assert got == 0
    live_got = store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE atom_id = ?", (live,)
    ).fetchone()[0]
    assert live_got == 1


def test_flat_index_build_excludes_superseded_atoms(embedder, store):
    # An atom embedded while live, then superseded, must not surface in search --
    # build filters to live atoms so stale embeddings never rank.
    old = _put(store, SENTENCES[0])
    new = _put(store, SENTENCES[1])
    embedMissing(store, embedder)  # both live -> both embedded
    assert _embeddingRows(store, MODEL_ID) == 2

    supersede(store, old, new, {"source": "claude-code"})
    index = FlatIndex().build(store, MODEL_ID)  # rebuilt after supersession
    qvec = embedder.embed([SENTENCES[0]])[0]
    hits = index.search(qvec, k=5)
    returned = {atomId for atomId, _ in hits}
    assert old not in returned
    assert new in returned


def test_search_on_empty_index_returns_empty(embedder, store):
    built = FlatIndex().build(store, MODEL_ID)  # store has no embeddings
    qvec = embedder.embed([SENTENCES[0]])[0]
    assert built.search(qvec, k=5) == []
    # An index that was never built is also empty, not an error.
    assert FlatIndex().search(qvec, k=5) == []


def test_search_with_nonpositive_k_returns_empty(embedder, store):
    _put(store, SENTENCES[0])
    embedMissing(store, embedder)
    index = FlatIndex().build(store, MODEL_ID)
    qvec = embedder.embed([SENTENCES[0]])[0]
    assert index.search(qvec, k=0) == []
    assert index.search(qvec, k=-3) == []


def test_search_normalizes_non_unit_query(embedder, store):
    # The defensive query-normalization path: a caller may pass an un-normalized
    # vector. Magnitude must not change the ranking, and scores must stay true
    # cosine (<= 1.0 within fp32 epsilon), not inflate with the query's scale.
    for s in SENTENCES[:6]:
        _put(store, s)
    embedMissing(store, embedder)
    index = FlatIndex().build(store, MODEL_ID)

    unit = embedder.embed([SENTENCES[0]])[0]
    scaled = unit * 3.0  # deliberately non-unit query

    unitHits = index.search(unit, k=6)
    scaledHits = index.search(scaled, k=6)

    assert [a for a, _ in scaledHits] == [a for a, _ in unitHits]
    for (a1, s1), (a2, s2) in zip(unitHits, scaledHits):
        assert a1 == a2
        assert s2 <= 1.0 + 1e-6  # true cosine, not scaled up by |query|=3
        assert math.isclose(s1, s2, rel_tol=0, abs_tol=1e-5)


def test_search_zero_vector_query_returns_empty(embedder, store):
    # A zero query has no direction; normalizing would divide by zero, so the
    # guard returns [] rather than raising or emitting NaNs.
    _put(store, SENTENCES[0])
    embedMissing(store, embedder)
    index = FlatIndex().build(store, MODEL_ID)
    zero = np.zeros(DIM, dtype=np.float32)
    assert index.search(zero, k=5) == []


def test_blob_round_trip_is_exact(embedder):
    # Write then read back one vector; float32 values must survive bit-for-bit.
    vec = embedder.embed([SENTENCES[0]])[0]
    restored = blobToVec(vecToBlob(vec))
    assert restored.dtype == np.float32
    assert restored.shape == vec.shape
    assert np.array_equal(restored, vec)


def test_blob_round_trip_exact_on_crafted_values():
    # Independent of the model: exact bytes for hand-picked float32 values,
    # including a negative and a subnormal-ish small magnitude.
    vec = np.array([1.0, -0.5, 3.1415927, 0.0, -2.7182817], dtype=np.float32)
    restored = blobToVec(vecToBlob(vec))
    assert np.array_equal(restored, vec)
    # 5 float32 == 20 bytes, little-endian.
    assert len(vecToBlob(vec)) == 5 * 4


def test_vector_index_is_an_abstract_seam():
    # The interface cannot be instantiated (Task 15's usearch index is the other
    # concrete impl); FlatIndex is a registered implementation of it.
    with pytest.raises(TypeError):
        VectorIndex()
    assert issubclass(FlatIndex, VectorIndex)
