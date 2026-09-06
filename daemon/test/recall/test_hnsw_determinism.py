"""Regression tests for restart-stable HNSW construction.

Risk rows map to ``RISK_MODEL_HNSW_DETERMINISM.md``.  These tests keep the
quality-affecting USearch policy explicit and prove that identical canonical
rows reconstruct identical derived graphs.
"""
from types import SimpleNamespace

import numpy as np

import recall.hnsw_index as hnsw_module
from recall.hnsw_index import HnswIndex
from recall.index_cache import indexFingerprint, saveIndexSnapshot


MODEL_ID = "test-model"


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, _params):
        return _Rows(self._rows)


class _Store:
    def __init__(self, rows):
        self._conn = _Connection(rows)


class _SpyIndex:
    created = []

    def __init__(self, **kwargs):
        self.init = kwargs
        self.adds = []
        self.searches = []
        type(self).created.append(self)

    def add(self, keys, vectors, **kwargs):
        self.adds.append((keys, vectors, kwargs))

    def search(self, vector, count, **kwargs):
        self.searches.append((vector, count, kwargs))
        return SimpleNamespace(
            keys=np.asarray([0], dtype=np.uint64),
            distances=np.asarray([0.0], dtype=np.float32),
        )


def _unitRows(count, dim=32):
    rng = np.random.default_rng(20260905)
    matrix = rng.standard_normal((count, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return [(f"atom-{i:05d}", vector.tobytes())
            for i, vector in enumerate(matrix)]


def test_full_build_pins_quality_and_serialization_parameters(monkeypatch):
    """I1/I2/C1/X1: a full build must not inherit USearch defaults."""
    _SpyIndex.created = []
    monkeypatch.setattr(hnsw_module, "Index", _SpyIndex)

    index = HnswIndex().build(_Store(_unitRows(3)), MODEL_ID)

    assert len(_SpyIndex.created) == 1, (
        "full-build constructor invariant violated: expected one USearch index, "
        f"created={len(_SpyIndex.created)}")
    compiled = _SpyIndex.created[0]
    assert compiled.init == {
        "ndim": 32,
        "metric": "cos",
        "dtype": "f32",
        "connectivity": 16,
        "expansion_add": 128,
        "expansion_search": 1024,
    }, f"full-build HNSW policy drifted: init={compiled.init!r}"
    assert len(compiled.adds) == 1, (
        "full-build insertion invariant violated: the ordered matrix must be "
        f"inserted exactly once, calls={len(compiled.adds)}")
    assert compiled.adds[0][2] == {"threads": 1}, (
        "serialized-build invariant violated: USearch bulk add must use one "
        f"worker, kwargs={compiled.adds[0][2]!r}")
    assert index._index is compiled, (
        "full-build publication invariant violated: HnswIndex did not retain "
        "the explicitly configured USearch index")


def test_first_incremental_add_uses_the_full_build_policy(monkeypatch):
    """I1/I2/S1/B2/C2: first-add construction shares the rebuild policy."""
    _SpyIndex.created = []
    monkeypatch.setattr(hnsw_module, "Index", _SpyIndex)
    vector = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    index = HnswIndex().add("first", vector)

    assert len(_SpyIndex.created) == 1, (
        "first-add constructor invariant violated: expected one USearch index, "
        f"created={len(_SpyIndex.created)}")
    compiled = _SpyIndex.created[0]
    assert compiled.init == {
        "ndim": 4,
        "metric": "cos",
        "dtype": "f32",
        "connectivity": 16,
        "expansion_add": 128,
        "expansion_search": 1024,
    }, f"first-add HNSW policy differs from full build: init={compiled.init!r}"
    assert len(compiled.adds) == 1, (
        "first-add insertion invariant violated: the first vector must be "
        f"inserted exactly once, calls={len(compiled.adds)}")
    assert compiled.adds[0][2] == {"threads": 1}, (
        "serialized incremental-insert invariant violated: first add must use "
        f"one worker, kwargs={compiled.adds[0][2]!r}")
    assert index._atomIds == ["first"], (
        "first-add key mapping invariant violated: key zero must name the first "
        f"atom, atomIds={index._atomIds!r}")


def test_repeated_builds_are_byte_and_query_identical():
    """I3/P1: unchanged ordered rows reproduce one graph and result order."""
    store = _Store(_unitRows(2048))

    first = HnswIndex().build(store, MODEL_ID)
    second = HnswIndex().build(store, MODEL_ID)

    firstBytes = first._index.save()
    secondBytes = second._index.save()
    assert firstBytes == secondBytes, (
        "restart determinism invariant violated: identical ordered vectors "
        f"produced graph sizes {len(firstBytes)} and {len(secondBytes)} with "
        "different serialized bytes")
    for atomId in ("atom-00000", "atom-00117", "atom-01023", "atom-02047"):
        key = first._atomIds.index(atomId)
        query = first._index.get(key)
        firstIds = [candidate for candidate, _score in first.search(query, 50)]
        secondIds = [candidate for candidate, _score in second.search(query, 50)]
        assert firstIds == secondIds, (
            "restart query-order invariant violated: byte-identical inputs must "
            f"return one order for query={atomId!r}; first={firstIds!r} "
            f"second={secondIds!r}")


def test_snapshot_cache_hit_bypasses_graph_rebuild(tmp_path, monkeypatch):
    """I5/S3/P2: a verified snapshot loads without reconstructing the graph."""
    store = _Store(_unitRows(256))
    cache = tmp_path / "index-cache"
    first = HnswIndex().build(store, MODEL_ID, kinds=("document_chunk",),
                              cacheDir=cache)
    assert cache.is_dir(), (
        "snapshot publication invariant violated: a cache miss did not create "
        f"the configured cache directory, path={cache}")
    assert len(list(cache.glob("*.usearch"))) == 1, (
        "snapshot publication invariant violated: a cache miss must publish one "
        f"completed graph, entries={[path.name for path in cache.iterdir()]!r}")

    def rebuildBomb(_matrix):
        raise AssertionError(
            "cache-hit invariant violated: verified snapshot rebuilt the graph")

    monkeypatch.setattr(hnsw_module, "_buildIndex", rebuildBomb)
    second = HnswIndex().build(store, MODEL_ID, kinds=("document_chunk",),
                               cacheDir=cache)

    assert second._index.save() == first._index.save(), (
        "snapshot identity invariant violated: loaded graph bytes differ from "
        "the graph published for the same canonical rows")
    assert second._index.expansion_search == 1024, (
        "snapshot search-policy invariant violated: loaded graph must use "
        f"expansion 1024, got {second._index.expansion_search}")


def test_invalid_snapshot_structure_rebuilds_from_canonical_rows(tmp_path):
    """I5/S3: valid file bytes with the wrong native shape are a cache miss."""
    rows = _unitRows(256)
    cache = tmp_path / "index-cache"
    namespace, fingerprint = indexFingerprint(
        rows, MODEL_ID, ("document_chunk",),
        hnsw_module._HNSW_CACHE_PARAMETERS,
    )
    wrong = hnsw_module._newIndex(32)
    wrong.add(
        np.arange(2, dtype=np.int64),
        np.stack([np.frombuffer(blob, dtype=np.float32) for _atom, blob in rows[:2]]),
        threads=1,
    )
    assert saveIndexSnapshot(cache, namespace, fingerprint, wrong) is True, (
        "invalid-snapshot setup invariant violated: native wrong-count graph "
        "did not publish for the validation test")

    rebuilt = HnswIndex().build(
        _Store(rows), MODEL_ID, kinds=("document_chunk",), cacheDir=cache)

    assert rebuilt._index.size == len(rows), (
        "snapshot native-count invariant violated: wrong-count graph was "
        f"accepted, expected={len(rows)} got={rebuilt._index.size}")
    assert np.array_equal(
        rebuilt._index.keys,
        np.arange(len(rows), dtype=np.uint64),
    ), (
        "snapshot native-key invariant violated: loaded keys must be exactly "
        f"0..{len(rows) - 1}, got={rebuilt._index.keys!r}")


def test_cache_failure_does_not_break_canonical_build(tmp_path, monkeypatch):
    """S3/P2: cache exceptions fall back to the ordered canonical rows."""
    def fingerprintBomb(*_args, **_kwargs):
        raise OSError("synthetic fingerprint failure")

    monkeypatch.setattr(hnsw_module, "indexFingerprint", fingerprintBomb)
    rows = _unitRows(64)
    built = HnswIndex().build(
        _Store(rows), MODEL_ID, kinds=("document_chunk",),
        cacheDir=tmp_path / "index-cache",
    )

    assert built._index.size == len(rows), (
        "cache-failure fallback invariant violated: canonical rows did not build "
        f"after a cache exception, expected={len(rows)} got={built._index.size}")


def test_empty_build_does_not_create_snapshot(tmp_path):
    """B4: an empty canonical population remains an uncached empty index."""
    cache = tmp_path / "index-cache"
    built = HnswIndex().build(
        _Store([]), MODEL_ID, kinds=("document_chunk",), cacheDir=cache)

    assert built._index is None, (
        "empty-build invariant violated: an empty canonical population created "
        f"a native index {built._index!r}")
    assert not cache.exists(), (
        "empty-cache invariant violated: empty build created derived cache state, "
        f"path={cache}")
