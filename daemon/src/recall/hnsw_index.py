"""The usearch-HNSW ``VectorIndex``: same seam, built for scale.

``FlatIndex`` (Task 5) scans an in-memory matrix per query -- exact, but O(n) per
query, which is honest to a few hundred thousand atoms and no further. This is the
approximate graph index that carries the decades property: usearch builds an HNSW
graph over the same LIVE-atom embeddings and answers a query in roughly O(log n)
by traversing it. It drops in behind the identical ``build``/``search`` two-method
seam, so neither the engine nor the ABC knows which concrete index it holds; a
size-threshold switch (:func:`recall.vector_index.selectIndex`) chooses between
them.

Two invariants keep it interchangeable with the flat reference:

- **Same candidate universe.** ``build`` runs the EXACT query ``FlatIndex.build``
  runs (LIVE atoms only, this ``model_id``, ordered by ``atom_id``). The two
  indexes must agree on WHO is recallable; only the ordering is approximate.
  Non-live atoms are excluded at build time -- query-time staleness is reconciled
  downstream (the Task 6 ledger note), same as flat.
- **Same score semantics.** Scores are cosine SIMILARITY in ``[-1, 1]``, higher is
  better, matching ``FlatIndex``'s normalized dot product. usearch's ``cos`` metric
  returns a DISTANCE defined as ``1 - cosine_similarity`` (verified empirically:
  an identical unit vector yields distance 0.0, and a pair with true cosine
  0.737161 yields distance 0.262839), so we convert back with
  ``similarity = 1 - distance``. Getting this conversion backwards would silently
  invert ranking inside fusion, so it is pinned here and in the tests.

Key mapping: usearch keys are integers, but atom ids are ULID strings. ``build``
adds vectors under keys ``0..n-1`` and keeps ``self._atomIds`` as the ``int ->
atomId`` lookup (``self._atomIds[key]`` is the atom for usearch key ``key``), in
the same ``ORDER BY atom_id`` order the rows were read.
"""
import heapq

import numpy as np

from usearch.index import Index

from recall.embedder import blobToVec
from recall.index_cache import (
    indexFingerprint,
    loadIndexSnapshot,
    saveIndexSnapshot,
)
from recall.vector_index import VectorIndex

__all__ = ["HnswIndex"]


# usearch has no metadata-filtered ANN call. Scoped search reads only allowed
# vectors through stable keys, in bounded batches, and maintains a top-k heap.
# Synthetic CPU timing on 2026-09-05 at 10k vectors x 64 dimensions, k=200
# (five warmups, 51 timed calls): unscoped ANN median/p95 0.115/0.177ms;
# exact scoped search at 10/100/1k/5k/10k allowed IDs had medians
# 0.247/0.329/1.141/4.694/9.008ms and p95s
# 0.291/0.342/1.212/5.706/10.442ms. This measures cardinality cost on one
# synthetic process, not a serving SLA. The unscoped path remains ANN.
_SCOPED_VECTOR_BATCH = 4096

# These parameters are part of recall quality, not incidental library defaults.
# USearch's threads=0 means every available core; parallel insertion makes graph
# topology depend on scheduler interleaving, so rebuilding unchanged canonical
# rows on restart can return a different candidate pool. One insertion worker
# preserves the ORDER BY atom_id sequence above. Wider search recovers more of
# the exact top candidates without changing graph construction breadth.
_HNSW_CONNECTIVITY = 16
_HNSW_EXPANSION_ADD = 128
_HNSW_EXPANSION_SEARCH = 1024
_HNSW_ADD_THREADS = 1
_HNSW_CACHE_PARAMETERS = {
    "version": 1,
    "metric": "cos",
    "dtype": "f32",
    "construction": {
        "connectivity": _HNSW_CONNECTIVITY,
        "expansion_add": _HNSW_EXPANSION_ADD,
        "threads": _HNSW_ADD_THREADS,
    },
    "search": {"expansion_search": _HNSW_EXPANSION_SEARCH},
}


def _newIndex(ndim):
    return Index(
        ndim=ndim,
        metric="cos",
        dtype="f32",
        connectivity=_HNSW_CONNECTIVITY,
        expansion_add=_HNSW_EXPANSION_ADD,
        expansion_search=_HNSW_EXPANSION_SEARCH,
    )


def _buildIndex(matrix):
    index = _newIndex(matrix.shape[1])
    index.add(
        np.arange(matrix.shape[0], dtype=np.int64),
        matrix,
        threads=_HNSW_ADD_THREADS,
    )
    return index


def _loadIndex(path, count, ndim):
    index = _newIndex(ndim)
    index.load(path)
    index.expansion_search = _HNSW_EXPANSION_SEARCH
    if index.size != count or index.ndim != ndim:
        raise ValueError(
            "cached HNSW shape does not match canonical rows: "
            f"size={index.size}/{count} ndim={index.ndim}/{ndim}"
        )
    keys = np.asarray(index.keys, dtype=np.uint64)
    expected = np.arange(count, dtype=np.uint64)
    if not np.array_equal(keys, expected):
        raise ValueError("cached HNSW keys are not the canonical 0..n-1 range")
    return index


class HnswIndex(VectorIndex):
    """usearch-HNSW cosine index, interchangeable with ``FlatIndex``.

    Approximate by construction (an HNSW graph, not an exhaustive scan), which is
    the point above the size threshold where a linear scan stops being free. Its
    top-k must still track the exact flat index closely -- the Step 1 test asserts
    >= 95% top-5 agreement over a 5k store, and this build measures ~98%.
    """

    def __init__(self):
        self._atomIds = []
        # None until build() sees at least one live embedding. usearch needs a
        # positive ndim to construct, so an empty store leaves the index unbuilt
        # and search() short-circuits to [] on it.
        self._index = None
        # Ids retired since the last build. usearch drops the key from the graph
        # and leaves no residue, so unlike FlatIndex there is no position mask to
        # read back; retirement has to be remembered explicitly.
        self._retiredIds = set()

    def build(self, store, modelId, kinds=None, cacheDir=None):
        # EXACTLY FlatIndex.build's query: same LIVE filter, same model, same
        # optional kinds restriction, same ORDER BY -- the flat and HNSW indexes
        # for a given class must load one identical candidate universe.
        from recall.strata import kindInClause

        kindClause, kindParams = kindInClause(kinds, alias="a")
        rows = store._conn.execute(
            "SELECT e.atom_id, e.vector FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live'" + kindClause + " "
            "ORDER BY e.atom_id",
            (modelId, *kindParams),
        ).fetchall()
        self._atomIds = [r[0] for r in rows]
        self._retiredIds = set()
        if not rows:
            self._index = None
            return self
        ndim = blobToVec(rows[0][1]).shape[0]
        namespace = fingerprint = None
        if cacheDir is not None:
            try:
                namespace, fingerprint = indexFingerprint(
                    rows, modelId, kinds, _HNSW_CACHE_PARAMETERS)
                cached = loadIndexSnapshot(
                    cacheDir,
                    namespace,
                    fingerprint,
                    lambda path: _loadIndex(path, len(rows), ndim),
                )
            except Exception:
                cached = None
            if cached is not None:
                self._index = cached
                return self
        # np.stack copies the read-only frombuffer views into one owned, writable
        # (n, dim) float32 array -- what usearch.add wants.
        matrix = np.stack([blobToVec(r[1]) for r in rows]).astype(
            np.float32, copy=False
        )
        # Keys 0..n-1 index straight into self._atomIds; the add order matches the
        # ORDER BY atom_id row order.
        index = _buildIndex(matrix)
        self._index = index
        if cacheDir is not None and namespace is not None:
            saveIndexSnapshot(cacheDir, namespace, fingerprint, index)
        return self

    def remove(self, atomId):
        if self._index is None:
            return False
        keys = [i for i, a in enumerate(self._atomIds) if a == atomId]
        if not keys:
            return False
        # usearch drops the key from the graph, so search stops returning it.
        # Verified discriminatingly rather than by documentation: an atom queried
        # with its OWN vector ranks first, and is absent from the top-3 after
        # remove. _atomIds keeps its slot so every later key still maps correctly.
        for key in keys:
            self._index.remove(key)
        self._retiredIds.add(atomId)
        return True

    def retiredIds(self):
        return set(self._retiredIds)

    def add(self, atomId, vec):
        # Idempotent by identity, same reason as FlatIndex.add.
        if atomId in self._atomIds:
            self.remove(atomId)
        row = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self._index is None:
            # build() leaves _index None on an empty store because usearch needs
            # a positive ndim to construct. The first added vector supplies it.
            self._index = _newIndex(row.shape[0])
        key = len(self._atomIds)
        # The key must equal the position this atom takes in _atomIds, because
        # search maps a usearch key straight back through _atomIds[key]. Appending
        # keeps that correspondence, and ULIDs are monotonic so a newly written
        # atom also sorts last under build()'s ORDER BY atom_id: the append lands
        # where a rebuild would have put it.
        self._index.add(key, row, threads=_HNSW_ADD_THREADS)
        self._atomIds.append(atomId)
        return self

    def search(self, vec, k, allowedIds=None):
        if self._index is None or k <= 0:
            return []
        if allowedIds is not None and not allowedIds:
            return []
        query = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        # Stored rows are unit-normalized; normalizing the query too makes the
        # cosine metric a true cosine for any caller-supplied vector, and guards
        # the zero-vector case above (division would produce NaNs).
        query = query / norm
        if allowedIds is not None:
            return self._searchScoped(query, k, allowedIds)

        n = len(self._atomIds)
        k = min(k, n)
        matches = self._index.search(query, k)
        # usearch returns matches ascending by distance (best first). Convert each
        # cos distance back to similarity and map the int key to its atom id.
        hits = [
            (self._atomIds[int(key)], 1.0 - float(dist))
            for key, dist in zip(matches.keys, matches.distances)
        ]
        # Defensive: assert the best-first, higher-is-better contract regardless of
        # what order usearch handed back.
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits

    def _searchScoped(self, query, k, allowedIds):
        """Exact cosine top-k over allowed live keys, using bounded vector reads."""
        allowed = set(allowedIds)
        keys = [
            key for key, atomId in enumerate(self._atomIds)
            if atomId in allowed and self._index.contains(key)
        ]
        if not keys:
            return []

        k = min(k, len(keys))
        best = []
        for start in range(0, len(keys), _SCOPED_VECTOR_BATCH):
            batchKeys = keys[start:start + _SCOPED_VECTOR_BATCH]
            fetched = self._index.get(np.asarray(batchKeys, dtype=np.uint64))
            present = [
                (key, vector) for key, vector in zip(batchKeys, fetched)
                if vector is not None
            ]
            if not present:
                continue
            matrix = np.stack([vector for _key, vector in present]).astype(
                np.float32, copy=False)
            scores = matrix @ query
            for (key, _vector), score in zip(present, scores):
                score = float(score)
                item = (score, -key, key)
                if len(best) < k:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)

        ordered = sorted(best, key=lambda item: (-item[0], item[2]))
        return [(self._atomIds[key], score) for score, _negKey, key in ordered]
