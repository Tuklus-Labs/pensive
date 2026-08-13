"""The vector-search seam and its flat (brute-force) implementation.

``VectorIndex`` is the interface Phase 2 recall searches against. It is kept
deliberately minimal -- ``build`` then ``search`` -- so that the Task 15
usearch-HNSW index can drop in behind the same two methods and a size-threshold
switch (:func:`selectIndex`) can pick between them without either caller or the
interface knowing which concrete index it holds. Nothing here leaks a flat-scan
assumption: ``search`` takes a query vector and a ``k`` and returns ranked
``(atomId, score)`` pairs; how those are found (linear scan vs. approximate graph
traversal) is the implementation's business.

Scores are cosine similarity in ``[-1, 1]``, higher is better. Stored vectors are
unit-normalized by the embedder, so the flat implementation's cosine reduces to a
matrix-vector dot product.
"""
import abc

import numpy as np

from recall.embedder import blobToVec
from recall.strata import KIND_CLASSES, kindInClause

__all__ = ["VectorIndex", "FlatIndex", "selectIndex", "buildClassIndexes",
           "HNSW_THRESHOLD"]

# Above this many embedded LIVE atoms, the exact flat scan's O(n)-per-query cost
# stops being free and :func:`selectIndex` switches to the approximate HNSW index.
# The plan's default; a named constant so a change is loud and tests can pin it.
# Serving today is far below it (shadow scale), so the switch is a scale path, not
# a behavior change now.
# Lowered from 200,000 on 2026-08-13, on a measurement the original number could
# not have accounted for.
#
# The old rationale was that below the threshold an exact flat scan is FREE, and
# on its own terms that is still true: flat search over the 16,657-vector memory
# class costs 4.22ms, which is not slow. The cost it misses is what a flat scan
# does to the stage that runs NEXT. A flat search streams the whole 24.5 MiB
# float32 matrix through CPU cache and evicts the embedding model's weights, so
# the following query's encode has to refill from RAM:
#
#     memory index    search      encode-after-search
#     FLAT            4.22 ms     60.48 ms
#     HNSW            0.30 ms      6.82 ms
#
# That alternation IS the serving pattern -- request N searches, request N+1
# encodes -- so the pathological interleave is the normal case, and the flat
# index was costing 8.9x on a stage that does not appear in its own measurement.
# An HNSW walk touches a small fraction of the same data and leaves the model
# resident.
#
# 5,000 rather than 0: for a genuinely tiny population the graph build is not
# worth its own overhead, and an exact scan over a few thousand vectors has a
# cache footprint small enough not to evict anything. The deciding factor is
# FOOTPRINT, not search latency, which is the correction this constant encodes.
HNSW_THRESHOLD = 5_000


class VectorIndex(abc.ABC):
    """Interface: load vectors for a model, then rank atoms by similarity.

    Contract shared by every implementation (flat here, usearch-HNSW in Task 15):

    - ``build(store, modelId)`` populates the index from the ``embeddings`` rows
      of ``modelId`` for LIVE atoms only (superseded/tombstone atoms are not
      recallable), and returns ``self``.
    - ``search(vec, k)`` returns up to ``k`` ``(atomId, score)`` pairs sorted by
      descending ``score`` (cosine similarity). An empty index, an empty query,
      or ``k <= 0`` returns ``[]`` rather than raising.
    """

    @abc.abstractmethod
    def build(self, store, modelId):
        """Populate the index from ``modelId``'s embeddings; return ``self``."""

    @abc.abstractmethod
    def search(self, vec, k):
        """Return up to ``k`` ``(atomId, score)`` pairs, best score first."""


class FlatIndex(VectorIndex):
    """Brute-force cosine index: an in-memory matrix scanned per query.

    Exact by construction and dependency-free, which makes it the reference the
    approximate Task 15 index is measured against, and the right choice below the
    size threshold where an HNSW graph is not worth its overhead.
    """

    def __init__(self):
        self._atomIds = []
        # (n, dim) float32 of unit-normalized rows; empty until build().
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def build(self, store, modelId, kinds=None):
        kindClause, kindParams = kindInClause(kinds, alias="a")
        rows = store._conn.execute(
            "SELECT e.atom_id, e.vector FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live'" + kindClause + " "
            "ORDER BY e.atom_id",
            (modelId, *kindParams),
        ).fetchall()
        self._atomIds = [r[0] for r in rows]
        if rows:
            # np.stack copies the read-only frombuffer views into one owned,
            # writable (n, dim) array.
            self._matrix = np.stack([blobToVec(r[1]) for r in rows]).astype(
                np.float32, copy=False
            )
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        return self

    def search(self, vec, k):
        if self._matrix.shape[0] == 0 or k <= 0:
            return []
        query = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        # Stored rows are unit-normalized; normalizing the query too makes the dot
        # product a true cosine for any caller-supplied vector.
        query = query / norm
        scores = self._matrix @ query
        n = scores.shape[0]
        k = min(k, n)
        if k >= n:
            order = np.argsort(-scores, kind="stable")
        else:
            # argpartition finds the top-k unordered in O(n), then we sort only
            # those k by descending score. Deterministic for a given input, but
            # (unlike the k >= n branch) the selected subset is not in atom-id
            # order, so tie order among equal scores here is unspecified.
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part], kind="stable")]
        return [(self._atomIds[i], float(scores[i])) for i in order]


def _countEmbeddedLive(store, modelId, kinds=None):
    """Count embedded LIVE atoms for ``modelId`` -- the size the switch keys on.

    Mirrors the build-time filter EXACTLY (``embeddings`` joined to LIVE ``atoms``
    for this model, same optional ``kinds`` restriction), so the count equals the
    number of vectors the matching index would actually load. A raw ``embeddings``
    row count would over-count superseded atoms that are never in the index and
    could pick HNSW for a store that is small once the dead rows are excluded.
    """
    kindClause, kindParams = kindInClause(kinds, alias="a")
    return store._conn.execute(
        "SELECT COUNT(*) FROM embeddings e "
        "JOIN atoms a ON a.id = e.atom_id "
        "WHERE e.model_id = ? AND a.status = 'live'" + kindClause,
        (modelId, *kindParams),
    ).fetchone()[0]


def selectIndex(store, modelId, kinds=None):
    """Build and return the right ``VectorIndex`` for this population's size.

    Below :data:`HNSW_THRESHOLD` embedded live atoms (of ``kinds``, if given), an
    exact ``FlatIndex``; at or above it, the approximate ``HnswIndex``. The
    threshold is set by CACHE FOOTPRINT rather than by search latency -- see the
    constant, where the measurement is recorded. Returns the
    index already BUILT. ``kinds`` scopes the index to one kind-class so a caller
    can hold one index per population; ``kinds=None`` preserves the original
    single-index contract (every live embedded atom). The count is taken over the
    SAME ``kinds``, so a small minority class rides the exact flat scan even when
    the whole store is past the threshold. ``HnswIndex`` is imported lazily so that
    importing this module (and using ``FlatIndex`` at shadow scale) never requires
    usearch.
    """
    if _countEmbeddedLive(store, modelId, kinds) >= HNSW_THRESHOLD:
        from recall.hnsw_index import HnswIndex

        return HnswIndex().build(store, modelId, kinds)
    return FlatIndex().build(store, modelId, kinds)


def buildClassIndexes(store, modelId):
    """One built ``VectorIndex`` per kind-class in ``KIND_CLASSES``, keyed by name.

    ``{"memory": <index>, "code": <index>}`` today. Each class picks flat vs HNSW
    independently by its own live-embedded count, so the small memory population
    gets an exact scan while the large code population gets the approximate graph.
    An empty class (no live embedded atoms of its kinds) yields an empty but valid
    index whose ``search`` returns ``[]``.
    """
    return {
        name: selectIndex(store, modelId, kinds)
        for name, kinds in KIND_CLASSES
    }
