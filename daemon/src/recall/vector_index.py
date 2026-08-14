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
    - ``add`` and ``remove`` maintain the index incrementally, so a write does
      not pay a rebuild. They are a PAIR; see below for why neither is optional.

    WHY ``add`` WITHOUT ``remove`` IS WRONG, recorded once here because both
    methods depend on it. A rebuild loads live atoms only, which enforced "a
    retracted atom is not recallable" implicitly. Incremental maintenance has to
    enforce it explicitly, and the tempting argument against that is wrong:
    ``recall.trust.assessTrust`` does resolve ``status`` per query, but it does
    not DROP a superseded atom, it annotates it with ``supersededBy`` at a capped
    confidence, so the retired claim still reaches the payload beside the current
    one. `test_correct_supersedes_and_recall_shows_current_truth` is what caught
    that, by asking a question whose retired answer reappeared next to its own
    correction.
    """

    @abc.abstractmethod
    def build(self, store, modelId):
        """Populate the index from ``modelId``'s embeddings; return ``self``."""

    @abc.abstractmethod
    def search(self, vec, k):
        """Return up to ``k`` ``(atomId, score)`` pairs, best score first."""

    @abc.abstractmethod
    def add(self, atomId, vec):
        """Add ONE already-embedded atom without rebuilding; return ``self``.

        ``build`` costs 272ms for the memory class and 18.6s for the code class
        on this store, and every emit used to pay it synchronously on the
        event-loop thread. One ``add`` costs 0.115ms.

        ``vec`` must be unit-normalized, as ``build`` requires, because search
        treats stored rows as unit length. Pairs with ``remove``.
        """

    @abc.abstractmethod
    def remove(self, atomId):
        """Stop returning ``atomId``; return True if it was present.

        Not optional: see the class docstring for why ``add`` alone leaves a
        retracted atom recallable.
        """


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
        # Row positions retired since the last build. A rebuild loads live atoms
        # only, so it starts empty again.
        self._retired = set()

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
        self._retired = set()
        if rows:
            # np.stack copies the read-only frombuffer views into one owned,
            # writable (n, dim) array.
            self._matrix = np.stack([blobToVec(r[1]) for r in rows]).astype(
                np.float32, copy=False
            )
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        return self

    def remove(self, atomId):
        # Masked, not deleted. A row's POSITION is its identity here (_atomIds[i]
        # names _matrix[i]), so deleting a row would renumber every atom after it
        # and hand back wrong ids for correct vectors. The mask costs one bool
        # per row and is applied in search.
        try:
            pos = self._atomIds.index(atomId)
        except ValueError:
            return False
        self._retired.add(pos)
        return True

    def add(self, atomId, vec):
        row = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self._matrix.shape[0] == 0:
            # A never-built or empty index carries shape (0, 0), so vstack would
            # raise on the dimension mismatch. The first row defines the width.
            self._matrix = row.copy()
        else:
            self._matrix = np.vstack([self._matrix, row])
        self._atomIds.append(atomId)
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
        # Retired rows are EXCLUDED, not merely scored low. Scoring them -inf was
        # the first attempt and it does not work: when k >= n this method returns
        # every row sorted by score, so a retired atom still comes back, just
        # ranked last. That is exactly the failure
        # `test_correct_supersedes_and_recall_shows_current_truth` reports, since
        # a corrected fact reappearing at the bottom of the payload is still the
        # corrected fact reappearing.
        candidates = np.arange(scores.shape[0])
        if self._retired:
            keep = np.ones(scores.shape[0], dtype=bool)
            keep[list(self._retired)] = False
            candidates = candidates[keep]
            if candidates.size == 0:
                return []
        sub = scores[candidates]
        n = sub.shape[0]
        k = min(k, n)
        if k >= n:
            order = candidates[np.argsort(-sub, kind="stable")]
        else:
            # argpartition finds the top-k unordered in O(n), then we sort only
            # those k by descending score. Deterministic for a given input, but
            # (unlike the k >= n branch) the selected subset is not in atom-id
            # order, so tie order among equal scores here is unspecified.
            part = np.argpartition(-sub, k - 1)[:k]
            order = candidates[part[np.argsort(-sub[part], kind="stable")]]
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
