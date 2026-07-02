"""The vector-search seam and its flat (brute-force) implementation.

``VectorIndex`` is the interface Phase 2 recall searches against. It is kept
deliberately minimal -- ``build`` then ``search`` -- so that the Task 15
usearch-HNSW index can drop in behind the same two methods and a size-threshold
switch can pick between them without either caller or the interface knowing which
concrete index it holds. Nothing here leaks a flat-scan assumption: ``search``
takes a query vector and a ``k`` and returns ranked ``(atomId, score)`` pairs;
how those are found (linear scan vs. approximate graph traversal) is the
implementation's business.

Scores are cosine similarity in ``[-1, 1]``, higher is better. Stored vectors are
unit-normalized by the embedder, so the flat implementation's cosine reduces to a
matrix-vector dot product.
"""
import abc

import numpy as np

from recall.embedder import blobToVec

__all__ = ["VectorIndex", "FlatIndex"]


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

    def build(self, store, modelId):
        rows = store._conn.execute(
            "SELECT e.atom_id, e.vector FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live' "
            "ORDER BY e.atom_id",
            (modelId,),
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
            # those k. Stable sort keeps ties in atom-id order (build sorts by id).
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part], kind="stable")]
        return [(self._atomIds[i], float(scores[i])) for i in order]
