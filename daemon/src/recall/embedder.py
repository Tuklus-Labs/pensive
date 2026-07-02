"""Model-keyed text embedder and the embeddings-table writer.

Embeddings are DERIVED data (cattle, not pet): rebuildable from atom text at any
time, and keyed by ``model_id`` so re-embedding under a new model is additive --
it never overwrites another model's vectors. The blob format is fixed by
``schema.sql``: float32 little-endian bytes, one vector per row, dimension set by
the model (384 for bge-small-en-v1.5).

Runtime note: torch + sentence-transformers are the Phase 0 runtime decision (the
only stack that reaches this box's AMD 7900 XTX via ROCm), so they are sanctioned
here. They are imported lazily inside ``Embedder`` so that importing this module
for the blob helpers alone stays cheap and side-effect free.
"""
import os
import time

import numpy as np

__all__ = ["Embedder", "embedMissing", "vecToBlob", "blobToVec"]

# Plan-specified batch size. The spike used 32; the plan's number governs here.
_BATCH_SIZE = 64

# schema.sql fixes the on-disk vector encoding as float32 little-endian. "<f4"
# forces that byte order regardless of host endianness, so the store stays
# portable even though this box happens to be little-endian already.
_BLOB_DTYPE = "<f4"


def vecToBlob(vec):
    """Serialize one vector to the schema's float32 little-endian BLOB bytes."""
    return np.asarray(vec, dtype=_BLOB_DTYPE).tobytes()


def blobToVec(blob):
    """Read one vector back from a float32 little-endian BLOB.

    Returns a 1-D float32 array. ``np.frombuffer`` yields a read-only view over
    the bytes; callers that need to mutate or stack copy first (``FlatIndex.build``
    stacks, which copies). The round-trip ``blobToVec(vecToBlob(v))`` reproduces
    every float32 value exactly -- the cast is float32 -> float32.
    """
    return np.frombuffer(blob, dtype=_BLOB_DTYPE)


class Embedder:
    """Loads one sentence-transformers model and embeds text to unit vectors.

    Construct once and reuse: the model is ~130MB and shares VRAM with other
    processes on this box, so a single resident copy is the rule (tests load it in
    a session-scoped fixture). ``embed`` batches internally and returns
    unit-normalized float32 vectors, so a flat cosine search reduces to a dot
    product.
    """

    def __init__(self, modelId):
        # Force offline resolution from the HF cache. setdefault (not assignment)
        # so an explicit caller/CI override still wins.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer

        # ROCm presents as CUDA to torch, so cuda.is_available() is the GPU probe
        # on this box; fall back to CPU where there is no device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelId = modelId
        self._model = SentenceTransformer(modelId, device=self.device)
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension; prefer the new name, keep the old for the
        # >=2.2 floor. Reading the attribute this way avoids the FutureWarning
        # the deprecated call emits on 5.x.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts):
        """Embed ``texts`` -> list of unit-normalized float32 vectors (dim ``self.dim``).

        Batched at ``_BATCH_SIZE`` in one ``encode`` call (not a per-text loop);
        an empty input yields an empty list without touching the model.
        """
        if not texts:
            return []
        matrix = self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        # .copy() so each returned vector owns its buffer -- a row view would
        # alias the batch matrix, and a consumer's in-place op would silently
        # mutate its siblings. 384 float32s per copy, negligible.
        return [matrix[i].copy() for i in range(matrix.shape[0])]


def embedMissing(store, embedder):
    """Embed every LIVE atom lacking an ``embeddings`` row for ``embedder.modelId``.

    Idempotent and additive: it selects only atoms with no row under this model,
    so a second call with nothing new to do embeds nothing and returns 0. Because
    embeddings are keyed by ``(atom_id, model_id)``, embedding under a different
    model adds rows rather than replacing this model's. Non-live atoms
    (superseded/tombstone) are skipped -- their text stays readable but they are
    not recallable, so spending GPU time embedding them would be waste.

    Returns the number of atoms embedded this call. All writes commit as one
    transaction; a failure mid-batch rolls back and leaves the table untouched.
    """
    conn = store._conn
    rows = conn.execute(
        "SELECT a.id, a.text FROM atoms a "
        "WHERE a.status = 'live' AND NOT EXISTS ("
        "  SELECT 1 FROM embeddings e "
        "  WHERE e.atom_id = a.id AND e.model_id = ?"
        ") ORDER BY a.id",
        (embedder.modelId,),
    ).fetchall()
    if not rows:
        return 0

    atomIds = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vecs = embedder.embed(texts)
    now = int(time.time())
    try:
        conn.executemany(
            "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (atomId, embedder.modelId, vecToBlob(vec), now)
                for atomId, vec in zip(atomIds, vecs)
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(atomIds)
