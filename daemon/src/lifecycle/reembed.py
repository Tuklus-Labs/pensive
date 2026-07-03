"""Add embeddings for a new model without disturbing existing model rows.

The active recall model id is owned by the serving/config surface that constructs
``Embedder(modelId)`` and calls ``recall.vector_index.selectIndex(store,
modelId)``. This job only prepares additive rows for ``newModelId``; callers cut
over that active model id after their external eval gate passes, and old rows are
dropped only by a separate explicit operation outside this module.
"""
import struct
import time

__all__ = ["reembed"]


def _vecToBlob(vec):
    return struct.pack("<" + "f" * len(vec), *[float(v) for v in vec])


def reembed(store, newModelId, embedder):
    """Embed live atoms missing ``newModelId`` rows.

    Idempotent and strictly additive: only ``embeddings`` rows for
    ``newModelId`` are inserted, old model rows remain untouched, and atoms,
    edges, facets, and provenance are never modified.
    """
    conn = store._conn
    rows = conn.execute(
        "SELECT a.id, a.text FROM atoms a "
        "WHERE a.status = 'live' AND NOT EXISTS ("
        "  SELECT 1 FROM embeddings e "
        "  WHERE e.atom_id = a.id AND e.model_id = ?"
        ") ORDER BY a.id",
        (newModelId,),
    ).fetchall()
    if not rows:
        return 0

    atomIds = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    vectors = embedder.embed(texts)
    if len(vectors) != len(atomIds):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(atomIds)} texts"
        )

    now = int(time.time())
    try:
        cursor = conn.executemany(
            "INSERT OR IGNORE INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (atomId, newModelId, _vecToBlob(vec), now)
                for atomId, vec in zip(atomIds, vectors)
            ],
        )
        inserted = cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return inserted
