"""Add embeddings for a new model without disturbing existing model rows.

The active recall model id is owned by the serving/config surface that constructs
``Embedder(modelId)`` and calls ``recall.vector_index.selectIndex(store,
modelId)``. This job only prepares additive rows for ``newModelId``; callers cut
over that active model id after their external eval gate passes, and old rows are
dropped only by a separate explicit operation outside this module.
"""
import struct
import time

__all__ = ["dropOldModel", "reembed"]


def _vecToBlob(vec):
    return struct.pack("<" + "f" * len(vec), *[float(v) for v in vec])


def reembed(store, newModelId, embedder):
    """Embed live atoms missing ``newModelId`` rows.

    Idempotent and strictly additive: only ``embeddings`` rows for
    ``newModelId`` are inserted, old model rows remain untouched, and atoms,
    edges, facets, and provenance are never modified.
    """
    embedderModelId = getattr(embedder, "modelId", None)
    if embedderModelId is not None and embedderModelId != newModelId:
        raise ValueError(
            f"embedder modelId {embedderModelId!r} does not match newModelId "
            f"{newModelId!r}"
        )
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


def dropOldModel(store, oldModelId, activeModelId):
    """Delete embeddings for ``oldModelId`` after structural safety checks."""
    if oldModelId == activeModelId:
        raise ValueError("refusing to drop the active model")

    conn = store._conn
    live = conn.execute("SELECT COUNT(*) FROM atoms WHERE status = 'live'").fetchone()[0]
    covered = conn.execute(
        "SELECT COUNT(*) FROM embeddings e "
        "JOIN atoms a ON a.id = e.atom_id "
        "WHERE e.model_id = ? AND a.status = 'live'",
        (activeModelId,),
    ).fetchone()[0]
    if covered != live:
        raise ValueError(
            f"active model coverage is incomplete: {covered}/{live} live atoms"
        )

    try:
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE model_id = ?",
            (oldModelId,),
        )
        deleted = cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return deleted
