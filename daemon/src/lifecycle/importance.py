"""Retrieval-driven importance accrual.

Near-duplicate reinforcement happens during capture in the ambient distiller.
This lifecycle job handles the between-session retrieval signal recorded in
``recall_log``. Serve-layer wiring is intentionally separate: callers should add
one recall_log row per retrieved atom, then this job marks rows processed so
re-runs never double-count.
"""
import time

__all__ = ["accrueImportance"]

_POINTS_PER_WEIGHT = 0.01
_IMPORTANCE_CAP = 1.0


def accrueImportance(store):
    """Apply unprocessed retrieval usage to atom importance.

    Returns ``{"processed": recall_rows, "updated": atom_rows}``. Importance is
    capped at 1.0 to match recall's ranking cap, and processed recall_log rows are
    stamped in the same transaction as the atom updates.
    """
    conn = store._conn
    rows = conn.execute(
        "SELECT atom_id, SUM(weight), COUNT(*) FROM recall_log "
        "WHERE processed_at IS NULL GROUP BY atom_id ORDER BY atom_id"
    ).fetchall()
    if not rows:
        return {"processed": 0, "updated": 0}

    now = int(time.time())
    processed = sum(row[2] for row in rows)
    updated = 0
    try:
        for atomId, weight, _count in rows:
            before = conn.execute(
                "SELECT importance FROM atoms WHERE id = ?", (atomId,)
            ).fetchone()
            if before is None:
                raise ValueError(f"recall_log references missing atom {atomId!r}")
            after = min(_IMPORTANCE_CAP, before[0] + max(0.0, weight) * _POINTS_PER_WEIGHT)
            if after != before[0]:
                conn.execute(
                    "UPDATE atoms SET importance = ? WHERE id = ?", (after, atomId)
                )
                updated += 1
        conn.execute(
            "UPDATE recall_log SET processed_at = ? WHERE processed_at IS NULL",
            (now,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"processed": processed, "updated": updated}
