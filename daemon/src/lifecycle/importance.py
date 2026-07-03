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

    Returns ``{"processed": recall_rows, "updated": atom_rows, "missingAtoms":
    skipped_rows}``. Importance is capped at 1.0 to match recall's ranking cap.
    Missing-atom recall rows are stamped and counted so a corrupt/log-skew row
    cannot crash every future run. Only the row ids selected at the start of the
    job are stamped; rows inserted while the job runs stay unprocessed for the
    next invocation.
    """
    conn = store._conn
    rows = conn.execute(
        "SELECT id, atom_id, weight FROM recall_log "
        "WHERE processed_at IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return {"processed": 0, "updated": 0, "missingAtoms": 0}

    rowIds = [row[0] for row in rows]
    weightByAtom = {}
    countByAtom = {}
    for _rowId, atomId, weight in rows:
        weightByAtom[atomId] = weightByAtom.get(atomId, 0.0) + weight
        countByAtom[atomId] = countByAtom.get(atomId, 0) + 1

    now = int(time.time())
    processed = len(rows)
    updated = 0
    missing = 0
    try:
        for atomId, weight in weightByAtom.items():
            before = conn.execute(
                "SELECT importance FROM atoms WHERE id = ?", (atomId,)
            ).fetchone()
            if before is None:
                missing += countByAtom[atomId]
                continue
            after = min(_IMPORTANCE_CAP, before[0] + max(0.0, weight) * _POINTS_PER_WEIGHT)
            if after != before[0]:
                conn.execute(
                    "UPDATE atoms SET importance = ? WHERE id = ?", (after, atomId)
                )
                updated += 1
        placeholders = ", ".join("?" for _ in rowIds)
        conn.execute(
            f"UPDATE recall_log SET processed_at = ? WHERE id IN ({placeholders})",
            (now, *rowIds),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"processed": processed, "updated": updated, "missingAtoms": missing}
