"""Store-mutation passes for the Phase A corpus repair.

Each pass takes an open store (and whatever inputs it needs), mutates in one
commit, and returns a flat report dict of counts. Passes are idempotent: the
row predicates that select work exclude already-repaired rows, so a second run
reports zeros. Nothing here deletes an atom; duplicate cleanup (Task 3) uses
the store's supersede().

Daemon-internal reach-through convention: like the recall modules, passes read
and write through store._conn for batched work the public API does not expose.
"""
import re
import sqlite3
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_SRC = _TOOLS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from repair_lib import parseFilesSummary, abspathToRef, refToProject  # noqa: E402
from store.store import supersede  # noqa: E402

__all__ = [
    "repairKvCacheRefs", "dedupReferenceLibrary", "backfillProjects",
    "totals", "verifyRepair",
]

_ROWID_RE = re.compile(r"^kv_cache/vector_meta\.db#rowid=(\d+)$")
_REFLIB_DUP_ROOT = "reference-library/"
_REFLIB_CANON_ROOT = "projects/Aegis/AEGIS/docs/reference-library/"


def repairKvCacheRefs(store, oldDbPath):
    """Rewrite kv_cache rowid refs from the retired store's summaries.

    For every provenance row on a live-or-superseded document_chunk whose
    source_ref matches ``kv_cache/vector_meta.db#rowid=N``: look up rowid N in
    the retired db, parse the absolute path out of its summary, and rewrite the
    ref to the standard root convention. Backfill atoms.project from the path
    only when project is NULL. Unparseable summaries and missing rowids are
    counted and left untouched (never guess). Idempotent: rewritten refs no
    longer match the predicate.
    """
    report = {"rewritten": 0, "projectBackfilled": 0,
              "noPathInSummary": 0, "rowidMissing": 0}

    rows = store._conn.execute(
        "SELECT p.id, p.atom_id, p.source_ref FROM provenance p "
        "JOIN atoms a ON a.id = p.atom_id "
        "WHERE a.kind = 'document_chunk' AND p.source_ref LIKE 'kv_cache%'"
    ).fetchall()
    if not rows:
        return report

    old = sqlite3.connect(f"file:{oldDbPath}?mode=ro", uri=True)
    try:
        summaryOf = {}
        for provId, atomId, ref in rows:
            m = _ROWID_RE.match(ref)
            if m is None:
                report["noPathInSummary"] += 1
                continue
            rowid = int(m.group(1))
            if rowid not in summaryOf:
                got = old.execute(
                    "SELECT summary FROM meta WHERE rowid = ?", (rowid,)
                ).fetchone()
                summaryOf[rowid] = got[0] if got else None
                if got is None:
                    summaryOf[rowid] = False  # sentinel: rowid absent
            summary = summaryOf[rowid]
            if summary is False:
                report["rowidMissing"] += 1
                continue
            path = parseFilesSummary(summary)
            if path is None:
                report["noPathInSummary"] += 1
                continue
            mapped = abspathToRef(path)
            if mapped is None:
                report["noPathInSummary"] += 1
                continue
            newRef, project = mapped
            store._conn.execute(
                "UPDATE provenance SET source_ref = ? WHERE id = ?",
                (newRef, provId))
            report["rewritten"] += 1
            if project is not None:
                cur = store._conn.execute(
                    "UPDATE atoms SET project = ? "
                    "WHERE id = ? AND project IS NULL",
                    (project, atomId))
                report["projectBackfilled"] += cur.rowcount
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise
    finally:
        old.close()
    return report


def dedupReferenceLibrary(store, sessionId=None):
    """Supersede null-root reflib chunks that duplicate the canonical copies.

    For each LIVE document_chunk whose ref is ``reference-library/<tail>``:
    find LIVE canonical twins at ``projects/Aegis/.../reference-library/<tail>``.
    Exactly one twin with EXACTLY matching text: supersede the null-root copy
    (survivor = the canonical copy, which carries project attribution). Zero
    twins, multiple twins, or text drift: count and leave live; a forced merge
    of drifted content would silently lose the difference. Idempotent: the
    LIVE predicate excludes already-superseded copies, and the candidate scan
    excludes this pass's own provenance rows (source='repair-tool') so a
    survivor's supersede-record -- written with the loser's old source_ref --
    is never mistaken for a fresh duplicate on rerun. Provenance rows this
    pass writes (source='repair-tool') are excluded from the candidate scan
    by design, so the pass's own writes can never re-trigger it.

    The candidate scan JOINs provenance, so an atom with more than one
    matching ``reference-library/...`` provenance row would otherwise surface
    once per row. The loop dedupes candidate atom ids (first occurrence wins)
    and re-checks each dup's status immediately before superseding it, so a
    multi-provenance dup is superseded at most once even if the dedupe above
    were ever bypassed.
    """
    report = {"superseded": 0, "noTwin": 0, "textMismatch": 0,
              "multipleTwins": 0}
    dups = store._conn.execute(
        "SELECT a.id, p.source_ref, a.text FROM atoms a "
        "JOIN provenance p ON p.atom_id = a.id "
        "WHERE a.kind = 'document_chunk' AND a.status = 'live' "
        "AND p.source_ref LIKE ? AND p.source != 'repair-tool'",
        (_REFLIB_DUP_ROOT + "%",)
    ).fetchall()
    seen = set()
    for dupId, ref, text in dups:
        if dupId in seen:
            continue
        seen.add(dupId)
        tail = ref[len(_REFLIB_DUP_ROOT):]
        twins = store._conn.execute(
            "SELECT a.id, a.text FROM atoms a "
            "JOIN provenance p ON p.atom_id = a.id "
            "WHERE a.kind = 'document_chunk' AND a.status = 'live' "
            "AND p.source_ref = ?",
            (_REFLIB_CANON_ROOT + tail,)
        ).fetchall()
        if not twins:
            report["noTwin"] += 1
            continue
        if len(twins) > 1:
            report["multipleTwins"] += 1
            continue
        twinId, twinText = twins[0]
        if twinText != text:
            report["textMismatch"] += 1
            continue
        # Fresh single-row status check right before the write: a
        # multi-provenance dup could otherwise be superseded twice (inflated
        # count, redundant edge+provenance) if it ever reached this point
        # more than once. Not counted as superseded when already handled.
        current = store._conn.execute(
            "SELECT status FROM atoms WHERE id = ?", (dupId,)
        ).fetchone()
        if current is None or current[0] != 'live':
            continue
        prov = {"source": "repair-tool", "sourceRef": ref}
        if sessionId is not None:
            prov["sessionId"] = sessionId
        supersede(store, dupId, twinId, prov)
        report["superseded"] += 1
    return report


def backfillProjects(store):
    """Backfill NULL projects derivable from refs; count the rest.

    Only document_chunk atoms; only where project IS NULL; the derivation is
    refToProject (projects/<name>/ and reference-library/ resolve, dotfile and
    kv_cache roots do not). Idempotent: backfilled rows leave the predicate.
    """
    report = {"backfilled": 0, "unresolvable": 0}
    rows = store._conn.execute(
        "SELECT a.id, p.source_ref FROM atoms a "
        "JOIN provenance p ON p.atom_id = a.id "
        "WHERE a.kind = 'document_chunk' AND a.project IS NULL"
    ).fetchall()
    try:
        for atomId, ref in rows:
            project = refToProject(ref) if ref else None
            if project is None:
                report["unresolvable"] += 1
                continue
            store._conn.execute(
                "UPDATE atoms SET project = ? WHERE id = ? AND project IS NULL",
                (project, atomId))
            report["backfilled"] += 1
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise
    return report


def totals(store):
    """The invariant counters verifyRepair checks against."""
    total = store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    live = store._conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE status='live'").fetchone()[0]
    return {"total": total, "live": live}


def verifyRepair(store, preTotals):
    """Post-run invariants: no atom created or destroyed; kv refs gone or known.

    ``atomTotalDelta`` must be zero (supersession changes status, never count).
    ``liveDelta`` is informational (dedup reduces live count by design).
    ``ok`` is True when the total held.
    """
    post = totals(store)
    kvRemaining = store._conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE source_ref LIKE 'kv_cache%'"
    ).fetchone()[0]
    delta = post["total"] - preTotals["total"]
    return {
        "ok": delta == 0,
        "atomTotalDelta": delta,
        "liveDelta": post["live"] - preTotals["live"],
        "kvRefsRemaining": kvRemaining,
    }
