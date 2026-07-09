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

from repair_lib import parseFilesSummary, abspathToRef  # noqa: E402

__all__ = ["repairKvCacheRefs"]

_ROWID_RE = re.compile(r"^kv_cache/vector_meta\.db#rowid=(\d+)$")


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
