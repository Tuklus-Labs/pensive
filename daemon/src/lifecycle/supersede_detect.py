"""Supersession proposal job.

This job proposes only. It never calls ``store.supersede()``, never writes
``supersedes`` edges, and never changes atom status. The v3.1 verbatim seam is
bound here now: any atom with provenance from ``person-import`` or a
``person-*`` source is excluded from proposals entirely, because a person's own
words are not candidates for automatic supersession.
"""
import math
import time

from util.ulid import ulid

__all__ = ["detectSupersession"]

_SIMILARITY_THRESHOLD = 0.98
_FACET_KEYS = ("entity", "tag", "src")


def _isPersonSource(source):
    return source == "person-import" or source.startswith("person-")


def _dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(vec):
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _cosine(a, b):
    denom = _norm(a) * _norm(b)
    if denom == 0.0:
        return 0.0
    return _dot(a, b) / denom


def _candidateRows(conn):
    personSources = conn.execute(
        "SELECT DISTINCT atom_id FROM provenance"
    ).fetchall()
    excluded = set()
    for (atomId,) in personSources:
        sources = conn.execute(
            "SELECT source FROM provenance WHERE atom_id = ?", (atomId,)
        ).fetchall()
        if any(_isPersonSource(row[0]) for row in sources):
            excluded.add(atomId)

    rows = conn.execute(
        "SELECT rowid, id, text, created_at FROM atoms "
        "WHERE status = 'live' ORDER BY created_at, rowid"
    ).fetchall()
    return [row for row in rows if row[1] not in excluded]


def _facets(conn, atomId):
    return {
        (row[0], row[1])
        for row in conn.execute(
            "SELECT key, value FROM facets WHERE atom_id = ? AND key IN (?, ?, ?)",
            (atomId, *_FACET_KEYS),
        ).fetchall()
    }


def detectSupersession(store, embedder):
    """Write reviewable supersession proposals for similar same-facet atoms.

    The proposal surface is the ``supersession_proposals`` table. Rows are unique
    by ``(old_atom_id, new_atom_id)``, so re-running the job is idempotent.
    """
    conn = store._conn
    rows = _candidateRows(conn)
    if len(rows) < 2:
        return {"scanned": len(rows), "proposed": 0}

    vectors = embedder.embed([row[2] for row in rows])
    if len(vectors) != len(rows):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(rows)} texts")

    facetById = {row[1]: _facets(conn, row[1]) for row in rows}
    now = int(time.time())
    proposed = 0
    try:
        for i, older in enumerate(rows):
            for j in range(i + 1, len(rows)):
                newer = rows[j]
                oldId = older[1]
                newId = newer[1]
                if older[2] == newer[2]:
                    continue
                shared = facetById[oldId] & facetById[newId]
                if not shared:
                    continue
                similarity = _cosine(vectors[i], vectors[j])
                if similarity < _SIMILARITY_THRESHOLD:
                    continue
                reason = "high-similarity same-facet newer atom; human review required"
                conn.execute(
                    "INSERT OR IGNORE INTO supersession_proposals("
                    "id, old_atom_id, new_atom_id, similarity, reason, status, created_at"
                    ") VALUES (?, ?, ?, ?, ?, 'proposed', ?)",
                    (ulid(), oldId, newId, similarity, reason, now),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    proposed += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"scanned": len(rows), "proposed": proposed}
