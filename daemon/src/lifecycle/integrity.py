"""Integrity scan for canonical memory stores.

The scan returns a plain dict. Nerve-center publishing is deliberately an
injectable caller concern; this module has no socket or HTTP dependency.
"""
import hashlib

__all__ = ["integrityScan"]


def _dicts(rows, keys):
    return [dict(zip(keys, row)) for row in rows]


def _orphanEdges(conn):
    rows = conn.execute(
        "SELECT e.id, e.src_atom, e.dst_atom, "
        "src.id IS NULL AS missing_src, dst.id IS NULL AS missing_dst "
        "FROM edges e "
        "LEFT JOIN atoms src ON src.id = e.src_atom "
        "LEFT JOIN atoms dst ON dst.id = e.dst_atom "
        "WHERE src.id IS NULL OR dst.id IS NULL "
        "ORDER BY e.id"
    ).fetchall()
    out = []
    for edgeId, src, dst, missingSrc, missingDst in rows:
        missing = []
        if missingSrc:
            missing.append("srcAtom")
        if missingDst:
            missing.append("dstAtom")
        out.append({"id": edgeId, "srcAtom": src, "dstAtom": dst, "missing": missing})
    return out


def _embeddingCoverage(conn):
    live = conn.execute("SELECT COUNT(*) FROM atoms WHERE status = 'live'").fetchone()[0]
    models = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT model_id FROM embeddings ORDER BY model_id"
        ).fetchall()
    ]
    coverage = {}
    for modelId in models:
        embedded = conn.execute(
            "SELECT COUNT(*) FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live'",
            (modelId,),
        ).fetchone()[0]
        coverage[modelId] = {
            "liveAtoms": live,
            "embeddedLiveAtoms": embedded,
            "missingLiveAtoms": live - embedded,
        }
    return coverage


def _orphanRows(conn, table):
    rows = conn.execute(
        f"SELECT t.* FROM {table} t LEFT JOIN atoms a ON a.id = t.atom_id "
        "WHERE a.id IS NULL"
    ).fetchall()
    keys = [col[0] for col in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    return _dicts(rows, keys)


def _supersessionSanity(conn):
    rows = conn.execute(
        "SELECT e.id, e.src_atom, e.dst_atom FROM edges e "
        "WHERE e.type = 'supersedes' ORDER BY e.id"
    ).fetchall()
    dangling = []
    graph = {}
    for edgeId, src, dst in rows:
        srcExists = conn.execute("SELECT 1 FROM atoms WHERE id = ?", (src,)).fetchone()
        dstExists = conn.execute("SELECT 1 FROM atoms WHERE id = ?", (dst,)).fetchone()
        if srcExists is None or dstExists is None:
            dangling.append({"id": edgeId, "srcAtom": src, "dstAtom": dst})
        graph.setdefault(src, []).append(dst)

    cycles = []
    for start in graph:
        seen = set()
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in graph.get(node, []):
                if nxt == start:
                    cycles.append(path + [nxt])
                elif nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, path + [nxt]))
    return {"danglingSuccessors": dangling, "cycles": cycles}


def _checksums(conn):
    rows = conn.execute("SELECT id, text FROM atoms ORDER BY id").fetchall()
    h = hashlib.sha256()
    for atomId, text in rows:
        h.update(atomId.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    ftsMissing = conn.execute(
        "SELECT a.id FROM atoms a LEFT JOIN fts ON fts.rowid = a.rowid "
        "WHERE fts.rowid IS NULL ORDER BY a.id"
    ).fetchall()
    return {
        "atomTextSha256": h.hexdigest(),
        "ftsMissingAtoms": [row[0] for row in ftsMissing],
    }


def integrityScan(store):
    """Return a plain integrity report and raise only for operational faults."""
    conn = store._conn
    report = {
        "checksums": _checksums(conn),
        "orphanEdges": _orphanEdges(conn),
        "embeddingCoverage": _embeddingCoverage(conn),
        "orphanFacets": _orphanRows(conn, "facets"),
        "orphanProvenance": _orphanRows(conn, "provenance"),
        "supersessionChains": _supersessionSanity(conn),
    }
    report["ok"] = not (
        report["orphanEdges"]
        or report["orphanFacets"]
        or report["orphanProvenance"]
        or report["supersessionChains"]["danglingSuccessors"]
        or report["supersessionChains"]["cycles"]
        or report["checksums"]["ftsMissingAtoms"]
    )
    return report
