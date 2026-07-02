"""Rebuild a fresh canonical store from a JSONL export -- the decades guarantee.

This is the proof that the derived layers are genuinely derived: given only the
four JSONL files :func:`store.export.exportJSONL` writes, ``rebuild`` reconstructs
a store whose canonical tables are byte-for-byte identical to the original. The
``embeddings`` and ``fts`` tables are absent from the export on purpose -- the
fts index regenerates because atoms are inserted through the normal triggers, and
embeddings are re-derived later by the Phase 2 embedder.

Two rules make it safe to lean on for decades:

  * Rows are inserted in FK order -- atoms, then provenance, then edges, then
    facets -- so every foreign key resolves at insert time (edges may reference a
    provenance row, so provenance must precede edges; all three reference atoms).
    ``PRAGMA foreign_keys`` is on, so a malformed export fails loudly here rather
    than producing a silently broken store.
  * ``rebuild`` refuses to write over an existing database. Overwriting a live
    store mid-rebuild is exactly the irreversible loss the canonical layer exists
    to prevent, so a pre-existing target is an error, never a clobber.
"""
import json
from pathlib import Path

from store.store import openStore
from store.export import ATOM_COLS, PROVENANCE_COLS, EDGE_COLS, FACET_COLS

__all__ = ["rebuild"]

# (filename, table, columns) in FK-safe insertion order: when each table loads,
# every foreign key it carries already points at a loaded row.
_LOAD_ORDER = (
    ("atoms.jsonl", "atoms", ATOM_COLS),
    ("provenance.jsonl", "provenance", PROVENANCE_COLS),
    ("edges.jsonl", "edges", EDGE_COLS),
    ("facets.jsonl", "facets", FACET_COLS),
)


def _readRows(path):
    """Yield each JSONL line of ``path`` as a dict.

    A missing file yields nothing: an empty table exports to an empty file, and a
    rebuild must tolerate that file being absent just the same as being empty.
    """
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _insertRows(conn, table, cols, rows):
    sql = (
        f"INSERT INTO {table}({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    for obj in rows:
        conn.execute(sql, tuple(obj[c] for c in cols))


def rebuild(fromDir, newDbPath):
    """Build a fresh store at ``newDbPath`` from the JSONL export in ``fromDir``.

    Returns the open :class:`store.store.Store`. The database is created by
    :func:`store.store.openStore` (schema installed, version stamped), then the
    four canonical tables are re-inserted in FK-safe order via plain INSERTs --
    which fire the atoms fts triggers and regenerate the full-text index. Every
    column value is preserved verbatim from the export (ids, timestamps,
    statuses, weights, per-row ``schema_version``).

    Refuses to overwrite: if ``newDbPath`` already exists, raises
    ``FileExistsError`` rather than clobbering a possibly-live store.
    """
    srcDir = Path(fromDir)
    target = Path(newDbPath)
    if target.exists():
        raise FileExistsError(
            f"rebuild target already exists: {target}; refusing to overwrite "
            "(canonical data must never be clobbered)"
        )
    store = openStore(target)
    conn = store._conn
    try:
        for filename, table, cols in _LOAD_ORDER:
            _insertRows(conn, table, cols, _readRows(srcDir / filename))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return store
