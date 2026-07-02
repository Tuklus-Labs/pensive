"""Rebuild a fresh canonical store from a JSONL export -- the decades guarantee.

This is the proof that the derived layers are genuinely derived: given only the
four JSONL files :func:`store.export.exportJSONL` writes, ``rebuild`` reconstructs
a store whose canonical tables are byte-for-byte identical to the original. The
``embeddings`` and ``fts`` tables are absent from the export on purpose -- the
fts index regenerates because atoms are inserted through the normal triggers, and
embeddings are re-derived later by the Phase 2 embedder.

Three rules make it safe to lean on for decades:

  * The export must be COMPLETE. ``exportJSONL`` always writes all four files (an
    empty table yields an empty file), so a missing file is a truncated or
    damaged export, never a legitimately empty table. ``rebuild`` refuses it up
    front rather than silently reconstructing a store missing an entire canonical
    table.
  * Rows are inserted in FK order -- atoms, then provenance, then edges, then
    facets -- so every foreign key resolves at insert time (edges may reference a
    provenance row, so provenance must precede edges; all three reference atoms).
    ``PRAGMA foreign_keys`` is on, so a malformed export fails loudly here rather
    than producing a silently broken store.
  * ``rebuild`` refuses to write over an existing database, and if its own insert
    pass fails it removes the half-built file it just created. Either way the
    target path never ends up holding a partial or clobbered store.
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

    The caller guarantees the file exists (``rebuild`` checks export completeness
    up front). An empty file yields no rows -- that is how an empty table round
    trips.
    """
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

    Preconditions and cleanup, so the target path is never left in a bad state:

    * If ``newDbPath`` already exists, raises ``FileExistsError`` rather than
      clobbering a possibly-live store.
    * The export must be complete: all four files must exist (empty files are
      fine -- they mean empty tables). A missing file raises ``FileNotFoundError``
      naming it, before any database is created.
    * If the insert pass fails partway, the half-built database is removed and the
      error re-raised, so a corrected retry is not blocked by our own leftover.
    """
    srcDir = Path(fromDir)
    target = Path(newDbPath)
    # The guard keys on the main db file only. openStore would create newDbPath
    # plus -wal/-shm sidecars, but checking the main file is sufficient: on a
    # freshly created db SQLite resets any orphan WAL, so a sidecar without a main
    # file only arises from external tampering, which is out of scope here.
    if target.exists():
        raise FileExistsError(
            f"rebuild target already exists: {target}; refusing to overwrite "
            "(canonical data must never be clobbered)"
        )

    # Require a COMPLETE export before creating anything, so a truncated dump can
    # never produce a store silently missing a whole canonical table.
    missing = [fn for fn, _t, _c in _LOAD_ORDER if not (srcDir / fn).exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete export in {srcDir}: missing {', '.join(missing)}; "
            "refusing to rebuild a partial store"
        )

    store = openStore(target)
    conn = store._conn
    try:
        for filename, table, cols in _LOAD_ORDER:
            _insertRows(conn, table, cols, _readRows(srcDir / filename))
        conn.commit()
    except Exception:
        conn.rollback()
        # The refuse-overwrite guard above proved newDbPath did not exist when
        # this call began, so the db file and any -wal/-shm sidecar here are
        # provably this call's own creation -- removing them can never touch
        # pre-existing data. Leaving the path clean lets a corrected retry run
        # instead of tripping the refuse-overwrite guard on our own leftover.
        store.close()
        for artifact in (target, Path(f"{target}-wal"), Path(f"{target}-shm")):
            if artifact.exists():
                artifact.unlink()
        raise
    return store
