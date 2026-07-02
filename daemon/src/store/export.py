"""JSONL export of the canonical tables -- the raw material ``rebuild`` needs.

Only the four canonical tables are exported: atoms, provenance, edges, facets.
Everything else in the schema is *derived* and deliberately excluded --
``embeddings`` are re-computed by the Phase 2 embedder, and the ``fts`` index
regenerates itself from the atoms triggers on rebuild. Exporting derived data
would bloat the dump and, worse, let a stale copy of it masquerade as canonical.

The dump is at the DB layer, not the API layer. JSON keys are the raw snake_case
column names from ``schema.sql`` (``created_at``, ``src_atom``, ...), NOT the
camelCase names ``getAtom``/``edgesFrom`` present to callers. That is deliberate:
``rebuild`` reinserts these rows column-for-column, so the export must mirror the
DDL exactly, and a DB-level dump reading like the DDL is easier to audit against
the schema decades from now than one translated through the API contract.

Output is deterministic. Rows are ordered by primary key, each row is one JSON
object on its own line, and every object's keys are emitted in a fixed column
order. Re-exporting an unchanged store therefore produces byte-identical files.
"""
import json
from pathlib import Path

__all__ = [
    "exportJSONL",
    "ATOM_COLS",
    "PROVENANCE_COLS",
    "EDGE_COLS",
    "FACET_COLS",
]

# Column lists in DDL order. Each list fixes BOTH the SELECT projection and the
# JSON key order, so the export is deterministic and mirrors schema.sql. rebuild
# imports these so the two sides can never drift apart.
ATOM_COLS = (
    "id", "text", "kind", "project", "created_at", "occurred_at",
    "importance", "status", "schema_version",
)
PROVENANCE_COLS = (
    "id", "atom_id", "source", "session_id", "agent", "source_ref", "recorded_at",
)
EDGE_COLS = (
    "id", "src_atom", "dst_atom", "type", "weight", "created_at", "provenance_id",
)
FACET_COLS = ("atom_id", "key", "value")

# (filename, table, columns, ORDER BY) for each canonical table. ORDER BY is the
# table's primary key so the row order is stable: ULID pk for the first three,
# the composite (atom_id, key, value) pk for facets.
_TABLES = (
    ("atoms.jsonl", "atoms", ATOM_COLS, "id"),
    ("provenance.jsonl", "provenance", PROVENANCE_COLS, "id"),
    ("edges.jsonl", "edges", EDGE_COLS, "id"),
    ("facets.jsonl", "facets", FACET_COLS, "atom_id, key, value"),
)


def _writeTable(conn, dirPath, filename, table, cols, orderBy):
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM {table} ORDER BY {orderBy}"
    ).fetchall()
    # newline="" suppresses platform newline translation, so the "\n" we write is
    # the byte that lands on disk everywhere -- required for deterministic bytes.
    with open(dirPath / filename, "w", encoding="utf-8", newline="") as fh:
        for row in rows:
            obj = dict(zip(cols, row))
            fh.write(json.dumps(obj, ensure_ascii=False))
            fh.write("\n")


def exportJSONL(store, dir):
    """Dump the canonical tables of ``store`` as JSONL into ``dir``.

    Writes exactly four files -- ``atoms.jsonl``, ``provenance.jsonl``,
    ``edges.jsonl``, ``facets.jsonl`` -- one JSON object per row, rows ordered by
    primary key. ``dir`` is created if absent. Each of the four files is
    truncated and rewritten; no other file in ``dir`` is touched. Derived data
    (``embeddings``, ``fts``) is intentionally NOT exported -- it is regenerated
    by :func:`store.rebuild.rebuild`.
    """
    dirPath = Path(dir)
    dirPath.mkdir(parents=True, exist_ok=True)
    conn = store._conn
    for filename, table, cols, orderBy in _TABLES:
        _writeTable(conn, dirPath, filename, table, cols, orderBy)
