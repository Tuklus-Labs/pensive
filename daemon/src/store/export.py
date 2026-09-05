"""Portable JSONL snapshots of memories and their retained history.

Format 2 includes recall usage and supersession-review state alongside atoms,
provenance, edges and facets. Embeddings and FTS remain derived. Every table is
read from one SQLite snapshot, staged, and published with a manifest containing
its SHA-256 digest. Rebuild rejects an interrupted or mixed-generation export.

The dump is at the DB layer, not the API layer. JSON keys are the raw snake_case
column names from ``schema.sql`` (``created_at``, ``src_atom``, ...), NOT the
camelCase names ``getAtom``/``edgesFrom`` present to callers. That is deliberate:
``rebuild`` reinserts these rows column-for-column, so the export must mirror the
DDL exactly, and a DB-level dump reading like the DDL is easier to audit against
the schema decades from now than one translated through the API contract.

Output is deterministic. Rows are ordered by primary key, each row is one JSON
object on its own line, and every object's keys are emitted in a fixed column
order. Re-exporting an unchanged store therefore produces byte-identical files.
Table and manifest bytes are fsynced before publication. On POSIX, directory
entries are fsynced around the marker and replacements so a filesystem crash is
fail-closed. Hardware and filesystem implementations still define the ultimate
power-loss guarantee; non-POSIX systems get atomic per-file replacement only.
"""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

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
RECALL_LOG_COLS = ("id", "atom_id", "query", "source_ref", "weight",
                   "recorded_at", "processed_at")
PROPOSAL_COLS = ("id", "old_atom_id", "new_atom_id", "similarity", "reason",
                 "status", "created_at")

FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
PUBLICATION_MARKER = ".pensive-export-incomplete"

# (filename, table, columns, ORDER BY) for each canonical table. ORDER BY is the
# table's primary key so the row order is stable: ULID pk for the first three,
# the composite (atom_id, key, value) pk for facets.
_TABLES = (
    ("atoms.jsonl", "atoms", ATOM_COLS, "id"),
    ("provenance.jsonl", "provenance", PROVENANCE_COLS, "id"),
    ("edges.jsonl", "edges", EDGE_COLS, "id"),
    ("facets.jsonl", "facets", FACET_COLS, "atom_id, key, value"),
    ("recall_log.jsonl", "recall_log", RECALL_LOG_COLS, "id"),
    ("supersession_proposals.jsonl", "supersession_proposals", PROPOSAL_COLS, "id"),
)


def _writeTable(conn, dirPath, filename, table, cols, orderBy):
    digest = hashlib.sha256()
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM {table} ORDER BY {orderBy}"
    )
    with open(dirPath / filename, "wb") as fh:
        for row in rows:
            obj = dict(zip(cols, row))
            encoded = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            digest.update(encoded)
            fh.write(encoded)
        fh.flush()
        os.fsync(fh.fileno())
    return digest.hexdigest()


def _writeDurable(path, data, *, exclusive=False):
    mode = "xb" if exclusive else "wb"
    with open(path, mode) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsyncDirectory(path):
    """Persist directory-entry changes where POSIX exposes directory fsync."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exportJSONL(store, dir):
    """Write six JSONL tables and a versioned manifest, preserving unrelated files.

    Staging failure leaves a previous completed dump intact. Publication failure
    leaves PUBLICATION_MARKER in place: rebuild rejects that directory, and the
    operator can export again to a fresh directory. Exclusive marker creation
    serializes concurrent publishers; manifest digests let a concurrent reader
    detect replacement that began after its initial marker check.
    """
    dirPath = Path(dir)
    dirPath.mkdir(parents=True, exist_ok=True)
    conn = store._conn
    marker = dirPath / PUBLICATION_MARKER
    if marker.exists():
        raise FileExistsError(f"incomplete export at {dirPath}; use a fresh directory")
    with TemporaryDirectory(prefix=".pensive-stage-", dir=dirPath) as temporary:
        staging = Path(temporary)
        conn.execute("SAVEPOINT pensive_export")
        try:
            hashes = {}
            for filename, table, cols, orderBy in _TABLES:
                hashes[filename] = _writeTable(
                    conn, staging, filename, table, cols, orderBy)
            manifest = {"format": "pensive-jsonl", "formatVersion": FORMAT_VERSION,
                        "schemaVersion": store.schemaVersion(),
                        "files": [row[0] for row in _TABLES],
                        "sha256": hashes}
            _writeDurable(
                staging / MANIFEST_NAME,
                (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
            )
            _fsyncDirectory(staging)
        except BaseException:
            conn.execute("ROLLBACK TO pensive_export")
            conn.execute("RELEASE pensive_export")
            raise
        conn.execute("RELEASE pensive_export")

        _writeDurable(
            marker,
            b"Publication incomplete. Rebuild must not read this file set.\n",
            exclusive=True,
        )
        _fsyncDirectory(dirPath)
        for filename in [row[0] for row in _TABLES] + [MANIFEST_NAME]:
            os.replace(staging / filename, dirPath / filename)
        _fsyncDirectory(dirPath)
        marker.unlink()
        _fsyncDirectory(dirPath)
