"""Canonical SQLite-backed store: open/close, schema install, versioning.

The store is the durable backbone every later task builds on, so the rules are
conservative: the schema is versioned, migrations are forward-only, and every
connection independently enforces the invariants SQLite tracks per-connection
(foreign keys). WAL journal mode persists in the database file, so it only
needs to be set once by ``schema.sql``.
"""
import sqlite3
import time
from pathlib import Path

from store.migrate import migrate, CURRENT_SCHEMA_VERSION
from util.ulid import ulid

__all__ = [
    "openStore",
    "Store",
    "migrate",
    "CURRENT_SCHEMA_VERSION",
    "putAtom",
    "getAtom",
    "atomCount",
]

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _loadSchema():
    return _SCHEMA_PATH.read_text()


def _connect(path):
    conn = sqlite3.connect(str(path))
    # foreign_keys is a per-connection pragma; schema.sql cannot make it stick.
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def openStore(path):
    """Open (creating if needed) the canonical store at ``path``.

    A brand-new database gets ``schema.sql`` applied wholesale, then is stamped
    to the current version. An existing database is run through the forward-only
    migration ladder.
    """
    conn = _connect(path)
    store = Store(conn)
    if store._readVersion() is None:
        conn.executescript(_loadSchema())
        conn.commit()
    migrate(store)
    return store


class Store:
    def __init__(self, conn):
        self._conn = conn
        self._closed = False

    def schemaVersion(self):
        version = self._readVersion()
        if version is None:
            raise RuntimeError("store has no schema_version; not initialized")
        return version

    def close(self):
        if not self._closed:
            self._conn.close()
            self._closed = True

    def _metaTableExists(self):
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
        ).fetchone()
        return row is not None

    def _readVersion(self):
        # A fresh file has no meta table yet; that is the only "not initialized"
        # signal. Probe for it explicitly so genuine operational errors (locked
        # db, I/O, corruption) propagate instead of being misread as a fresh file.
        if not self._metaTableExists():
            return None
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def _setVersion(self, version):
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(version),),
        )

    def _commit(self):
        self._conn.commit()


def _now():
    """Current wall clock in unix seconds (UTC). Epoch seconds are inherently
    UTC, so no timezone handling is needed."""
    return int(time.time())


def putAtom(store, atomInput):
    """Write one atom and its single provenance row in one transaction.

    ``atomInput`` is a dict shaped ``{text, kind, project?, occurredAt?,
    importance?, provenance: {source, sessionId?, agent?, sourceRef?}}``.
    ``text``, ``kind`` and ``provenance.source`` are required; the ``?`` fields
    default to NULL (importance to 0.0). ``created_at`` (atom) and
    ``recorded_at`` (provenance) are stamped to now, ``status`` to ``'live'``,
    ``schema_version`` to current. Returns the generated atom ULID.

    The two inserts are one transaction: the atom row goes in first, then the
    provenance row. If the provenance insert fails (e.g. a NULL source hits the
    NOT NULL constraint) the whole thing rolls back, so a half-written atom with
    no provenance can never be committed.
    """
    conn = store._conn
    now = _now()
    atomId = ulid()
    prov = atomInput["provenance"]
    try:
        conn.execute(
            "INSERT INTO atoms(id, text, kind, project, created_at, occurred_at, "
            "importance, status, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                atomId,
                atomInput["text"],
                atomInput["kind"],
                atomInput.get("project"),
                now,
                atomInput.get("occurredAt"),
                atomInput.get("importance", 0.0),
                "live",
                CURRENT_SCHEMA_VERSION,
            ),
        )
        conn.execute(
            "INSERT INTO provenance(id, atom_id, source, session_id, agent, "
            "source_ref, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ulid(),
                atomId,
                prov["source"],
                prov.get("sessionId"),
                prov.get("agent"),
                prov.get("sourceRef"),
                now,
            ),
        )
        conn.commit()
    except Exception:
        # The atom insert already ran inside this transaction; rollback undoes it
        # (and its fts trigger row) so the failed put leaves the store untouched.
        conn.rollback()
        raise
    return atomId


def getAtom(store, atomId):
    """Return the atom ``atomId`` as a dict with its provenance list, or None.

    The returned shape is ``{id, text, kind, project, createdAt, occurredAt,
    importance, status, schemaVersion, provenance: [ {id, atomId, source,
    sessionId, agent, sourceRef, recordedAt}, ... ]}``. Provenance rows are
    ordered by their ULID, i.e. write order. Missing atom -> None.
    """
    conn = store._conn
    atomRow = conn.execute(
        "SELECT id, text, kind, project, created_at, occurred_at, importance, "
        "status, schema_version FROM atoms WHERE id = ?",
        (atomId,),
    ).fetchone()
    if atomRow is None:
        return None
    provRows = conn.execute(
        "SELECT id, atom_id, source, session_id, agent, source_ref, recorded_at "
        "FROM provenance WHERE atom_id = ? ORDER BY id",
        (atomId,),
    ).fetchall()
    return {
        "id": atomRow[0],
        "text": atomRow[1],
        "kind": atomRow[2],
        "project": atomRow[3],
        "createdAt": atomRow[4],
        "occurredAt": atomRow[5],
        "importance": atomRow[6],
        "status": atomRow[7],
        "schemaVersion": atomRow[8],
        "provenance": [
            {
                "id": p[0],
                "atomId": p[1],
                "source": p[2],
                "sessionId": p[3],
                "agent": p[4],
                "sourceRef": p[5],
                "recordedAt": p[6],
            }
            for p in provRows
        ],
    }


def atomCount(store):
    """Number of atom rows currently in the store."""
    return store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
