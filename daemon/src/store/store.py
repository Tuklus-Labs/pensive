"""Canonical SQLite-backed store: open/close, schema install, versioning.

The store is the durable backbone every later task builds on, so the rules are
conservative: the schema is versioned, migrations are forward-only, and every
connection independently enforces the invariants SQLite tracks per-connection
(foreign keys). WAL journal mode persists in the database file, so it only
needs to be set once by ``schema.sql``.
"""
import sqlite3
from pathlib import Path

from store.migrate import migrate, CURRENT_SCHEMA_VERSION

__all__ = ["openStore", "Store", "migrate", "CURRENT_SCHEMA_VERSION"]

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

    def _readVersion(self):
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None  # meta table does not exist yet (fresh file)
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
