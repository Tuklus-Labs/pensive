import sqlite3

import pytest

from store.store import openStore, CURRENT_SCHEMA_VERSION


EXPECTED_TABLES = [
    "meta", "atoms", "provenance", "edges", "facets", "embeddings", "fts",
]


def tableExists(store, name):
    row = store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def test_open_creates_schema_at_current_version(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        # Deliberate pin: a schema migration must consciously bump this
        # assertion (v2 = Task 19 lifecycle: recall_log, proposals,
        # idx_prov_source_ref).
        assert CURRENT_SCHEMA_VERSION == 2
        assert s.schemaVersion() == CURRENT_SCHEMA_VERSION
        assert tableExists(s, "atoms")
        assert tableExists(s, "edges")
        for name in EXPECTED_TABLES:
            assert tableExists(s, name), f"missing table {name}"
    finally:
        s.close()


def test_foreign_keys_enforced_on_the_connection(tmp_path):
    # PRAGMA foreign_keys is per-connection; openStore must turn it on for
    # every connection, not rely on schema.sql alone. An edge to a missing
    # atom must be rejected.
    s = openStore(tmp_path / "mem.db")
    try:
        assert s._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            s._conn.execute(
                "INSERT INTO edges(id, src_atom, dst_atom, type, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("e1", "no-such-atom", "also-missing", "relates", 1),
            )
            s._conn.commit()
    finally:
        s.close()


def test_reopen_preserves_version_and_data(tmp_path):
    dbfile = tmp_path / "mem.db"
    s = openStore(dbfile)
    s._conn.execute(
        "INSERT INTO atoms(id, text, kind, created_at, schema_version) "
        "VALUES (?, ?, ?, ?, ?)",
        ("01ABCDEF", "hello world", "atom", 1234567890, CURRENT_SCHEMA_VERSION),
    )
    s._conn.commit()
    s.close()

    s2 = openStore(dbfile)
    try:
        assert s2.schemaVersion() == CURRENT_SCHEMA_VERSION  # unchanged across re-open
        row = s2._conn.execute(
            "SELECT text FROM atoms WHERE id = ?", ("01ABCDEF",)
        ).fetchone()
        assert row is not None and row[0] == "hello world"  # data survived
        # the INSERT trigger populated fts, so the row is searchable after reopen
        fts_hit = s2._conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH 'hello'"
        ).fetchone()
        assert fts_hit is not None
    finally:
        s2.close()


def test_double_close_is_safe(tmp_path):
    s = openStore(tmp_path / "mem.db")
    s.close()
    s.close()  # must not raise


def test_open_refuses_future_schema_version(tmp_path):
    # A database recorded at a version newer than this build must be refused,
    # not silently mis-opened or downgraded (forward-only / never-unrecoverable).
    dbfile = tmp_path / "mem.db"
    s = openStore(dbfile)
    s._conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'schema_version'",
        (str(CURRENT_SCHEMA_VERSION + 1),),
    )
    s._conn.commit()
    s.close()

    with pytest.raises(RuntimeError):
        openStore(dbfile)
