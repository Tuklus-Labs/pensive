"""Regression tests for the v3 to v4 schema ladder."""

import pytest

import store.migrate as migration
from store.store import CURRENT_SCHEMA_VERSION, Store, openStore


V4_TABLES = (
    "task_checkpoints",
    "recall_receipts",
    "recall_exposures",
    "recall_feedback",
    "memory_credits",
)


def _object_names(store):
    return {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        )
    }


def _v3_store(path):
    """Create a database carrying v3 objects without v4 objects."""
    store = openStore(path)
    conn = store._conn
    conn.execute("PRAGMA foreign_keys = OFF")
    for table in reversed(V4_TABLES):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    for index in (
        "idx_task_checkpoints_scope_revision",
        "idx_task_checkpoints_task_agent_project",
        "idx_recall_receipts_task_agent_time",
        "idx_recall_exposures_atom",
        "idx_recall_feedback_pending_type",
        "idx_recall_feedback_receipt",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {index}")
    conn.execute(
        "UPDATE meta SET value = '3' WHERE key = 'schema_version'"
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    return store


def test_fresh_and_migrated_schemas_agree(tmp_path):
    # contract: fresh-and-upgraded-schema
    fresh = openStore(tmp_path / "fresh.db")
    migrated = _v3_store(tmp_path / "migrated.db")
    try:
        assert_rule(
            CURRENT_SCHEMA_VERSION == 4
            and fresh.schemaVersion() == 4
            and migrated.schemaVersion() == 3,
            "fresh and v3 fixtures establish the expected version boundary",
            fresh=fresh.schemaVersion(),
            migrated=migrated.schemaVersion(),
        )
        migration.migrate(migrated)
        assert_rule(
            migrated.schemaVersion() == fresh.schemaVersion() == 4,
            "migration stamps the current schema version",
            fresh=fresh.schemaVersion(),
            migrated=migrated.schemaVersion(),
        )
        for table in V4_TABLES:
            fresh_info = fresh._conn.execute(f"PRAGMA table_info({table})").fetchall()
            migrated_info = migrated._conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            assert_rule(
                migrated_info == fresh_info,
                "fresh and migrated table columns agree",
                table=table,
                fresh=fresh_info,
                migrated=migrated_info,
            )
            fresh_sql = fresh._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()[0]
            migrated_sql = migrated._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()[0]
            assert_rule(
                migrated_sql == fresh_sql,
                "fresh and migrated table definitions agree byte-for-byte",
                table=table,
                fresh=fresh_sql,
                migrated=migrated_sql,
            )
            if table == "memory_credits":
                feedback_id = next(row for row in fresh_info if row[1] == "feedback_id")
                assert_rule(
                    feedback_id[3] == 1,
                    "memory credits require helpful feedback lineage",
                    column=feedback_id,
                )
        assert_rule(
            {
                name for name in _object_names(fresh) if name.startswith("idx_")
            }
            == {
                name for name in _object_names(migrated) if name.startswith("idx_")
            },
            "fresh and migrated indexes agree",
            fresh=sorted(_object_names(fresh)),
            migrated=sorted(_object_names(migrated)),
        )
    finally:
        fresh.close()
        migrated.close()


def assert_rule(condition, message, **state):
    assert condition, f"{message}: state={state!r}"


def test_v3_to_v4_failure_rolls_back_ddl_and_version(tmp_path, monkeypatch):
    # persistence: atomic-migration-after-ddl-failure
    store = _v3_store(tmp_path / "v3.db")
    original = migration._upgrade_3_to_4

    def fail_after_ddl(current):
        original(current)
        raise RuntimeError("injected failure after v4 DDL")

    monkeypatch.setattr(migration, "_upgrade_3_to_4", fail_after_ddl)
    try:
        with pytest.raises(RuntimeError, match="after v4 DDL"):
            migration.migrate(store)
        assert_rule(
            store.schemaVersion() == 3,
            "failed v4 migration keeps the prior version stamp",
            version=store.schemaVersion(),
        )
        assert_rule(
            all(name not in _object_names(store) for name in V4_TABLES),
            "failed v4 migration removes every newly-created table",
            objects=sorted(_object_names(store)),
        )
        assert_rule(
            all(
                name not in _object_names(store)
                for name in (
                    "idx_task_checkpoints_scope_revision",
                    "idx_task_checkpoints_task_agent_project",
                    "idx_recall_receipts_task_agent_time",
                    "idx_recall_exposures_atom",
                    "idx_recall_feedback_pending_type",
                    "idx_recall_feedback_receipt",
                )
            ),
            "failed v4 migration removes every newly-created index",
            objects=sorted(_object_names(store)),
        )
    finally:
        store.close()


def test_v3_to_v4_preserves_existing_rows(tmp_path):
    # persistence: additive-migration-preserves-old-rows
    store = _v3_store(tmp_path / "v3.db")
    try:
        store._conn.execute(
            "INSERT INTO atoms(id, text, kind, project, created_at, schema_version) "
            "VALUES ('atom-v3', 'old text', 'atom', 'old-project', 17, 3)"
        )
        store._conn.commit()
        migration.migrate(store)
        row = store._conn.execute(
            "SELECT id, text, project, created_at, schema_version FROM atoms"
        ).fetchone()
        assert_rule(
            row == ("atom-v3", "old text", "old-project", 17, 3),
            "v4 migration leaves preexisting atom columns byte-for-byte intact",
            row=row,
        )
    finally:
        store.close()
