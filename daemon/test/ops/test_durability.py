import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from store.backup import SNAPSHOT_KEEP, restoreSnapshot, snapshot
from store.store import (
    CURRENT_SCHEMA_VERSION,
    addEdge,
    addFacet,
    getAtom,
    openStore,
    putAtom,
)


CRASH_DRILL_ITERATIONS = 5


def _populate_store(store):
    first = putAtom(
        store,
        {
            "text": "durability source atom",
            "kind": "atom",
            "project": "pensive",
            "importance": 0.7,
            "provenance": {
                "source": "codex",
                "sessionId": "snapshot-test",
                "agent": "task-20",
                "sourceRef": "ops:source",
            },
        },
    )
    second = putAtom(
        store,
        {
            "text": "durability linked atom",
            "kind": "narrative",
            "project": "pensive",
            "occurredAt": 1_700_000_001,
            "provenance": {"source": "bulk-import", "sourceRef": "ops:linked"},
        },
    )
    addEdge(store, {"src": first, "dst": second, "type": "relates", "weight": 0.25})
    addFacet(store, first, "tag", "durability")
    addFacet(store, second, "entity", "sqlite")
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        (first, "test-model", sqlite3.Binary(b"\x00\x00\x80?"), 1_700_000_100),
    )
    store._conn.commit()
    return first, second


def _sqlite_objects(path):
    conn = sqlite3.connect(path)
    try:
        return {
            (kind, name)
            for kind, name in conn.execute(
                """
                SELECT type, name
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger')
                  AND name NOT LIKE 'sqlite_%'
                """
            )
        }
    finally:
        conn.close()


def _user_tables(path):
    conn = sqlite3.connect(path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                  AND name NOT IN ('fts', 'fts_data', 'fts_idx', 'fts_content',
                                   'fts_docsize', 'fts_config')
                ORDER BY name
                """
            )
        ]
        return tables
    finally:
        conn.close()


def _ordered_rows(conn, table, include_snapshot_meta=False):
    cols = [
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    ]
    order_by = ", ".join(cols)
    where = ""
    if table == "meta" and not include_snapshot_meta:
        where = "WHERE key NOT IN ('snapshot_of', 'snapshot_at')"
    return conn.execute(f"SELECT * FROM {table} {where} ORDER BY {order_by}").fetchall()


def _assert_same_snapshot_content(left, right, include_snapshot_meta=False):
    assert _sqlite_objects(left) == _sqlite_objects(right)
    assert _user_tables(left) == _user_tables(right)

    left_conn = sqlite3.connect(left)
    right_conn = sqlite3.connect(right)
    try:
        for table in _user_tables(left):
            left_rows = _ordered_rows(left_conn, table, include_snapshot_meta)
            right_rows = _ordered_rows(right_conn, table, include_snapshot_meta)
            assert len(right_rows) == len(left_rows), table
            assert right_rows == left_rows

        probe = left_conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH 'durability' ORDER BY rowid"
        ).fetchall()
        assert right_conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH 'durability' ORDER BY rowid"
        ).fetchall() == probe
    finally:
        left_conn.close()
        right_conn.close()


def _integrity_ok(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_snapshot_is_loadable_standalone_db_and_restore_round_trips(tmp_path):
    db_path = tmp_path / "memory.db"
    restore_path = tmp_path / "restored.db"
    snapshot_dir = tmp_path / "snapshots"

    store = openStore(db_path)
    try:
        first, second = _populate_store(store)
        snap_path = snapshot(store, snapshot_dir)
    finally:
        store.close()

    assert snap_path.parent == snapshot_dir
    assert snap_path.exists()
    assert _integrity_ok(snap_path)

    snap_store = openStore(snap_path)
    try:
        assert snap_store.schemaVersion() == CURRENT_SCHEMA_VERSION
        assert getAtom(snap_store, first)["text"] == "durability source atom"
        assert getAtom(snap_store, second)["kind"] == "narrative"
    finally:
        snap_store.close()

    restored = restoreSnapshot(snap_path, restore_path)
    assert restored == restore_path

    restored_store = openStore(restore_path)
    try:
        assert restored_store.schemaVersion() == CURRENT_SCHEMA_VERSION
        assert getAtom(restored_store, first)["provenance"][0]["source"] == "codex"
    finally:
        restored_store.close()

    _assert_same_snapshot_content(db_path, snap_path)
    _assert_same_snapshot_content(db_path, restore_path)
    _assert_same_snapshot_content(snap_path, restore_path, include_snapshot_meta=True)


def test_snapshot_rotation_prunes_only_older_snapshots_for_same_store(tmp_path):
    db_path = tmp_path / "memory.db"
    other_db_path = tmp_path / "other.db"
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    keeper = snapshot_dir / "operator-note.txt"
    keeper.write_text("do not prune me")

    store = openStore(db_path)
    other = openStore(other_db_path)
    try:
        _populate_store(store)
        _populate_store(other)

        first_snap = snapshot(store, snapshot_dir)
        prefix = first_snap.name.rsplit("-", 2)[0]
        decoy_file = snapshot_dir / f"{prefix}-19990101T000000Z-1.sqlite3"
        decoy_file.write_text("operator copy, not a pensive snapshot")
        symlink_decoy = snapshot_dir / f"{prefix}-19990101T000000Z-2.sqlite3"
        symlink_decoy.symlink_to(keeper)

        mine = [first_snap]
        for _ in range(SNAPSHOT_KEEP + 2):
            mine.append(snapshot(store, snapshot_dir))
            time.sleep(0.001)
        other_snap = snapshot(other, snapshot_dir)
    finally:
        store.close()
        other.close()

    remaining = sorted(p for p in snapshot_dir.iterdir() if p.name != keeper.name)
    assert keeper.read_text() == "do not prune me"
    assert decoy_file.read_text() == "operator copy, not a pensive snapshot"
    assert symlink_decoy.is_symlink()
    assert symlink_decoy.resolve() == keeper
    assert other_snap in remaining
    assert sorted(p for p in mine if p.exists()) == mine[-SNAPSHOT_KEEP:]
    assert len([p for p in mine if p.exists()]) == SNAPSHOT_KEEP


def test_restore_refuses_unrelated_sqlite_and_markerless_store_copy(tmp_path):
    unrelated = tmp_path / "unrelated.db"
    markerless_copy = tmp_path / "markerless.db"
    db_path = tmp_path / "memory.db"
    snapshot_dir = tmp_path / "snapshots"

    conn = sqlite3.connect(unrelated)
    try:
        conn.execute("CREATE TABLE notes(body TEXT)")
        conn.execute("INSERT INTO notes(body) VALUES ('not a pensive snapshot')")
        conn.commit()
    finally:
        conn.close()

    store = openStore(db_path)
    try:
        _populate_store(store)
        store._conn.execute("VACUUM main INTO ?", (str(markerless_copy),))
        snap_path = snapshot(store, snapshot_dir)
    finally:
        store.close()

    with pytest.raises(RuntimeError, match="not a Pensive snapshot"):
        restoreSnapshot(unrelated, tmp_path / "restore-unrelated.db")
    with pytest.raises(RuntimeError, match="not a Pensive snapshot"):
        restoreSnapshot(markerless_copy, tmp_path / "restore-copy.db")

    restored = restoreSnapshot(snap_path, tmp_path / "restore-snapshot.db")
    assert restored.exists()


def test_restore_refuses_to_overwrite_existing_target(tmp_path):
    db_path = tmp_path / "memory.db"
    snapshot_dir = tmp_path / "snapshots"
    target_path = tmp_path / "already-open.db"

    store = openStore(db_path)
    target = openStore(target_path)
    try:
        _populate_store(store)
        snap_path = snapshot(store, snapshot_dir)
        with pytest.raises(FileExistsError):
            restoreSnapshot(snap_path, target_path)
    finally:
        target.close()
        store.close()


def _run_crash_drill_once(tmp_path, iteration):
    db_path = tmp_path / f"crash-{iteration}.db"
    progress_path = tmp_path / f"crash-{iteration}.acks"
    store = openStore(db_path)
    try:
        seed_id = putAtom(
            store,
            {
                "text": f"pre-kill committed seed {iteration}",
                "kind": "atom",
                "provenance": {"source": "codex"},
            },
        )
    finally:
        store.close()

    child_code = """
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from store.store import openStore, putAtom

db_path = Path(sys.argv[1])
progress_path = Path(sys.argv[2])
store = openStore(db_path)
try:
    with progress_path.open("a") as progress:
        for index in range(100_000):
            atom_id = putAtom(
                store,
                {
                    "text": f"crash drill atom {index}",
                    "kind": "atom",
                    "project": "pensive",
                    "importance": index / 1000,
                    "provenance": {"source": "codex", "sourceRef": f"crash:{index}"},
                },
            )
            progress.write(atom_id + "\\n")
            progress.flush()
            os.fsync(progress.fileno())
            time.sleep(0.003)
finally:
    store.close()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(db_path), str(progress_path)],
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        deadline = time.time() + 5
        acknowledged = []
        while time.time() < deadline:
            if progress_path.exists():
                acknowledged = [
                    line.strip()
                    for line in progress_path.read_text().splitlines()
                    if line.strip()
                ]
                if len(acknowledged) >= 8:
                    break
            if proc.poll() is not None:
                break
            time.sleep(0.01)
        assert len(acknowledged) >= 8
        assert proc.poll() is None
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)

    recovered = openStore(db_path)
    try:
        assert recovered.schemaVersion() == CURRENT_SCHEMA_VERSION
        assert recovered._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert getAtom(recovered, seed_id)["text"] == f"pre-kill committed seed {iteration}"
        for atom_id in acknowledged:
            atom = getAtom(recovered, atom_id)
            assert atom is not None
            assert atom["id"] == atom_id
            assert atom["text"]
            assert atom["kind"]
            assert atom["status"] == "live"
            assert atom["schemaVersion"] == CURRENT_SCHEMA_VERSION
            assert len(atom["provenance"]) == 1
            assert atom["provenance"][0]["source"] == "codex"
            assert atom["provenance"][0]["recordedAt"] is not None

        bad_acked_provenance = recovered._conn.execute(
            """
            SELECT COUNT(*)
            FROM provenance
            WHERE atom_id IN (
                SELECT value FROM json_each(?)
            )
              AND (source IS NULL OR recorded_at IS NULL)
            """,
            ("[" + ",".join(f'"{atom_id}"' for atom_id in acknowledged) + "]",),
        ).fetchone()[0]
        assert bad_acked_provenance == 0

        incomplete = recovered._conn.execute(
            """
            SELECT COUNT(*)
            FROM atoms AS a
            LEFT JOIN provenance AS p ON p.atom_id = a.id
            WHERE a.id IS NULL
               OR a.text IS NULL
               OR a.kind IS NULL
               OR a.created_at IS NULL
               OR a.importance IS NULL
               OR a.status IS NULL
               OR a.schema_version IS NULL
               OR p.id IS NULL
               OR p.atom_id IS NULL
               OR p.source IS NULL
               OR p.recorded_at IS NULL
            """
        ).fetchone()[0]
        assert incomplete == 0
    finally:
        recovered.close()


def test_wal_recovers_cleanly_after_child_process_sigkill_mid_write(tmp_path):
    # Five bounded iterations catch obvious timing-sensitive WAL recovery bugs
    # without making the ops gate a long soak test.
    for iteration in range(CRASH_DRILL_ITERATIONS):
        _run_crash_drill_once(tmp_path, iteration)
