"""Regression tests for format 3 export and task-state restore."""

import hashlib
import json

import pytest

import store.export as exporter
from store.checkpoints import putTaskCheckpoint
from store.export import exportJSONL
from store.rebuild import rebuild
from store.store import openStore, putAtom


NEW_TABLES = (
    "task_checkpoints",
    "recall_receipts",
    "recall_exposures",
    "recall_feedback",
    "memory_credits",
)


def _assert_rule(condition, message, **state):
    assert condition, f"{message}: state={state!r}"


def _seed_all_new_tables(store):
    atom = putAtom(store, {
        "text": "portable task state with Unicode α",
        "kind": "atom",
        "provenance": {"source": "test", "sourceRef": "test:atom"},
    })
    checkpoint = putTaskCheckpoint(
        store,
        project="pensive",
        agent="agent-a",
        taskId="task-1",
        expectedRevision=0,
        requestId="request-1",
        state="blocked",
        body="checkpoint body\nwith exact text",
        source="test",
        sessionId="session-a",
        sourceRef="test:checkpoint",
    )
    conn = store._conn
    conn.execute(
        "INSERT INTO recall_receipts(id, query, project, agent, task_id, "
        "source_ref, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("receipt-1", "why?", "pensive", "agent-a", "task-1", "test:receipt", 101),
    )
    conn.execute(
        "INSERT INTO recall_exposures(receipt_id, atom_id, rank, score, delivery) "
        "VALUES (?, ?, ?, ?, ?)",
        ("receipt-1", atom, 1, 0.75, "body"),
    )
    conn.execute(
        "INSERT INTO recall_feedback(event_id, receipt_id, atom_id, feedback_type, "
        "source, agent, task_id, session_id, source_ref, note, recorded_at, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "feedback-1", "receipt-1", atom, "helpful", "test", "agent-a", "task-1",
            "session-a", "test:feedback", "kept", 102, 103,
        ),
    )
    conn.execute(
        "INSERT INTO memory_credits(atom_id, agent, task_id, feedback_id, awarded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (atom, "agent-a", "task-1", "feedback-1", 104),
    )
    conn.commit()
    return atom, checkpoint


def _rows(store, table):
    return store._conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()


def test_format3_export_has_stable_eleven_table_digest_set(tmp_path):
    # contract: format3-table-set-and-digests
    store = openStore(tmp_path / "source.db")
    try:
        _seed_all_new_tables(store)
        directory = tmp_path / "dump"
        exportJSONL(store, directory)
        manifest = json.loads((directory / exporter.MANIFEST_NAME).read_text())
        expected = [row[0] for row in exporter._TABLES]
        _assert_rule(
            manifest["formatVersion"] == 3
            and manifest["files"] == expected
            and set(manifest["sha256"]) == set(expected)
            and len(expected) == 11,
            "format 3 manifest names all eleven portable files",
            manifest=manifest,
        )
        for filename in expected:
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            _assert_rule(
                manifest["sha256"][filename] == digest,
                "format 3 manifest digest matches exact file bytes",
                filename=filename,
                expected=manifest["sha256"][filename],
                actual=digest,
            )
    finally:
        store.close()


def test_format3_round_trip_preserves_all_new_tables(tmp_path):
    # persistence: eleven-table-round-trip
    source = openStore(tmp_path / "source.db")
    restored = None
    try:
        atom, checkpoint = _seed_all_new_tables(source)
        directory = tmp_path / "dump"
        exportJSONL(source, directory)
        restored = rebuild(directory, tmp_path / "restored.db")
        for table in NEW_TABLES:
            source_rows = _rows(source, table)
            restored_rows = _rows(restored, table)
            _assert_rule(
                restored_rows == source_rows,
                "format 3 rebuild preserves every new table row",
                table=table,
                source=source_rows,
                restored=restored_rows,
            )
        _assert_rule(
            restored._conn.execute("PRAGMA foreign_key_check").fetchall() == [],
            "format 3 rebuild leaves all foreign keys valid",
            checkpoint=checkpoint,
            atom=atom,
        )
    finally:
        source.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("damage", ["missing", "digest"])
def test_format3_rebuild_rejects_missing_file_and_digest_mismatch(tmp_path, damage):
    # io: complete-file-and-digest-guard
    source = openStore(tmp_path / "source.db")
    try:
        _seed_all_new_tables(source)
        directory = tmp_path / "dump"
        exportJSONL(source, directory)
    finally:
        source.close()
    target = tmp_path / f"restored-{damage}.db"
    filename = "recall_feedback.jsonl"
    if damage == "missing":
        (directory / filename).unlink()
        expected = FileNotFoundError
        match = filename
    else:
        with (directory / filename).open("ab") as handle:
            handle.write(b"\n")
        expected = ValueError
        match = "digest mismatch"
    with pytest.raises(expected, match=match):
        rebuild(directory, target)
    _assert_rule(
        not target.exists(),
        "rejected format 3 export never publishes a target",
        target=str(target),
        damage=damage,
    )


def test_format2_and_legacy_four_file_exports_remain_readable(tmp_path):
    # contract: format2-and-legacy-backcompat
    source = openStore(tmp_path / "source.db")
    try:
        atom = putAtom(source, {
            "text": "legacy memory",
            "kind": "atom",
            "provenance": {"source": "test"},
        })
        source._conn.execute(
            "UPDATE meta SET value = '3' WHERE key = 'schema_version'"
        )
        source._conn.commit()
        format2 = tmp_path / "format2"
        exportJSONL(source, format2, formatVersion=2)
    finally:
        source.close()
    manifest = json.loads((format2 / exporter.MANIFEST_NAME).read_text())
    expected_v2 = [row[0] for row in exporter._TABLES_V2]
    _assert_rule(
        manifest["formatVersion"] == 2
        and manifest["schemaVersion"] == 3
        and manifest["files"] == expected_v2
        and len(expected_v2) == 6,
        "format 2 retains its exact original six-file manifest",
        manifest=manifest,
    )
    restored = rebuild(format2, tmp_path / "format2.db")
    try:
        _assert_rule(
            restored._conn.execute("SELECT text FROM atoms WHERE id = ?", (atom,)).fetchone()
            == ("legacy memory",),
            "format 2 rebuild preserves legacy canonical text",
            atom=atom,
        )
    finally:
        restored.close()

    legacy = tmp_path / "legacy4"
    legacy.mkdir()
    for filename in ("atoms.jsonl", "provenance.jsonl", "edges.jsonl", "facets.jsonl"):
        (legacy / filename).write_bytes((format2 / filename).read_bytes())
    legacy_restored = rebuild(legacy, tmp_path / "legacy.db")
    try:
        _assert_rule(
            legacy_restored._conn.execute(
                "SELECT COUNT(*) FROM task_checkpoints"
            ).fetchone()[0] == 0,
            "legacy four-file restore does not invent task checkpoints",
            atom=atom,
        )
    finally:
        legacy_restored.close()


@pytest.mark.parametrize("table, field, value, expected", [
    ("recall_feedback.jsonl", "agent", "rogue-agent", "feedback scope"),
    ("memory_credits.jsonl", "agent", "rogue-agent", "memory credit"),
    ("memory_credits.jsonl", "feedback_id", None, "credit"),
    ("recall_feedback.jsonl", "note", "", "feedback note"),
    ("recall_feedback.jsonl", "note", None, "feedback note"),
    ("recall_feedback.jsonl", "processed_at", None, "memory credit"),
    ("recall_exposures.jsonl", "delivery", "structured", "exposure delivery"),
    ("task_checkpoints.jsonl", "revision", True, "checkpoint revision"),
    ("task_checkpoints.jsonl", "body", "x" * 32001, "checkpoint body"),
])
def test_format3_rebuild_rejects_invalid_v4_rows(
        tmp_path, table, field, value, expected):
    # contract: imported-feedback-scope-integrity
    source = openStore(tmp_path / "source.db")
    try:
        _seed_all_new_tables(source)
        directory = tmp_path / "dump"
        exportJSONL(source, directory)
    finally:
        source.close()

    path = directory / table
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(encoded)
    manifest_path = directory / exporter.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][table] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    target = tmp_path / "rejected.db"
    with pytest.raises(ValueError, match=expected):
        rebuild(directory, target)
    _assert_rule(
        not target.exists(),
        "scope-invalid format 3 export never publishes a target",
        target=str(target),
        table=table,
    )


def test_format3_rebuild_rejects_processed_helpful_without_scope_credit(tmp_path):
    # persistence: processed-helpful-requires-scope-credit
    source = openStore(tmp_path / "source.db")
    try:
        _seed_all_new_tables(source)
        directory = tmp_path / "dump"
        exportJSONL(source, directory)
    finally:
        source.close()

    credits = directory / "memory_credits.jsonl"
    credits.write_bytes(b"")
    manifest_path = directory / exporter.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][credits.name] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    target = tmp_path / "rejected-missing-credit.db"
    with pytest.raises(ValueError, match="processed helpful feedback"):
        rebuild(directory, target)
    _assert_rule(
        not target.exists(),
        "processed helpful feedback without a scope credit never publishes",
        target=str(target),
    )


def test_format3_restore_allows_processed_helpful_rows_to_share_scope_credit(tmp_path):
    # state: repeated-helpful-events-share-credit
    source = openStore(tmp_path / "source.db")
    restored = None
    try:
        atom, _checkpoint = _seed_all_new_tables(source)
        source._conn.execute(
            "INSERT INTO recall_feedback(event_id, receipt_id, atom_id, feedback_type, "
            "source, agent, task_id, session_id, source_ref, note, recorded_at, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "feedback-2", "receipt-1", atom, "helpful", "test", "agent-a", "task-1",
                "session-a", "test:feedback-2", "still useful", 105, 106,
            ),
        )
        source._conn.commit()
        directory = tmp_path / "dump"
        exportJSONL(source, directory)
        restored = rebuild(directory, tmp_path / "restored.db")
        feedback_count = restored._conn.execute(
            "SELECT COUNT(*) FROM recall_feedback WHERE feedback_type = 'helpful'"
        ).fetchone()[0]
        credit_count = restored._conn.execute(
            "SELECT COUNT(*) FROM memory_credits WHERE atom_id = ? AND agent = ? AND task_id = ?",
            (atom, "agent-a", "task-1"),
        ).fetchone()[0]
        _assert_rule(
            (feedback_count, credit_count) == (2, 1),
            "processed helpful events share one atom-agent-task credit",
            feedback_count=feedback_count,
            credit_count=credit_count,
        )
    finally:
        source.close()
        if restored is not None:
            restored.close()
