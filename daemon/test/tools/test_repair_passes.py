"""Store-mutation passes for the Phase A corpus repair."""
import sqlite3
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from repair_passes import repairKvCacheRefs
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _putChunk(store, text, sourceRef, project=None):
    return putAtom(store, {
        "text": text, "kind": "document_chunk", "project": project,
        "importance": 0.0,
        "provenance": {"source": "bulk-import", "sourceRef": sourceRef},
    })


def _makeOldDb(tmp_path, rows):
    """rows: list of (rowid, summary). Returns the db path."""
    p = tmp_path / "vector_meta.db"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, summary TEXT)")
    conn.executemany("INSERT INTO meta(rowid, summary) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return p


def test_repairs_ref_and_backfills_project(store, tmp_path):
    aid = _putChunk(store, "chunk body", "kv_cache/vector_meta.db#rowid=7")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created rate.go"),
    ])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 1
    assert report["projectBackfilled"] == 1
    ref = store._conn.execute(
        "SELECT source_ref FROM provenance WHERE atom_id=?", (aid,)
    ).fetchone()[0]
    assert ref == "projects/obol/api/rate.go"
    project = store._conn.execute(
        "SELECT project FROM atoms WHERE id=?", (aid,)).fetchone()[0]
    assert project == "obol"


def test_non_files_summary_reported_not_guessed(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=3")
    old = _makeOldDb(tmp_path, [(3, "[claude] on GPU things, no path here")])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 0
    assert report["noPathInSummary"] == 1
    ref = store._conn.execute(
        "SELECT source_ref FROM provenance LIMIT 1").fetchone()[0]
    assert ref == "kv_cache/vector_meta.db#rowid=3"  # untouched


def test_missing_rowid_reported(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=999")
    old = _makeOldDb(tmp_path, [(1, "[files] file /home/aegis/Projects/x/y.go z")])
    report = repairKvCacheRefs(store, old)
    assert report["rowidMissing"] == 1
    assert report["rewritten"] == 0


def test_existing_project_not_overwritten(store, tmp_path):
    aid = _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=7",
                    project="keep-me")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created"),
    ])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 1
    assert report["projectBackfilled"] == 0
    assert store._conn.execute(
        "SELECT project FROM atoms WHERE id=?", (aid,)).fetchone()[0] == "keep-me"


def test_idempotent_second_run_is_noop(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=7")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created"),
    ])
    repairKvCacheRefs(store, old)
    second = repairKvCacheRefs(store, old)
    assert second == {"rewritten": 0, "projectBackfilled": 0,
                      "noPathInSummary": 0, "rowidMissing": 0}
