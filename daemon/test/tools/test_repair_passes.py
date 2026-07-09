"""Store-mutation passes for the Phase A corpus repair."""
import sqlite3
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from repair_passes import repairKvCacheRefs, dedupReferenceLibrary
from store.store import openStore, putAtom
from util.ulid import ulid


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


def test_null_summary_reported_not_crashed(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=5")
    old = _makeOldDb(tmp_path, [(5, None)])
    report = repairKvCacheRefs(store, old)
    assert report["noPathInSummary"] == 1
    assert report["rewritten"] == 0
    ref = store._conn.execute(
        "SELECT source_ref FROM provenance LIMIT 1").fetchone()[0]
    assert ref == "kv_cache/vector_meta.db#rowid=5"  # untouched


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


_CANON_ROOT = "projects/Aegis/AEGIS/docs/reference-library/"


def _putPair(store, fname, cN, text, driftedText=None):
    """A reflib duplicate pair: null-project copy + canonical Aegis copy."""
    dup = _putChunk(store, text, f"reference-library/{fname}#c{cN}")
    canon = _putChunk(store, driftedText if driftedText is not None else text,
                      f"{_CANON_ROOT}{fname}#c{cN}", project="Aegis")
    return dup, canon


def test_dedup_supersedes_null_copy_keeps_canonical(store):
    dup, canon = _putPair(store, "53-hw.md", 2, "identical body text")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 1
    statuses = dict(store._conn.execute(
        "SELECT id, status FROM atoms WHERE kind='document_chunk'").fetchall())
    assert statuses[dup] == "superseded"
    assert statuses[canon] == "live"
    # A supersedes edge canon -> dup exists (new -> old convention).
    edge = store._conn.execute(
        "SELECT src_atom, dst_atom FROM edges WHERE type='supersedes'"
    ).fetchone()
    assert edge == (canon, dup)


def test_text_mismatch_left_live_and_reported(store):
    dup, canon = _putPair(store, "99-drift.md", 0, "old text", "revised text")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["textMismatch"] == 1
    statuses = {r[1] for r in store._conn.execute(
        "SELECT id, status FROM atoms").fetchall()}
    assert statuses == {"live"}


def test_no_twin_reported(store):
    _putChunk(store, "orphan body", "reference-library/only-here.md#c0")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["noTwin"] == 1


def test_multiple_twins_reported_not_guessed(store):
    _putChunk(store, "same", "reference-library/multi.md#c1")
    _putChunk(store, "same", f"{_CANON_ROOT}multi.md#c1", project="Aegis")
    _putChunk(store, "same", f"{_CANON_ROOT}multi.md#c1", project="Aegis")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["multipleTwins"] == 1


def test_dedup_multi_provenance_dup_superseded_once(store):
    """A dup atom with TWO reference-library provenance rows must be
    superseded exactly once, not once per matching provenance row."""
    dup, canon = _putPair(store, "77-multi.md", 1, "identical body text")
    store._conn.execute(
        "INSERT INTO provenance(id, atom_id, source, session_id, agent, "
        "source_ref, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ulid(), dup, "bulk-import", None, None,
         "reference-library/77-multi.md#c1", 0),
    )
    store._conn.commit()
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 1
    edges = store._conn.execute(
        "SELECT src_atom, dst_atom FROM edges WHERE type='supersedes'"
    ).fetchall()
    assert edges == [(canon, dup)]
    statuses = dict(store._conn.execute(
        "SELECT id, status FROM atoms WHERE kind='document_chunk'").fetchall())
    assert statuses[dup] == "superseded"
    assert statuses[canon] == "live"
    second = dedupReferenceLibrary(store)
    assert second == {"superseded": 0, "noTwin": 0,
                      "textMismatch": 0, "multipleTwins": 0}


def test_dedup_idempotent(store):
    _putPair(store, "53-hw.md", 2, "identical body text")
    dedupReferenceLibrary(store)
    second = dedupReferenceLibrary(store)
    assert second == {"superseded": 0, "noTwin": 0,
                      "textMismatch": 0, "multipleTwins": 0}
