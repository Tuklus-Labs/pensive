"""Edge writer: idempotent relates edges with campaign provenance + rollback."""
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from edge_writer import writeVerdicts, rollbackCampaign, sampleCampaignEdges
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _pair(store):
    mem = putAtom(store, {
        "text": "memory", "kind": "atom", "project": "p",
        "importance": 0.0, "provenance": {"source": "claude-code"}})
    chunk = putAtom(store, {
        "text": "chunk", "kind": "document_chunk", "project": "p",
        "importance": 0.0, "provenance": {"source": "bulk-import"}})
    return mem, chunk


def _verdict(mem, chunk, keep=True, confidence=0.9):
    return {"memId": mem, "chunkId": chunk, "keep": keep,
            "confidence": confidence}


def test_writes_edge_chunk_to_memory_with_campaign_provenance(store):
    mem, chunk = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, chunk)], "camp-1")
    assert report["written"] == 1
    edge = store._conn.execute(
        "SELECT src_atom, dst_atom, type, weight, provenance_id "
        "FROM edges WHERE type='relates'").fetchone()
    assert edge[0] == chunk and edge[1] == mem      # DIRECTION: src=chunk
    assert edge[3] == pytest.approx(0.9)
    prov = store._conn.execute(
        "SELECT source, source_ref FROM provenance WHERE id=?",
        (edge[4],)).fetchone()
    assert prov == ("edge-campaign", "camp-1")


def test_rejected_verdicts_write_nothing(store):
    mem, chunk = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, chunk, keep=False)], "camp-1")
    assert report["rejected"] == 1
    assert store._conn.execute(
        "SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_idempotent_rerun_updates_weight_not_duplicate(store):
    mem, chunk = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk, confidence=0.9)], "camp-1")
    report = writeVerdicts(store, [_verdict(mem, chunk, confidence=0.4)],
                           "camp-1")
    assert report["written"] == 0
    assert report["updated"] == 1
    rows = store._conn.execute(
        "SELECT weight FROM edges WHERE type='relates'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.4)


def test_missing_atom_skipped_not_crash(store):
    mem, _ = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, "01FAKEID")], "camp-1")
    assert report["skippedMissingAtom"] == 1
    assert store._conn.execute(
        "SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_malformed_line_counted(store):
    report = writeVerdicts(store, [{"nonsense": True}], "camp-1")
    assert report["malformed"] == 1


def test_rollback_deletes_only_campaign_rows(store):
    mem, chunk = _pair(store)
    mem2, chunk2 = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk)], "camp-A")
    writeVerdicts(store, [_verdict(mem2, chunk2)], "camp-B")
    report = rollbackCampaign(store, "camp-A")
    assert report["edgesDeleted"] == 1
    assert report["provenanceDeleted"] == 1
    left = store._conn.execute(
        "SELECT src_atom FROM edges WHERE type='relates'").fetchall()
    assert left == [(chunk2,)]                 # camp-B untouched
    # Idempotent: rolling back again deletes nothing.
    assert rollbackCampaign(store, "camp-A") == {
        "edgesDeleted": 0, "provenanceDeleted": 0}


def test_sample_returns_written_edges_with_texts(store):
    mem, chunk = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk)], "camp-1")
    got = sampleCampaignEdges(store, "camp-1", 5)
    assert len(got) == 1
    assert got[0]["memId"] == mem and got[0]["chunkId"] == chunk
    assert got[0]["memText"] == "memory" and got[0]["chunkText"] == "chunk"
