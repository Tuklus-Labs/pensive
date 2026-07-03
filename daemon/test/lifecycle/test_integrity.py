import pytest

from lifecycle.integrity import integrityScan
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def test_integrity_scan_flags_orphaned_edge(store):
    live = putAtom(store, {
        "text": "valid atom",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "valid"},
    })
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("edge-orphan", live, "missing-atom", "relates", 1.0, 123, None),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["orphanEdges"] == [{
        "id": "edge-orphan",
        "srcAtom": live,
        "dstAtom": "missing-atom",
        "missing": ["dstAtom"],
    }]
