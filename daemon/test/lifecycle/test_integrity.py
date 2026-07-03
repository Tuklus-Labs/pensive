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


def test_integrity_scan_reports_coverage_gap_as_warning_not_failure(store):
    covered = putAtom(store, {
        "text": "covered live atom",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "covered"},
    })
    putAtom(store, {
        "text": "fresh live atom not embedded yet",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "fresh"},
    })
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        (covered, "model-a", b"vec", 100),
    )
    store._conn.commit()

    report = integrityScan(store)

    assert report["ok"] is True
    assert report["warnings"] == [{
        "type": "embeddingCoverage",
        "modelId": "model-a",
        "liveAtoms": 2,
        "embeddedLiveAtoms": 1,
        "missingLiveAtoms": 1,
    }]


def test_integrity_scan_emits_finished_report(store):
    seen = []

    report = integrityScan(store, emit=seen.append)

    assert seen == [report]
    assert report["ok"] is True


def test_integrity_scan_flags_missing_src_endpoint(store):
    live = putAtom(store, {
        "text": "valid dst",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "dst"},
    })
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("edge-missing-src", "missing-src", live, "relates", 1.0, 123, None),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["orphanEdges"][0]["missing"] == ["srcAtom"]


def test_integrity_scan_flags_missing_both_edge_endpoints(store):
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("edge-missing-both", "missing-src", "missing-dst", "relates", 1.0, 123, None),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["orphanEdges"][0]["missing"] == ["srcAtom", "dstAtom"]


def test_integrity_scan_flags_orphan_facet(store):
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO facets(atom_id, key, value) VALUES (?, ?, ?)",
        ("missing-atom", "entity", "submarine"),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["orphanFacets"] == [{
        "atom_id": "missing-atom",
        "key": "entity",
        "value": "submarine",
    }]


def test_integrity_scan_flags_orphan_provenance(store):
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO provenance(id, atom_id, source, source_ref, recorded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("prov-orphan", "missing-atom", "codex", "missing", 123),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["orphanProvenance"][0]["id"] == "prov-orphan"


def test_integrity_scan_flags_dangling_supersession_successor(store):
    old = putAtom(store, {
        "text": "old memory",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "old"},
    })
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("sup-missing-new", "missing-new", old, "supersedes", 1.0, 123, None),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = integrityScan(store)

    assert report["ok"] is False
    assert report["supersessionChains"]["danglingSuccessors"] == [{
        "id": "sup-missing-new",
        "srcAtom": "missing-new",
        "dstAtom": old,
    }]


def test_integrity_scan_flags_supersession_cycle(store):
    first = putAtom(store, {
        "text": "first memory",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "first"},
    })
    second = putAtom(store, {
        "text": "second memory",
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": "second"},
    })
    store._conn.executemany(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("sup-1", first, second, "supersedes", 1.0, 123, None),
            ("sup-2", second, first, "supersedes", 1.0, 124, None),
        ],
    )
    store._conn.commit()

    report = integrityScan(store)

    assert report["ok"] is False
    assert [first, second, first] in report["supersessionChains"]["cycles"]
