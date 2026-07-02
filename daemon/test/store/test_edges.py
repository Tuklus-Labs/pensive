import sqlite3

import pytest

from store.store import (
    openStore,
    putAtom,
    getAtom,
    atomCount,
    addEdge,
    addFacet,
    facetsOf,
    supersede,
    edgesFrom,
    edgesTo,
)


def _open(tmp_path):
    return openStore(tmp_path / "mem.db")


def _put(store, text, source="claude-code"):
    return putAtom(
        store,
        {"text": text, "kind": "atom", "provenance": {"source": source}},
    )


# --- supersession: the load-bearing semantics -----------------------------

def test_supersede_marks_old_superseded_and_links_new_to_old(tmp_path):
    # Step 1 (RED target): supersede sets old.status='superseded' and creates a
    # supersedes edge new->old, in one call. The old atom is NOT deleted -- its
    # text stays intact and readable (the decades rule).
    s = _open(tmp_path)
    try:
        oldId = _put(s, "the hull rates to 400m")
        newId = _put(s, "the hull rates to 488m")

        supersede(s, oldId, newId, {"source": "distiller"})

        old = getAtom(s, oldId)
        assert old is not None                       # not deleted
        assert old["text"] == "the hull rates to 400m"  # text intact
        assert old["status"] == "superseded"
        # the successor stays live
        assert getAtom(s, newId)["status"] == "live"

        # the supersedes edge points new->old and is discoverable from both ends
        incoming = edgesTo(s, oldId, "supersedes")
        assert len(incoming) == 1
        edge = incoming[0]
        assert edge["srcAtom"] == newId
        assert edge["dstAtom"] == oldId
        assert edge["type"] == "supersedes"
        outgoing = edgesFrom(s, newId, "supersedes")
        assert len(outgoing) == 1
        assert outgoing[0]["id"] == edge["id"]
    finally:
        s.close()


def test_supersede_invariant_live_filter_hides_old_but_canonical_intact(tmp_path):
    # Step 5 (the critical invariant): after supersede, a plain "live" status
    # filter excludes the old atom -- but nothing canonical is lost. getAtom
    # still returns the full text, the incoming supersedes edge chains it to its
    # successor, both atoms still exist, and the old text is still searchable.
    s = _open(tmp_path)
    try:
        oldId = _put(s, "CANARY-OLD unique marker text")
        newId = _put(s, "CANARY-NEW replacement text")
        assert atomCount(s) == 2

        supersede(s, oldId, newId, {"source": "distiller"})

        # a plain live filter (what an unaware reader would run) hides old...
        live = [
            r[0]
            for r in s._conn.execute(
                "SELECT id FROM atoms WHERE status = 'live'"
            ).fetchall()
        ]
        assert oldId not in live
        assert newId in live

        # ...but the canonical record is fully intact: nothing deleted,
        assert atomCount(s) == 2
        old = getAtom(s, oldId)
        assert old is not None
        assert old["text"] == "CANARY-OLD unique marker text"  # full text preserved
        # the old text is still in the FTS index (proves the row was not deleted)
        fts_hit = s._conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH 'CANARY'"
        ).fetchall()
        assert len(fts_hit) == 2

        # and the successor is reachable by chaining the incoming supersedes edge
        incoming = edgesTo(s, oldId, "supersedes")
        assert len(incoming) == 1
        assert incoming[0]["srcAtom"] == newId
    finally:
        s.close()


def test_supersede_provenance_attaches_to_successor_and_links_edge(tmp_path):
    # The supersession event's provenance row is written against the successor
    # (the atom that carries the assertion) and the supersedes edge points at it
    # via provenanceId, so the trust layer can later explain *why* old was
    # replaced.
    s = _open(tmp_path)
    try:
        oldId = _put(s, "old fact", source="claude-code")
        newId = _put(s, "new fact", source="claude-code")

        supersede(s, oldId, newId, {"source": "distiller", "agent": "heph"})

        new = getAtom(s, newId)
        # successor now has TWO provenance rows: its own creation + supersession
        assert len(new["provenance"]) == 2
        supProv = [p for p in new["provenance"] if p["source"] == "distiller"]
        assert len(supProv) == 1
        assert supProv[0]["agent"] == "heph"

        edge = edgesTo(s, oldId, "supersedes")[0]
        assert edge["provenanceId"] == supProv[0]["id"]
    finally:
        s.close()


def test_supersede_missing_new_leaves_old_unchanged(tmp_path):
    # Sabotage / atomicity: superseding against a non-existent successor must
    # fail on the FK and leave old.status untouched -- no half-applied state.
    s = _open(tmp_path)
    try:
        oldId = _put(s, "durable old fact")
        assert getAtom(s, oldId)["status"] == "live"

        with pytest.raises(sqlite3.IntegrityError):
            supersede(s, oldId, "MISSINGNEWATOM0000000000000", {"source": "distiller"})

        assert getAtom(s, oldId)["status"] == "live"     # unchanged
        assert edgesTo(s, oldId, "supersedes") == []     # no edge leaked
    finally:
        s.close()


def test_supersede_missing_old_is_atomic(tmp_path):
    # Symmetric sabotage: a non-existent old atom fails the edge FK. The
    # successor must be untouched -- crucially, the supersession provenance row
    # (which attaches to the successor and is written first) must be rolled back,
    # so the successor's provenance list stays at its original single row.
    s = _open(tmp_path)
    try:
        newId = _put(s, "the successor")
        assert len(getAtom(s, newId)["provenance"]) == 1

        with pytest.raises(sqlite3.IntegrityError):
            supersede(s, "MISSINGOLDATOM0000000000000", newId, {"source": "distiller"})

        new = getAtom(s, newId)
        assert new["status"] == "live"
        assert len(new["provenance"]) == 1              # rolled back, no orphan prov
        assert edgesFrom(s, newId, "supersedes") == []  # no edge leaked
    finally:
        s.close()


# --- typed edges ----------------------------------------------------------

def test_add_edge_round_trips_and_is_directional(tmp_path):
    s = _open(tmp_path)
    try:
        a = _put(s, "atom a")
        b = _put(s, "atom b")

        edgeId = addEdge(s, {"src": a, "dst": b, "type": "relates"})
        assert isinstance(edgeId, str) and len(edgeId) == 26

        outs = edgesFrom(s, a)
        assert len(outs) == 1
        edge = outs[0]
        # full documented Edge shape
        assert set(edge.keys()) == {
            "id", "srcAtom", "dstAtom", "type", "weight", "createdAt", "provenanceId",
        }
        assert edge["id"] == edgeId
        assert edge["srcAtom"] == a
        assert edge["dstAtom"] == b
        assert edge["type"] == "relates"
        assert edge["weight"] == 1.0          # DDL default
        assert edge["provenanceId"] is None
        assert isinstance(edge["createdAt"], int)

        # directional: b has an incoming edge but no outgoing one
        assert edgesTo(s, b)[0]["id"] == edgeId
        assert edgesFrom(s, b) == []
        assert edgesTo(s, a) == []
    finally:
        s.close()


def test_add_edge_custom_weight_and_provenance(tmp_path):
    s = _open(tmp_path)
    try:
        a = _put(s, "cause")
        b = _put(s, "effect")
        # a real provenance id to satisfy the edges.provenance_id FK
        provId = s._conn.execute(
            "SELECT id FROM provenance WHERE atom_id = ?", (a,)
        ).fetchone()[0]

        edgeId = addEdge(
            s,
            {"src": a, "dst": b, "type": "causes", "weight": 0.5, "provenanceId": provId},
        )
        edge = edgesFrom(s, a, "causes")[0]
        assert edge["id"] == edgeId
        assert edge["weight"] == 0.5
        assert edge["provenanceId"] == provId
    finally:
        s.close()


def test_edges_filter_by_type(tmp_path):
    s = _open(tmp_path)
    try:
        a = _put(s, "a")
        b = _put(s, "b")
        c = _put(s, "c")
        addEdge(s, {"src": a, "dst": b, "type": "relates"})
        addEdge(s, {"src": a, "dst": c, "type": "causes"})

        assert len(edgesFrom(s, a)) == 2                     # unfiltered: both
        assert [e["type"] for e in edgesFrom(s, a, "causes")] == ["causes"]
        assert [e["dstAtom"] for e in edgesFrom(s, a, "relates")] == [b]
        assert edgesFrom(s, a, "supersedes") == []           # none of that type
    finally:
        s.close()


def test_add_edge_rejects_missing_endpoint(tmp_path):
    # FK guard + atomicity: an edge to a non-existent atom is rejected and no
    # partial row is committed.
    s = _open(tmp_path)
    try:
        a = _put(s, "real atom")
        before = s._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            addEdge(s, {"src": a, "dst": "NOSUCHATOM00000000000000000", "type": "relates"})
        after = s._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert before == after == 0
    finally:
        s.close()


# --- facets ---------------------------------------------------------------

def test_facets_add_and_list(tmp_path):
    s = _open(tmp_path)
    try:
        a = _put(s, "atom with facets")
        addFacet(s, a, "tag", "rf")
        addFacet(s, a, "project", "sentinel")
        addFacet(s, a, "tag", "sigint")

        facets = facetsOf(s, a)
        # deterministic order (key, value)
        assert facets == [
            {"key": "project", "value": "sentinel"},
            {"key": "tag", "value": "rf"},
            {"key": "tag", "value": "sigint"},
        ]
        assert facetsOf(s, "no-such-atom") == []
    finally:
        s.close()


def test_facet_duplicate_triple_is_idempotent(tmp_path):
    # Composite PK (atom_id, key, value): re-adding the identical triple is a
    # no-op, not an error. The fact "this atom has this (key,value)" either
    # exists or it does not.
    s = _open(tmp_path)
    try:
        a = _put(s, "atom")
        addFacet(s, a, "tag", "rf")
        addFacet(s, a, "tag", "rf")  # duplicate -> silently ignored
        assert facetsOf(s, a) == [{"key": "tag", "value": "rf"}]
    finally:
        s.close()


def test_facet_on_missing_atom_raises(tmp_path):
    # OR IGNORE swallows only the PK duplicate, never the FK violation: a facet
    # pointed at a non-existent atom is a real error and must surface.
    s = _open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            addFacet(s, "NOSUCHATOM00000000000000000", "tag", "rf")
    finally:
        s.close()
