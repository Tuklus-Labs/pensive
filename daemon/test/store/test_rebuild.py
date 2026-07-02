import json

import pytest

from store.store import (
    openStore,
    putAtom,
    addEdge,
    addFacet,
    supersede,
    atomCount,
)
from store.export import exportJSONL
from store.rebuild import rebuild


# A genuinely hostile atom text: embedded newline, carriage return, tab, double
# quote, backslash, an emoji (astral-plane codepoint), and low control chars
# adjacent to NUL. If any layer (JSON escaping, sqlite text storage, the file
# round-trip) mangles these, the byte-identical assertion below fails loudly.
NASTY = 'line1\nline2\r\ttab "quoted" back\\slash \U0001f9e0 brain ctrl\x01\x1f end'

# Full-table SELECT dumps of every canonical table, ordered by primary key.
# rowid is deliberately excluded: rebuild reassigns rowids, but no canonical
# column depends on the rowid value.
CANONICAL_DUMPS = {
    "atoms": "SELECT id, text, kind, project, created_at, occurred_at, "
             "importance, status, schema_version FROM atoms ORDER BY id",
    "provenance": "SELECT id, atom_id, source, session_id, agent, source_ref, "
                  "recorded_at FROM provenance ORDER BY id",
    "edges": "SELECT id, src_atom, dst_atom, type, weight, created_at, "
             "provenance_id FROM edges ORDER BY id",
    "facets": "SELECT atom_id, key, value FROM facets ORDER BY atom_id, key, value",
}


def _dump(store):
    """Full-table dump of all four canonical tables as plain tuples."""
    return {
        name: store._conn.execute(sql).fetchall()
        for name, sql in CANONICAL_DUMPS.items()
    }


def _ftsIds(store, query):
    """Atom ids whose text matches an fts query, resolved via the shared rowid.

    Comparing ids (not rowids) is deliberate: rebuild reassigns rowids as it
    reinserts atoms, but the SET of matching atoms must be identical on both
    stores if the fts index regenerated correctly from the triggers.
    """
    return [
        r[0]
        for r in store._conn.execute(
            "SELECT a.id FROM atoms a JOIN fts f ON f.rowid = a.rowid "
            "WHERE fts MATCH ? ORDER BY a.id",
            (query,),
        ).fetchall()
    ]


def _seed(store):
    """~50 atoms plus edges, facets, multi-row provenance, one superseded atom,
    and one nasty-unicode atom. Returns ``(ids, nastyId)``.

    All writes go through the public store API, so the seeded state is exactly
    what the store produces in real use -- including the second provenance row
    ``supersede`` attaches to a successor (the multi-row provenance case).
    """
    ids = []
    for i in range(50):
        aid = putAtom(
            store,
            {
                "text": f"hull segment {i} rated to depth marker",
                "kind": ["atom", "narrative", "snapshot"][i % 3],
                "project": None if i % 4 == 0 else f"proj-{i % 3}",
                "occurredAt": None if i % 2 == 0 else 1_600_000_000 + i,
                "importance": (i % 10) / 10.0,
                "provenance": {
                    "source": "claude-code",
                    "sessionId": None if i % 3 == 0 else f"sess-{i}",
                    "agent": None if i % 2 == 0 else "heph",
                    "sourceRef": None if i % 5 == 0 else f"transcript:{i}",
                },
            },
        )
        ids.append(aid)

    # nasty-unicode atom, searchable via the clean token "brain"
    nastyId = putAtom(
        store,
        {"text": NASTY, "kind": "atom", "provenance": {"source": "explicit-emit"}},
    )
    ids.append(nastyId)

    # typed edges with weights + a real provenance link (edges.provenance_id FK)
    provId = store._conn.execute(
        "SELECT id FROM provenance WHERE atom_id = ?", (ids[0],)
    ).fetchone()[0]
    addEdge(store, {"src": ids[0], "dst": ids[1], "type": "relates"})
    addEdge(store, {"src": ids[1], "dst": ids[2], "type": "causes", "weight": 0.25})
    addEdge(
        store,
        {"src": ids[2], "dst": ids[3], "type": "contradicts",
         "weight": 0.5, "provenanceId": provId},
    )

    # facets, several per atom (idempotent set membership)
    addFacet(store, ids[0], "tag", "rf")
    addFacet(store, ids[0], "tag", "sigint")
    addFacet(store, ids[0], "project", "sentinel")
    addFacet(store, ids[5], "era", "2026")

    # supersession: ids[10] -> superseded; ids[11] gains a second provenance row
    supersede(store, ids[10], ids[11], {"source": "distiller", "agent": "heph"})

    return ids, nastyId


def test_rebuild_round_trip_is_byte_identical(tmp_path):
    # The centerpiece: seed a rich store, export, rebuild into a fresh file, and
    # prove every canonical row survived byte-for-byte and the derived fts index
    # regenerated to the same matches.
    src = openStore(tmp_path / "src.db")
    export_dir = tmp_path / "export"
    try:
        ids, nastyId = _seed(src)
        assert atomCount(src) == 51

        exportJSONL(src, export_dir)
        for name in ("atoms", "provenance", "edges", "facets"):
            assert (export_dir / f"{name}.jsonl").exists()

        rebuilt = rebuild(export_dir, tmp_path / "rebuilt.db")
        try:
            # every canonical table byte-identical, via FULL-table dumps
            assert _dump(src) == _dump(rebuilt)

            # the superseded atom's status survived the round-trip
            old_status = rebuilt._conn.execute(
                "SELECT status FROM atoms WHERE id = ?", (ids[10],)
            ).fetchone()[0]
            assert old_status == "superseded"

            # the nasty-unicode text is preserved exactly
            nasty_text = rebuilt._conn.execute(
                "SELECT text FROM atoms WHERE id = ?", (nastyId,)
            ).fetchone()[0]
            assert nasty_text == NASTY

            # fts regenerated by the atoms triggers: identical matches on both
            for q in ("hull", "segment", "brain"):
                assert _ftsIds(src, q) == _ftsIds(rebuilt, q)
            # the superseded atom is still searchable (its row was not deleted)
            assert ids[10] in _ftsIds(rebuilt, "hull")
        finally:
            rebuilt.close()
    finally:
        src.close()


def test_rebuild_refuses_to_overwrite_existing_db(tmp_path):
    # Decades rule: rebuild must never clobber an existing database. The first
    # rebuild succeeds; a second into the same path must raise, not overwrite.
    src = openStore(tmp_path / "src.db")
    export_dir = tmp_path / "export"
    try:
        _seed(src)
        exportJSONL(src, export_dir)
    finally:
        src.close()

    target = tmp_path / "rebuilt.db"
    first = rebuild(export_dir, target)
    first.close()
    assert target.exists()

    with pytest.raises(FileExistsError):
        rebuild(export_dir, target)


def test_export_empty_store_rebuilds_to_empty_store(tmp_path):
    # An empty store round-trips to an empty store: four empty files out, zero
    # rows back in.
    src = openStore(tmp_path / "src.db")
    export_dir = tmp_path / "export"
    try:
        assert atomCount(src) == 0
        exportJSONL(src, export_dir)
        for name in ("atoms", "provenance", "edges", "facets"):
            f = export_dir / f"{name}.jsonl"
            assert f.exists()
            assert f.read_text(encoding="utf-8") == ""
    finally:
        src.close()

    rebuilt = rebuild(export_dir, tmp_path / "rebuilt.db")
    try:
        assert atomCount(rebuilt) == 0
        for sql in CANONICAL_DUMPS.values():
            assert rebuilt._conn.execute(sql).fetchall() == []
    finally:
        rebuilt.close()


def test_export_is_deterministic_reexport_identical_bytes(tmp_path):
    # Deterministic output: exporting the same store twice (to different dirs and
    # re-exporting into the same dir) yields byte-identical files.
    src = openStore(tmp_path / "src.db")
    try:
        _seed(src)
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        exportJSONL(src, dir_a)
        exportJSONL(src, dir_b)
        for name in ("atoms", "provenance", "edges", "facets"):
            assert (dir_a / f"{name}.jsonl").read_bytes() == \
                   (dir_b / f"{name}.jsonl").read_bytes()

        # re-exporting into the same dir is idempotent at the byte level
        before = (dir_a / "atoms.jsonl").read_bytes()
        exportJSONL(src, dir_a)
        assert (dir_a / "atoms.jsonl").read_bytes() == before
    finally:
        src.close()


def test_export_overwrites_only_its_four_files(tmp_path):
    # Export truncates its own four files and touches nothing else in the dir.
    src = openStore(tmp_path / "src.db")
    try:
        _seed(src)
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        sentinel = export_dir / "README.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        stale = export_dir / "atoms.jsonl"
        stale.write_text("STALE-JUNK-LINE\n" * 100, encoding="utf-8")

        exportJSONL(src, export_dir)

        # unrelated file untouched
        assert sentinel.read_text(encoding="utf-8") == "keep me"
        # stale content truncated (not appended): no junk survives
        text = stale.read_text(encoding="utf-8")
        assert "STALE-JUNK-LINE" not in text
        # and the file is valid JSONL, one line per atom
        lines = [l for l in text.splitlines() if l]
        assert len(lines) == atomCount(src)
        assert all(json.loads(l)["id"] for l in lines)
    finally:
        src.close()
