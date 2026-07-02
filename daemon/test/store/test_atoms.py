import sqlite3

import pytest

from store.store import (
    openStore,
    putAtom,
    getAtom,
    atomCount,
    CURRENT_SCHEMA_VERSION,
)


def _open(tmp_path):
    return openStore(tmp_path / "mem.db")


def test_put_then_get_round_trips_atom_and_provenance(tmp_path):
    s = _open(tmp_path)
    try:
        atomId = putAtom(
            s,
            {
                "text": "the pressure hull rates to 488m",
                "kind": "atom",
                "project": "aegis-auv",
                "provenance": {"source": "claude-code"},
            },
        )
        assert isinstance(atomId, str) and len(atomId) == 26

        atom = getAtom(s, atomId)
        assert atom is not None
        assert atom["id"] == atomId
        assert atom["text"] == "the pressure hull rates to 488m"
        assert atom["kind"] == "atom"
        assert atom["project"] == "aegis-auv"
        assert atom["status"] == "live"
        assert atom["schemaVersion"] == CURRENT_SCHEMA_VERSION
        assert len(atom["provenance"]) == 1
        assert atom["provenance"][0]["source"] == "claude-code"
        assert atom["provenance"][0]["atomId"] == atomId
        assert atomCount(s) == 1
    finally:
        s.close()


def test_put_atom_minimal_fields_default_the_optionals(tmp_path):
    s = _open(tmp_path)
    try:
        atomId = putAtom(
            s,
            {
                "text": "bare minimum atom",
                "kind": "atom",
                "provenance": {"source": "explicit-emit"},
            },
        )
        atom = getAtom(s, atomId)
        assert atom["project"] is None
        assert atom["occurredAt"] is None
        assert atom["importance"] == 0.0
        assert atom["status"] == "live"
        prov = atom["provenance"][0]
        assert prov["source"] == "explicit-emit"
        assert prov["sessionId"] is None
        assert prov["agent"] is None
        assert prov["sourceRef"] is None
    finally:
        s.close()


def test_put_atom_full_fields_round_trip(tmp_path):
    s = _open(tmp_path)
    try:
        atomId = putAtom(
            s,
            {
                "text": "fully specified atom",
                "kind": "narrative",
                "project": "pensive",
                "occurredAt": 1_600_000_000,
                "importance": 0.75,
                "provenance": {
                    "source": "codex",
                    "sessionId": "sess-42",
                    "agent": "heph",
                    "sourceRef": "transcript:120-135",
                },
            },
        )
        atom = getAtom(s, atomId)
        assert atom["kind"] == "narrative"
        assert atom["project"] == "pensive"
        assert atom["occurredAt"] == 1_600_000_000
        assert atom["importance"] == 0.75
        prov = atom["provenance"][0]
        assert prov["source"] == "codex"
        assert prov["sessionId"] == "sess-42"
        assert prov["agent"] == "heph"
        assert prov["sourceRef"] == "transcript:120-135"
    finally:
        s.close()


def test_get_atom_missing_id_returns_none(tmp_path):
    s = _open(tmp_path)
    try:
        assert getAtom(s, "00000000000000000000000000") is None
    finally:
        s.close()


def test_atom_count_tracks_inserts(tmp_path):
    s = _open(tmp_path)
    try:
        assert atomCount(s) == 0
        for i in range(3):
            putAtom(
                s,
                {
                    "text": f"atom {i}",
                    "kind": "atom",
                    "provenance": {"source": "bulk-import"},
                },
            )
        assert atomCount(s) == 3
    finally:
        s.close()


def test_put_atom_rolls_back_when_provenance_insert_fails(tmp_path):
    # Sabotage / atomicity: a provenance row with a NULL source violates the
    # NOT NULL constraint. Because putAtom inserts the atom FIRST and only then
    # the provenance row, this is a genuine mid-transaction failure -- the atom
    # row (and its fts trigger row) already exist in the open transaction when
    # the provenance insert raises, and the rollback must remove them as a unit.
    s = _open(tmp_path)
    try:
        assert atomCount(s) == 0
        with pytest.raises(sqlite3.IntegrityError):
            putAtom(
                s,
                {
                    "text": "SABOTAGE-CANARY-must-not-persist",
                    "kind": "atom",
                    "provenance": {"source": None},
                },
            )
        # No atom row leaked...
        assert atomCount(s) == 0
        # ...and the AFTER INSERT fts trigger row was rolled back with it (no
        # orphan in the full-text index).
        fts_hit = s._conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH 'SABOTAGE'"
        ).fetchone()
        assert fts_hit is None
    finally:
        s.close()


def test_put_atom_rollback_preserves_prior_committed_atoms(tmp_path):
    # The rollback must undo ONLY the failed put, not previously committed data.
    s = _open(tmp_path)
    try:
        goodId = putAtom(
            s,
            {
                "text": "durable atom",
                "kind": "atom",
                "provenance": {"source": "claude-code"},
            },
        )
        assert atomCount(s) == 1
        with pytest.raises(sqlite3.IntegrityError):
            putAtom(
                s,
                {
                    "text": "doomed atom",
                    "kind": "atom",
                    "provenance": {"source": None},
                },
            )
        assert atomCount(s) == 1
        assert getAtom(s, goodId) is not None
    finally:
        s.close()


def test_ulid_is_26_char_crockford_and_sortable():
    from util.ulid import ulid

    _CROCKFORD = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")

    ids = []
    for _ in range(500):
        u = ulid()
        assert len(u) == 26
        assert set(u) <= _CROCKFORD
        ids.append(u)

    # Unique across a tight loop (80 bits of randomness make collisions absurd).
    assert len(set(ids)) == len(ids)

    # A ULID minted with a later timestamp sorts lexically after an earlier one,
    # regardless of the random tail.
    earlier = ulid(timestampMs=1_000_000_000_000)
    later = ulid(timestampMs=2_000_000_000_000)
    assert earlier < later
