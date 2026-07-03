import pytest

from lifecycle.supersede_detect import detectSupersession
from store.store import addFacet, getAtom, openStore, putAtom


class FakeEmbedder:
    modelId = "fake"

    def embed(self, texts):
        out = []
        for text in texts:
            if "pressure hull" in text:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, source="codex"):
    return putAtom(store, {
        "text": text,
        "kind": "atom",
        "provenance": {"source": source, "sourceRef": text},
    })


def test_supersession_detect_proposes_excludes_person_source_and_applies_nothing(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is now 500m")
    person = _put(store, "pressure hull rating is 700m in my own words", source="person-import")
    for atom_id in (old, new, person):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, FakeEmbedder())
    again = detectSupersession(store, FakeEmbedder())

    assert report["proposed"] == 1
    assert again["proposed"] == 0
    proposals = store._conn.execute(
        "SELECT old_atom_id, new_atom_id, status FROM supersession_proposals"
    ).fetchall()
    assert proposals == [(old, new, "proposed")]
    assert getAtom(store, old)["status"] == "live"
    assert store._conn.execute("SELECT COUNT(*) FROM edges WHERE type = 'supersedes'").fetchone()[0] == 0
    assert person not in {row[0] for row in store._conn.execute(
        "SELECT old_atom_id FROM supersession_proposals"
    ).fetchall()}
