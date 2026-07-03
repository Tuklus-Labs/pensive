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


class SimilarityEmbedder:
    modelId = "fake"

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors[text] for text in texts]


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


def test_supersession_detect_excludes_any_person_prefix_source(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is now 500m")
    person = _put(store, "pressure hull rating is now 600m", source="person-note")
    for atom_id in (old, new, person):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, FakeEmbedder())

    assert report["proposed"] == 1
    proposals = store._conn.execute(
        "SELECT old_atom_id, new_atom_id FROM supersession_proposals"
    ).fetchall()
    assert proposals == [(old, new)]


def test_supersession_detect_excludes_atom_with_any_person_provenance(store):
    old = _put(store, "pressure hull rating is 300m")
    mixed = _put(store, "pressure hull rating is now 500m")
    store._conn.execute(
        "INSERT INTO provenance(id, atom_id, source, source_ref, recorded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("prov-person-extra", mixed, "person-note", "mixed", 123),
    )
    for atom_id in (old, mixed):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, FakeEmbedder())

    assert report == {"scanned": 1, "proposed": 0}
    assert store._conn.execute("SELECT COUNT(*) FROM supersession_proposals").fetchone()[0] == 0


def test_supersession_detect_skips_when_no_shared_facet(store):
    _put(store, "pressure hull rating is 300m")
    _put(store, "pressure hull rating is now 500m")

    report = detectSupersession(store, FakeEmbedder())

    assert report["proposed"] == 0


def test_supersession_detect_skips_low_similarity(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is now 500m")
    for atom_id in (old, new):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, SimilarityEmbedder({
        "pressure hull rating is 300m": [1.0, 0.0],
        "pressure hull rating is now 500m": [0.0, 1.0],
    }))

    assert report["proposed"] == 0


def test_supersession_detect_skips_identical_text(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is 300m")
    for atom_id in (old, new):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, SimilarityEmbedder({
        "pressure hull rating is 300m": [1.0, 0.0],
    }))

    assert report["proposed"] == 0


def test_supersession_detect_ignores_non_live_atoms(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is now 500m")
    store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (old,))
    store._conn.commit()
    for atom_id in (old, new):
        addFacet(store, atom_id, "entity", "submarine")

    report = detectSupersession(store, FakeEmbedder())

    assert report == {"scanned": 1, "proposed": 0}


def test_supersession_detect_direction_is_newer_supersedes_older(store):
    old = _put(store, "pressure hull rating is 300m")
    new = _put(store, "pressure hull rating is now 500m")
    store._conn.execute("UPDATE atoms SET created_at = ? WHERE id = ?", (100, old))
    store._conn.execute("UPDATE atoms SET created_at = ? WHERE id = ?", (200, new))
    store._conn.commit()
    for atom_id in (old, new):
        addFacet(store, atom_id, "entity", "submarine")

    detectSupersession(store, FakeEmbedder())

    assert store._conn.execute(
        "SELECT old_atom_id, new_atom_id FROM supersession_proposals"
    ).fetchall() == [(old, new)]


def test_supersession_detect_tag_and_src_facets_participate(store):
    old_tag = _put(store, "pressure hull rating is 300m")
    new_tag = _put(store, "pressure hull rating is now 500m")
    addFacet(store, old_tag, "tag", "pressure")
    addFacet(store, new_tag, "tag", "pressure")
    old_src = _put(store, "pressure hull rating is 250m")
    new_src = _put(store, "pressure hull rating is now 550m")
    addFacet(store, old_src, "src", "manual")
    addFacet(store, new_src, "src", "manual")

    report = detectSupersession(store, FakeEmbedder())

    assert report["proposed"] == 2
