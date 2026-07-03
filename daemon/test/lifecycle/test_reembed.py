import struct

import pytest

from lifecycle.reembed import dropOldModel, reembed
from store.store import openStore, putAtom


class FakeEmbedder:
    def __init__(self, modelId):
        self.modelId = modelId
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text):
    return putAtom(store, {
        "text": text,
        "kind": "atom",
        "provenance": {"source": "codex", "sourceRef": text},
    })


def _blob(*values):
    return struct.pack("<" + "f" * len(values), *values)


def _counts(store):
    return {
        name: store._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in ("atoms", "edges", "facets", "provenance", "embeddings")
    }


def test_reembed_adds_new_model_rows_without_touching_old_rows(store):
    first = _put(store, "sonar calibration")
    second = _put(store, "battery chemistry")
    old_blob = _blob(0.25, 0.75)
    store._conn.executemany(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        [(first, "old-model", old_blob, 101), (second, "old-model", old_blob, 101)],
    )
    store._conn.commit()

    embedded = reembed(store, "new-model", FakeEmbedder("new-model"))

    assert embedded == 2
    old_rows = store._conn.execute(
        "SELECT atom_id, vector, embedded_at FROM embeddings WHERE model_id = ? ORDER BY atom_id",
        ("old-model",),
    ).fetchall()
    assert set(old_rows) == {(first, old_blob, 101), (second, old_blob, 101)}
    assert store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", ("new-model",)
    ).fetchone()[0] == 2

    again = reembed(store, "new-model", FakeEmbedder("new-model"))

    assert again == 0
    assert store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", ("new-model",)
    ).fetchone()[0] == 2


def test_reembed_rejects_embedder_model_mismatch_before_touching_store(store):
    _put(store, "sonar calibration")
    before = _counts(store)
    embedder = FakeEmbedder("wrong-model")

    with pytest.raises(ValueError, match="embedder modelId"):
        reembed(store, "new-model", embedder)

    assert embedder.calls == []
    assert _counts(store) == before


def test_reembed_accepts_matching_embedder_model_id(store):
    _put(store, "sonar calibration")

    assert reembed(store, "new-model", FakeEmbedder("new-model")) == 1


def test_reembed_completes_partial_new_model_coverage(store):
    first = _put(store, "already embedded")
    second = _put(store, "missing embedding")
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        (first, "new-model", _blob(0.5, 0.5), 100),
    )
    store._conn.commit()

    embedded = reembed(store, "new-model", FakeEmbedder("new-model"))

    assert embedded == 1
    assert store._conn.execute(
        "SELECT atom_id FROM embeddings WHERE model_id = ? ORDER BY atom_id",
        ("new-model",),
    ).fetchall() == sorted([(first,), (second,)])


def test_reembed_skips_non_live_atoms(store):
    live = _put(store, "live atom")
    dead = _put(store, "dead atom")
    store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (dead,))
    store._conn.commit()

    embedded = reembed(store, "new-model", FakeEmbedder("new-model"))

    assert embedded == 1
    assert store._conn.execute(
        "SELECT atom_id FROM embeddings WHERE model_id = ?",
        ("new-model",),
    ).fetchall() == [(live,)]


def test_reembed_only_changes_embedding_row_counts(store):
    first = _put(store, "first atom")
    second = _put(store, "second atom")
    store._conn.execute(
        "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("edge-1", first, second, "relates", 1.0, 100),
    )
    store._conn.execute(
        "INSERT INTO facets(atom_id, key, value) VALUES (?, ?, ?)",
        (first, "entity", "submarine"),
    )
    store._conn.commit()
    before = _counts(store)

    reembed(store, "new-model", FakeEmbedder("new-model"))

    after = _counts(store)
    assert after["embeddings"] == before["embeddings"] + 2
    for table in ("atoms", "edges", "facets", "provenance"):
        assert after[table] == before[table], f"{table} count changed during reembed"


def test_drop_old_model_deletes_only_old_model_rows(store):
    first = _put(store, "first atom")
    second = _put(store, "second atom")
    store._conn.executemany(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        [
            (first, "old-model", _blob(1.0), 100),
            (second, "old-model", _blob(1.0), 100),
            (first, "new-model", _blob(2.0), 101),
            (second, "new-model", _blob(2.0), 101),
        ],
    )
    store._conn.commit()

    deleted = dropOldModel(store, "old-model", activeModelId="new-model")

    assert deleted == 2
    assert store._conn.execute(
        "SELECT atom_id, model_id FROM embeddings ORDER BY atom_id, model_id"
    ).fetchall() == sorted([(first, "new-model"), (second, "new-model")])


def test_drop_old_model_refuses_to_drop_active_model(store):
    _put(store, "first atom")

    with pytest.raises(ValueError, match="active model"):
        dropOldModel(store, "new-model", activeModelId="new-model")


def test_drop_old_model_refuses_when_active_model_coverage_is_incomplete(store):
    first = _put(store, "first atom")
    second = _put(store, "second atom")
    store._conn.executemany(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        [
            (first, "old-model", _blob(1.0), 100),
            (second, "old-model", _blob(1.0), 100),
            (first, "new-model", _blob(2.0), 101),
        ],
    )
    store._conn.commit()

    with pytest.raises(ValueError, match="coverage"):
        dropOldModel(store, "old-model", activeModelId="new-model")

    assert store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", ("old-model",)
    ).fetchone()[0] == 2
