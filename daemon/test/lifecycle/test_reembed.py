import struct

import pytest

from lifecycle.reembed import reembed
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


def test_reembed_adds_new_model_rows_without_touching_old_rows(store):
    first = _put(store, "sonar calibration")
    second = _put(store, "battery chemistry")
    old_blob = _blob(0.25, 0.75)
    store._conn.executemany(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) VALUES (?, ?, ?, ?)",
        [(first, "old-model", old_blob, 101), (second, "old-model", old_blob, 101)],
    )
    store._conn.commit()

    embedded = reembed(store, "new-model", FakeEmbedder("ignored-model"))

    assert embedded == 2
    old_rows = store._conn.execute(
        "SELECT atom_id, vector, embedded_at FROM embeddings WHERE model_id = ? ORDER BY atom_id",
        ("old-model",),
    ).fetchall()
    assert set(old_rows) == {(first, old_blob, 101), (second, old_blob, 101)}
    assert store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", ("new-model",)
    ).fetchone()[0] == 2

    again = reembed(store, "new-model", FakeEmbedder("ignored-model"))

    assert again == 0
    assert store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", ("new-model",)
    ).fetchone()[0] == 2
