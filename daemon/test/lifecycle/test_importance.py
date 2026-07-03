import pytest

from lifecycle.importance import accrueImportance
from store.store import getAtom, openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def test_importance_accrual_is_bounded_and_idempotent(store):
    atom_id = putAtom(store, {
        "text": "retrieved memory",
        "kind": "atom",
        "importance": 0.95,
        "provenance": {"source": "codex", "sourceRef": "retrieved"},
    })
    store._conn.execute(
        "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("recall-1", atom_id, "memory", "turn-1", 10.0, 100),
    )
    store._conn.commit()

    first = accrueImportance(store)
    second = accrueImportance(store)

    assert first == {"processed": 1, "updated": 1}
    assert second == {"processed": 0, "updated": 0}
    assert getAtom(store, atom_id)["importance"] == 1.0
    assert store._conn.execute(
        "SELECT processed_at FROM recall_log WHERE id = ?", ("recall-1",)
    ).fetchone()[0] is not None
