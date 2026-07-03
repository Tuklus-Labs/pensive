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

    assert first == {"processed": 1, "updated": 1, "missingAtoms": 0}
    assert second == {"processed": 0, "updated": 0, "missingAtoms": 0}
    assert getAtom(store, atom_id)["importance"] == 1.0
    assert store._conn.execute(
        "SELECT processed_at FROM recall_log WHERE id = ?", ("recall-1",)
    ).fetchone()[0] is not None


def test_importance_empty_recall_log_is_noop(store):
    assert accrueImportance(store) == {"processed": 0, "updated": 0, "missingAtoms": 0}


def test_importance_accrues_multiple_rows_to_multiple_atoms(store):
    first = putAtom(store, {
        "text": "first retrieved memory",
        "kind": "atom",
        "importance": 0.10,
        "provenance": {"source": "codex", "sourceRef": "first"},
    })
    second = putAtom(store, {
        "text": "second retrieved memory",
        "kind": "atom",
        "importance": 0.20,
        "provenance": {"source": "codex", "sourceRef": "second"},
    })
    store._conn.executemany(
        "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("recall-1", first, "q", "turn-1", 2.0, 100),
            ("recall-2", first, "q", "turn-2", 3.0, 101),
            ("recall-3", second, "q", "turn-3", 4.0, 102),
        ],
    )
    store._conn.commit()

    report = accrueImportance(store)

    assert report == {"processed": 3, "updated": 2, "missingAtoms": 0}
    assert getAtom(store, first)["importance"] == pytest.approx(0.15)
    assert getAtom(store, second)["importance"] == pytest.approx(0.24)


def test_importance_counts_only_unprocessed_rows(store):
    atom_id = putAtom(store, {
        "text": "partly retrieved memory",
        "kind": "atom",
        "importance": 0.0,
        "provenance": {"source": "codex", "sourceRef": "partial"},
    })
    store._conn.executemany(
        "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at, processed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("old", atom_id, "q", "turn-old", 50.0, 99, 123),
            ("new", atom_id, "q", "turn-new", 2.0, 100, None),
        ],
    )
    store._conn.commit()

    report = accrueImportance(store)

    assert report == {"processed": 1, "updated": 1, "missingAtoms": 0}
    assert getAtom(store, atom_id)["importance"] == pytest.approx(0.02)
    assert store._conn.execute(
        "SELECT processed_at FROM recall_log WHERE id = ?", ("old",)
    ).fetchone()[0] == 123


class _LateInsertConn:
    def __init__(self, conn, atom_id):
        self._conn = conn
        self._atom_id = atom_id
        self._inserted = False

    def execute(self, sql, params=()):
        if (
            not self._inserted
            and sql.startswith("UPDATE recall_log SET processed_at")
        ):
            self._inserted = True
            self._conn.execute(
                "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("late", self._atom_id, "q", "turn-late", 9.0, 999),
            )
        return self._conn.execute(sql, params)

    def executemany(self, sql, params):
        return self._conn.executemany(sql, params)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()


def test_importance_marks_only_rows_consumed_by_the_job(store):
    atom_id = putAtom(store, {
        "text": "race-shaped retrieval memory",
        "kind": "atom",
        "importance": 0.0,
        "provenance": {"source": "codex", "sourceRef": "race"},
    })
    store._conn.execute(
        "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("selected", atom_id, "q", "turn-selected", 1.0, 100),
    )
    store._conn.commit()
    original_conn = store._conn
    store._conn = _LateInsertConn(original_conn, atom_id)
    try:
        report = accrueImportance(store)
    finally:
        store._conn = original_conn

    assert report == {"processed": 1, "updated": 1, "missingAtoms": 0}
    assert store._conn.execute(
        "SELECT processed_at IS NOT NULL FROM recall_log WHERE id = ?", ("selected",)
    ).fetchone()[0] == 1
    assert store._conn.execute(
        "SELECT processed_at FROM recall_log WHERE id = ?", ("late",)
    ).fetchone()[0] is None


def test_importance_missing_atom_row_is_stamped_and_counted(store):
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.execute(
        "INSERT INTO recall_log(id, atom_id, query, source_ref, weight, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("missing", "no-such-atom", "q", "turn-missing", 4.0, 100),
    )
    store._conn.commit()
    store._conn.execute("PRAGMA foreign_keys=ON")

    report = accrueImportance(store)

    assert report == {"processed": 1, "updated": 0, "missingAtoms": 1}
    assert store._conn.execute(
        "SELECT processed_at FROM recall_log WHERE id = ?", ("missing",)
    ).fetchone()[0] is not None
