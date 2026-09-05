"""Regression tests for the append-only task checkpoint contract."""

import sqlite3
import threading

import pytest

import store.checkpoints as checkpoint_store
from store.store import atomCount, openStore, putAtom


def _put(store, **overrides):
    values = {
        "project": "pensive",
        "agent": "agent-a",
        "taskId": "task-1",
        "expectedRevision": 0,
        "requestId": "request-1",
        "state": "active",
        "body": "first body",
        "source": "test",
        "sessionId": "session-1",
        "sourceRef": "test:1",
    }
    values.update(overrides)
    return checkpoint_store.putTaskCheckpoint(store, **values)


def _assert_rule(condition, message, **state):
    assert condition, f"{message}: state={state!r}"


def test_checkpoint_wire_shape_preserves_authorship_and_timestamp(tmp_path):
    # invariant: canonical-wire-row
    store = openStore(tmp_path / "store.db")
    try:
        checkpoint = _put(store)
        expected = {
            "id", "project", "agent", "taskId", "revision", "requestId",
            "state", "body", "source", "sessionId", "sourceRef", "recordedAt",
        }
        _assert_rule(
            set(checkpoint) == expected,
            "canonical checkpoint wire fields stay exact",
            keys=sorted(checkpoint),
        )
        _assert_rule(
            checkpoint["project"] == "pensive"
            and checkpoint["agent"] == "agent-a"
            and checkpoint["taskId"] == "task-1"
            and checkpoint["revision"] == 1
            and checkpoint["requestId"] == "request-1"
            and checkpoint["state"] == "active"
            and checkpoint["body"] == "first body"
            and checkpoint["source"] == "test"
            and checkpoint["sessionId"] == "session-1"
            and checkpoint["sourceRef"] == "test:1"
            and type(checkpoint["recordedAt"]) is int,
            "checkpoint preserves authorship and server timestamp",
            checkpoint=checkpoint,
        )
    finally:
        store.close()


def test_checkpoint_write_does_not_touch_atoms_or_embeddings(tmp_path):
    # invariant: atom-embedding-isolation
    store = openStore(tmp_path / "store.db")
    try:
        atom = putAtom(store, {
            "text": "unrelated canonical atom",
            "kind": "atom",
            "provenance": {"source": "test"},
        })
        store._conn.execute(
            "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            (atom, "test-model", sqlite3.Binary(b"vector"), 123),
        )
        store._conn.commit()
        before = store._conn.execute(
            "SELECT COUNT(*) FROM atoms"
        ).fetchone()[0], store._conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        _put(store)
        after = store._conn.execute(
            "SELECT COUNT(*) FROM atoms"
        ).fetchone()[0], store._conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        _assert_rule(
            after == before,
            "checkpoint write leaves semantic atoms and embeddings unchanged",
            before=before,
            after=after,
        )
    finally:
        store.close()


def test_checkpoint_replay_is_idempotent_and_changed_replay_fails(tmp_path):
    # invariant: request-replay
    store = openStore(tmp_path / "store.db")
    try:
        first = _put(store)
        replay = _put(store)
        _assert_rule(
            replay == first,
            "identical request replay returns original canonical row",
            first=first,
            replay=replay,
        )
        with pytest.raises(ValueError, match="requestId"):
            _put(store, body="changed body")
        count = store._conn.execute(
            "SELECT COUNT(*) FROM task_checkpoints"
        ).fetchone()[0]
        _assert_rule(
            count == 1,
            "changed replay does not append a checkpoint",
            count=count,
        )
    finally:
        store.close()


def test_checkpoint_cas_race_allows_one_writer(tmp_path):
    # concurrency: writer-lock-before-head-read
    db = tmp_path / "store.db"
    barrier = threading.Barrier(2)
    results = []

    def race(request_id):
        store = openStore(db)
        barrier.wait()
        try:
            results.append(("ok", checkpoint_store.putTaskCheckpoint(
                store,
                project="pensive",
                agent="agent-a",
                taskId="task-1",
                expectedRevision=0,
                requestId=request_id,
                state="active",
                body=request_id,
                source="test",
            )))
        except Exception as exc:  # result is asserted below, preserving error type
            results.append(("error", exc))
        finally:
            store.close()

    threads = [
        threading.Thread(target=race, args=("request-a",)),
        threading.Thread(target=race, args=("request-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    store = None
    try:
        _assert_rule(
            len(results) == 2 and all(item[0] in {"ok", "error"} for item in results),
            "both concurrent checkpoint attempts finish",
            results=results,
        )
        _assert_rule(
            sum(item[0] == "ok" for item in results) == 1
            and sum(item[0] == "error" for item in results) == 1
            and isinstance(next(item[1] for item in results if item[0] == "error"), ValueError),
            "CAS race commits one revision and rejects the loser",
            results=results,
        )
        store = openStore(db)
        count, revision = store._conn.execute(
            "SELECT COUNT(*), MAX(revision) FROM task_checkpoints"
        ).fetchone()
        _assert_rule(
            (count, revision) == (1, 1),
            "CAS race leaves one contiguous head",
            count=count,
            revision=revision,
        )
        store.close()
        store = None
    finally:
        if store is not None:
            store.close()


def test_checkpoint_scope_isolation(tmp_path):
    # invariant: scope-isolation
    store = openStore(tmp_path / "store.db")
    try:
        _put(store, requestId="a", project="project-a", agent="agent-a")
        _put(store, requestId="b", project="project-b", agent="agent-a")
        _put(store, requestId="c", project="project-a", agent="agent-b")
        all_heads = checkpoint_store.getTaskStates(store, taskId="task-1")
        scoped = checkpoint_store.getTaskStates(
            store, taskId="task-1", project="project-a", agent="agent-a")
        _assert_rule(
            len(all_heads["checkpoints"]) == 3,
            "unscoped current lookup returns one head per matching scope",
            result=all_heads,
        )
        _assert_rule(
            len(scoped["checkpoints"]) == 1
            and scoped["checkpoints"][0]["project"] == "project-a"
            and scoped["checkpoints"][0]["agent"] == "agent-a",
            "scoped current lookup cannot cross project or agent boundaries",
            result=scoped,
        )
    finally:
        store.close()


def test_current_history_and_asof_selection(tmp_path, monkeypatch):
    # invariant: stable-selection
    timestamps = iter([100, 200, 300])
    monkeypatch.setattr(checkpoint_store, "_now", lambda: next(timestamps))
    store = openStore(tmp_path / "store.db")
    try:
        _put(store)
        _put(store, expectedRevision=1, requestId="request-2", state="blocked")
        _put(store, expectedRevision=2, requestId="request-3", state="completed")
        current = checkpoint_store.getTaskStates(
            store, taskId="task-1", project="pensive", agent="agent-a")
        revision = checkpoint_store.getTaskStates(
            store, taskId="task-1", project="pensive", agent="agent-a", revision=2)
        asof = checkpoint_store.getTaskStates(
            store, taskId="task-1", project="pensive", agent="agent-a", asOf=250)
        _assert_rule(
            current["checkpoints"][0]["revision"] == 3
            and current["checkpoints"][0]["state"] == "completed",
            "current lookup returns the stream head",
            current=current,
        )
        _assert_rule(
            revision["checkpoints"][0]["revision"] == 2
            and asof["checkpoints"][0]["revision"] == 2,
            "revision and recordedAt as-of return the latest eligible row",
            revision=revision,
            asof=asof,
        )
        with pytest.raises(ValueError, match="project.*agent"):
            checkpoint_store.getTaskStates(store, taskId="task-1", revision=1)
        with pytest.raises(ValueError, match="mutually exclusive"):
            checkpoint_store.getTaskStates(
                store,
                taskId="task-1",
                project="pensive",
                agent="agent-a",
                revision=1,
                asOf=100,
            )
    finally:
        store.close()


def test_history_and_recent_pages_are_explicit(tmp_path, monkeypatch):
    # boundary: pagination
    timestamps = iter([100, 200, 150, 300, 400])
    monkeypatch.setattr(checkpoint_store, "_now", lambda: next(timestamps))
    store = openStore(tmp_path / "store.db")
    try:
        _put(store)
        _put(store, expectedRevision=1, requestId="request-2", state="blocked")
        _put(store, taskId="task-2", requestId="request-4")
        _put(store, expectedRevision=2, requestId="request-3", state="completed")
        _put(store, taskId="task-3", requestId="request-5")
        page = checkpoint_store.listTaskHistory(
            store, project="pensive", agent="agent-a", taskId="task-1", limit=2)
        tail = checkpoint_store.listTaskHistory(
            store, project="pensive", agent="agent-a", taskId="task-1",
            afterRevision=page["nextRevision"], limit=2)
        recent = checkpoint_store.listRecentTaskStates(
            store, project="pensive", limit=2)
        _assert_rule(
            [row["revision"] for row in page["checkpoints"]] == [1, 2]
            and page["nextRevision"] == 2 and page["truncated"] is True,
            "history page exposes a reusable exclusive cursor and truncation",
            page=page,
        )
        _assert_rule(
            [row["revision"] for row in tail["checkpoints"]] == [3]
            and tail["nextRevision"] is None
            and any(row["state"] == "completed" for row in recent["checkpoints"])
            and recent["truncated"] is True,
            "history tail and recent page expose completion and truncation",
            tail=tail,
            recent=recent,
        )
    finally:
        store.close()


def test_checkpoint_rejects_bool_and_bound_violations(tmp_path):
    # boundary: real-integer-and-length-guards
    store = openStore(tmp_path / "store.db")
    try:
        for field, value in (("expectedRevision", True), ("expectedRevision", 1.0)):
            with pytest.raises((TypeError, ValueError), match=field):
                _put(store, requestId=f"bad-{field}-{value}", **{field: value})
        with pytest.raises(ValueError, match="project"):
            _put(store, project="")
        with pytest.raises(ValueError, match="project"):
            _put(store, project="   ")
        with pytest.raises(ValueError, match="agent"):
            _put(store, agent="x" * 65)
        with pytest.raises(ValueError, match="agent"):
            _put(store, agent=" \t ")
        with pytest.raises(ValueError, match="taskId"):
            _put(store, taskId="x" * 257)
        with pytest.raises(ValueError, match="taskId"):
            _put(store, taskId="\n")
        with pytest.raises(ValueError, match="requestId"):
            _put(store, requestId="x" * 257)
        with pytest.raises(ValueError, match="requestId"):
            _put(store, requestId=" \n")
        with pytest.raises(ValueError, match="body"):
            _put(store, body="x" * 32001)
        with pytest.raises(ValueError, match="state"):
            _put(store, state="paused")
        count = store._conn.execute(
            "SELECT COUNT(*) FROM task_checkpoints"
        ).fetchone()[0]
        _assert_rule(
            count == 0,
            "invalid checkpoint inputs fail before any append",
            count=count,
        )
    finally:
        store.close()


def test_checkpoint_rejects_an_open_caller_transaction_without_rollback(tmp_path):
    # resource: caller-transaction-ownership
    store = openStore(tmp_path / "store.db")
    try:
        store._conn.execute(
            "INSERT INTO atoms(id, text, kind, created_at, schema_version) "
            "VALUES ('caller-atom', 'caller work', 'atom', 1, 4)"
        )
        with pytest.raises(RuntimeError, match="own transaction"):
            _put(store)
        store._conn.commit()
        row = store._conn.execute(
            "SELECT text FROM atoms WHERE id = 'caller-atom'"
        ).fetchone()
        count = store._conn.execute(
            "SELECT COUNT(*) FROM task_checkpoints"
        ).fetchone()[0]
        _assert_rule(
            row == ("caller work",) and count == 0,
            "checkpoint begin guard preserves the caller transaction and appends nothing",
            row=row,
            count=count,
        )
    finally:
        store.close()
