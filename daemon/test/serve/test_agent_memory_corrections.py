"""Regression tests for atomic corrections and pasted MCP handles.

Every test name maps to a row in ``RISK_MODEL_AGENT_MEMORY.md``. These tests use
temporary SQLite stores and small fake contexts/indexes; no model or live daemon
is involved.
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import serve.mcp as mcp
import store.store as store_module
from lifecycle.integrity import integrityScan
from store.store import (
    addEdge,
    addFacet,
    atomCount,
    edgesTo,
    facetsOf,
    getAtom,
    openStore,
    putAtom,
    supersede,
)


def _put(store, text, *, kind="atom", project="aegis", importance=0.0,
         occurred_at=None, source="bulk-import"):
    value = {
        "text": text,
        "kind": kind,
        "project": project,
        "importance": importance,
        "provenance": {"source": source},
    }
    if occurred_at is not None:
        value["occurredAt"] = occurred_at
    return putAtom(store, value)


def _ctx(store, *, index_atom=None, reindex=None, retire_atom=None):
    return SimpleNamespace(
        store=store,
        agent="test-agent",
        indexAtom=index_atom or (lambda _atom_id, _kind: None),
        reindex=reindex or (lambda kinds=None: None),
        retireAtom=retire_atom or (lambda _atom_id, _kind: None),
    )


def test_correct_atom_commits_successor_edge_and_metadata_atomically(tmp_path):
    store = openStore(tmp_path / "mem.db")
    try:
        old = _put(store, "old correction body", kind="narrative", project="pensive",
                   importance=0.75, occurred_at=1_700_000_000)
        old_before = getAtom(store, old)
        addFacet(store, old, "pin", "true")
        addFacet(store, old, "tag", "for:grok")
        addFacet(store, old, "tag", "reviewed")
        addFacet(store, old, "entity", "content-derived")

        new = store_module.correctAtom(store, old, {
            "text": "new correction body",
            "provenance": {"source": "explicit-emit", "agent": "grok"},
        })

        old_row = getAtom(store, old)
        new_row = getAtom(store, new)
        assert old_row["status"] == "superseded", (
            f"correction-state invariant violated: old={old_row!r}"
        )
        assert new_row["status"] == "live", (
            f"correction-successor-live invariant violated: new={new_row!r}"
        )
        assert atomCount(store) == 2, (
            f"correction-single-successor invariant violated: atomCount={atomCount(store)}"
        )
        edge_rows = edgesTo(store, old, "supersedes")
        assert len(edge_rows) == 1 and edge_rows[0]["srcAtom"] == new, (
            f"correction-edge invariant violated: old={old!r} edges={edge_rows!r}"
        )
        assert new_row["kind"] == "narrative" and new_row["project"] == "pensive", (
            f"correction-class-metadata invariant violated: new={new_row!r}"
        )
        assert new_row["importance"] == pytest.approx(0.75), (
            f"correction-importance-continuity invariant violated: new={new_row!r}"
        )
        assert new_row["occurredAt"] == 1_700_000_000, (
            f"correction-occurred-time continuity invariant violated: new={new_row!r}"
        )
        assert facetsOf(store, new) == [
            {"key": "pin", "value": "true"},
            {"key": "tag", "value": "for:grok"},
            {"key": "tag", "value": "reviewed"},
        ], (
            f"correction-facet-continuity invariant violated: facets={facetsOf(store, new)!r}"
        )
        assert [p["source"] for p in new_row["provenance"]] == [
            "explicit-emit", "explicit-emit"
        ], (
            f"correction-provenance-authorship invariant violated: provenance={new_row['provenance']!r}"
        )
        assert all(p["source"] != "bulk-import" for p in new_row["provenance"]), (
            f"correction-old-provenance-copy invariant violated: provenance={new_row['provenance']!r}"
        )
        assert getAtom(store, old)["text"] == old_before["text"], (
            f"correction-old-text retention invariant violated: before={old_before!r} after={getAtom(store, old)!r}"
        )
        assert getAtom(store, old)["provenance"] == old_before["provenance"], (
            f"correction-old-provenance retention invariant violated: before={old_before!r} after={getAtom(store, old)!r}"
        )
        edge_provenance_id = edge_rows[0]["provenanceId"]
        assert edge_provenance_id in {
            row["id"] for row in new_row["provenance"]
        }, (
            f"correction-edge-provenance-link invariant violated: edge={edge_rows[0]!r} provenance={new_row['provenance']!r}"
        )
    finally:
        store.close()


def test_correct_atom_rolls_back_without_orphan(tmp_path, monkeypatch):
    store = openStore(tmp_path / "mem.db")
    try:
        old = _put(store, "rollback source")
        before = atomCount(store)

        def fail_edge(*_args, **_kwargs):
            raise RuntimeError("edge write sabotage")

        monkeypatch.setattr(store_module, "_insertEdge", fail_edge)
        with pytest.raises(RuntimeError, match="edge write sabotage"):
            store_module.correctAtom(store, old, {
                "text": "must not persist",
                "provenance": {"source": "explicit-emit"},
            })

        assert atomCount(store) == before, (
            f"correction-rollback atom invariant violated: before={before} after={atomCount(store)}"
        )
        assert getAtom(store, old)["status"] == "live", (
            f"correction-rollback status invariant violated: old={getAtom(store, old)!r}"
        )
        assert store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE type = 'supersedes'"
        ).fetchone()[0] == 0, "correction-rollback edge invariant violated: supersedes row leaked"
        assert store._conn.execute(
            "SELECT COUNT(*) FROM fts WHERE fts MATCH 'persist'"
        ).fetchone()[0] == 0, "correction-rollback FTS invariant violated: orphan text leaked"
    finally:
        store.close()


def test_correct_atom_rolls_back_every_write_on_late_status_failure(tmp_path):
    store = openStore(tmp_path / "late-rollback.db")
    try:
        old = _put(store, "late rollback source")
        addFacet(store, old, "pin", "true")
        before = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM facets").fetchone()[0],
        )
        store._conn.execute(
            "CREATE TRIGGER reject_superseded_status "
            "BEFORE UPDATE OF status ON atoms "
            "WHEN new.status = 'superseded' "
            "BEGIN SELECT RAISE(ABORT, 'status sabotage'); END"
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="status sabotage"):
            store_module.correctAtom(store, old, {
                "text": "must roll back after all inserts",
                "provenance": {"source": "explicit-emit"},
            })

        after = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM facets").fetchone()[0],
        )
        assert after == before, (
            f"correction-late-rollback invariant violated: before={before!r} after={after!r}"
        )
        assert getAtom(store, old)["status"] == "live", (
            f"correction-late-rollback status invariant violated: old={getAtom(store, old)!r}"
        )
    finally:
        store.close()


def test_correct_atom_persists_canonical_state_after_reopen(tmp_path):
    path = tmp_path / "persistent-correction.db"
    store = openStore(path)
    old = _put(store, "persistent old", importance=0.625,
               occurred_at=1_650_000_000)
    addFacet(store, old, "pin", "true")
    addFacet(store, old, "tag", "for:grok")
    new = store_module.correctAtom(store, old, {
        "text": "persistent successor",
        "provenance": {"source": "explicit-emit", "agent": "grok"},
    })
    store.close()

    reopened = openStore(path)
    try:
        old_row = getAtom(reopened, old)
        new_row = getAtom(reopened, new)
        edge_rows = edgesTo(reopened, old, "supersedes")
        assert old_row["status"] == "superseded" and old_row["text"] == "persistent old", (
            f"correction-persistence predecessor invariant violated: old={old_row!r}"
        )
        assert new_row["status"] == "live" and new_row["text"] == "persistent successor", (
            f"correction-persistence successor invariant violated: new={new_row!r}"
        )
        assert new_row["importance"] == pytest.approx(0.625), (
            f"correction-persistence importance invariant violated: new={new_row!r}"
        )
        assert facetsOf(reopened, new) == [
            {"key": "pin", "value": "true"},
            {"key": "tag", "value": "for:grok"},
        ], (
            f"correction-persistence facet invariant violated: facets={facetsOf(reopened, new)!r}"
        )
        assert len(edge_rows) == 1 and edge_rows[0]["srcAtom"] == new, (
            f"correction-persistence edge invariant violated: edges={edge_rows!r}"
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("status", ["superseded", "tombstone"])
def test_correct_atom_refuses_stale_target_with_successor_guidance(tmp_path, status):
    store = openStore(tmp_path / f"{status}.db")
    try:
        old = _put(store, f"{status} source")
        if status == "superseded":
            successor = _put(store, "existing successor")
            supersede(store, old, successor, {"source": "distiller"})
        else:
            successor = None
            store._conn.execute(
                "UPDATE atoms SET status = 'tombstone' WHERE id = ?", (old,))
            store._conn.commit()
        before = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        )

        with pytest.raises(ValueError) as raised:
            store_module.correctAtom(store, old, {
                "text": "must not overwrite stale state",
                "provenance": {"source": "explicit-emit"},
            })

        message = str(raised.value)
        assert "not live" in message and "history" in message, (
            f"stale-correction diagnostic invariant violated: status={status!r} message={message!r}"
        )
        if successor is not None:
            assert f"p3://{successor}" in message, (
                f"stale-correction successor-guidance invariant violated: message={message!r}"
            )
        after = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        )
        assert after == before, (
            f"stale-correction zero-write invariant violated: status={status!r} before={before} after={after}"
        )
    finally:
        store.close()


def test_concurrent_corrections_create_one_successor(tmp_path):
    path = tmp_path / "concurrent.db"
    seed = openStore(path)
    old = _put(seed, "one live source")
    seed.close()

    def correct_in_independent_connection(number):
        store = openStore(path)
        try:
            try:
                return store_module.correctAtom(store, old, {
                    "text": f"winner {number}",
                    "provenance": {"source": "explicit-emit", "agent": f"agent-{number}"},
                })
            except Exception as exc:
                return exc
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(correct_in_independent_connection, (1, 2)))

    successes = [item for item in outcomes if isinstance(item, str)]
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    assert len(successes) == 1 and len(failures) == 1, (
        f"correction-CAS invariant violated: outcomes={outcomes!r} successes={successes!r} failures={failures!r}"
    )
    assert isinstance(failures[0], ValueError) and "history" in str(failures[0]), (
        f"correction-stale-loser diagnostic invariant violated: failure={failures[0]!r}"
    )
    check = openStore(path)
    try:
        assert atomCount(check) == 2, (
            f"correction-concurrent-single-successor invariant violated: atomCount={atomCount(check)}"
        )
        assert len(edgesTo(check, old, "supersedes")) == 1, (
            f"correction-concurrent-edge invariant violated: edges={edgesTo(check, old, 'supersedes')!r}"
        )
    finally:
        check.close()


def test_history_lists_all_fork_branches(tmp_path, monkeypatch):
    store = openStore(tmp_path / "fork.db")
    try:
        old = _put(store, "fork root")
        first = _put(store, "first legacy branch")
        second = _put(store, "second legacy branch")
        supersede(store, old, first, {"source": "distiller"})
        supersede(store, old, second, {"source": "distiller"})
        monkeypatch.setattr(mcp, "assembleTier2", lambda _store, _atom_id: "tier2")

        text = mcp.handle_history(_ctx(store), {"atomId": old})

        assert all(f"p3://{atom_id}" in text for atom_id in (old, first, second)), (
            f"history-fork-visibility invariant violated: ids={(old, first, second)!r} text={text!r}"
        )
        assert "supersession graph" in text and "fork" in text.lower(), (
            f"history-fork-label invariant violated: text={text!r}"
        )
        fork_lines = [line for line in text.splitlines() if line.startswith("fork ")]
        assert len(fork_lines) == 1, (
            f"history-single-fork-line invariant violated: forkLines={fork_lines!r} text={text!r}"
        )
        fork_line = fork_lines[0]
        assert all(f"p3://{atom_id}" in fork_line for atom_id in (old, first, second)), (
            f"history-fork-edge invariant violated: forkLine={fork_line!r}"
        )
        branch_order = [
            edge["srcAtom"] for edge in sorted(
                edgesTo(store, old, "supersedes"), key=lambda item: item["id"])
        ]
        assert [
            text.index(f"p3://{atom_id}") for atom_id in branch_order
        ] == sorted(text.index(f"p3://{atom_id}") for atom_id in branch_order), (
            f"history-fork-determinism invariant violated: branchOrder={branch_order!r} text={text!r}"
        )
    finally:
        store.close()


def test_history_from_fork_descendant_lists_sibling_branch(tmp_path, monkeypatch):
    store = openStore(tmp_path / "fork-descendant.db")
    try:
        root = _put(store, "fork root")
        first = _put(store, "first legacy branch")
        second = _put(store, "second legacy branch")
        leaf = _put(store, "first branch leaf")
        supersede(store, root, first, {"source": "distiller"})
        supersede(store, root, second, {"source": "distiller"})
        supersede(store, first, leaf, {"source": "distiller"})
        monkeypatch.setattr(mcp, "assembleTier2", lambda _store, _atom_id: "tier2")

        text = mcp.handle_history(_ctx(store), {"atomId": leaf})

        expected = (root, first, second, leaf)
        assert all(f"p3://{atom_id}" in text for atom_id in expected), (
            f"history-connected-fork invariant violated: expected={expected!r} text={text!r}"
        )
        assert "supersession graph" in text and "fork" in text.lower(), (
            f"history-connected-fork label invariant violated: text={text!r}"
        )
    finally:
        store.close()


def test_history_cycle_is_finite(tmp_path, monkeypatch):
    store = openStore(tmp_path / "cycle.db")
    try:
        first = _put(store, "cycle first")
        second = _put(store, "cycle second")
        store._conn.execute(
            "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
            "VALUES (?, ?, ?, 'supersedes', 1.0, 1, NULL)",
            ("cycle-edge-a", second, first),
        )
        store._conn.execute(
            "INSERT INTO edges(id, src_atom, dst_atom, type, weight, created_at, provenance_id) "
            "VALUES (?, ?, ?, 'supersedes', 1.0, 2, NULL)",
            ("cycle-edge-b", first, second),
        )
        store._conn.commit()
        monkeypatch.setattr(mcp, "assembleTier2", lambda _store, _atom_id: "tier2")

        text = mcp.handle_history(_ctx(store), {"atomId": first})

        assert text.count("p3://") == 2, (
            f"history-cycle-bound invariant violated: handleCount={text.count('p3://')} text={text!r}"
        )
    finally:
        store.close()


def test_integrity_marks_fork_unhealthy(tmp_path):
    store = openStore(tmp_path / "integrity.db")
    try:
        old = _put(store, "fork root")
        first = _put(store, "branch one")
        second = _put(store, "branch two")
        supersede(store, old, first, {"source": "distiller"})
        supersede(store, old, second, {"source": "distiller"})

        report = integrityScan(store)

        assert report["ok"] is False, (
            f"integrity-fork-health invariant violated: report={report!r}"
        )
        assert report["supersessionChains"]["forkedPredecessors"] == [{
            "atomId": old,
            "successors": sorted((first, second)),
        }], (
            f"integrity-fork-report invariant violated: forks={report['supersessionChains'].get('forkedPredecessors')!r}"
        )
    finally:
        store.close()


def test_integrity_reports_each_fork_branch_once(tmp_path):
    store = openStore(tmp_path / "integrity-duplicate-edge.db")
    try:
        old = _put(store, "fork root")
        first = _put(store, "branch one")
        second = _put(store, "branch two")
        supersede(store, old, first, {"source": "distiller"})
        supersede(store, old, second, {"source": "distiller"})
        addEdge(store, {"src": first, "dst": old, "type": "supersedes"})

        report = integrityScan(store)

        assert report["supersessionChains"]["forkedPredecessors"] == [{
            "atomId": old,
            "successors": sorted((first, second)),
        }], (
            f"integrity-unique-fork-branch invariant violated: forks={report['supersessionChains']['forkedPredecessors']!r}"
        )
    finally:
        store.close()


def test_correct_reports_committed_handle_after_index_failure(tmp_path):
    store = openStore(tmp_path / "index-failure.db")
    calls = []
    try:
        old = _put(store, "index failure source")

        def fail_index(_atom_id, _kind):
            calls.append("index")
            raise RuntimeError("incremental index failed")

        def fail_rebuild(kinds=None):
            calls.append(("rebuild", kinds))
            raise RuntimeError("class rebuild failed")

        def retire(_atom_id, _kind):
            calls.append("retire")

        result, is_error = mcp.dispatch(
            _ctx(store, index_atom=fail_index, reindex=fail_rebuild, retire_atom=retire),
            "correct",
            {"oldAtomId": old, "newText": "committed despite index failure"},
        )

        edge_rows = edgesTo(store, old, "supersedes")
        assert len(edge_rows) == 1, (
            f"postcommit-edge invariant violated: old={old!r} edges={edge_rows!r}"
        )
        new = edge_rows[0]["srcAtom"]
        assert is_error is True and "COMMITTED" in result and f"p3://{new}" in result, (
            f"postcommit-diagnostic invariant violated: result={result!r} new={new!r}"
        )
        assert getAtom(store, new)["status"] == "live", (
            f"postcommit-durability invariant violated: new={getAtom(store, new)!r}"
        )
        assert calls == ["index", ("rebuild", ("atom",)), "retire"], (
            f"postcommit-recovery invariant violated: calls={calls!r}"
        )
    finally:
        store.close()


def test_correct_add_failure_recovers_with_rebuild_without_retire(tmp_path):
    store = openStore(tmp_path / "add-recovery.db")
    calls = []
    try:
        old = _put(store, "add recovery source")

        def fail_index(_atom_id, _kind):
            calls.append("index")
            raise RuntimeError("incremental add failed")

        def rebuild(kinds=None):
            calls.append(("rebuild", kinds))

        def retire(_atom_id, _kind):
            calls.append("retire")
            raise RuntimeError("retire must not run after a complete rebuild")

        result, is_error = mcp.dispatch(
            _ctx(store, index_atom=fail_index, reindex=rebuild, retire_atom=retire),
            "correct",
            {"oldAtomId": old, "newText": "recovered by class rebuild"},
        )

        assert is_error is False and "corrected p3://" in result, (
            f"postcommit-add-rebuild recovery invariant violated: result={result!r} error={is_error}"
        )
        assert calls == ["index", ("rebuild", ("atom",))], (
            f"postcommit-add-rebuild call invariant violated: calls={calls!r}"
        )
    finally:
        store.close()


def test_correct_retire_failure_recovers_with_rebuild(tmp_path):
    store = openStore(tmp_path / "retire-recovery.db")
    calls = []
    try:
        old = _put(store, "retire recovery source")

        def index(_atom_id, _kind):
            calls.append("index")

        def fail_retire(_atom_id, _kind):
            calls.append("retire")
            raise RuntimeError("incremental remove failed")

        def rebuild(kinds=None):
            calls.append(("rebuild", kinds))

        result, is_error = mcp.dispatch(
            _ctx(store, index_atom=index, reindex=rebuild, retire_atom=fail_retire),
            "correct",
            {"oldAtomId": old, "newText": "retirement recovered by rebuild"},
        )

        assert is_error is False and "corrected p3://" in result, (
            f"postcommit-retire-rebuild recovery invariant violated: result={result!r} error={is_error}"
        )
        assert calls == ["index", "retire", ("rebuild", ("atom",))], (
            f"postcommit-retire-rebuild call invariant violated: calls={calls!r}"
        )
    finally:
        store.close()


def test_correct_reports_committed_handle_when_retire_and_rebuild_fail(tmp_path):
    store = openStore(tmp_path / "retire-rebuild-failure.db")
    calls = []
    try:
        old = _put(store, "retire and rebuild failure source")

        def index(_atom_id, _kind):
            calls.append("index")

        def fail_retire(_atom_id, _kind):
            calls.append("retire")
            raise RuntimeError("incremental remove failed")

        def fail_rebuild(kinds=None):
            calls.append(("rebuild", kinds))
            raise RuntimeError("class rebuild failed")

        result, is_error = mcp.dispatch(
            _ctx(store, index_atom=index, reindex=fail_rebuild,
                 retire_atom=fail_retire),
            "correct",
            {"oldAtomId": old, "newText": "committed before index recovery failed"},
        )

        edge_rows = edgesTo(store, old, "supersedes")
        assert len(edge_rows) == 1, (
            f"postcommit-retire-edge invariant violated: old={old!r} edges={edge_rows!r}"
        )
        new = edge_rows[0]["srcAtom"]
        assert is_error is True and "COMMITTED" in result and f"p3://{new}" in result, (
            f"postcommit-retire-failure diagnostic invariant violated: result={result!r} new={new!r}"
        )
        assert "incremental remove failed" in result and "class rebuild failed" in result, (
            f"postcommit-retire-failure cause invariant violated: result={result!r}"
        )
        assert calls == ["index", "retire", ("rebuild", ("atom",))], (
            f"postcommit-retire-failure call invariant violated: calls={calls!r}"
        )
    finally:
        store.close()


def test_native_handlers_accept_bare_and_prefixed_ids(tmp_path, monkeypatch):
    store = openStore(tmp_path / "handles.db")
    try:
        monkeypatch.setattr(mcp, "assembleTier2", lambda _store, _atom_id: "tier2")
        ctx = _ctx(store)
        for prefix in ("", "p3://"):
            old = _put(store, f"{prefix or 'bare'} handle source")
            handle = f"{prefix}{old}"

            pinned, pin_error = mcp.dispatch(ctx, "pin", {"atomId": handle})
            history, history_error = mcp.dispatch(ctx, "history", {"atomId": handle})
            corrected, correct_error = mcp.dispatch(ctx, "correct", {
                "oldAtomId": handle, "newText": f"{prefix or 'bare'} handle successor",
            })
            edge_rows = edgesTo(store, old, "supersedes")
            assert pin_error is False and pinned == f"pinned p3://{old} (ok)", (
                f"handle-pin normalization invariant violated: prefix={prefix!r} result={pinned!r} error={pin_error}"
            )
            assert history_error is False and "tier2" in history, (
                f"handle-history normalization invariant violated: prefix={prefix!r} result={history!r} error={history_error}"
            )
            assert correct_error is False and len(edge_rows) == 1, (
                f"handle-correct normalization invariant violated: prefix={prefix!r} result={corrected!r} error={correct_error} edges={edge_rows!r}"
            )
            new = edge_rows[0]["srcAtom"]
            new_handle = f"{prefix}{new}"
            unpinned, unpin_error = mcp.dispatch(
                ctx, "unpin", {"atomId": new_handle})

            assert correct_error is False and f"p3://{new}" in corrected, (
                f"handle-correct normalization invariant violated: prefix={prefix!r} result={corrected!r} error={correct_error}"
            )
            assert unpin_error is False and unpinned == f"unpinned p3://{new} (ok)", (
                f"handle-unpin normalization invariant violated: prefix={prefix!r} result={unpinned!r} error={unpin_error}"
            )
    finally:
        store.close()


def test_correct_missing_valid_id_is_zero_write(tmp_path):
    store = openStore(tmp_path / "missing-valid-id.db")
    try:
        existing = _put(store, "existing atom")
        before = atomCount(store)

        result, is_error = mcp.dispatch(_ctx(store), "correct", {
            "oldAtomId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "newText": "must not be written",
        })

        assert is_error is True and "not found" in result, (
            f"correction-missing-target diagnostic invariant violated: result={result!r}"
        )
        assert atomCount(store) == before and getAtom(store, existing)["status"] == "live", (
            f"correction-missing-target zero-write invariant violated: before={before} after={atomCount(store)}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("bad_id", [
    None,
    7,
    ["01ARZ3NDEKTSV4RRFFQ69G5FAV"],
    "",
    "p3://p3://01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "p2://01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ref:01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "p3:/01ARZ3NDEKTSV4RRFFQ69G5FAV",
    " 01ARZ3NDEKTSV4RRFFQ69G5FAV",
])
def test_native_handlers_reject_malformed_handles_before_write(tmp_path, bad_id):
    store = openStore(tmp_path / "malformed.db")
    try:
        old = _put(store, "malformed target")
        before = atomCount(store)
        result, is_error = mcp.dispatch(
            _ctx(store), "correct", {"oldAtomId": bad_id, "newText": "must reject"})

        shape_terms = ("atom id", "prefix", "scheme", "ulid", "required")
        assert is_error is True and any(term in result.lower() for term in shape_terms), (
            f"handle-shape validation invariant violated: bad_id={bad_id!r} result={result!r}"
        )
        assert atomCount(store) == before and getAtom(store, old)["status"] == "live", (
            f"handle-shape zero-write invariant violated: bad_id={bad_id!r} atomCount={atomCount(store)} old={getAtom(store, old)!r}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("bad_args", [
    {"newText": 7},
    {"newText": ["not", "text"]},
    {"newText": "valid text", "provenance": []},
    {"newText": "valid text", "provenance": ["not", "an", "object"]},
])
def test_correct_rejects_malformed_content_before_write(tmp_path, bad_args):
    store = openStore(tmp_path / "malformed-content.db")
    try:
        old = _put(store, "malformed content target")
        before = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        )

        result, is_error = mcp.dispatch(
            _ctx(store), "correct", {"oldAtomId": old, **bad_args})

        after = (
            atomCount(store),
            store._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()[0],
            store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        )
        assert is_error is True and (
            "newtext" in result.lower() or "provenance" in result.lower()
        ), (
            f"correction-content validation invariant violated: args={bad_args!r} result={result!r}"
        )
        assert after == before and getAtom(store, old)["status"] == "live", (
            f"correction-content zero-write invariant violated: args={bad_args!r} before={before!r} after={after!r}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("bad_limit", [0, -1, 1.5, True, "2"])
def test_get_atom_provenance_limit_requires_positive_integer(tmp_path, bad_limit):
    store = openStore(tmp_path / "provenance-limit-validation.db")
    try:
        atom_id = _put(store, "bounded provenance target")

        with pytest.raises(ValueError, match="provenanceLimit"):
            getAtom(store, atom_id, provenanceLimit=bad_limit)
    finally:
        store.close()


@pytest.mark.parametrize("limit", [1, 3, 10])
def test_get_atom_provenance_limit_preserves_prefix_and_default_full_read(
        tmp_path, limit):
    store = openStore(tmp_path / "provenance-limit.db")
    try:
        atom_id = _put(store, "bounded provenance target")
        for index in range(4):
            store._conn.execute(
                "INSERT INTO provenance(id, atom_id, source, recorded_at) "
                "VALUES (?, ?, ?, ?)",
                (f"0000000000000000000000000{index}", atom_id, f"source-{index}", index),
            )
        store._conn.commit()

        full_default = getAtom(store, atom_id)
        full_explicit = getAtom(store, atom_id, provenanceLimit=None)
        bounded = getAtom(store, atom_id, provenanceLimit=limit)

        assert full_default == full_explicit and len(full_default["provenance"]) == 5, (
            f"get-atom-full-provenance invariant violated: default={full_default!r} explicit={full_explicit!r}"
        )
        assert bounded["provenance"] == full_default["provenance"][:limit], (
            f"get-atom-provenance-prefix invariant violated: limit={limit} bounded={bounded['provenance']!r} full={full_default['provenance']!r}"
        )
    finally:
        store.close()
