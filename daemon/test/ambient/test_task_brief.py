"""Regression tests for task-aware working-set briefs."""

from types import SimpleNamespace

from store.checkpoints import putTaskCheckpoint
from store.store import addFacet, openStore, putAtom

from ambient.briefer import brief


def _assert_rule(condition, message, **state):
    assert condition, f"{message}: state={state!r}"


def _checkpoint(store, *, task_id="task-1", request_id="request-1", body="current task body",
                state="active", project="pensive", agent="agent-a"):
    return putTaskCheckpoint(
        store,
        project=project,
        agent=agent,
        taskId=task_id,
        expectedRevision=0,
        requestId=request_id,
        state=state,
        body=body,
        source="test",
    )


def test_scoped_task_brief_shows_current_state_and_keeps_pins_and_loose_ends(tmp_path):
    # integration: scoped-task-view
    store = openStore(tmp_path / "store.db")
    try:
        task = _checkpoint(store)
        generic = putAtom(store, {
            "text": "generic active memory must be suppressed",
            "kind": "atom",
            "project": "pensive",
            "provenance": {"source": "test", "agent": "agent-a"},
        })
        pin = putAtom(store, {
            "text": "standing pin remains visible in a task brief",
            "kind": "atom",
            "project": "pensive",
            "provenance": {"source": "test", "agent": "agent-a"},
        })
        loose = putAtom(store, {
            "text": "loose end remains visible in a task brief",
            "kind": "atom",
            "project": "pensive",
            "provenance": {"source": "test", "agent": "agent-a"},
        })
        addFacet(store, pin, "pin", "1")
        addFacet(store, loose, "tag", "for:agent-a")
        output = brief(store, {
            "taskId": "task-1", "project": "pensive", "agent": "agent-a",
            "budget": 1500, "now": 2_000_000_000,
        })
        _assert_rule(
            "current task state:" in output
            and "task task-1 revision 1" in output
            and "state=active" in output
            and task["body"] in output
            and "standing pin remains visible" in output
            and "loose end remains visible" in output
            and "generic active memory must be suppressed" not in output
            and f"p3://{generic}" not in output,
            "scoped task brief contains explicit state and retained view sections",
            output=output,
            generic=generic,
        )
    finally:
        store.close()


def test_scoped_task_brief_small_budget_keeps_metadata_and_omits_body(tmp_path):
    # boundary: task-metadata-budget-fallback
    store = openStore(tmp_path / "store.db")
    try:
        _checkpoint(store, body="long body " * 1000)
        output = brief(store, {
            "taskId": "task-1", "project": "pensive", "agent": "agent-a",
            "budget": 64, "now": 2_000_000_000,
        })
        _assert_rule(
            "task task-1 revision 1" in output
            and "state=active" in output
            and "body omitted" in output
            and "long body" not in output,
            "small task brief budget retains revision identity and states omitted body",
            output=output,
        )
    finally:
        store.close()


def test_scoped_task_brief_uses_completed_head_and_unscoped_brief_keeps_active(tmp_path):
    # state: completed-head-and-unscoped-compatibility
    store = openStore(tmp_path / "store.db")
    try:
        _checkpoint(store, state="completed", body="completed task body")
        generic = putAtom(store, {
            "text": "unscoped active memory remains visible",
            "kind": "atom",
            "project": "pensive",
            "provenance": {"source": "test", "agent": "agent-a"},
        })
        scoped = brief(store, {
            "taskId": "task-1", "project": "pensive", "agent": "agent-a",
            "budget": 1500, "now": 2_000_000_000,
        })
        unscoped = brief(store, {
            "agent": "agent-a", "budget": 1500, "now": 2_000_000_000,
        })
        _assert_rule(
            "state=completed" in scoped
            and "completed task body" in scoped
            and "unscoped active memory remains visible" not in scoped
            and f"p3://{generic}" in unscoped,
            "completed scoped state and unscoped active compatibility remain distinct",
            scoped=scoped,
            unscoped=unscoped,
        )
    finally:
        store.close()


def test_scoped_task_brief_is_read_only(tmp_path, monkeypatch):
    # persistence: brief-view-does-not-write
    import ambient.briefer as briefer

    store = openStore(tmp_path / "store.db")
    try:
        _checkpoint(store)
        def forbidden_active_ranking(*args, **kwargs):
            raise AssertionError("scoped brief must skip generic active ranking")

        monkeypatch.setattr(briefer, "_activeRanked", forbidden_active_ranking)
        before = {
            table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "atoms", "provenance", "facets", "task_checkpoints", "recall_log",
            )
        }
        brief(store, {
            "taskId": "task-1", "project": "pensive", "agent": "agent-a",
            "budget": 1500, "now": 2_000_000_000,
        })
        after = {
            table: store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        _assert_rule(
            after == before,
            "scoped brief reads task state without mutating canonical tables",
            before=before,
            after=after,
        )
    finally:
        store.close()


def test_scoped_task_brief_caps_heads_and_fits_budget(tmp_path):
    # boundary: many-task-heads-small-budget
    store = openStore(tmp_path / "store.db")
    try:
        for index in range(75):
            _checkpoint(
                store,
                request_id=f"request-{index}",
                body=f"body-{index}",
                agent=f"agent-{index}",
            )
        output = brief(store, {
            "taskId": "task-1", "project": "pensive", "budget": 32,
            "now": 2_000_000_000,
        })
        _assert_rule(
            len(output) <= 32 * 3
            and "body omitted" in output
            and "more omitted" in output
            and output.count("task task-1 revision") <= 8,
            "scoped task heads stay capped and disclose budget/task truncation",
            tokens=(len(output) + 2) // 3,
            output=output,
        )
    finally:
        store.close()


def test_brief_endpoint_passes_project_and_task_id(tmp_path, monkeypatch):
    # contract: brief-endpoint-task-query
    from starlette.testclient import TestClient

    import serve.daemon as daemon

    observed = []

    def fake_brief(store, options):
        observed.append((store, options))
        return "current task state: task task-1 revision 1 state=active"

    monkeypatch.setattr(daemon, "brief", fake_brief)
    store = openStore(tmp_path / "store.db")
    try:
        ctx = SimpleNamespace(store=store, indexes={}, embedder=None, modelId="test",
                              aux=None, agent="agent-a")
        app = daemon.buildApp(ctx)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.get(
                "/brief",
                params={"project": "pensive", "taskId": "task-1", "agent": "agent-a", "budget": "64"},
            )
            unscoped = client.get("/brief", params={"agent": "agent-a", "budget": "64"})
        _assert_rule(
            response.status_code == 200
            and observed
            and observed[0][1] == {
                "project": "pensive", "taskId": "task-1", "agent": "agent-a", "budget": 64,
            }
            and response.json()["taskId"] == "task-1"
            and response.json()["project"] == "pensive"
            and set(unscoped.json()) == {"brief", "agent", "budget"},
            "brief endpoint forwards task scope and returns it in the envelope",
            status=response.status_code,
            observed=observed,
            payload=response.json(),
        )
    finally:
        store.close()
