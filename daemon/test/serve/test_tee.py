"""Double-write tee + shadow A/B logging: the Phase 3 parallel-run boundary.

Risk model (what could silently break, and the test that catches it):

- **A tee'd emit never reaches the v3 store.** The whole point of the parallel
  run is that every OLD-path emit ALSO lands in v3; if the tee endpoint drops it,
  the A/B corpus is empty and the cutover has no evidence. Caught by
  ``test_tee_emit_atom_lands_in_v3_store`` (row present, right kind/project/
  provenance) and ``test_tee_all_emit_tools_route_and_write`` (every emit name
  routes to its handler).

- **A v3 failure bleeds into the old path.** The old server fire-and-forgets to
  the tee AFTER its own authoritative write; a v3 store error (or a malformed
  payload) that raised or hung would corrupt or delay the production response.
  The daemon-side guarantee is that the boundary CONTAINS every failure: it
  returns a normal ``(status, body)`` -- never an exception, never a block -- and
  counts it. Caught by ``test_tee_forced_v3_write_error_is_contained_and_counted``
  (a forced store error -> 500 + ``teeFailed``, no raise, nothing written) and
  ``test_tee_malformed_payload_400_and_keeps_serving`` (bad payloads -> 400 +
  counter, the next valid emit still writes).

- **Dedup at the wrong layer.** The old path is authoritative; the tee must mirror
  it faithfully, NOT dedup -- dedup is the distiller's job (a later task). A tee
  that quietly collapsed duplicates would hide real double-emits from the judge.
  Caught by ``test_tee_duplicate_emit_writes_two_rows``.

- **Cross-repo payload drift.** The Engram-side forward and this receiver live in
  two repos and must agree on the wire shape without importing across the repo
  boundary. Pinned by POSTing the EXACT shapes the Engram patch sends
  (``{"tool", "args"}`` for tee, ``{"query", "oldResultText"}`` for shadow) in
  ``test_tee_emit_payload_shape_matches_engram_forward`` and
  ``test_shadow_payload_shape_matches_engram_forward``.

- **The shadow line loses the evidence.** The judge needs the OLD answer verbatim
  (no trimming), the v3 payload, its top-10 ids, and the trust/latency metadata,
  one JSON object per line, append-only. A dropped field, a trimmed old text, or
  an overwrite instead of an append silently ruins the A/B set. Caught by
  ``test_shadow_recall_logs_one_line_full_schema`` (every field, old text equal to
  input), ``test_shadow_ids_are_top10_real_atomids``, and
  ``test_shadow_appends_multiple_lines``.

- **A shadow failure bleeds into the serving path.** Same boundary contract as the
  tee: a recall error or an unwritable log path is contained and counted, never
  raised. Caught by ``test_shadow_recall_failure_is_contained`` (recall raises ->
  ``shadowFailed``, no raise) and ``test_shadow_unwritable_log_is_contained``
  (bad path -> ``shadowFailed`` + serving continues).

- **The endpoints are not wired / counters not exposed.** The Phase 3 gate reads
  the counters off a live daemon; if ``buildApp`` never mounts the routes or the
  ``/status`` surface, the gate is blind. Caught by
  ``test_daemon_wires_tee_shadow_status_routes`` and
  ``test_counters_snapshot_reflects_activity``.

Real components end to end: the embedder + reranker load once per session (the
test_mcp/test_engine session-fixture pattern), a fresh store per test. The tee and
shadow BOUNDARY handlers (``handleTeeEmit``/``runShadow``) are exercised directly
-- payload bytes in, ``(status, body)`` out plus the store/log side effect -- which
is the in-process path the Starlette routes call; the LIVE end-to-end (the real old
server with the flag on) is the controller-owned Phase 3 gate step, not this test.
"""
import json
import re

import pytest

from serve.mcp import ServeContext, dispatch
import serve.mcp as mcp
from serve.tee import Counters, handleTeeEmit, TEE_EMIT_TOOLS
from serve.shadow import (
    runShadow,
    buildShadowRecord,
    appendShadowLine,
    defaultShadowLogPath,
    MAX_SHADOW_IDS,
)
from recall.embedder import Embedder
from store.store import openStore, putAtom, getAtom, facetsOf

MODEL_ID = "BAAI/bge-small-en-v1.5"

_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# The GPU stack emits two SwigPy DeprecationWarnings on first import under CPython
# 3.14; filter exactly those, mirroring the recall/mcp tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


# --------------------------------------------------------------------------- #
# Fixtures (same session-model pattern as test_mcp)                            #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture(scope="session")
def _rerankerWarm():
    from recall.rerank import _getReranker

    _getReranker()


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ctx(store, embedder):
    return ServeContext(store, embedder, MODEL_ID, agent="heph")


@pytest.fixture
def counters():
    return Counters()


def _put(store, text, project="aegis", kind="atom"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": project,
        "provenance": {"source": "bulk-import"},
    })


def _emitBody(project="pensive", **over):
    """The full engram_emit_atom arg shape, as the Engram forward would send it."""
    args = {
        "project": project,
        "shape": "recall must beat BM25 on the eval gate",
        "approach": "RRF fusion + cross-encoder rerank + trust layer",
        "outcome": "succeeded",
        "reason": "reranking pulls paraphrases the dense signal missed",
        "principle": "joint query-document scoring beats bag-of-words at the top",
    }
    args.update(over)
    return json.dumps({"tool": "engram_emit_atom", "args": args}).encode("utf-8")


def _atomRows(store):
    return store._conn.execute("SELECT id, kind, project FROM atoms").fetchall()


# --------------------------------------------------------------------------- #
# TEE: a tee'd emit lands in the v3 store                                      #
# --------------------------------------------------------------------------- #


def test_tee_emit_atom_lands_in_v3_store(ctx, counters):
    status, body = handleTeeEmit(ctx, counters, _emitBody())

    assert status == 200
    assert body.get("ok") is True
    # The legacy emit result string comes back through the tee (proves it ran the
    # real Task 12 handler, not a stub).
    assert re.search(
        r"atom \[succeeded\] emitted \(emission_id: " + _UUID_RE + r"\).*\(ok\)",
        body.get("result", ""),
    )
    # The atom really landed: one row, kind='atom', project in the column,
    # provenance source='explicit-emit', importance 0.0.
    rows = ctx.store._conn.execute(
        "SELECT id, kind, project, importance FROM atoms").fetchall()
    assert len(rows) == 1
    atomId, kind, project, importance = rows[0]
    assert kind == "atom" and project == "pensive" and importance == 0.0
    atom = getAtom(ctx.store, atomId)
    assert atom["provenance"][0]["source"] == "explicit-emit"
    # Counters: one arrival, zero failures.
    assert counters.teeReceived == 1
    assert counters.teeFailed == 0


def test_tee_all_emit_tools_route_and_write(ctx, counters):
    payloads = {
        "engram_emit_atom": {
            "project": "p", "shape": "s", "approach": "a", "outcome": "succeeded",
            "reason": "r", "principle": "atom principle",
        },
        "engram_emit_discovery": {"project": "p", "principle": "discovered a thing"},
        "engram_emit_failure": {"project": "p", "principle": "a thing failed"},
        "engram_emit_narrative": {"project": "p", "narrative": "a quiet narrative"},
        "engram_emit_snapshot": {"project": "p", "hypothesis": "a working theory"},
    }
    assert set(payloads) == set(TEE_EMIT_TOOLS)      # every accepted emit covered
    for i, (tool, args) in enumerate(payloads.items(), start=1):
        body = json.dumps({"tool": tool, "args": args}).encode("utf-8")
        status, resp = handleTeeEmit(ctx, counters, body)
        assert status == 200, f"{tool} did not tee: {resp}"
    # Five emits, five rows.
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 5
    assert counters.teeReceived == 5 and counters.teeFailed == 0


def test_tee_emit_payload_shape_matches_engram_forward(ctx, counters):
    # Pin the exact wire shape the Engram patch sends: {"tool", "args"} with the
    # raw MCP arguments as "args". If the receiver expected a different envelope
    # the parallel run would silently drop every emit.
    body = json.dumps({
        "tool": "engram_emit_atom",
        "args": {
            "project": "pensive", "shape": "s", "approach": "a",
            "outcome": "partial", "reason": "r", "principle": "cross-repo contract",
            "tags": "recall, rerank",
        },
    }).encode("utf-8")
    status, resp = handleTeeEmit(ctx, counters, body)
    assert status == 200 and resp["ok"] is True
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# TEE: a v3 failure is CONTAINED, counted, and cannot touch the old path        #
# --------------------------------------------------------------------------- #


def test_tee_forced_v3_write_error_is_contained_and_counted(ctx, counters, monkeypatch):
    # Force the v3 store write to fail (the realistic "v3 is unhealthy" case) by
    # making putAtom raise. The tee must contain it: a normal (500, error) return
    # -- NOT a raised exception -- so an old server that already completed its own
    # authoritative write is provably undisturbed. Nothing lands in v3.
    def _boom(*a, **k):
        raise RuntimeError("v3 store write failed")

    monkeypatch.setattr(mcp, "putAtom", _boom)

    status, body = handleTeeEmit(ctx, counters, _emitBody())      # must not raise

    assert status == 500
    assert "error" in body
    assert counters.teeReceived == 1
    assert counters.teeFailed == 1
    # No atom was written by the failed tee.
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    # By construction the tee holds only the v3 store via ctx; it has no legacy
    # store/socket handle, so a v3 failure structurally cannot reach the old path.
    assert not hasattr(ctx, "oldStore") and not hasattr(ctx, "legacyStore")


def test_tee_dispatch_raise_is_contained_and_counted(ctx, counters, monkeypatch):
    # dispatch() is designed not to raise, but defend against an UNMODELED raise:
    # it must still be counted as teeFailed and returned as a clean 500, never leak
    # a raw exception to Starlette (which would 500 with teeReceived incremented but
    # teeFailed stuck at zero).
    import serve.tee as tee

    def _boom(*a, **k):
        raise RuntimeError("dispatch blew up unexpectedly")

    monkeypatch.setattr(tee, "dispatch", _boom)

    status, body = handleTeeEmit(ctx, counters, _emitBody())   # must not raise

    assert status == 500 and "error" in body
    assert counters.teeReceived == 1 and counters.teeFailed == 1


def test_tee_malformed_payload_400_and_keeps_serving(ctx, counters):
    bad_bodies = [
        b"not json at all",                                   # unparseable
        json.dumps([1, 2, 3]).encode(),                       # JSON but not an object
        json.dumps({"args": {"project": "p"}}).encode(),      # missing "tool"
        json.dumps({"tool": "pensive_recall",                 # a non-emit tool
                    "args": {"query": "x"}}).encode(),
        json.dumps({"tool": "engram_emit_atom",               # "args" not an object
                    "args": "oops"}).encode(),
    ]
    for raw in bad_bodies:
        status, body = handleTeeEmit(ctx, counters, raw)
        assert status == 400, f"expected 400 for {raw!r}, got {status}"
        assert "error" in body
    assert counters.teeReceived == len(bad_bodies)
    assert counters.teeFailed == len(bad_bodies)
    # Nothing was written by any malformed request.
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0

    # The daemon keeps serving: a valid emit right after the bad ones still writes.
    status, body = handleTeeEmit(ctx, counters, _emitBody())
    assert status == 200
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_tee_duplicate_emit_writes_two_rows(ctx, counters):
    # The old path is authoritative; the tee mirrors it faithfully and does NOT
    # dedup (dedup is the distiller's job). Two identical emits -> two rows.
    body = _emitBody()
    handleTeeEmit(ctx, counters, body)
    handleTeeEmit(ctx, counters, body)
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert counters.teeReceived == 2 and counters.teeFailed == 0


# --------------------------------------------------------------------------- #
# SHADOW: one JSONL line, full schema, old text verbatim                        #
# --------------------------------------------------------------------------- #


def _shadowBody(query, oldResultText):
    return json.dumps({"query": query, "oldResultText": oldResultText}).encode("utf-8")


def test_shadow_recall_logs_one_line_full_schema(ctx, counters, tmp_path, _rerankerWarm):
    for i in range(6):
        _put(ctx.store, f"acoustic modems trade range for data rate at station {i}")
    ctx.reindex()

    logPath = tmp_path / "shadow.jsonl"
    # A deliberately long old answer: the judge needs it verbatim, so the line must
    # store it WHOLE (no trimming).
    oldText = "Found 3 memories:\n" + "\n".join(f"- [{80 - i}%] (aegis) line {i} " + "x" * 200 for i in range(3))

    status, body = runShadow(ctx, counters, _shadowBody("acoustic modem range", oldText), logPath)

    assert status == 200 and body.get("ok") is True
    assert counters.shadowLogged == 1 and counters.shadowFailed == 0

    lines = logPath.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1                                     # exactly one line
    rec = json.loads(lines[0])                                 # valid JSON object

    assert rec["query"] == "acoustic modem range"
    assert isinstance(rec["ts"], str) and rec["ts"]            # a timestamp present
    assert "project" in rec and rec["project"] is None         # unscoped query -> null
    # Old answer stored WHOLE -- byte-for-byte equal to the input, not truncated.
    assert rec["old"]["resultText"] == oldText
    v3 = rec["v3"]
    assert isinstance(v3["ids"], list) and v3["ids"]           # v3 found hits
    assert all(isinstance(i, str) for i in v3["ids"])
    assert isinstance(v3["payload"], str) and v3["payload"]
    assert isinstance(v3["tokensUsed"], int)
    assert isinstance(v3["lowConfidence"], bool)
    assert isinstance(v3["latencyMs"], (int, float)) and v3["latencyMs"] >= 0
    # The v3 ids are real atoms in the store.
    for aid in v3["ids"]:
        assert getAtom(ctx.store, aid) is not None


def test_shadow_ids_are_top10_real_atomids(ctx, counters, tmp_path, _rerankerWarm):
    # More than 10 strong matches: the line records at most the top-10 ids.
    for i in range(15):
        _put(ctx.store, f"the extended kalman filter fuses INS and DVL at tick {i}")
    ctx.reindex()

    logPath = tmp_path / "shadow.jsonl"
    status, _ = runShadow(ctx, counters, _shadowBody("kalman filter INS DVL fusion", "old"), logPath)
    assert status == 200

    rec = json.loads(logPath.read_text(encoding="utf-8").splitlines()[0])
    ids = rec["v3"]["ids"]
    assert 0 < len(ids) <= MAX_SHADOW_IDS
    assert len(ids) == len(set(ids))                           # no duplicate ids


def test_shadow_appends_multiple_lines(ctx, counters, tmp_path, _rerankerWarm):
    _put(ctx.store, "sonar bathymetry swath mapping run")
    ctx.reindex()
    logPath = tmp_path / "shadow.jsonl"

    runShadow(ctx, counters, _shadowBody("sonar bathymetry", "old A"), logPath)
    runShadow(ctx, counters, _shadowBody("swath mapping", "old B"), logPath)

    lines = logPath.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2                                     # appended, not overwritten
    assert json.loads(lines[0])["query"] == "sonar bathymetry"
    assert json.loads(lines[1])["query"] == "swath mapping"
    assert counters.shadowLogged == 2


def test_shadow_payload_shape_matches_engram_forward(ctx, counters, tmp_path, _rerankerWarm):
    # Pin the exact wire shape the Engram patch sends: {"query", "oldResultText"}.
    _put(ctx.store, "titanium hull biofouling note")
    ctx.reindex()
    logPath = tmp_path / "shadow.jsonl"
    body = json.dumps({"query": "biofouling", "oldResultText": "No memories found for query: biofouling"}).encode()
    status, resp = runShadow(ctx, counters, body, logPath)
    assert status == 200 and resp["ok"] is True
    rec = json.loads(logPath.read_text(encoding="utf-8").splitlines()[0])
    assert rec["old"]["resultText"] == "No memories found for query: biofouling"


def test_shadow_records_and_scopes_project(ctx, counters, tmp_path, _rerankerWarm):
    # project makes the A/B apples-to-apples: the shadow recall is scoped to the
    # same project the old pensive_recall used, and the line records that scope.
    # Two atoms with the SAME query-matching text in different projects; a
    # project="aegis" shadow must record "aegis" AND return only aegis atoms.
    _put(ctx.store, "the shared spreading activation topic appears here", project="aegis")
    _put(ctx.store, "the shared spreading activation topic appears here", project="pensive")
    ctx.reindex()

    logPath = tmp_path / "shadow.jsonl"
    body = json.dumps({
        "query": "shared spreading activation topic",
        "oldResultText": "old", "project": "aegis",
    }).encode("utf-8")
    status, _ = runShadow(ctx, counters, body, logPath)
    assert status == 200

    rec = json.loads(logPath.read_text(encoding="utf-8").splitlines()[0])
    assert rec["project"] == "aegis"                          # scope recorded
    ids = rec["v3"]["ids"]
    assert ids                                                # v3 found the aegis atom
    for aid in ids:                                           # scoped: never the pensive one
        assert getAtom(ctx.store, aid)["project"] == "aegis"


def test_shadow_empty_project_normalizes_to_null(ctx, counters, tmp_path, _rerankerWarm):
    # The legacy default project="" must mean "whole store" (mirroring the old
    # server's `if project:`), and the line records null, not "".
    _put(ctx.store, "an unscoped note about depth")
    ctx.reindex()
    logPath = tmp_path / "shadow.jsonl"
    body = json.dumps({"query": "depth", "oldResultText": "old", "project": ""}).encode()
    status, _ = runShadow(ctx, counters, body, logPath)
    assert status == 200
    rec = json.loads(logPath.read_text(encoding="utf-8").splitlines()[0])
    assert rec["project"] is None


# --------------------------------------------------------------------------- #
# SHADOW: failures are CONTAINED and counted, serving continues                 #
# --------------------------------------------------------------------------- #


def test_shadow_recall_failure_is_contained(ctx, counters, tmp_path, monkeypatch):
    # A recall that raises inside the v3 pipeline must be contained at the shadow
    # boundary: shadowFailed++ and a normal (500, error) return, never a raise.
    import serve.shadow as shadow

    def _boom(*a, **k):
        raise RuntimeError("recall desync")

    monkeypatch.setattr(shadow, "recall", _boom)
    logPath = tmp_path / "shadow.jsonl"

    status, body = runShadow(ctx, counters, _shadowBody("q", "old"), logPath)   # no raise

    assert status == 500 and "error" in body
    assert counters.shadowFailed == 1 and counters.shadowLogged == 0
    assert not logPath.exists()                                # nothing written


def test_shadow_unwritable_log_is_contained_and_keeps_serving(ctx, counters, tmp_path, _rerankerWarm):
    _put(ctx.store, "a recallable aegis note about depth rating")
    ctx.reindex()

    # Point the log at a path whose parent is a regular FILE -> the append cannot
    # create/open it. The boundary must contain the write error, count it, and not
    # raise into the serving path.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    badPath = blocker / "shadow.jsonl"

    status, body = runShadow(ctx, counters, _shadowBody("depth rating", "old"), badPath)

    assert status == 500 and "error" in body
    assert counters.shadowFailed == 1 and counters.shadowLogged == 0

    # Serving continues: the same request to a GOOD path now succeeds.
    goodPath = tmp_path / "shadow.jsonl"
    status2, _ = runShadow(ctx, counters, _shadowBody("depth rating", "old"), goodPath)
    assert status2 == 200
    assert counters.shadowLogged == 1
    assert len(goodPath.read_text(encoding="utf-8").splitlines()) == 1


def test_shadow_malformed_payload_400(ctx, counters, tmp_path):
    logPath = tmp_path / "shadow.jsonl"
    status, body = runShadow(ctx, counters, b"not json", logPath)
    assert status == 400 and "error" in body
    assert counters.shadowFailed == 1 and counters.shadowLogged == 0
    assert not logPath.exists()


# --------------------------------------------------------------------------- #
# WIRING: routes mounted, counters exposed                                     #
# --------------------------------------------------------------------------- #


def test_daemon_wires_tee_shadow_status_routes(ctx):
    from serve.daemon import buildApp

    app = buildApp(ctx)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/tee/emit" in paths
    assert "/shadow/recall" in paths
    assert "/status" in paths
    assert "/mcp" in paths                                     # Task 12 mount intact


def test_counters_snapshot_reflects_activity(ctx, counters, tmp_path, _rerankerWarm):
    _put(ctx.store, "a note the shadow recall can find about sonar")
    ctx.reindex()
    logPath = tmp_path / "shadow.jsonl"

    handleTeeEmit(ctx, counters, _emitBody())                 # one good tee
    handleTeeEmit(ctx, counters, b"garbage")                  # one bad tee
    runShadow(ctx, counters, _shadowBody("sonar", "old"), logPath)   # one good shadow

    snap = counters.snapshot()
    assert snap == {
        "teeReceived": 2, "teeFailed": 1, "shadowLogged": 1, "shadowFailed": 0,
    }


def test_http_surface_end_to_end_moves_counters(embedder, _rerankerWarm, tmp_path, monkeypatch):
    # Drive the REAL Starlette app end to end: routing, request-body parsing, JSON
    # responses, and the closure-shared Counters that GET /status reports. This is
    # the in-process HTTP surface the live daemon serves (the live cross-process
    # end-to-end is the controller's Phase 3 gate).
    #
    # Starlette's TestClient runs the ASGI app in a portal thread, so the store's
    # sqlite connection must tolerate cross-thread use -- open it
    # check_same_thread=False for THIS test only. The live daemon creates and uses
    # its store on the single uvicorn event-loop thread, so it never needs this.
    import sqlite3
    from starlette.testclient import TestClient
    from serve.daemon import buildApp

    _real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **k: _real_connect(*a, **{**k, "check_same_thread": False}))
    monkeypatch.setenv("PENSIVE_V3_SHADOW_LOG", str(tmp_path / "shadow.jsonl"))

    s = openStore(tmp_path / "mem.db")
    try:
        _put(s, "acoustic modems trade range for data rate at the surface buoy")
        c = ServeContext(s, embedder, MODEL_ID, agent="heph")
        app = buildApp(c)
        with TestClient(app) as client:
            assert client.get("/status").json()["counters"] == {
                "teeReceived": 0, "teeFailed": 0, "shadowLogged": 0, "shadowFailed": 0}

            r1 = client.post("/tee/emit", content=_emitBody())
            assert r1.status_code == 200 and r1.json()["ok"] is True

            r2 = client.post("/shadow/recall",
                             content=_shadowBody("acoustic modem range", "old answer"))
            assert r2.status_code == 200 and r2.json()["ok"] is True

            r3 = client.post("/tee/emit", content=b"garbage")     # contained 400
            assert r3.status_code == 400

            # The closure-shared counters moved, visible on the status surface.
            assert client.get("/status").json()["counters"] == {
                "teeReceived": 2, "teeFailed": 1, "shadowLogged": 1, "shadowFailed": 0}
        # The tee'd emit really landed in the v3 store.
        assert s._conn.execute(
            "SELECT COUNT(*) FROM atoms WHERE project = 'pensive'").fetchone()[0] == 1
    finally:
        s.close()


def test_default_shadow_log_path_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PENSIVE_V3_SHADOW_LOG", str(tmp_path / "custom.jsonl"))
    assert defaultShadowLogPath() == tmp_path / "custom.jsonl"
    monkeypatch.delenv("PENSIVE_V3_SHADOW_LOG")
    # Default resolves under daemon/eval/, regardless of cwd.
    assert defaultShadowLogPath().name == "shadow.jsonl"
    assert defaultShadowLogPath().parent.name == "eval"
