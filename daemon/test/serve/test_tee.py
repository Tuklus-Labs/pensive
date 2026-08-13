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


# The local-write guard's wire contract, pinned as LITERALS on purpose: these are
# what the Engram-side forward has to send. Importing the constants from serve.tee
# would let a rename there drag every test along and still pass while every real
# caller broke. See the CSRF section below for the guard's risk model.
_LOCAL_WRITE_HEADER = "x-pensive-tee-secret"
_SECRET_FILE_ENV = "PENSIVE_V3_TEE_SECRET_FILE"
_TEE_SECRET = "scratch-loopback-secret-not-the-real-one"


def _localHeaders(secret=_TEE_SECRET, contentType="application/json", **extra):
    """The header set a legitimate local (non-browser) tee caller sends."""
    h = {"content-type": contentType}
    if secret is not None:
        h[_LOCAL_WRITE_HEADER] = secret
    h.update(extra)
    return h


# --------------------------------------------------------------------------- #
# TEE: a tee'd emit lands in the v3 store                                      #
# --------------------------------------------------------------------------- #


def test_tee_emit_atom_lands_in_v3_store(ctx, counters):
    status, body = handleTeeEmit(
        ctx, counters, _emitBody(), _localHeaders(), _TEE_SECRET)

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
        status, resp = handleTeeEmit(ctx, counters, body, _localHeaders(), _TEE_SECRET)
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
    status, resp = handleTeeEmit(ctx, counters, body, _localHeaders(), _TEE_SECRET)
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

    status, body = handleTeeEmit(ctx, counters, _emitBody(), _localHeaders(), _TEE_SECRET)      # must not raise

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

    status, body = handleTeeEmit(ctx, counters, _emitBody(), _localHeaders(), _TEE_SECRET)   # must not raise

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
        status, body = handleTeeEmit(ctx, counters, raw, _localHeaders(), _TEE_SECRET)
        assert status == 400, f"expected 400 for {raw!r}, got {status}"
        assert "error" in body
    assert counters.teeReceived == len(bad_bodies)
    assert counters.teeFailed == len(bad_bodies)
    # Nothing was written by any malformed request.
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0

    # The daemon keeps serving: a valid emit right after the bad ones still writes.
    status, body = handleTeeEmit(ctx, counters, _emitBody(), _localHeaders(), _TEE_SECRET)
    assert status == 200
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_tee_duplicate_emit_writes_two_rows(ctx, counters):
    # The old path is authoritative; the tee mirrors it faithfully and does NOT
    # dedup (dedup is the distiller's job). Two identical emits -> two rows.
    body = _emitBody()
    handleTeeEmit(ctx, counters, body, _localHeaders(), _TEE_SECRET)
    handleTeeEmit(ctx, counters, body, _localHeaders(), _TEE_SECRET)
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
# CSRF: the loopback write routes must refuse a browser-forgeable request      #
# --------------------------------------------------------------------------- #
#
# Risk model for this section. Binding 127.0.0.1 is NOT a defense against CSRF:
# a page the operator visits runs in a browser that CAN reach loopback, and a
# "simple" cross-origin POST (Content-Type text/plain, form-urlencoded, or
# multipart) is transmitted WITHOUT a preflight. CORS only stops the page from
# READING the response -- the write still lands. /tee/emit and /shadow/recall are
# state-changing routes into the memory every agent on this box reads and trusts
# as its own, so a forged emit becomes another agent's belief.
#
# Three independent guards, and each test below must fail for its OWN reason --
# a 4xx is an outcome, not a mechanism, and two guards can both produce one. Every
# rejection therefore carries a machine-readable "reason", and every test asserts
# BOTH the reason AND that the store is untouched.
#
#   1. a shared secret in a CUSTOM header  -- a custom header is not on the CORS
#      safelist, so setting one forces a preflight this server never approves;
#      the secret itself lives in a 0600 file a browser cannot read, so the guard
#      survives even if a permissive CORS middleware is ever added.
#   2. Content-Type must be application/json -- text/plain and
#      x-www-form-urlencoded are exactly the types a no-preflight request can set.
#   3. Origin/Referer, when present, must be loopback -- catches DNS rebinding,
#      where the attacker's name resolves to 127.0.0.1 and the browser considers
#      itself same-origin.

# The exact shape of the attack: a plain HTML page doing
# fetch("http://127.0.0.1:5999/tee/emit", {method:"POST", body: JSON.stringify(...)})
# sends text/plain and an Origin, and cannot set a custom header without a
# preflight. No secret, because the page has no way to read the file.
_BROWSER_SIMPLE_POST = {
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://evil.example",
    "referer": "https://evil.example/post",
}


@pytest.fixture
def httpApp(embedder, tmp_path, monkeypatch):
    """Real Starlette app + TestClient over a SCRATCH store and a SCRATCH secret.

    CSRF is an HTTP-level defect, so these tests drive the actual request surface
    rather than the handler function -- headers, content-type negotiation and
    routing are the thing under test.

    Two redirections keep this off anything live: the secret file is pointed at
    tmp_path (never the real ~/.local/share/pensive-v3/tee.secret, which the live
    daemon's callers depend on) and the store is a fresh temp db. As in
    test_http_surface_end_to_end_moves_counters, TestClient runs the ASGI app in a
    portal thread, so sqlite must tolerate cross-thread use for the test only; the
    live daemon creates and uses its store on the one uvicorn event-loop thread.
    """
    import sqlite3
    from starlette.testclient import TestClient
    from serve.daemon import buildApp

    _real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **k: _real_connect(*a, **{**k, "check_same_thread": False}))

    secretFile = tmp_path / "tee.secret"
    secretFile.write_text(_TEE_SECRET)
    monkeypatch.setenv(_SECRET_FILE_ENV, str(secretFile))
    monkeypatch.setenv("PENSIVE_V3_SHADOW_LOG", str(tmp_path / "shadow.jsonl"))

    s = openStore(tmp_path / "mem.db")
    try:
        c = ServeContext(s, embedder, MODEL_ID, agent="heph")
        with TestClient(buildApp(c), base_url="http://127.0.0.1") as client:
            yield client, s
    finally:
        s.close()


def test_tee_emit_rejects_browser_simple_cross_origin_post(httpApp):
    # THE headline case: the whole forgeable request, exactly as a hostile page
    # sends it. Rejected, and -- the assertion that actually matters -- no row.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(), headers=_BROWSER_SIMPLE_POST)

    # This request trips all three guards at once, so the status is deliberately
    # NOT pinned to one code: which guard answers first is an implementation
    # detail, and defense in depth means ANY of them catching it is a pass. The
    # security property is the second assertion. (A sabotage run that deleted only
    # the origin check caught this: the request was still safely refused as a 415,
    # and an == 403 here would have reported a breach that had not happened.)
    assert 400 <= r.status_code < 500, \
        f"forged cross-origin POST was accepted: {r.status_code} {r.text}"
    assert _atomRows(store) == [], "a forged cross-origin POST wrote to the store"


def test_tee_emit_rejects_request_without_local_secret(httpApp):
    # A well-formed local-looking POST that simply cannot prove it read the secret
    # file. Rejected for the SECRET reason specifically: content-type and origin
    # are both clean here, so no other guard can be the one firing.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(secret=None))

    assert r.status_code == 403
    assert r.json()["reason"] == "missing-local-secret"
    assert _atomRows(store) == []


def test_tee_emit_rejects_wrong_local_secret(httpApp):
    # A guessed/stale secret must fail closed the same way a missing one does.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(secret="not-the-secret"))

    assert r.status_code == 403
    assert r.json()["reason"] == "bad-local-secret"
    assert _atomRows(store) == []


def test_tee_emit_rejects_cross_origin_even_with_valid_secret(httpApp):
    # Defense in depth: even if the secret ever leaks to a page (a stray fetch of
    # a file:// path, a leaked log), a cross-origin Origin is still refused. The
    # secret is valid here, so ONLY the origin guard can produce this rejection.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(origin="https://evil.example"))

    assert r.status_code == 403
    assert r.json()["reason"] == "cross-origin"
    assert _atomRows(store) == []


def test_tee_emit_rejects_null_origin(httpApp):
    # "Origin: null" is what a sandboxed iframe or a file:// page sends. It is not
    # loopback and must not be mistaken for "no origin at all" -- treating an
    # unparseable origin as absent is the classic bypass.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(origin="null"))

    assert r.status_code == 403
    assert r.json()["reason"] == "cross-origin"
    assert _atomRows(store) == []


@pytest.mark.parametrize("contentType", [
    "text/plain;charset=UTF-8",              # fetch() with a string body
    "application/x-www-form-urlencoded",     # a plain auto-submitting <form>
    "multipart/form-data; boundary=x",       # the third no-preflight type
])
def test_tee_emit_rejects_non_json_write_content_types(httpApp, contentType):
    # The three content types a cross-origin request can set WITHOUT a preflight.
    # A valid secret and no Origin are supplied so the content-type guard is
    # provably the one firing.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(contentType=contentType))

    assert r.status_code == 415
    assert r.json()["reason"] == "bad-content-type"
    assert _atomRows(store) == []


def test_tee_emit_accepts_legitimate_local_caller(httpApp):
    # REGRESSION GUARD: the guards must not break the real caller. A local process
    # that read the secret file, sends JSON, and sets no Origin still writes.
    # Without this, "reject everything" would pass every test above.
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(), headers=_localHeaders())

    assert r.status_code == 200, f"legitimate local tee was rejected: {r.text}"
    assert r.json()["ok"] is True
    rows = _atomRows(store)
    assert len(rows) == 1 and rows[0][2] == "pensive"


def test_tee_emit_accepts_loopback_origin_from_the_daemons_own_page(httpApp):
    # The daemon serves /viz off the same origin. A same-origin loopback Origin is
    # legitimate and must survive the origin guard (which exists to catch
    # CROSS-origin, not to ban browsers outright).
    client, store = httpApp

    r = client.post("/tee/emit", content=_emitBody(),
                    headers=_localHeaders(origin="http://127.0.0.1:5999"))

    assert r.status_code == 200, f"same-origin loopback was rejected: {r.text}"
    assert len(_atomRows(store)) == 1


def test_rejected_tee_does_not_move_the_phase3_gate_counters(httpApp):
    # Instrument hygiene: teeReceived/teeFailed feed the Phase 3 cutover gate
    # ("did every old-path emit land in v3?"). If a forged request bumped them,
    # an attacker could move the gate's number at will and a hostile POST would
    # read as a tee bug. Rejections get their OWN counter and leave the gate alone.
    client, _store = httpApp

    client.post("/tee/emit", content=_emitBody(), headers=_BROWSER_SIMPLE_POST)
    client.post("/shadow/recall", content=_shadowBody("q", "old"),
                headers=_BROWSER_SIMPLE_POST)

    counts = client.get("/status").json()["counters"]
    assert counts["teeReceived"] == 0 and counts["teeFailed"] == 0
    assert counts["shadowLogged"] == 0 and counts["shadowFailed"] == 0
    assert counts["teeRejected"] == 1 and counts["shadowRejected"] == 1


def test_handle_tee_emit_refuses_hostile_headers_directly(ctx, counters):
    # The guard lives in the HANDLER, not only in the route, so it cannot be
    # forgotten by a future route that also replays emits. This is the exact call
    # the audit used as its proof of exploit; it must now refuse and write nothing.
    status, body = handleTeeEmit(
        ctx, counters, _emitBody(), _BROWSER_SIMPLE_POST, _TEE_SECRET)

    assert status == 403
    assert body["reason"] == "cross-origin"
    assert _atomRows(ctx.store) == []
    assert counters.teeReceived == 0 and counters.teeRejected == 1


def test_handle_tee_emit_fails_closed_when_no_secret_is_available(ctx, counters):
    # If the secret could not be loaded (unreadable file, bad permissions) the
    # guard must fail CLOSED. A guard that waves requests through when its own
    # configuration is missing is worse than no guard, because it reads as armed.
    status, body = handleTeeEmit(
        ctx, counters, _emitBody(), _localHeaders(), None)

    assert status == 503
    assert body["reason"] == "local-secret-unavailable"
    assert _atomRows(ctx.store) == []


# --------------------------------------------------------------------------- #
# CSRF: the sibling state-changing routes get the same treatment               #
# --------------------------------------------------------------------------- #


def test_shadow_recall_rejects_browser_simple_cross_origin_post(httpApp, _rerankerWarm):
    # /shadow/recall is the other POST on this app. It writes the A/B corpus the
    # cutover decision is read off, so a forged line is evidence tampering even
    # though it does not touch the atom store.
    client, _store = httpApp

    r = client.post("/shadow/recall", content=_shadowBody("sonar", "old"),
                    headers=_BROWSER_SIMPLE_POST)

    # Status not pinned, for the same reason as the tee's headline case above; the
    # property under test is that no line reached the A/B corpus.
    assert 400 <= r.status_code < 500, \
        f"forged cross-origin POST was accepted: {r.status_code} {r.text}"
    assert client.get("/status").json()["counters"]["shadowLogged"] == 0


def test_shadow_recall_accepts_legitimate_local_caller(httpApp, _rerankerWarm):
    # REGRESSION GUARD for the sibling route, same reason as the tee's.
    client, store = httpApp
    _put(store, "acoustic modems trade range for data rate at the surface buoy")

    r = client.post("/shadow/recall",
                    content=_shadowBody("acoustic modem range", "old answer"),
                    headers=_localHeaders())

    assert r.status_code == 200, f"legitimate local shadow was rejected: {r.text}"
    assert client.get("/status").json()["counters"]["shadowLogged"] == 1


def test_mcp_mount_rejects_cross_origin_browser_post(httpApp):
    # /mcp dispatches the SAME five emit tools. It is much harder to forge (the
    # SDK requires application/json, which a no-preflight request cannot set), but
    # that leaves DNS rebinding: the attacker's name resolves to 127.0.0.1, the
    # browser believes it is same-origin, and the content-type barrier is gone.
    # An Origin check is the documented MCP defense and costs non-browser clients
    # nothing, since they send no Origin at all.
    client, _store = httpApp

    r = client.post("/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"accept": "application/json, text/event-stream",
                             "origin": "https://evil.example"})

    assert r.status_code == 403
    assert r.json()["reason"] == "cross-origin"


def test_mcp_mount_still_serves_a_client_that_sends_no_origin(httpApp):
    # REGRESSION GUARD, and the one that matters most operationally: every live
    # agent on this box talks to /mcp. Non-browser clients send no Origin, so the
    # rebinding guard must be invisible to them.
    client, _store = httpApp

    r = client.post("/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"accept": "application/json, text/event-stream"})

    # Assert the call really WORKED, not merely that it dodged our 403: a mount
    # broken into some other error would sail past a bare `!= 403`.
    assert r.status_code == 200, f"the origin guard broke a normal MCP client: {r.text}"
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert TEE_EMIT_TOOLS <= names


def test_mcp_mount_allows_same_origin_loopback(httpApp):
    # The rebinding guard keys on CROSS-origin. A loopback Origin -- what the
    # daemon's own /viz page would send -- must still be served.
    client, _store = httpApp

    r = client.post("/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"accept": "application/json, text/event-stream",
                             "origin": "http://127.0.0.1:5999"})

    assert r.status_code == 200, f"same-origin loopback MCP was rejected: {r.text}"


def test_build_app_never_provisions_a_secret_on_the_filesystem(ctx, tmp_path, monkeypatch):
    # buildApp() must have NO filesystem side effect. It is called by tests in
    # test_viz.py and test_briefer.py as well as here, so a create-on-read would
    # have every one of them writing a fresh credential into the operator's real
    # ~/.local/share/pensive-v3/. That is not hypothetical: it happened once during
    # this fix, which is why loadTeeSecret() defaults to create=False and only
    # main() provisions.
    from serve.daemon import buildApp

    secretFile = tmp_path / "not-created" / "tee.secret"
    monkeypatch.setenv(_SECRET_FILE_ENV, str(secretFile))

    buildApp(ctx)

    assert not secretFile.exists(), "buildApp() provisioned a secret file"
    assert not secretFile.parent.exists(), "buildApp() created the secret's parent dir"


def test_missing_secret_makes_writes_fail_closed_not_open(ctx, counters, tmp_path, monkeypatch):
    # The end-to-end consequence of the above: with no secret on disk, the write
    # routes refuse rather than serve unguarded. A daemon that cannot find its
    # secret must not silently become the vulnerable version of itself.
    import sqlite3
    from starlette.testclient import TestClient
    from serve.daemon import buildApp

    _real_connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect",
        lambda *a, **k: _real_connect(*a, **{**k, "check_same_thread": False}))
    monkeypatch.setenv(_SECRET_FILE_ENV, str(tmp_path / "absent" / "tee.secret"))

    with TestClient(buildApp(ctx), base_url="http://127.0.0.1") as client:
        r = client.post("/tee/emit", content=_emitBody(), headers=_localHeaders())

    assert r.status_code == 503
    assert r.json()["reason"] == "local-secret-unavailable"
    assert _atomRows(ctx.store) == []


def test_read_only_routes_are_not_gated(httpApp):
    # The guards belong on WRITES. /status and /brief change nothing, and gating
    # them would break the SessionStart hook that curls /brief for no security
    # gain -- a read of this store is not the threat being defended against.
    client, _store = httpApp

    assert client.get("/status").status_code == 200
    assert client.get("/brief").status_code == 200


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

    handleTeeEmit(ctx, counters, _emitBody(), _localHeaders(), _TEE_SECRET)                 # one good tee
    handleTeeEmit(ctx, counters, b"garbage", _localHeaders(), _TEE_SECRET)                  # one bad tee
    runShadow(ctx, counters, _shadowBody("sonar", "old"), logPath)   # one good shadow

    snap = counters.snapshot()
    # The four gate counters are unchanged by the CSRF guard; the two rejection
    # counters are new and stay at zero here because every request above was
    # legitimate (see test_rejected_tee_does_not_move_the_phase3_gate_counters).
    assert snap == {
        "teeReceived": 2, "teeFailed": 1, "shadowLogged": 1, "shadowFailed": 0,
        "teeRejected": 0, "shadowRejected": 0,
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
    # Scratch write secret: never read or create the live daemon's tee.secret.
    secretFile = tmp_path / "tee.secret"
    secretFile.write_text(_TEE_SECRET)
    monkeypatch.setenv(_SECRET_FILE_ENV, str(secretFile))

    s = openStore(tmp_path / "mem.db")
    try:
        _put(s, "acoustic modems trade range for data rate at the surface buoy")
        c = ServeContext(s, embedder, MODEL_ID, agent="heph")
        app = buildApp(c)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            assert client.get("/status").json()["counters"] == {
                "teeReceived": 0, "teeFailed": 0, "shadowLogged": 0, "shadowFailed": 0,
                "teeRejected": 0, "shadowRejected": 0}

            r1 = client.post("/tee/emit", content=_emitBody(),
                             headers=_localHeaders())
            assert r1.status_code == 200 and r1.json()["ok"] is True

            r2 = client.post("/shadow/recall",
                             content=_shadowBody("acoustic modem range", "old answer"),
                             headers=_localHeaders())
            assert r2.status_code == 200 and r2.json()["ok"] is True

            r3 = client.post("/tee/emit", content=b"garbage",     # contained 400
                             headers=_localHeaders())
            assert r3.status_code == 400

            # The closure-shared counters moved, visible on the status surface.
            assert client.get("/status").json()["counters"] == {
                "teeReceived": 2, "teeFailed": 1, "shadowLogged": 1, "shadowFailed": 0,
                "teeRejected": 0, "shadowRejected": 0}
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
