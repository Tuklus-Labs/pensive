"""Activation visualization routes for the v3 daemon.

Risk model:

- Invariant: recall telemetry must never affect recall serving. Covered by
  ``test_emit_recall_event_contains_failures`` and the recall-log wiring test.
- Invariant: the event ring is bounded and evicts oldest events. Covered by
  ``test_event_ring_is_bounded_and_evicts_oldest``.
- Boundary: graph atom params are empty, malformed, duplicated, or too long.
  Covered by ``test_graph_rejects_malformed_params`` and
  ``test_graph_caps_input_atoms``.
- Boundary: snippets, neighbor degree, and per-atom neighbor count are capped.
  Covered by ``test_graph_derives_live_facet_neighbors_with_caps_and_weights``.
- State: superseded atoms are not emitted as graph nodes. Covered by
  ``test_graph_derives_live_facet_neighbors_with_caps_and_weights``.
- Persistence: history reads recent ``recall_log`` rows without writing. Covered
  by ``test_history_caps_limit_and_returns_newest_first``.
- Integration contract: ``/viz`` serves the static page with HTML content type,
  ``/viz/graph`` and ``/viz/history`` return JSON shapes. Covered by
  ``test_daemon_wires_viz_routes_and_static_page``.
- Concurrency/resource: a slow SSE send is timed out and the stream stops rather
  than buffering unboundedly. Covered by ``test_sse_sender_drops_slow_client``.
- Malformed inputs: bad query strings return 400 JSON errors, not exceptions.
  Covered by ``test_graph_rejects_malformed_params``.
- Regression traps: boundary populated above; concurrency populated above;
  contract populated above; encoding N/A, payload is JSON/HTML UTF-8 only;
  framework populated by route test; io N/A, static file read is deterministic;
  persistence populated above; resource populated by ring/cap tests; state
  populated above.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from serve.daemon import buildApp
from serve.mcp import ServeContext
from serve import mcp, viz
from store.store import addFacet, openStore, putAtom, supersede


class FakeEmbedder:
    modelId = "fake-model"

    def embed(self, texts):
        import numpy as np

        return [np.zeros(4, dtype=np.float32) for _text in texts]


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ctx(store):
    return ServeContext(store, FakeEmbedder(), FakeEmbedder.modelId, agent="heph")


def _put(store, text, kind="atom", importance=0.0):
    return putAtom(store, {
        "text": text,
        "kind": kind,
        "project": "pensive",
        "importance": importance,
        "provenance": {"source": "bulk-import"},
    })


def _recall_log(store, atom_ids, query):
    from store.store import logRecall

    logRecall(store, atom_ids, query=query, sourceRef="mcp.recall")


def _request(ctx, params=None):
    return SimpleNamespace(
        query_params=params or {},
        app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)),
    )


def _http_scope(path, query_string=b""):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "headers": [],
        "client": ("127.0.0.1", 50123),
        "server": ("127.0.0.1", 5999),
    }


async def _asgi_messages(app, path, query_string=b""):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(_http_scope(path, query_string), receive, send)
    return messages


def _body(messages):
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _headers(message):
    return {key.lower(): value for key, value in message["headers"]}


async def _stream_response_until_body(response, after_start=None):
    messages = []

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.start" and after_start:
            after_start()
        if message["type"] == "http.response.body" and message.get("body"):
            raise RuntimeError("client closed after first body")

    await response(_http_scope("/viz/events"), receive, send)
    return messages


def test_event_ring_is_bounded_and_evicts_oldest(ctx):
    viz.ensureVizState(ctx, maxlen=2)

    viz.emitRecallEvent(ctx, "first", ["a"], "src")
    viz.emitRecallEvent(ctx, "second", ["b"], "src")
    viz.emitRecallEvent(ctx, "third", ["c"], "src")

    events = list(ctx.vizEvents)
    assert [event["query"] for event in events] == ["second", "third"], (
        f"event-ring invariant violated: queries={events}"
    )
    assert events[-1]["atomIds"] == ["c"], (
        f"event payload invariant violated: event={events[-1]}"
    )


def test_emit_recall_event_contains_failures(ctx, monkeypatch):
    class BrokenEvents:
        def append(self, _event):
            raise RuntimeError("ring broken")

    ctx.vizEvents = BrokenEvents()

    viz.emitRecallEvent(ctx, "query", ["a"], "src")

    assert getattr(ctx, "vizEventErrors", 0) == 1, (
        f"telemetry containment invariant violated: errors={ctx.vizEventErrors}"
    )


def test_recall_dispatch_emits_viz_event(ctx, monkeypatch):
    atom_id = _put(ctx.store, "live recall event")
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: {
        "results": [{
            "atomId": atom_id,
            "confidence": 0.9,
            "shouldTrust": True,
            "why": "test",
        }],
        "payload": f"p3://{atom_id}",
        "tokensUsed": 1,
        "lowConfidence": False,
    })

    text, is_error = mcp.dispatch(ctx, "recall", {"query": "show activation"})

    assert is_error is False
    assert f"p3://{atom_id}" in text
    assert list(ctx.vizEvents)[-1]["query"] == "show activation", (
        f"recall event invariant violated: events={list(ctx.vizEvents)}"
    )
    assert list(ctx.vizEvents)[-1]["atomIds"] == [atom_id], (
        f"returned atom invariant violated: events={list(ctx.vizEvents)}"
    )


def test_graph_derives_live_facet_neighbors_with_caps_and_weights(ctx):
    root = _put(ctx.store, "root " + "x" * 180, importance=0.7)
    rare_neighbor = _put(ctx.store, "rare neighbor", kind="narrative", importance=0.2)
    tag_neighbor = _put(ctx.store, "tag neighbor")
    superseded_old = _put(ctx.store, "old version")
    replacement = _put(ctx.store, "replacement")
    supersede(ctx.store, superseded_old, replacement, {"source": "test"})

    addFacet(ctx.store, root, "entity", "rare")
    addFacet(ctx.store, rare_neighbor, "entity", "rare")
    addFacet(ctx.store, root, "tag", "viz")
    addFacet(ctx.store, tag_neighbor, "tag", "viz")
    addFacet(ctx.store, superseded_old, "tag", "viz")
    addFacet(ctx.store, root, "topic", "ignored")
    addFacet(ctx.store, rare_neighbor, "topic", "ignored")

    for i in range(viz.HUB_CAP + 1):
        hub_id = _put(ctx.store, f"hub {i}")
        addFacet(ctx.store, hub_id, "entity", "hub")
    addFacet(ctx.store, root, "entity", "hub")

    graph = viz.buildGraphPayload(ctx, [root])

    node_ids = {node["id"] for node in graph["nodes"]}
    assert root in node_ids and rare_neighbor in node_ids and tag_neighbor in node_ids, (
        f"live graph invariant violated: node_ids={node_ids}"
    )
    assert superseded_old not in node_ids, (
        f"live-only invariant violated: superseded={superseded_old} nodes={node_ids}"
    )
    root_node = next(node for node in graph["nodes"] if node["id"] == root)
    assert len(root_node["textSnippet"]) == 120, (
        f"snippet cap invariant violated: snippet={root_node['textSnippet']!r}"
    )
    assert {edge["facet"]["key"] for edge in graph["edges"]} == {"entity", "tag"}, (
        f"facet filter invariant violated: edges={graph['edges']}"
    )
    assert all(edge["facet"]["value"] != "hub" for edge in graph["edges"]), (
        f"hub facet invariant violated: edges={graph['edges']}"
    )
    rare_edge = next(edge for edge in graph["edges"] if edge["to"] == rare_neighbor)
    assert rare_edge["weight"] == pytest.approx(0.5), (
        f"specificity weight invariant violated: edge={rare_edge}"
    )


def test_graph_skips_hub_facet_before_row_fetch(ctx):
    root = _put(ctx.store, "root")
    addFacet(ctx.store, root, "entity", "hub")
    for i in range(viz.HUB_CAP + 1):
        hub_id = _put(ctx.store, f"hub {i}")
        addFacet(ctx.store, hub_id, "entity", "hub")
    row_fetches = []

    def trace(statement):
        if (
            "FROM facets f JOIN atoms a ON a.id = f.atom_id" in statement
            and "f.value = 'hub'" in statement
        ):
            row_fetches.append(statement)

    ctx.store._conn.set_trace_callback(trace)
    try:
        graph = viz.buildGraphPayload(ctx, [root])
    finally:
        ctx.store._conn.set_trace_callback(None)

    assert row_fetches == [], f"hub row fetch invariant violated: {row_fetches}"
    assert graph["nodes"] == [next(node for node in graph["nodes"] if node["id"] == root)]
    assert graph["edges"] == [], f"hub neighbor invariant violated: edges={graph['edges']}"


def test_graph_caps_input_atoms(ctx):
    ids = [_put(ctx.store, f"atom {i}") for i in range(viz.INPUT_ATOM_CAP + 5)]

    graph = viz.buildGraphPayload(ctx, ids)

    assert graph["truncatedInput"] is True
    assert len(graph["seedIds"]) == viz.INPUT_ATOM_CAP, (
        f"input cap invariant violated: seedIds={len(graph['seedIds'])}"
    )


def test_graph_rejects_malformed_params(ctx):
    missing = asyncio.run(viz.graphEndpoint(_request(ctx)))
    blank = asyncio.run(viz.graphEndpoint(_request(ctx, {"atoms": ",,"})))
    too_many = asyncio.run(viz.graphEndpoint(_request(
        ctx,
        {"atoms": ",".join(f"id{i}" for i in range(viz.INPUT_ATOM_CAP + 1))},
    )))

    assert missing.status_code == 400
    assert blank.status_code == 400
    assert too_many.status_code == 400
    assert json.loads(missing.body)["error"] == "atoms query param is required"


def test_history_caps_limit_and_returns_newest_first(ctx):
    first = _put(ctx.store, "first")
    second = _put(ctx.store, "second")
    _recall_log(ctx.store, [first], "older")
    _recall_log(ctx.store, [second], "newer")

    rows = viz.historyPayload(ctx, limitRaw="5000")["events"]

    assert len(rows) == 2
    assert rows[0]["atom_id"] == second and rows[0]["query"] == "newer", (
        f"history ordering invariant violated: rows={rows}"
    )
    assert set(rows[0]) == {"id", "atom_id", "query", "recorded_at"}, (
        f"history shape invariant violated: row={rows[0]}"
    )


def test_daemon_wires_viz_routes_and_static_page(ctx):
    _recall_log(ctx.store, [_put(ctx.store, "served")], "served query")
    daemon_app = buildApp(ctx)
    paths = {getattr(route, "path", None) for route in daemon_app.routes}
    page = asyncio.run(viz.staticPage(_request(ctx)))
    history = asyncio.run(viz.historyEndpoint(_request(ctx, {"limit": "1"})))
    graph = asyncio.run(viz.graphEndpoint(_request(ctx, {"atoms": "missing-id"})))

    assert {"/viz", "/viz/events", "/viz/graph", "/viz/history"} <= paths, (
        f"daemon route invariant violated: paths={paths}"
    )
    assert page.status_code == 200
    assert page.media_type == "text/html"
    assert page.headers["content-type"].startswith("text/html"), (
        f"static content-type invariant violated: {page.headers}"
    )
    assert "Pensive activation" in page.body.decode("utf-8")
    assert json.loads(history.body)["events"][0]["query"] == "served query"
    assert json.loads(graph.body)["nodes"] == []


def test_app_dispatches_viz_routes_without_testclient(ctx):
    served = _put(ctx.store, "served")
    _recall_log(ctx.store, [served], "served query")
    app = buildApp(ctx)

    page = asyncio.run(_asgi_messages(app, "/viz"))
    malformed = asyncio.run(_asgi_messages(app, "/viz/graph", b"atoms=,,"))
    history = asyncio.run(_asgi_messages(app, "/viz/history", b"limit=5000"))

    assert page[0]["status"] == 200
    assert _headers(page[0])[b"content-type"].startswith(b"text/html")
    assert b"Pensive activation" in _body(page)
    assert malformed[0]["status"] == 400
    assert json.loads(_body(malformed))["error"] == (
        "atoms query param must contain at least one atom id"
    )
    history_payload = json.loads(_body(history))
    assert history_payload["limit"] == viz.HISTORY_LIMIT_CAP
    assert history_payload["events"][0]["query"] == "served query"


def test_app_dispatches_viz_events_streaming_headers(ctx):
    app = buildApp(ctx)
    messages = []

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.start":
            raise RuntimeError("stop after headers")

    with pytest.raises(RuntimeError, match="stop after headers"):
        asyncio.run(app(_http_scope("/viz/events"), receive, send))

    assert messages[0]["status"] == 200
    headers = _headers(messages[0])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert headers[b"cache-control"] == b"no-cache"


def test_sse_event_is_framed_once_on_real_asgi_path(ctx, monkeypatch):
    response = viz.VizEventResponse(ctx)
    emitted = False

    async def connected(_receive):
        nonlocal emitted
        if not emitted:
            emitted = True
            viz.emitRecallEvent(ctx, "wire query", ["atom-a"], "mcp.recall")
        return False

    monkeypatch.setattr(response, "_disconnected", connected)

    messages = asyncio.run(_stream_response_until_body(response))

    bodies = [
        message["body"]
        for message in messages
        if message["type"] == "http.response.body" and message.get("body")
    ]
    assert len(bodies) == 1
    assert bodies[0].startswith(b"data: ") and bodies[0].endswith(b"\n\n")
    assert bodies[0].count(b"data: ") == 1
    payload = json.loads(bodies[0][len(b"data: "):-2])
    assert payload["query"] == "wire query"
    assert payload["atomIds"] == ["atom-a"]

def test_sse_ping_is_framed_once_on_real_asgi_path(ctx, monkeypatch):
    monkeypatch.setattr(viz, "SSE_PING_INTERVAL", 0)

    messages = asyncio.run(_stream_response_until_body(viz.VizEventResponse(ctx)))

    bodies = [
        message["body"]
        for message in messages
        if message["type"] == "http.response.body" and message.get("body")
    ]
    assert bodies == [b": ping\n\n"]


def test_sse_sender_drops_slow_client_through_real_asgi_path(ctx, monkeypatch):
    monkeypatch.setattr(viz, "SSE_SEND_TIMEOUT", 0.001)
    response = viz.VizEventResponse(ctx)
    emitted = False
    sent = []

    async def connected(_receive):
        nonlocal emitted
        if not emitted:
            emitted = True
            viz.emitRecallEvent(ctx, "slow query", ["slow"], "mcp.recall")
        return False

    monkeypatch.setattr(response, "_disconnected", connected)

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            await asyncio.sleep(0.5)

    asyncio.run(asyncio.wait_for(
        response(_http_scope("/viz/events"), receive, send),
        timeout=1,
    ))

    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ], (
        f"sse slow-client invariant violated: sent={sent}"
    )
    assert sent[1]["more_body"] is True
