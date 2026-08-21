import json
import math
from types import SimpleNamespace

import anyio
import jsonschema
import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    TextContent,
)

import serve.mcp as mcp_module
from recall.payload import estimateTokens
from serve.mcp import NATIVE_TOOLS, buildServer, dispatch
from store.store import getAtom, openStore, putAtom


def _tool(name):
    for tool in NATIVE_TOOLS:
        if tool.name == name:
            return tool
    raise AssertionError(f"native-tool registration rule violated: missing tool={name!r}")


def _assertStrictObjects(schema, path="$Schema"):
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, (
            f"strict-schema rule violated: object path={path!r} "
            f"additionalProperties={schema.get('additionalProperties')!r}"
        )
    for name, child in schema.get("properties", {}).items():
        _assertStrictObjects(child, f"{path}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        _assertStrictObjects(items, f"{path}[]")


@pytest.fixture
def store(tmp_path):
    value = openStore(tmp_path / "structured.db")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def ctx(store):
    return SimpleNamespace(
        store=store,
        indexes={"memory": object()},
        embedder=object(),
        defaultK=10,
        defaultTokenBudget=1500,
        recallLogErrors=0,
        aux=None,
    )


def _put(store, content, *, kind="atom", project="grok-kernel", importance=0.0,
         occurredAt=None, source="explicit-emit", agent="grok-kernel"):
    atomInput = {
        "text": content,
        "kind": kind,
        "project": project,
        "importance": importance,
        "provenance": {"source": source, "agent": agent},
    }
    if occurredAt is not None:
        atomInput["occurredAt"] = occurredAt
    return putAtom(store, atomInput)


def _engineResult(results=(), *, lowConfidence=False):
    return {
        "results": list(results),
        "payload": "unused by structured recall",
        "tokensUsed": 0,
        "lowConfidence": lowConfidence,
    }


def _trust(atomId, *, score=0.9, confidence=0.8, shouldTrust=True,
           why="both signals agree", supersededBy=None):
    value = {
        "atomId": atomId,
        "score": score,
        "confidence": confidence,
        "shouldTrust": shouldTrust,
        "why": why,
    }
    if supersededBy is not None:
        value["supersededBy"] = supersededBy
    return value


def _wireResult(structured, isError=False):
    return CallToolResult(
        content=[TextContent(type="text", text=structured.text)],
        structuredContent=structured.value,
        isError=isError,
    )


def _wireBytes(structured, isError=False):
    return len(_wireResult(structured, isError).model_dump_json().encode("utf-8"))


def _modelBytes(result):
    return len(result.model_dump_json().encode("utf-8"))


def _serverHandler(ctx):
    return buildServer(ctx).request_handlers[CallToolRequest]


async def _serverCall(handler, name, arguments):
    response = await handler(CallToolRequest(
        params=CallToolRequestParams(name=name, arguments=arguments)))
    return response.root


def _resultSummary(result):
    if isinstance(result, mcp_module.StructuredResult):
        return {
            "recordIds": [record["id"] for record in result.value["records"]],
            "truncated": result.value["truncated"],
            "wireBytes": _wireBytes(result),
        }
    return result


def _structuredValue(value):
    return SimpleNamespace(
        value=value,
        text=json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _recordValue(atom, trust):
    return {
        "id": atom["id"],
        "kind": atom["kind"],
        "project": atom["project"],
        "content": atom["text"],
        "createdAt": atom["createdAt"],
        "occurredAt": atom["occurredAt"],
        "importance": atom["importance"],
        "status": atom["status"],
        "atomSchemaVersion": atom["schemaVersion"],
        "provenance": atom["provenance"],
        "score": trust["score"],
        "confidence": trust["confidence"],
        "shouldTrust": trust["shouldTrust"],
        "why": trust["why"],
        "supersededBy": trust.get("supersededBy"),
        "estimatedTokens": estimateTokens(atom["text"]),
    }


def _recallValue(query, records, *, lowConfidence=False, truncated=False):
    return {
        "schemaVersion": 1,
        "query": query,
        "project": None,
        "records": records,
        "estimatedTokens": sum(record["estimatedTokens"] for record in records),
        "lowConfidence": lowConfidence,
        "truncated": truncated,
    }


def _addProvenanceRows(
        store, atomId, count, *, fill="p", sourceChars=256, textChars=2048):
    rows = []
    for index in range(count):
        unique = f"{index:026d}"
        rows.append((
            unique,
            atomId,
            fill * sourceChars,
            fill * textChars,
            fill * textChars,
            fill * textChars,
            1_780_000_000 + index,
        ))
    store._conn.executemany(
        "INSERT INTO provenance(id, atom_id, source, session_id, agent, source_ref, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    store._conn.commit()


def test_recall_records_schemas_are_strict():  # I1, B8, C2
    tool = _tool("recall_records")
    inputSchema = tool.inputSchema
    outputSchema = tool.outputSchema

    jsonschema.Draft202012Validator.check_schema(inputSchema)
    jsonschema.Draft202012Validator.check_schema(outputSchema)
    _assertStrictObjects(inputSchema, "input")
    _assertStrictObjects(outputSchema, "output")

    assert inputSchema["required"] == ["query"], (
        f"required-input rule violated: required={inputSchema['required']!r}"
    )
    assert set(inputSchema["properties"]) == {
        "query", "project", "timeScope", "kinds", "k", "tokenBudget", "agent",
    }, f"input-field rule violated: properties={sorted(inputSchema['properties'])!r}"
    assert inputSchema["properties"]["query"]["minLength"] == 1, (
        f"query-min rule violated: schema={inputSchema['properties']['query']!r}"
    )
    assert inputSchema["properties"]["query"]["maxLength"] == 8192, (
        f"query-max rule violated: schema={inputSchema['properties']['query']!r}"
    )
    assert inputSchema["properties"]["k"]["minimum"] == 1, (
        f"k-min rule violated: schema={inputSchema['properties']['k']!r}"
    )
    assert inputSchema["properties"]["k"]["maximum"] == 32, (
        f"k-max rule violated: schema={inputSchema['properties']['k']!r}"
    )
    assert inputSchema["properties"]["tokenBudget"]["minimum"] == 1, (
        f"budget-min rule violated: schema={inputSchema['properties']['tokenBudget']!r}"
    )
    assert inputSchema["properties"]["tokenBudget"]["maximum"] == 8000, (
        f"budget-max rule violated: schema={inputSchema['properties']['tokenBudget']!r}"
    )

    expectedTop = {
        "schemaVersion", "query", "project", "records", "estimatedTokens",
        "lowConfidence", "truncated",
    }
    expectedRecord = {
        "id", "kind", "project", "content", "createdAt", "occurredAt",
        "importance", "status", "atomSchemaVersion", "provenance", "score",
        "confidence", "shouldTrust", "why", "supersededBy", "estimatedTokens",
    }
    expectedProvenance = {
        "id", "atomId", "source", "sessionId", "agent", "sourceRef", "recordedAt",
    }
    assert set(outputSchema["required"]) == expectedTop, (
        f"output-required rule violated: required={sorted(outputSchema['required'])!r}"
    )
    recordSchema = outputSchema["properties"]["records"]["items"]
    provenanceSchema = recordSchema["properties"]["provenance"]["items"]
    assert set(recordSchema["required"]) == expectedRecord, (
        f"record-required rule violated: required={sorted(recordSchema['required'])!r}"
    )
    assert set(provenanceSchema["required"]) == expectedProvenance, (
        f"provenance-required rule violated: required={sorted(provenanceSchema['required'])!r}"
    )
    assert outputSchema["properties"]["records"]["maxItems"] == 32, (
        f"record-cap rule violated: schema={outputSchema['properties']['records']!r}"
    )
    assert recordSchema["properties"]["provenance"]["maxItems"] == 64, (
        f"provenance-cap rule violated: schema={recordSchema['properties']['provenance']!r}"
    )
    assert recordSchema["properties"]["provenance"]["minItems"] == 1, (
        f"provenance-required-content rule violated: "
        f"schema={recordSchema['properties']['provenance']!r}"
    )
    assert recordSchema["properties"]["status"]["enum"] == ["live", "superseded"], (
        f"recallable-status rule violated: schema={recordSchema['properties']['status']!r}"
    )


@pytest.mark.parametrize(("args", "field"), [
    ({}, "query"),
    ({"query": None}, "query"),
    ({"query": 7}, "query"),
    ({"query": "   "}, "query"),
    ({"query": "q" * 8193}, "query"),
    ({"query": "q", "project": ""}, "project"),
    ({"query": "q", "project": "p" * 257}, "project"),
    ({"query": "q", "project": False}, "project"),
    ({"query": "q", "k": True}, "k"),
    ({"query": "q", "k": 0}, "k"),
    ({"query": "q", "k": 33}, "k"),
    ({"query": "q", "k": 1.0}, "k"),
    ({"query": "q", "k": "10"}, "k"),
    ({"query": "q", "tokenBudget": False}, "tokenBudget"),
    ({"query": "q", "tokenBudget": 0}, "tokenBudget"),
    ({"query": "q", "tokenBudget": 8001}, "tokenBudget"),
    ({"query": "q", "tokenBudget": 1.0}, "tokenBudget"),
    ({"query": "q", "timeScope": []}, "timeScope"),
    ({"query": "q", "timeScope": [0]}, "timeScope"),
    ({"query": "q", "timeScope": [0, 1, 2]}, "timeScope"),
    ({"query": "q", "timeScope": [False, 1]}, "timeScope"),
    ({"query": "q", "timeScope": [-1, 1]}, "timeScope"),
    ({"query": "q", "timeScope": [0, 9_007_199_254_740_992]}, "timeScope"),
    ({"query": "q", "timeScope": [2, 1]}, "timeScope"),
    ({"query": "q", "kinds": []}, "kinds"),
    ({"query": "q", "kinds": "atom"}, "kinds"),
    ({"query": "q", "kinds": ["atom", "atom"]}, "kinds"),
    ({"query": "q", "kinds": ["unknown"]}, "kinds"),
    ({"query": "q", "kinds": [1]}, "kinds"),
    ({"query": "q", "surprise": 1}, "unknown fields"),
])
def test_recall_records_argument_contract(monkeypatch, ctx, args, field):  # B1-B6, M1, C5, S1
    calls = []
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: calls.append((a, kw)))

    text, isError = dispatch(ctx, "recall_records", args)

    assert isError is True, (
        f"invalid-argument containment rule violated: args={args!r} result={text!r}"
    )
    expectedError = f"recall_records: {field}"
    assert expectedError in text, (
        f"field-naming error rule violated: field={field!r} args={args!r} result={text!r}"
    )
    assert calls == [], (
        f"validate-before-recall rule violated: args={args!r} engineCalls={calls!r}"
    )


def test_recall_records_calls_engine_once_with_normalized_arguments(monkeypatch, ctx):  # C1
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append((args, kwargs))
        return _engineResult()

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    query = "  preserve this query exactly  "
    value, isError = dispatch(ctx, "recall_records", {
        "query": query,
        "project": None,
        "timeScope": [0, 9_007_199_254_740_991],
        "kinds": ["atom", "document_chunk"],
        "k": 32,
        "tokenBudget": 8000,
    })

    assert isError is False, (
        f"valid-argument rule violated: isError={isError} result={value!r}"
    )
    assert len(calls) == 1, (
        f"single-engine-call rule violated: callCount={len(calls)} calls={calls!r}"
    )
    positional, keywords = calls[0]
    assert positional == (ctx.store, ctx.indexes, ctx.embedder, query), (
        f"engine-positional contract violated: positional={positional!r}"
    )
    assert keywords == {
        "project": None,
        "timeScope": (0, 9_007_199_254_740_991),
        "kinds": ("atom", "document_chunk"),
        "k": 32,
        "tokenBudget": 8000,
        "aux": ctx.aux,
        # rerankEnabled=False ADDED 2026-08-14. The compat handlers passed no
        # rerank hint, so recall()'s signature default of True applied and the
        # cross-encoder ran on every call: 620.6ms for pensive_recall and
        # 606.6ms for recall_records against 8.4ms for the native tier-routed
        # recall, and ~3,190ms on the real store. The tools live agents call
        # were the slow ones, and the gate never exercised them.
        "rerankEnabled": False,
    }, f"engine-keyword contract violated: keywords={keywords!r}"


def test_recall_records_accepts_inclusive_argument_endpoints(monkeypatch, ctx):  # B1-B6
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append((args, kwargs))
        return _engineResult()

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    cases = [
        {"query": "q", "project": "p", "k": 1, "tokenBudget": 1},
        {"query": "q" * 8192, "project": "p" * 256, "k": 32,
         "tokenBudget": 8000},
        {"query": "q", "timeScope": [0, 0],
         "kinds": ["atom", "narrative", "snapshot", "document_chunk"]},
        {"query": "q", "project": None, "timeScope": None, "kinds": None},
    ]

    results = [dispatch(ctx, "recall_records", args) for args in cases]

    assert all(isError is False for _, isError in results), (
        f"inclusive-endpoint rule violated: cases={cases!r} results={results!r}"
    )
    assert len(calls) == len(cases), (
        f"endpoint-forwarding rule violated: callCount={len(calls)} cases={len(cases)}"
    )
    assert calls[0][1]["k"] == 1 and calls[0][1]["tokenBudget"] == 1, (
        f"minimum-endpoint forwarding rule violated: keywords={calls[0][1]!r}"
    )
    assert calls[1][1]["k"] == 32 and calls[1][1]["tokenBudget"] == 8000, (
        f"maximum-endpoint forwarding rule violated: keywords={calls[1][1]!r}"
    )
    assert calls[2][1]["timeScope"] == (0, 0), (
        f"zero-timescope forwarding rule violated: keywords={calls[2][1]!r}"
    )
    assert calls[3][1] == {
        "project": None,
        "timeScope": None,
        "kinds": None,
        "k": 10,
        "tokenBudget": 1500,
        "aux": ctx.aux,
        # rerankEnabled=False ADDED 2026-08-14. The compat handlers passed no
        # rerank hint, so recall()'s signature default of True applied and the
        # cross-encoder ran on every call: 620.6ms for pensive_recall and
        # 606.6ms for recall_records against 8.4ms for the native tier-routed
        # recall, and ~3,190ms on the real store. The tools live agents call
        # were the slow ones, and the gate never exercised them.
        "rerankEnabled": False,
    }, f"default-argument rule violated: keywords={calls[3][1]!r}"


def test_pensive_recall_passes_l2_tier(monkeypatch, ctx):
    # I11: the listing tool must not walk document_chunk by default.
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append(kwargs)
        return _engineResult()

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    dispatch(ctx, "pensive_recall", {"query": "q"})
    assert calls and calls[0].get("tier") == "L2", (
        f"pensive_recall L2-default rule violated: keywords={calls[0] if calls else None!r}"
    )
    assert "rerankEnabled" not in calls[0] or calls[0].get("rerankEnabled") is False, (
        f"pensive_recall rerank-off rule violated: keywords={calls[0]!r}"
    )


def test_handle_recall_forwards_kinds_instead_of_default_tier(monkeypatch, ctx):
    # M3: kinds extracted then dropped was the clean-pass finding.
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append(kwargs)
        return {"payload": "ok", "results": [], "tokensUsed": 0, "lowConfidence": True}

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    dispatch(ctx, "recall", {"query": "q", "kinds": ["narrative"]})
    assert calls, "handle_recall kinds-honor rule violated: engine not called"
    assert calls[0].get("kinds") == ("narrative",), (
        f"handle_recall kinds-honor rule violated: keywords={calls[0]!r}"
    )
    assert "tier" not in calls[0], (
        f"handle_recall kinds-honor rule violated: default tier overrode kinds "
        f"keywords={calls[0]!r}"
    )


def test_handle_recall_does_not_forward_transport_agent(monkeypatch, ctx):
    # I10: ServeContext.agent is a write stamp. Unscoped recall stays unscoped.
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append(kwargs)
        return {"payload": "ok", "results": [], "tokensUsed": 0, "lowConfidence": True}

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    ctx.agent = "heph"
    dispatch(ctx, "recall", {"query": "q"})
    assert "agent" not in calls[0], (
        f"transport-autoscope rule violated: ctx.agent leaked into retrieve "
        f"keywords={calls[0]!r}"
    )


def test_recall_records_forwards_agent_only_when_set(monkeypatch, ctx):
    # C2: unset must omit the keyword (existing exact-dict tests). Set must
    # reach the engine. Connection identity is not this path.
    calls = []

    def fakeRecall(*args, **kwargs):
        calls.append(kwargs)
        return _engineResult()

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)
    dispatch(ctx, "recall_records", {"query": "q"})
    assert "agent" not in calls[0], (
        f"unset-agent-omit rule violated: keywords={calls[0]!r}"
    )
    dispatch(ctx, "recall_records", {"query": "q", "agent": "grok"})
    assert calls[1].get("agent") == "grok", (
        f"agent-forwarding rule violated: keywords={calls[1]!r}"
    )


def test_recall_records_preserves_rank_metadata_provenance_and_trust(
        monkeypatch, ctx, store):  # I2, I4, M4, P1, P2
    firstId = _put(
        store, "first complete body", kind="narrative", project=None,
        importance=0.75, occurredAt=1_780_000_000, source="explicit-emit",
    )
    secondId = _put(store, "historical body", project="grok-kernel")
    successorId = _put(store, "current body", project="grok-kernel")
    store._conn.execute(
        "INSERT INTO provenance(id, atom_id, source, session_id, agent, source_ref, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("01SECONDPROVENANCE00000000", firstId, "codex", "session-7", None,
         "turn/14", 1_780_000_001),
    )
    store._conn.execute("UPDATE atoms SET status = 'superseded' WHERE id = ?", (secondId,))
    store._conn.commit()
    ranked = [
        _trust(firstId, score=1.25, confidence=0.91, why="lexical+dense agreement"),
        _trust(secondId, score=0.5, confidence=0.4, shouldTrust=False,
               why="superseded by newer atom", supersededBy=successorId),
    ]
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))

    result, isError = dispatch(ctx, "recall_records", {
        "query": "durable context", "project": "grok-kernel", "tokenBudget": 8000,
    })

    assert isError is False, (
        f"structured-record success rule violated: isError={isError} result={result!r}"
    )
    value = result.value
    assert [record["id"] for record in value["records"]] == [firstId, secondId], (
        f"rank-order rule violated: records={value['records']!r}"
    )
    first, second = value["records"]
    atom = getAtom(store, firstId)
    assert first == {
        "id": firstId,
        "kind": "narrative",
        "project": None,
        "content": "first complete body",
        "createdAt": atom["createdAt"],
        "occurredAt": 1_780_000_000,
        "importance": 0.75,
        "status": "live",
        "atomSchemaVersion": atom["schemaVersion"],
        "provenance": atom["provenance"],
        "score": 1.25,
        "confidence": 0.91,
        "shouldTrust": True,
        "why": "lexical+dense agreement",
        "supersededBy": None,
        "estimatedTokens": estimateTokens("first complete body"),
    }, f"metadata-provenance rule violated: record={first!r} atom={atom!r}"
    assert second["status"] == "superseded" and second["supersededBy"] == successorId, (
        f"supersession-field rule violated: record={second!r}"
    )
    expectedTokens = sum(record["estimatedTokens"] for record in value["records"])
    assert value["estimatedTokens"] == expectedTokens <= 8000, (
        f"token-sum rule violated: total={value['estimatedTokens']} "
        f"recordTokens={[r['estimatedTokens'] for r in value['records']]!r}"
    )
    assert json.loads(result.text) == value, (
        f"deterministic-fallback equivalence rule violated: text={result.text!r} value={value!r}"
    )
    assert result.text == json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")), (
        f"deterministic-fallback formatting rule violated: text={result.text!r}"
    )
    jsonschema.validate(value, _tool("recall_records").outputSchema)


def test_recall_records_applies_atomic_rank_order_budget(monkeypatch, ctx, store):  # I3, B7, C6
    firstId = _put(store, "abcdef")                 # 2 tokens
    secondId = _put(store, "x" * 30)                # 10 tokens
    thirdId = _put(store, "end")                    # 1 token
    ranked = [_trust(firstId), _trust(secondId), _trust(thirdId)]
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))

    exact, exactError = dispatch(ctx, "recall_records", {
        "query": "context precedence", "tokenBudget": 12,
    })
    short, shortError = dispatch(ctx, "recall_records", {
        "query": "context precedence", "tokenBudget": 11,
    })

    assert exactError is False and shortError is False, (
        f"budget-call success rule violated: exactError={exactError} shortError={shortError}"
    )
    assert [r["id"] for r in exact.value["records"]] == [firstId, secondId], (
        f"exact-fit admission rule violated: records={exact.value['records']!r}"
    )
    assert exact.value["estimatedTokens"] == 12 and exact.value["truncated"] is True, (
        f"tail-truncation rule violated: value={exact.value!r}"
    )
    assert [r["id"] for r in short.value["records"]] == [firstId], (
        f"first-nonfit tail-drop rule violated: records={short.value['records']!r}"
    )
    assert short.value["estimatedTokens"] == 2 and short.value["truncated"] is True, (
        f"short-budget accounting rule violated: value={short.value!r}"
    )
    originalBodies = {"abcdef", "x" * 30, "end"}
    assert all(r["content"] in originalBodies for r in exact.value["records"]), (
        f"atomic-body rule violated: records={exact.value['records']!r}"
    )
    logged = store._conn.execute(
        "SELECT atom_id, source_ref FROM recall_log ORDER BY rowid"
    ).fetchall()
    assert logged == [
        (firstId, "mcp.recall_records"),
        (secondId, "mcp.recall_records"),
        (firstId, "mcp.recall_records"),
    ], f"served-record telemetry rule violated: rows={logged!r}"


def test_recall_records_wire_cap_admits_exact_size_and_drops_one_byte_over(
        monkeypatch, ctx, store):  # I7, I8, B9, C6
    atomId = _put(store, "whole record")
    ranked = [_trust(atomId)]
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))
    atom = getAtom(store, atomId)
    record = _recordValue(atom, ranked[0])
    candidate = _structuredValue(_recallValue("wire boundary", [record]))
    exactBytes = _wireBytes(candidate)

    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", exactBytes, raising=False)
    exact, exactError = dispatch(ctx, "recall_records", {"query": "wire boundary"})
    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", exactBytes - 1, raising=False)
    over, overError = dispatch(ctx, "recall_records", {"query": "wire boundary"})

    assert exactError is False and [r["id"] for r in exact.value["records"]] == [atomId], (
        f"exact-wire-cap admission rule violated: cap={exactBytes} "
        f"result={_resultSummary(exact)!r} "
        f"isError={exactError}"
    )
    assert _wireBytes(exact) == exactBytes, (
        f"exact-wire-size rule violated: expected={exactBytes} actual={_wireBytes(exact)}"
    )
    assert overError is False and over.value["records"] == [], (
        f"one-byte-over atomic omission rule violated: cap={exactBytes - 1} "
        f"result={_resultSummary(over)!r} isError={overError}"
    )
    assert over.value["truncated"] is True and _wireBytes(over) <= exactBytes - 1, (
        f"one-byte-over bounded-envelope rule violated: cap={exactBytes - 1} "
        f"result={_resultSummary(over)!r}"
    )
    logged = store._conn.execute(
        "SELECT atom_id, source_ref FROM recall_log ORDER BY rowid"
    ).fetchall()
    assert logged == [(atomId, "mcp.recall_records")], (
        f"wire-admission telemetry rule violated: admitted={[atomId]!r} rows={logged!r}"
    )


def test_recall_records_wire_cap_counts_multibyte_utf8_and_logs_only_admitted(
        monkeypatch, ctx, store):  # I7, I8, B10, C6
    firstId = _put(store, "first")
    secondId = _put(store, "second")
    _addProvenanceRows(store, secondId, 63, fill="\U0001f9e0")
    ranked = [_trust(firstId), _trust(secondId)]
    firstRecord = _recordValue(getAtom(store, firstId), ranked[0])
    secondRecord = _recordValue(getAtom(store, secondId), ranked[1])
    firstEnvelope = _structuredValue(
        _recallValue("unicode wire", [firstRecord], truncated=True))
    bothEnvelope = _structuredValue(
        _recallValue("unicode wire", [firstRecord, secondRecord]))
    characterCount = len(_wireResult(bothEnvelope).model_dump_json())
    byteCount = _wireBytes(bothEnvelope)
    cap = mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES
    assert cap == 1 << 20, (
        f"aggregate-wire-cap constant rule violated: cap={cap} expected={1 << 20}"
    )
    assert _wireBytes(firstEnvelope) <= cap, (
        f"multibyte-test first-record precondition violated: "
        f"wireBytes={_wireBytes(firstEnvelope)} cap={cap}"
    )
    assert characterCount < cap < byteCount, (
        f"multibyte-test precondition violated: chars={characterCount} cap={cap} bytes={byteCount}"
    )
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))

    result, isError = dispatch(ctx, "recall_records", {"query": "unicode wire"})

    assert isError is False and [r["id"] for r in result.value["records"]] == [firstId], (
        f"UTF-8 byte admission rule violated: chars={characterCount} cap={cap} "
        f"bytes={byteCount} result={_resultSummary(result)!r} isError={isError}"
    )
    assert result.value["truncated"] is True and _wireBytes(result) <= cap, (
        f"multibyte bounded-envelope rule violated: cap={cap} "
        f"result={_resultSummary(result)!r}"
    )
    logged = store._conn.execute(
        "SELECT atom_id, source_ref FROM recall_log ORDER BY rowid"
    ).fetchall()
    assert logged == [(firstId, "mcp.recall_records")], (
        f"multibyte telemetry rule violated: admitted={[firstId]!r} rows={logged!r}"
    )


def test_recall_records_base_envelope_over_cap_fails_loudly(monkeypatch, ctx):  # S4
    monkeypatch.setattr(
        mcp_module, "recall", lambda *a, **kw: _engineResult(lowConfidence=True))
    base = _structuredValue(
        _recallValue("base boundary", [], lowConfidence=True, truncated=False))
    baseBytes = _wireBytes(base)
    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", baseBytes - 1, raising=False)

    bad, badError = dispatch(ctx, "recall_records", {"query": "base boundary"})
    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", baseBytes, raising=False)
    good, goodError = dispatch(ctx, "recall_records", {"query": "base boundary"})

    assert badError is True and "base envelope" in bad and "cap" in bad, (
        f"base-envelope loud-failure rule violated: cap={baseBytes - 1} "
        f"result={bad!r} isError={badError}"
    )
    assert goodError is False and _wireBytes(good) == baseBytes, (
        f"base-envelope exact-cap rule violated: cap={baseBytes} "
        f"result={good!r} isError={goodError}"
    )


def test_recall_records_bounds_schema_maximum_shape_without_building_it_all(
        monkeypatch, ctx, store):  # I7, I8, B8
    atomId = _put(store, "bounded shape")
    _addProvenanceRows(
        store, atomId, 63, fill="\U0001f9e0", sourceChars=16, textChars=16)
    ranked = [_trust(atomId) for _ in range(32)]
    calls = []
    originalGetAtom = mcp_module.getAtom

    def trackedGetAtom(*args):
        calls.append(args[1])
        return originalGetAtom(*args)

    monkeypatch.setattr(mcp_module, "getAtom", trackedGetAtom)
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))
    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", 4096, raising=False)

    result, isError = dispatch(ctx, "recall_records", {
        "query": "bounded maximum shape", "k": 32, "tokenBudget": 8000,
    })

    assert isError is False and result.value["records"] == [], (
        f"maximum-shape aggregate bound rule violated: "
        f"result={_resultSummary(result)!r} isError={isError}"
    )
    assert result.value["truncated"] is True and _wireBytes(result) <= 4096, (
        f"maximum-shape bounded-envelope rule violated: "
        f"result={_resultSummary(result)!r}"
    )
    assert calls == [atomId], (
        f"first-oversize tail-stop rule violated: getAtomCalls={calls!r} expected={[atomId]!r}"
    )


def test_recall_records_empty_result_is_low_confidence(monkeypatch, ctx):  # contract: empty
    monkeypatch.setattr(
        mcp_module, "recall", lambda *a, **kw: _engineResult(lowConfidence=True))

    result, isError = dispatch(ctx, "recall_records", {"query": "no match"})

    assert isError is False, (
        f"empty-result success rule violated: isError={isError} result={result!r}"
    )
    assert result.value["records"] == [], (
        f"empty-collection rule violated: value={result.value!r}"
    )
    assert result.value["lowConfidence"] is True, (
        f"low-confidence propagation rule violated: value={result.value!r}"
    )
    assert result.value["estimatedTokens"] == 0 and result.value["truncated"] is False, (
        f"empty-accounting rule violated: value={result.value!r}"
    )


def test_recall_records_missing_atom_errors_then_next_call_succeeds(
        monkeypatch, ctx, store):  # S2, M2, P3
    goodId = _put(store, "present atom")

    def fakeRecall(*args, **kwargs):
        query = args[3]
        atomId = "01MISSINGXXXXXXXXXXXXXXXXXX" if query == "bad" else goodId
        return _engineResult([_trust(atomId)])

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)

    bad, badError = dispatch(ctx, "recall_records", {"query": "bad"})
    good, goodError = dispatch(ctx, "recall_records", {"query": "good"})

    assert badError is True and "absent from the store" in bad, (
        f"missing-record loudness rule violated: isError={badError} result={bad!r}"
    )
    assert goodError is False and good.value["records"][0]["id"] == goodId, (
        f"post-error serving rule violated: isError={goodError} result={good!r}"
    )


def test_recall_records_serialization_failure_is_contained(
        monkeypatch, ctx, store):  # I5, M3, S2
    atomId = _put(store, "finite atom")

    def fakeRecall(*args, **kwargs):
        score = math.nan if args[3] == "bad json" else 0.9
        return _engineResult([_trust(atomId, score=score)])

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)

    bad, badError = dispatch(ctx, "recall_records", {"query": "bad json"})
    good, goodError = dispatch(ctx, "recall_records", {"query": "good json"})

    assert badError is True and "JSON" in bad, (
        f"non-finite JSON containment rule violated: isError={badError} result={bad!r}"
    )
    assert goodError is False and json.loads(good.text) == good.value, (
        f"post-serialization-error rule violated: isError={goodError} result={good!r}"
    )


def test_recall_records_output_schema_failure_is_contained(
        monkeypatch, ctx):  # I1, S2, M4
    calls = 0

    def fakeRecall(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _engineResult(lowConfidence="yes" if calls == 1 else True)

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)

    bad, badError = dispatch(ctx, "recall_records", {"query": "bad shape"})
    good, goodError = dispatch(ctx, "recall_records", {"query": "good shape"})

    assert badError is True and "structured output" in bad, (
        f"output-schema containment rule violated: isError={badError} result={bad!r}"
    )
    assert goodError is False and good.value["lowConfidence"] is True, (
        f"post-output-schema-error rule violated: isError={goodError} result={good!r}"
    )


def test_recall_records_preserves_store_state(monkeypatch, ctx, store):  # S3
    atomId = _put(store, "read-only atom")
    monkeypatch.setattr(
        mcp_module, "recall", lambda *a, **kw: _engineResult([_trust(atomId)]))
    before = store._conn.execute(
        "SELECT id, text, kind, project, status, schema_version FROM atoms ORDER BY id"
    ).fetchall()

    result, isError = dispatch(ctx, "recall_records", {"query": "read only"})
    after = store._conn.execute(
        "SELECT id, text, kind, project, status, schema_version FROM atoms ORDER BY id"
    ).fetchall()

    assert isError is False and result.value["records"][0]["id"] == atomId, (
        f"read-only call success rule violated: isError={isError} result={result!r}"
    )
    assert after == before, (
        f"atom-store immutability rule violated: before={before!r} after={after!r}"
    )


def test_recall_records_contains_corrupt_provenance_at_served_boundary(
        monkeypatch, ctx, store):  # I9, I10, M5, P3, P4, S5
    corruptId = _put(store, "corrupt provenance")
    validId = _put(store, "valid provenance")
    marker = "RAW_CORRUPT_PROVENANCE_"
    corruptSource = marker + ("x" * (2 * 1024 * 1024))
    store._conn.execute(
        "UPDATE provenance SET source = ? WHERE atom_id = ?",
        (corruptSource, corruptId),
    )
    store._conn.commit()

    def fakeRecall(*args, **kwargs):
        atomId = corruptId if args[3] == "corrupt" else validId
        return _engineResult([_trust(atomId)])

    monkeypatch.setattr(mcp_module, "recall", fakeRecall)

    direct, directError = dispatch(
        ctx, "recall_records", {"query": "corrupt"})
    handler = _serverHandler(ctx)
    served = anyio.run(
        _serverCall, handler, "recall_records", {"query": "corrupt"})
    recovered = anyio.run(
        _serverCall, handler, "recall_records", {"query": "valid"})

    assert directError is True and len(direct) < 512, (
        f"bounded-schema-diagnostic rule violated: isError={directError} "
        f"errorChars={len(direct)} limit=511 markerPresent={marker in direct}"
    )
    assert marker not in direct and "validator=maxLength" in direct, (
        f"instance-free schema-diagnostic rule violated: errorChars={len(direct)} "
        f"markerPresent={marker in direct} diagnosticPrefix={direct[:160]!r}"
    )
    assert served.isError is True and _modelBytes(served) <= (
        mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES
    ), (
        f"corrupt-row served-bound rule violated: isError={served.isError} "
        f"wireBytes={_modelBytes(served)} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
    )
    assert marker not in served.content[0].text, (
        f"corrupt-instance non-echo rule violated: "
        f"textChars={len(served.content[0].text)} "
        f"markerPresent={marker in served.content[0].text}"
    )
    assert (
        recovered.isError is False
        and recovered.structuredContent["records"][0]["id"] == validId
    ), (
        f"post-corrupt-row recovery rule violated: isError={recovered.isError} "
        f"structuredContent={recovered.structuredContent!r}"
    )


def test_recall_records_served_boundary_replaces_oversized_dispatch_error(
        monkeypatch, ctx):  # I9, I11, B11, C9, S5
    marker = "OVERSIZED_RECALL_ERROR_"
    oversized = "error: " + marker + ("x" * (2 * 1024 * 1024))
    recoveredValue = _recallValue("recovered", [])
    recoveredResult = mcp_module.StructuredResult(
        value=recoveredValue,
        text=json.dumps(recoveredValue, sort_keys=True, separators=(",", ":")),
    )
    dispatchCalls = 0

    def fakeDispatch(*_args):
        nonlocal dispatchCalls
        dispatchCalls += 1
        if dispatchCalls == 1:
            return oversized, True
        return recoveredResult, False

    wrapped = []
    originalWrapper = mcp_module._callToolResult

    def trackedWrapper(result, isError):
        wrapped.append((result, isError))
        return originalWrapper(result, isError)

    monkeypatch.setattr(mcp_module, "dispatch", fakeDispatch)
    monkeypatch.setattr(mcp_module, "_callToolResult", trackedWrapper)
    handler = _serverHandler(ctx)

    response = anyio.run(
        _serverCall, handler, "recall_records", {"query": "oversized"})
    recovered = anyio.run(
        _serverCall, handler, "recall_records", {"query": "recovered"})
    text = response.content[0].text

    assert response.isError is True and _modelBytes(response) <= (
        mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES
    ), (
        f"final-error-envelope bound rule violated: isError={response.isError} "
        f"wireBytes={_modelBytes(response)} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
    )
    assert text == "error: recall_records response exceeds wire-size cap", (
        f"fixed-oversize-error rule violated: textChars={len(text)} "
        f"markerPresent={marker in text} prefix={text[:96]!r}"
    )
    assert all(result is not oversized for result, _ in wrapped), (
        f"raw-oversize preconstruction rule violated: wrapperCalls={len(wrapped)} "
        f"giantArgCalls={sum(result is oversized for result, _ in wrapped)}"
    )
    assert (
        dispatchCalls == 2
        and len(wrapped) == 2
        and recovered.isError is False
        and recovered.structuredContent == recoveredValue
    ), (
        f"post-oversize recovery rule violated: dispatchCalls={dispatchCalls} "
        f"wrapperCalls={len(wrapped)} isError={recovered.isError} "
        f"structuredContent={recovered.structuredContent!r}"
    )


def test_recall_records_served_boundary_counts_escaping_overhead(
        monkeypatch, ctx):  # I9, B11, C9
    escaping = "error: " + ("\x00" * 200_000)
    assert len(escaping.encode("utf-8")) < (
        mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES
    ), (
        f"escaping-overhead precondition violated: rawBytes={len(escaping.encode('utf-8'))} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
    )
    originalEnvelope = mcp_module._callToolResult(escaping, True)
    assert _modelBytes(originalEnvelope) > (
        mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES
    ), (
        f"escaping-envelope precondition violated: wireBytes={_modelBytes(originalEnvelope)} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
    )
    wrapped = []
    originalWrapper = mcp_module._callToolResult

    def trackedWrapper(result, isError):
        wrapped.append(result)
        return originalWrapper(result, isError)

    monkeypatch.setattr(
        mcp_module, "dispatch", lambda *_args: (escaping, True))
    monkeypatch.setattr(mcp_module, "_callToolResult", trackedWrapper)

    response = anyio.run(
        _serverCall, _serverHandler(ctx), "recall_records", {"query": "q"})

    assert wrapped and wrapped[0] is escaping, (
        f"within-raw-cap construction rule violated: wrapperCalls={len(wrapped)} "
        f"originalSeen={any(result is escaping for result in wrapped)}"
    )
    assert response.content[0].text == (
        "error: recall_records response exceeds wire-size cap"
    ) and _modelBytes(response) <= mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES, (
        f"escaping-overhead final-bound rule violated: wireBytes={_modelBytes(response)} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES} "
        f"text={response.content[0].text[:96]!r}"
    )


def test_recall_records_served_boundary_fails_when_cap_cannot_hold_error(
        monkeypatch, ctx):  # B12
    oversized = "error: " + ("x" * 1024)
    monkeypatch.setattr(
        mcp_module, "dispatch", lambda *_args: (oversized, True))
    monkeypatch.setattr(
        mcp_module, "MAX_RECALL_RECORDS_CALL_RESULT_BYTES", 1)

    response = anyio.run(
        _serverCall, _serverHandler(ctx), "recall_records", {"query": "q"})
    text = response.content[0].text

    assert response.isError is True and text == (
        "recall_records: wire cap cannot hold bounded error response"
    ), (
        f"unrepresentable-cap loud-failure rule violated: isError={response.isError} "
        f"textChars={len(text)} diagnostic={text[:96]!r}"
    )
    assert _modelBytes(response) < 512, (
        f"bounded cap-configuration diagnostic rule violated: "
        f"wireBytes={_modelBytes(response)} limit=511"
    )


def test_legacy_served_boundary_preserves_oversized_dispatch_error(
        monkeypatch, ctx):  # I6, C4, C8
    oversized = "error: LEGACY_BYTES_" + ("z" * (2 * 1024 * 1024))
    monkeypatch.setattr(
        mcp_module, "dispatch", lambda *_args: (oversized, True))

    response = anyio.run(
        _serverCall, _serverHandler(ctx), "pensive_recall", {"query": "q"})

    assert response.isError is True and response.content[0].text == oversized, (
        f"legacy-error byte-preservation rule violated: isError={response.isError} "
        f"expectedChars={len(oversized)} actualChars={len(response.content[0].text)}"
    )
    assert _modelBytes(response) > mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES, (
        f"recall-records-only guard scope rule violated: wireBytes={_modelBytes(response)} "
        f"cap={mcp_module.MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
    )


def test_call_tool_result_wrapper_matches_served_envelope():  # C7
    value = _recallValue("wrapper", [])
    structured = mcp_module.StructuredResult(
        value=value,
        text=json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    wrapper = getattr(mcp_module, "_callToolResult", None)

    assert callable(wrapper), (
        f"shared-wrapper existence rule violated: wrapper={wrapper!r}"
    )
    structuredResult = wrapper(structured, False)
    stringResult = wrapper("legacy string", True)
    assert structuredResult.model_dump_json() == _wireResult(structured).model_dump_json(), (
        f"structured-wrapper exact-envelope rule violated: result={structuredResult!r}"
    )
    assert stringResult == CallToolResult(
        content=[TextContent(type="text", text="legacy string")],
        isError=True,
    ), f"string-wrapper compatibility rule violated: result={stringResult!r}"


def test_server_and_wire_measurement_share_call_tool_result_wrapper(
        monkeypatch, ctx):  # C7
    import anyio
    from mcp.types import CallToolRequest, CallToolRequestParams

    wrapper = getattr(mcp_module, "_callToolResult", None)
    assert callable(wrapper), (
        f"shared-wrapper existence rule violated: wrapper={wrapper!r}"
    )
    calls = []

    def trackedWrapper(result, isError):
        calls.append((result, isError))
        return wrapper(result, isError)

    monkeypatch.setattr(mcp_module, "_callToolResult", trackedWrapper)
    monkeypatch.setattr(
        mcp_module, "recall", lambda *a, **kw: _engineResult(lowConfidence=True))

    measured, measuredError = dispatch(ctx, "recall_records", {"query": "measure"})
    assert measuredError is False and len(calls) == 1, (
        f"wire-measurement shared-wrapper rule violated: calls={calls!r} "
        f"result={measured!r} isError={measuredError}"
    )

    calls.clear()
    monkeypatch.setattr(
        mcp_module, "dispatch", lambda *_args: (measured, False))
    server = buildServer(ctx)
    handler = server.request_handlers[CallToolRequest]

    async def call():
        response = await handler(CallToolRequest(
            params=CallToolRequestParams(
                name="recall_records", arguments={"query": "serve"})))
        return response.root

    response = anyio.run(call)
    assert (
        len(calls) == 1
        and response.model_dump_json() == _wireResult(measured).model_dump_json()
    ), (
        f"server shared-wrapper rule violated: calls={calls!r} response={response!r}"
    )


def test_build_server_wraps_structured_and_string_results(monkeypatch, ctx):  # C3, C4
    import anyio
    from mcp.types import CallToolRequest, CallToolRequestParams

    value = {
        "schemaVersion": 1,
        "query": "q",
        "project": None,
        "records": [],
        "estimatedTokens": 0,
        "lowConfidence": True,
        "truncated": False,
    }
    structured = mcp_module.StructuredResult(
        value=value,
        text=json.dumps(value, sort_keys=True, separators=(",", ":")),
    )

    def fakeDispatch(_ctx, name, _args):
        if name == "recall_records":
            return structured, False
        return "legacy string", False

    monkeypatch.setattr(mcp_module, "dispatch", fakeDispatch)
    server = buildServer(ctx)
    handler = server.request_handlers[CallToolRequest]

    async def call(name, arguments):
        response = await handler(CallToolRequest(
            params=CallToolRequestParams(name=name, arguments=arguments)))
        return response.root

    recordsResponse = anyio.run(call, "recall_records", {"query": "q"})
    stringResponse = anyio.run(call, "recall", {"query": "q"})

    assert recordsResponse.isError is False, (
        f"structured-server success rule violated: response={recordsResponse!r}"
    )
    assert recordsResponse.structuredContent == value, (
        f"structured-server envelope rule violated: response={recordsResponse!r}"
    )
    assert recordsResponse.content[0].text == structured.text, (
        f"structured-server fallback rule violated: response={recordsResponse!r}"
    )
    assert stringResponse.isError is False and stringResponse.content[0].text == "legacy string", (
        f"string-server compatibility rule violated: response={stringResponse!r}"
    )
    assert stringResponse.structuredContent is None, (
        f"string-server unstructured rule violated: response={stringResponse!r}"
    )


def test_recall_records_budget_starvation_serves_stub_not_empty(
        monkeypatch, ctx, store):  # fail-open finding 2026-08-03: empty is not no-match
    bigId = _put(store, "y" * 60)                   # 20 tokens, exceeds budget alone
    ranked = [_trust(bigId)]
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult(ranked))

    starved, starvedError = dispatch(ctx, "recall_records", {
        "query": "budget starvation", "tokenBudget": 10,
    })
    assert starvedError is False, (
        f"starved-call success rule violated: error={starvedError}"
    )
    assert len(starved.value["records"]) == 1, (
        "budget-starvation stub rule violated: a budget smaller than the top "
        f"body must serve a stub, not records:[] — value={starved.value!r}"
    )
    stub = starved.value["records"][0]
    assert stub["id"] == bigId and stub["content"] != "y" * 60, (
        f"stub-identity rule violated: stub={stub!r}"
    )
    assert "tokenBudget 10" in stub["content"] and "20-token" in stub["content"], (
        f"stub-cost disclosure rule violated: content={stub['content']!r}"
    )
    assert stub["estimatedTokens"] == 20, (
        f"stub real-cost accounting rule violated: {stub['estimatedTokens']!r}"
    )
    assert starved.value["truncated"] is True, (
        f"starvation truncation-flag rule violated: value={starved.value!r}"
    )

    # Negative control: genuinely-empty results stay empty — the stub must not
    # invent a match where the engine returned none.
    monkeypatch.setattr(mcp_module, "recall", lambda *a, **kw: _engineResult([]))
    empty, emptyError = dispatch(ctx, "recall_records", {
        "query": "budget starvation", "tokenBudget": 10,
    })
    assert emptyError is False and empty.value["records"] == [], (
        f"no-match emptiness rule violated: value={empty.value!r}"
    )
    assert empty.value["truncated"] is False, (
        f"no-match truncation-flag rule violated: value={empty.value!r}"
    )
