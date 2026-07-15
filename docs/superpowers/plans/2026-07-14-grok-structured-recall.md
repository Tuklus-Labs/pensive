# Grok Structured Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned, bounded, machine-readable Pensive recall tool for the Grok kernel without changing the existing human-facing recall tools.

**Architecture:** `recall_records` is a new v3-native MCP tool over the existing recall engine. It returns strict `structuredContent` plus deterministic JSON text fallback, while `recall`, `pensive_recall`, every compat schema, and every existing result string remain unchanged. Record bodies are atomic and admitted in rank order under the caller's token budget; the result says when the tail was omitted.

**Tech Stack:** Python 3.14, Pensive v3 recall/store, MCP 1.27 streamable HTTP, pytest.

---

### Task 1: Freeze the structured recall contract

**Files:**
- Modify: `daemon/src/serve/mcp.py`
- Modify: `daemon/test/serve/test_mcp.py`
- Create: `daemon/test/serve/test_mcp_structured.py`

- [ ] **Step 1: Add failing registration and schema tests**

Pin `recall_records` as the fifth native tool. Require a strict input schema with `query`; optional `project`, `timeScope`, `kinds`, `k`, and `tokenBudget`; no unknown properties; `k` in `1..32`; and `tokenBudget` in `1..8000`. Pin a strict output schema for this versioned shape:

```python
{
    "schemaVersion": 1,
    "query": "how does Grok recall durable context",
    "project": "grok-kernel",
    "records": [{
        "id": "01...",
        "kind": "atom",
        "project": "grok-kernel",
        "content": "...",
        "createdAt": 1780000000,
        "occurredAt": None,
        "importance": 0.0,
        "status": "live",
        "atomSchemaVersion": 1,
        "provenance": [{
            "id": "01...",
            "atomId": "01...",
            "source": "explicit-emit",
            "sessionId": None,
            "agent": "grok-kernel",
            "sourceRef": None,
            "recordedAt": 1780000000,
        }],
        "score": 0.9,
        "confidence": 0.8,
        "shouldTrust": True,
        "why": "lexical+dense agreement",
        "supersededBy": None,
        "estimatedTokens": 12,
    }],
    "estimatedTokens": 12,
    "lowConfidence": False,
    "truncated": False,
}
```

Run:

```bash
PYTHONPATH=daemon/src python3 -m pytest \
  daemon/test/serve/test_mcp.py::test_native_tools_present_with_required_fields \
  daemon/test/serve/test_mcp_structured.py::test_recall_records_schemas_are_strict -q
```

Expected: FAIL because `recall_records` is absent.

- [ ] **Step 2: Add the minimal tool definitions**

Add module constants for the input and output schemas and append:

```python
Tool(
    name="recall_records",
    description="Versioned structured recall records for kernel-owned memory adapters.",
    inputSchema=RECALL_RECORDS_INPUT_SCHEMA,
    outputSchema=RECALL_RECORDS_OUTPUT_SCHEMA,
)
```

Use `additionalProperties: false` at every object boundary. Bound IDs and provenance text, cap records at 32 and provenance entries at 64, allow nullable stored fields explicitly, and keep every protocol integer at or below `9007199254740991`.

- [ ] **Step 3: Run schema and compat tests green**

```bash
PYTHONPATH=daemon/src python3 -m pytest \
  daemon/test/serve/test_mcp.py::test_compat_tool_schemas_are_verbatim \
  daemon/test/serve/test_mcp.py::test_native_tools_present_with_required_fields \
  daemon/test/serve/test_mcp_structured.py::test_recall_records_schemas_are_strict -q
```

Expected: PASS.

### Task 2: Build bounded records from the existing recall engine

**Files:**
- Modify: `daemon/src/serve/mcp.py`
- Modify: `daemon/test/serve/test_mcp_structured.py`

- [ ] **Step 1: Add failing argument, result, and budget tests**

Cover exact/over query, project, `k`, and `tokenBudget` bounds; bool-as-int rejection; malformed/inverted `timeScope`; unknown/duplicate kinds; unknown fields; project filtering; low-confidence output; complete provenance; superseded records; and missing-store records failing loudly without killing the next call. Prove record contents are admitted whole, in rank order, and the first record that does not fit drops the remaining tail:

```python
result, is_error = dispatch(ctx, "recall_records", {
    "query": "context precedence",
    "project": "grok-kernel",
    "k": 10,
    "tokenBudget": 12,
})
assert is_error is False
assert result.value["estimatedTokens"] <= 12
assert result.value["truncated"] is True
assert all(record["content"] in original_bodies for record in result.value["records"])
```

Run the new test file and expect handler-not-found or wrong-type failures.

- [ ] **Step 2: Implement strict argument normalization**

Add one private normalizer that rejects unknown fields and returns exactly the engine arguments:

```python
def _recallRecordsArgs(args):
    allowed = {"query", "project", "timeScope", "kinds", "k", "tokenBudget"}
    unknown = set(args) - allowed
    if unknown:
        raise ValueError(f"recall_records: unknown fields: {sorted(unknown)}")
    query = _boundedText(args.get("query"), "query", 1, 8192)
    project = args.get("project")
    if project is not None:
        project = _boundedText(project, "project", 1, 256)
    k = _boundedJSONInt(args.get("k", 10), "k", 1, 32)
    tokenBudget = _boundedJSONInt(
        args.get("tokenBudget", 1500), "tokenBudget", 1, 8000)
    timeScope = _timeScope(args.get("timeScope"))
    kinds = _recallKinds(args.get("kinds"))
    return query, project, timeScope, kinds, k, tokenBudget
```

Implement the named private helpers beside it: `_boundedText` checks type,
non-blank content, and inclusive character limits; `_boundedJSONInt` rejects
booleans and checks inclusive bounds; `_timeScope` accepts `None` or two
JavaScript-safe non-negative integers with `start <= end`; `_recallKinds`
accepts `None` or a non-empty unique subset of
`atom|narrative|snapshot|document_chunk`. Preserve query text and normalize
absent project to `None`. Errors remain `ValueError` so `dispatch` returns
`isError=True` and the daemon keeps serving.

- [ ] **Step 3: Add a serializable structured value**

```python
@dataclass(frozen=True)
class StructuredResult:
    value: dict
    text: str

def _structuredResult(value):
    return StructuredResult(
        value=value,
        text=json.dumps(value, ensure_ascii=False, allow_nan=False,
                        sort_keys=True, separators=(",", ":")),
    )
```

Construct it inside the handler so serialization failures stay inside `dispatch`'s containment boundary.

- [ ] **Step 4: Implement `handle_recall_records`**

Call existing `recall(...)` exactly once. For each ranked result, fetch its atom, compute `estimateTokens(atom["text"])`, and admit whole records until the next record exceeds the remaining budget; then set `truncated=True` and stop. Never cut a body. Preserve rank order, complete provenance, trust fields, and nullable `supersededBy`. Raise if the engine returns an atom absent from the store.

- [ ] **Step 5: Run focused tests green**

```bash
PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve/test_mcp_structured.py -q
```

Expected: PASS.

### Task 3: Carry structured content over real MCP without compat drift

**Files:**
- Modify: `daemon/src/serve/mcp.py`
- Modify: `daemon/test/serve/test_mcp.py`
- Modify: `daemon/test/serve/test_mcp_structured.py`

- [ ] **Step 1: Add failing server-envelope tests**

Test real streamable HTTP so `call_tool("recall_records", ...)` returns:

```python
assert response.isError is False
assert response.structuredContent["schemaVersion"] == 1
assert json.loads(response.content[0].text) == response.structuredContent
```

Also assert existing string-returning tools still have `structuredContent is None` and unchanged text.

- [ ] **Step 2: Extend only the response wrapper**

```python
if isinstance(result, StructuredResult):
    return CallToolResult(
        content=[TextContent(type="text", text=result.text)],
        structuredContent=result.value,
        isError=isError,
    )
```

Keep the existing string branch byte-for-byte equivalent.

- [ ] **Step 3: Prove transport and compatibility**

```bash
PYTHONPATH=daemon/src python3 -m pytest \
  daemon/test/serve/test_mcp.py daemon/test/serve/test_mcp_structured.py -q
```

Expected: PASS, including subprocess transport.

### Task 4: Adversarial verification and live-service gate

**Files:**
- Create: `daemon/test/serve/RISK_MODEL_STRUCTURED_RECALL.md`
- Create: `daemon/test/serve/SABOTAGE_LOG_STRUCTURED_RECALL.md`

- [ ] **Step 1: Record the risk and assertion matrix**

Cover schema drift, compat drift, unbounded output, partial record truncation, rank reordering, bool/int confusion, missing records, NaN serialization, provenance loss, MCP envelope drift, handler failure containment, and live-service rollback.

- [ ] **Step 2: Sabotage load-bearing behavior**

Temporarily remove the budget stop, reorder records, drop provenance, accept unknown arguments, return text without `structuredContent`, and route `recall_records` through `handle_recall`. Each mutation must fail a named assertion; restore every mutation. Run mutation coverage if the repository's Python mutation tooling is available; otherwise record exact manual sabotage results without inventing a score.

- [ ] **Step 3: Run full relevant verification**

```bash
PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve -q
PYTHONPATH=daemon/src python3 -m pytest daemon/test/recall daemon/test/store -q
python3 -m compileall -q daemon/src/serve daemon/src/recall daemon/src/store
git diff --check
```

Expected: PASS. Existing unrelated global `pip check` conflicts are not a Pensive test failure.

- [ ] **Step 4: Commit and review before deployment**

Commit the Pensive feature. Require specification review, quality review, a clean worktree, and a live temporary-port MCP probe before fast-forwarding `feat/pensive-v3` or restarting `pensive-v3.service`. After deployment, verify `tools/list`, one bounded `recall_records` call, one existing `recall` call, and service health. If any live probe fails, roll back the service checkout to its prior commit and restart.
