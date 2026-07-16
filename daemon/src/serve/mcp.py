"""The MCP tool surface: legacy compat names (identical shapes) plus v3 natives.

Two audiences share one server:

- **Compat** (``pensive_recall``, ``engram_emit_atom``, ``engram_emit_discovery``,
  ``engram_emit_failure``, ``engram_emit_narrative``, ``engram_emit_snapshot``,
  ``pensive_analytics``). A live agent's call sites are shaped to the PRODUCTION
  server's tool schemas, so these mirror them EXACTLY -- the ``inputSchema`` dicts
  are copied verbatim from ``~/Projects/Engram/tools/pensive-mcp-server`` and the
  emit RESULT strings reproduce that server's format
  (``atom [<outcome>] emitted (emission_id: <uuid>): <principle[:80]> (ok)`` and
  friends). During shadow (Task 13) old and new answers are diffed, so any shape
  drift here reads as a broken agent. The DATA is served from the v3 engine and
  written to the v3 store; only the envelope is legacy.

- **Natives** (``recall``, ``history``, ``correct``, ``pin``, ``recall_records``).
  These expose the v3 capabilities the legacy names cannot: the rich tiered
  ``recall`` payload, a ``history`` view (Tier-2 neighborhood + supersession
  chain), a one-flow ``correct`` (put + supersede, the trust layer's decades
  rule), ``pin``, and bounded machine-readable records.

Design rules fixed by the task brief:

- Emits map to ``putAtom``: ``kind='atom'`` (narrative -> ``'narrative'``,
  snapshot -> ``'snapshot'``), provenance ``source='explicit-emit'`` with ``agent``
  from the calling context when known, ``importance`` 0.0, and the emit's project
  in the project COLUMN (which is what ``recall.signals.facetSignal`` filters on).
- A tool error NEVER kills the daemon: :func:`dispatch` catches every handler
  exception and returns an MCP error response; the next call still works. Handlers
  fail LOUD (raise with the component's message) rather than swallowing.
- Emits write to the v3 store ONLY here; the double-write tee to the old store is
  Task 13's job, not this one.

Handlers return a string or ``StructuredResult`` through :func:`dispatch`, which
is exactly the path the server's ``call_tool`` takes. Tests can exercise the same
boundary directly against a real store and models without spawning the daemon.
"""
from dataclasses import dataclass
import json
import uuid

from jsonschema import Draft202012Validator
from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from recall.engine import recall
from recall.embedder import embedMissing
from recall.vector_index import buildClassIndexes
from recall.payload import assembleTier2, estimateTokens
from serve import viz
from store.store import (
    putAtom,
    getAtom,
    logRecall,
    supersede,
    addFacet,
    facetsOf,
    edgesFrom,
    edgesTo,
)

__all__ = [
    "ServeContext",
    "COMPAT_TOOLS",
    "NATIVE_TOOLS",
    "TOOLS",
    "HANDLERS",
    "StructuredResult",
    "MAX_RECALL_RECORDS_CALL_RESULT_BYTES",
    "dispatch",
    "buildServer",
]

SERVER_NAME = "pensived-v3-shadow"

# Fixed provenance source for anything an agent emits through this server (the
# schema enumerates this exact value). Distinct from bulk-import (backfill) and
# claude-code/codex (passive capture).
_EMIT_SOURCE = "explicit-emit"

# The handle scheme the payload layer uses; reused by history/correct so every
# atom reference across the server anchors on one token.
_HANDLE = "p3://"

# Legacy pensive_recall never truncated its summary; a v3 atom body can be long,
# so the compat listing collapses whitespace to one line and caps the length to
# keep each "- [NN%] (proj) ..." line readable. Structural shape, not content, is
# what the shadow diff pins.
_LISTING_GIST_CHARS = 300

_MAX_JSON_INTEGER = 9_007_199_254_740_991
_RECALL_KINDS = ["atom", "narrative", "snapshot", "document_chunk"]
MAX_RECALL_RECORDS_CALL_RESULT_BYTES = 1 << 20
_RECALL_RECORDS_OVERSIZE_ERROR = (
    "error: recall_records response exceeds wire-size cap"
)
_RECALL_RECORDS_UNREPRESENTABLE_CAP_ERROR = (
    "recall_records: wire cap cannot hold bounded error response"
)

_NULLABLE_ID_SCHEMA = {
    "type": ["string", "null"],
    "minLength": 1,
    "maxLength": 256,
}
_NULLABLE_TEXT_SCHEMA = {
    "type": ["string", "null"],
    "maxLength": 2048,
}

RECALL_RECORDS_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "type": "string", "minLength": 1, "maxLength": 8192,
            "pattern": r".*\S.*",
        },
        "project": {
            "type": ["string", "null"], "minLength": 1, "maxLength": 256,
            "pattern": r".*\S.*",
        },
        "timeScope": {
            "type": ["array", "null"],
            "items": {
                "type": "integer", "minimum": 0, "maximum": _MAX_JSON_INTEGER,
            },
            "minItems": 2,
            "maxItems": 2,
        },
        "kinds": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": _RECALL_KINDS},
            "minItems": 1,
            "maxItems": len(_RECALL_KINDS),
            "uniqueItems": True,
        },
        "k": {"type": "integer", "minimum": 1, "maximum": 32, "default": 10},
        "tokenBudget": {
            "type": "integer", "minimum": 1, "maximum": 8000, "default": 1500,
        },
    },
    "required": ["query"],
}

_PROVENANCE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 256},
        "atomId": {"type": "string", "minLength": 1, "maxLength": 256},
        "source": {"type": "string", "minLength": 1, "maxLength": 256},
        "sessionId": _NULLABLE_TEXT_SCHEMA,
        "agent": _NULLABLE_TEXT_SCHEMA,
        "sourceRef": _NULLABLE_TEXT_SCHEMA,
        "recordedAt": {
            "type": "integer", "minimum": 0, "maximum": _MAX_JSON_INTEGER,
        },
    },
    "required": [
        "id", "atomId", "source", "sessionId", "agent", "sourceRef", "recordedAt",
    ],
}

_RECORD_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 256},
        "kind": {"type": "string", "enum": _RECALL_KINDS},
        "project": {
            "type": ["string", "null"], "minLength": 1, "maxLength": 256,
        },
        "content": {"type": "string", "maxLength": 24_000},
        "createdAt": {
            "type": "integer", "minimum": 0, "maximum": _MAX_JSON_INTEGER,
        },
        "occurredAt": {
            "type": ["integer", "null"], "minimum": 0,
            "maximum": _MAX_JSON_INTEGER,
        },
        "importance": {"type": "number"},
        "status": {"type": "string", "enum": ["live", "superseded"]},
        "atomSchemaVersion": {
            "type": "integer", "minimum": 1, "maximum": _MAX_JSON_INTEGER,
        },
        "provenance": {
            "type": "array", "items": _PROVENANCE_OUTPUT_SCHEMA,
            "minItems": 1,
            "maxItems": 64,
        },
        "score": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "shouldTrust": {"type": "boolean"},
        "why": {"type": "string", "minLength": 1, "maxLength": 2048},
        "supersededBy": _NULLABLE_ID_SCHEMA,
        "estimatedTokens": {"type": "integer", "minimum": 0, "maximum": 8000},
    },
    "required": [
        "id", "kind", "project", "content", "createdAt", "occurredAt",
        "importance", "status", "atomSchemaVersion", "provenance", "score",
        "confidence", "shouldTrust", "why", "supersededBy", "estimatedTokens",
    ],
}

RECALL_RECORDS_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schemaVersion": {"type": "integer", "const": 1},
        "query": {"type": "string", "minLength": 1, "maxLength": 8192},
        "project": {"type": ["string", "null"], "minLength": 1, "maxLength": 256},
        "records": {"type": "array", "items": _RECORD_OUTPUT_SCHEMA, "maxItems": 32},
        "estimatedTokens": {"type": "integer", "minimum": 0, "maximum": 8000},
        "lowConfidence": {"type": "boolean"},
        "truncated": {"type": "boolean"},
    },
    "required": [
        "schemaVersion", "query", "project", "records", "estimatedTokens",
        "lowConfidence", "truncated",
    ],
}
_RECALL_RECORDS_OUTPUT_VALIDATOR = Draft202012Validator(RECALL_RECORDS_OUTPUT_SCHEMA)


# --------------------------------------------------------------------------- #
# Resident context: store + models + a rebuildable index                       #
# --------------------------------------------------------------------------- #


class ServeContext:
    """Everything the handlers need, loaded once and reused.

    Holds the canonical ``store``, a resident ``embedder``, the ``modelId`` they
    agree on, and ``indexes``: one dense index per kind-class (``memory``,
    ``code``, ...), built by ``buildClassIndexes`` and rebuilt by ``reindex``, so
    memory and code atoms are searched from separate pools instead of one mixed
    index. ``agent`` is stamped into emit provenance when the caller is known.

    ``reindex`` embeds any not-yet-embedded live atoms and rebuilds the per-class
    indexes, so an atom written by an emit/correct becomes recallable by BOTH the
    lexical (query-time) and dense (index) signals on the next call. It runs at
    construction and after every mutating tool. A full rebuild per emit is the
    known cost an incremental index would later retire; for the shadow daemon it is
    correct and cheap enough.
    """

    def __init__(self, store, embedder, modelId, agent=None,
                 defaultK=10, defaultTokenBudget=1500):
        self.store = store
        self.embedder = embedder
        self.modelId = modelId
        self.agent = agent
        self.defaultK = defaultK
        self.defaultTokenBudget = defaultTokenBudget
        self.indexes = {}
        self.recallLogErrors = 0
        self.reindex()

    def reindex(self):
        """Embed missing live atoms and rebuild the per-class dense indexes."""
        embedMissing(self.store, self.embedder)
        self.indexes = buildClassIndexes(self.store, self.modelId)


# --------------------------------------------------------------------------- #
# Small shared helpers                                                          #
# --------------------------------------------------------------------------- #


def _require(args, names):
    """Raise a clean ValueError if any required field is absent or blank.

    Loud over silent: an emit missing its principle must error with the field
    name, not write a half-formed atom. Blank ('' / None) counts as missing --
    the required reasoning fields are meaningless empty.
    """
    for name in names:
        value = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"missing required field: {name}")


def _boundedText(value, name, minimum, maximum):
    if not isinstance(value, str):
        raise ValueError(f"recall_records: {name} must be a string")
    if not value.strip():
        raise ValueError(f"recall_records: {name} must not be blank")
    if not minimum <= len(value) <= maximum:
        raise ValueError(
            f"recall_records: {name} length must be in {minimum}..{maximum}"
        )
    return value


def _boundedJSONInt(value, name, minimum, maximum):
    if type(value) is not int:
        raise ValueError(f"recall_records: {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"recall_records: {name} must be in {minimum}..{maximum}"
        )
    return value


def _timeScope(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("recall_records: timeScope must be [start, end]")
    start = _boundedJSONInt(value[0], "timeScope start", 0, _MAX_JSON_INTEGER)
    end = _boundedJSONInt(value[1], "timeScope end", 0, _MAX_JSON_INTEGER)
    if start > end:
        raise ValueError("recall_records: timeScope start must be <= end")
    return start, end


def _recallKinds(value):
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("recall_records: kinds must be a non-empty array")
    if any(not isinstance(kind, str) for kind in value):
        raise ValueError("recall_records: kinds entries must be strings")
    if len(value) != len(set(value)):
        raise ValueError("recall_records: kinds must not contain duplicates")
    unknown = sorted(set(value) - set(_RECALL_KINDS))
    if unknown:
        raise ValueError(f"recall_records: kinds contains unknown values: {unknown}")
    return tuple(value)


def _recallRecordsArgs(args):
    allowed = {"query", "project", "timeScope", "kinds", "k", "tokenBudget"}
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"recall_records: unknown fields: {unknown}")
    query = _boundedText(args.get("query"), "query", 1, 8192)
    project = args.get("project")
    if project is not None:
        project = _boundedText(project, "project", 1, 256)
    timeScope = _timeScope(args.get("timeScope"))
    kinds = _recallKinds(args.get("kinds"))
    k = _boundedJSONInt(args.get("k", 10), "k", 1, 32)
    tokenBudget = _boundedJSONInt(
        args.get("tokenBudget", 1500), "tokenBudget", 1, 8000
    )
    return query, project, timeScope, kinds, k, tokenBudget


@dataclass(frozen=True)
class StructuredResult:
    value: dict
    text: str


def _structuredResult(value):
    errors = sorted(
        _RECALL_RECORDS_OUTPUT_VALIDATOR.iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"recall_records: structured output violates schema at {path}: "
            f"validator={error.validator}"
        )
    return StructuredResult(
        value=value,
        text=json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _callToolResult(result, isError):
    if isinstance(result, StructuredResult):
        return CallToolResult(
            content=[TextContent(type="text", text=result.text)],
            structuredContent=result.value,
            isError=isError,
        )
    return CallToolResult(
        content=[TextContent(type="text", text=result)],
        isError=isError,
    )


def _callToolResultBytes(result, isError):
    return len(_callToolResult(result, isError).model_dump_json().encode("utf-8"))


def _gistLine(text):
    """One-line, whitespace-collapsed, length-capped preview of an atom body."""
    return " ".join((text or "").split())[:_LISTING_GIST_CHARS]


def _emissionId():
    """A fresh uuid4 string -- the compat emission id shape agents expect."""
    return str(uuid.uuid4())


def _splitCsv(value):
    """Legacy comma-split: trimmed, empties dropped."""
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def _emitProvenance(ctx):
    prov = {"source": _EMIT_SOURCE}
    if ctx.agent:
        prov["agent"] = ctx.agent
    return prov


def _composeAtomText(shape, approach, outcome, reason, principle,
                     domain="", narrative=""):
    """Fold the reasoning-atom fields into one recall-friendly text body.

    The v3 store is text-first (bm25 + dense over ``text``), so the structured
    legacy fields are composed into a single readable block. Keeps the fields the
    dense/lexical signals key on (shape, reason, principle) prominent; an inline
    narrative trails as its own paragraph.
    """
    # Keep this separate from ambient.distiller._composeEmitText deliberately:
    # mcp validates/requires legacy fields, while distiller tolerates arbitrary
    # transcript args captured from tool calls.
    lines = [shape.strip()]
    if approach and approach.strip():
        lines.append(f"approach: {approach.strip()}")
    lines.append(f"outcome: {outcome.strip()}. {reason.strip()}")
    lines.append(f"principle: {principle.strip()}")
    if domain and domain.strip():
        lines.append(f"domain: {domain.strip()}")
    body = "\n".join(lines)
    if narrative and narrative.strip():
        body += "\n\n" + narrative.strip()
    return body


def _truncateWords(text, limit=500):
    """Legacy narrative cap: at most ``limit`` words."""
    words = (text or "").split()
    if len(words) > limit:
        return " ".join(words[:limit])
    return text


def _returnedAtomIds(out):
    return [r["atomId"] for r in out["results"]]


def _logReturnedRecall(ctx, out, query, sourceRef, atomIds=None):
    try:
        if atomIds is None:
            atomIds = _returnedAtomIds(out)
        logRecall(ctx.store, atomIds, query=query, sourceRef=sourceRef)
        viz.emitRecallEvent(ctx, query, atomIds, sourceRef)
    except Exception:  # noqa: BLE001 -- recall serving wins over telemetry
        ctx.recallLogErrors += 1


# --------------------------------------------------------------------------- #
# Compat handlers -- legacy names, legacy result shapes, v3 store               #
# --------------------------------------------------------------------------- #


def handle_emit_atom(ctx, args):
    _require(args, ["project", "shape", "approach", "outcome", "reason", "principle"])
    outcome = args["outcome"]
    principle = args["principle"]
    text = _composeAtomText(
        args["shape"], args["approach"], outcome, args["reason"], principle,
        domain=args.get("domain", ""), narrative=args.get("narrative", ""),
    )
    atomId = putAtom(ctx.store, {
        "text": text,
        "kind": "atom",
        "project": args["project"],
        "importance": 0.0,
        "provenance": _emitProvenance(ctx),
    })
    for tag in dict.fromkeys(_splitCsv(args.get("tags", ""))):
        addFacet(ctx.store, atomId, "tag", tag)
    ctx.reindex()
    emissionId = _emissionId()
    return (f"atom [{outcome}] emitted (emission_id: {emissionId}): "
            f"{principle[:80]} (ok)")


def handle_emit_discovery(ctx, args):
    _require(args, ["project", "principle"])
    principle = args["principle"]
    delegated = {
        "project": args["project"], "shape": "discovery", "approach": "observed",
        "outcome": "succeeded", "reason": principle, "principle": principle,
    }
    for key in ("narrative", "stakes", "topic", "trigger", "domain", "tags"):
        if key in args:
            delegated[key] = args[key]
    return handle_emit_atom(ctx, delegated)


def handle_emit_failure(ctx, args):
    _require(args, ["project", "principle"])
    principle = args["principle"]
    delegated = {
        "project": args["project"], "shape": "failed approach",
        "approach": "attempted", "outcome": "failed",
        "reason": principle, "principle": principle,
    }
    for key in ("narrative", "stakes", "topic", "trigger", "domain", "tags"):
        if key in args:
            delegated[key] = args[key]
    return handle_emit_atom(ctx, delegated)


def handle_emit_narrative(ctx, args):
    _require(args, ["project", "narrative"])
    text = _truncateWords(args["narrative"])
    putAtom(ctx.store, {
        "text": text,
        "kind": "narrative",
        "project": args["project"],
        "importance": 0.0,
        "provenance": _emitProvenance(ctx),
    })
    ctx.reindex()
    return f"Narrative fragment emitted (emission_id: {_emissionId()})"


def handle_emit_snapshot(ctx, args):
    _require(args, ["project", "hypothesis"])
    hypothesis = args["hypothesis"]
    lines = [f"hypothesis: {hypothesis.strip()}"]
    deadEnds = _splitCsv(args.get("dead_ends", ""))
    nextSteps = _splitCsv(args.get("next_steps", ""))
    if deadEnds:
        lines.append("dead ends: " + "; ".join(deadEnds))
    if nextSteps:
        lines.append("next steps: " + "; ".join(nextSteps))
    putAtom(ctx.store, {
        "text": "\n".join(lines),
        "kind": "snapshot",
        "project": args["project"],
        "importance": 0.0,
        "provenance": _emitProvenance(ctx),
    })
    ctx.reindex()
    return f"snapshot emitted: {hypothesis[:80]} (ok)"


def handle_pensive_recall(ctx, args):
    _require(args, ["query"])
    query = args["query"]
    project = args.get("project", "") or None      # "" (legacy default) -> no filter
    limit = int(args.get("limit", 10))
    out = recall(
        ctx.store, ctx.indexes, ctx.embedder, query,
        project=project, k=limit, tokenBudget=ctx.defaultTokenBudget,
    )
    results = out["results"]
    if not results:
        return f"No memories found for query: {query}"
    lines = [f"Found {len(results)} memories:\n"]
    for r in results:
        atom = getAtom(ctx.store, r["atomId"])
        pct = int(r["confidence"] * 100)
        src = (atom["project"] or "") if atom else ""
        summary = _gistLine(atom["text"] if atom else "")
        tag = f"[{pct}%] ({src})" if src else f"[{pct}%]"
        lines.append(f"- {tag} {summary}")
    response = "\n".join(lines)
    _logReturnedRecall(ctx, out, query, "mcp.pensive_recall")
    return response


def handle_pensive_analytics(ctx, args):
    """A truthful v3 analogue of the legacy analytics JSON.

    The legacy tool surfaced the production vector service's latency percentiles
    and SA/L2 agreement rates -- metrics this shadow daemon does not own. Rather
    than fabricate them, report the v3 store's real shape (counts by kind/status,
    embedded coverage) as JSON in the same ``json.dumps(indent=2)`` envelope.
    """
    conn = ctx.store._conn
    total = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    byKind = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM atoms GROUP BY kind").fetchall())
    byStatus = dict(conn.execute(
        "SELECT status, COUNT(*) FROM atoms GROUP BY status").fetchall())
    embedded = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_id = ?",
        (ctx.modelId,),
    ).fetchone()[0]
    data = {
        "server": SERVER_NAME,
        "store": {
            "totalAtoms": total,
            "byKind": byKind,
            "byStatus": byStatus,
            "embedded": embedded,
            "modelId": ctx.modelId,
        },
        "note": ("v3 shadow store shape; legacy latency/agreement metrics are "
                 "served by the production analytics endpoint, not this daemon"),
    }
    return json.dumps(data, indent=2)


# --------------------------------------------------------------------------- #
# Native handlers -- the v3 capabilities                                        #
# --------------------------------------------------------------------------- #


def handle_recall(ctx, args):
    """v3 native recall: return the rich tiered payload (not the legacy listing)."""
    _require(args, ["query"])
    query = args["query"]
    project = args.get("project") or None
    k = int(args.get("k", ctx.defaultK))
    tokenBudget = int(args.get("tokenBudget", ctx.defaultTokenBudget))
    timeScope = args.get("timeScope")
    if timeScope is not None:
        timeScope = (int(timeScope[0]), int(timeScope[1]))
    kinds = args.get("kinds")
    out = recall(
        ctx.store, ctx.indexes, ctx.embedder, query,
        project=project, timeScope=timeScope, kinds=kinds,
        k=k, tokenBudget=tokenBudget, enrich=True,
    )
    response = out["payload"]
    _logReturnedRecall(ctx, out, query, "mcp.recall")
    return response


def handle_recall_records(ctx, args):
    query, project, timeScope, kinds, k, tokenBudget = _recallRecordsArgs(args)
    out = recall(
        ctx.store,
        ctx.indexes,
        ctx.embedder,
        query,
        project=project,
        timeScope=timeScope,
        kinds=kinds,
        k=k,
        tokenBudget=tokenBudget,
    )
    ranked = out["results"]
    if len(ranked) > k:
        raise ValueError(
            f"recall_records: engine returned {len(ranked)} results for k={k}"
        )

    records = []
    tokens = 0
    truncated = bool(ranked)

    def makeStructured(candidateRecords, candidateTokens, candidateTruncated):
        return _structuredResult({
            "schemaVersion": 1,
            "query": query,
            "project": project,
            "records": candidateRecords,
            "estimatedTokens": candidateTokens,
            "lowConfidence": out["lowConfidence"],
            "truncated": candidateTruncated,
        })

    structured = makeStructured(records, tokens, truncated)
    baseBytes = _callToolResultBytes(structured, False)
    if baseBytes > MAX_RECALL_RECORDS_CALL_RESULT_BYTES:
        raise ValueError(
            f"recall_records: base envelope is {baseBytes} bytes, exceeds cap "
            f"{MAX_RECALL_RECORDS_CALL_RESULT_BYTES}"
        )

    for index, result in enumerate(ranked):
        atom = getAtom(ctx.store, result["atomId"])
        if atom is None:
            raise ValueError(
                f"recall_records: atom {result['atomId']!r} absent from the store"
            )
        if len(atom["provenance"]) > 64:
            raise ValueError(
                f"recall_records: atom {atom['id']!r} has more than 64 provenance rows"
            )
        recordTokens = estimateTokens(atom["text"])
        if tokens + recordTokens > tokenBudget:
            truncated = True
            break
        record = {
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
            "score": result["score"],
            "confidence": result["confidence"],
            "shouldTrust": result["shouldTrust"],
            "why": result["why"],
            "supersededBy": result.get("supersededBy"),
            "estimatedTokens": recordTokens,
        }
        candidateRecords = [*records, record]
        candidateTokens = tokens + recordTokens
        candidateTruncated = index < len(ranked) - 1
        candidate = makeStructured(
            candidateRecords, candidateTokens, candidateTruncated)
        if (
            _callToolResultBytes(candidate, False)
            > MAX_RECALL_RECORDS_CALL_RESULT_BYTES
        ):
            truncated = True
            break
        records = candidateRecords
        tokens = candidateTokens
        truncated = candidateTruncated
        structured = candidate

    if truncated:
        structured = makeStructured(records, tokens, True)
    _logReturnedRecall(
        ctx, out, query, "mcp.recall_records",
        atomIds=[record["id"] for record in records],
    )
    return structured


def _supersessionChain(store, atomId):
    """The full supersession chain through ``atomId``, oldest id first.

    A supersedes edge points new -> old (src=new, dst=old). Walking
    ``edgesFrom(cur, 'supersedes')`` steps toward older atoms; walking
    ``edgesTo(cur, 'supersedes')`` steps toward newer ones. Both walks are guarded
    against a cycle by a seen-set (a corrupt cycle stops the walk rather than
    looping forever; the trust layer is where a cycle raises). A lone atom with no
    supersedes edge returns ``[atomId]``.
    """
    seen = {atomId}

    older = []
    cur = atomId
    while True:
        edges = edgesFrom(store, cur, "supersedes")
        if not edges:
            break
        dst = edges[0]["dstAtom"]
        if dst in seen:
            break
        older.append(dst)
        seen.add(dst)
        cur = dst
    older.reverse()   # oldest first

    newer = []
    cur = atomId
    while True:
        edges = edgesTo(store, cur, "supersedes")
        if not edges:
            break
        src = edges[0]["srcAtom"]
        if src in seen:
            break
        newer.append(src)
        seen.add(src)
        cur = src

    return older + [atomId] + newer


def handle_history(ctx, args):
    """Tier-2 neighborhood for ``atomId`` plus its supersession chain.

    The Tier-2 block (via :func:`recall.payload.assembleTier2`) renders the atom's
    body, provenance, and LIVE outgoing edges. Because a supersedes edge points at
    a NON-live (superseded) atom, the chain is rendered separately: one line per
    chain member, oldest to newest, each with its live/superseded status. A
    missing ``atomId`` raises (loud) via the payload layer's ``_fetch``.
    """
    _require(args, ["atomId"])
    atomId = args["atomId"]
    block = assembleTier2(ctx.store, atomId)
    chain = _supersessionChain(ctx.store, atomId)
    if len(chain) <= 1:
        return block
    lines = [block, "", "supersession chain (oldest to newest):"]
    for cid in chain:
        atom = getAtom(ctx.store, cid)
        status = atom["status"] if atom else "missing"
        gist = _gistLine(atom["text"]) if atom else ""
        marker = "  <- this atom" if cid == atomId else ""
        lines.append(f"{_HANDLE}{cid} [{status}] {gist}{marker}")
    return "\n".join(lines)


def handle_correct(ctx, args):
    """Create a corrected atom and supersede the old one, in one flow.

    The new atom inherits the old atom's ``project`` and ``kind`` (a correction of
    a narrative is still a narrative) and carries the caller-supplied provenance
    (defaulting ``source`` to ``explicit-emit``, ``agent`` to the context agent).
    The old atom's existence is checked BEFORE the put so a missing target errors
    cleanly without orphaning a freshly written atom. After both writes commit,
    :meth:`ServeContext.reindex` makes the new atom recallable and the old atom
    surfaces only chained (the trust layer's decades rule) on the next recall.
    """
    _require(args, ["oldAtomId", "newText"])
    oldId = args["oldAtomId"]
    newText = args["newText"]
    old = getAtom(ctx.store, oldId)
    if old is None:
        raise ValueError(f"correct: atom {oldId!r} not found")

    prov = args.get("provenance") or {}
    provenance = {"source": prov.get("source") or _EMIT_SOURCE}
    for key in ("agent", "sessionId", "sourceRef"):
        if prov.get(key) is not None:
            provenance[key] = prov[key]
    if "agent" not in provenance and ctx.agent:
        provenance["agent"] = ctx.agent

    newId = putAtom(ctx.store, {
        "text": newText,
        "kind": old["kind"],
        "project": old["project"],
        "importance": 0.0,
        "provenance": provenance,
    })
    supersede(ctx.store, oldId, newId, provenance)
    ctx.reindex()
    return f"corrected {_HANDLE}{oldId} -> {_HANDLE}{newId} (ok)"


def handle_pin(ctx, args):
    """Pin ``atomId`` via a ``key='pin'`` facet. Idempotent (facet INSERT OR
    IGNORE); a missing atom trips the facet foreign key and errors cleanly."""
    _require(args, ["atomId"])
    atomId = args["atomId"]
    addFacet(ctx.store, atomId, "pin", "true")
    return f"pinned {_HANDLE}{atomId} (ok)"


# --------------------------------------------------------------------------- #
# Tool definitions                                                              #
# --------------------------------------------------------------------------- #

# COMPAT: inputSchema dicts copied VERBATIM from the production server
# (~/Projects/Engram/tools/pensive-mcp-server). Do not "improve" these -- an agent
# mid-shadow is calling them exactly as written.
COMPAT_TOOLS = [
    Tool(
        name="engram_emit_atom",
        description="Emit a reasoning atom (structured insight) to Engram's living memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "project":   {"type": "string", "description": "Project name"},
                "shape":     {"type": "string", "description": "Abstract problem description"},
                "approach":  {"type": "string", "description": "What was tried"},
                "outcome":   {"type": "string", "description": "Result of the approach",
                              "enum": ["succeeded", "failed", "partial", "abandoned"]},
                "reason":    {"type": "string", "description": "Why it worked or failed"},
                "principle": {"type": "string", "description": "Transferable insight extracted"},
                "tags":      {"type": "string", "description": "Comma-separated tags", "default": ""},
                "domain":    {"type": "string", "description": "Domain area", "default": ""},
                "narrative": {"type": "string", "description": "First-person experiential context of this moment (1-4 sentences, max 500 words)"},
                "trigger":   {"type": "string", "enum": ["spontaneous", "checkpoint"], "description": "What triggered this emission"},
                "stakes":    {"type": "string", "enum": ["high", "medium", "low"], "description": "How much this moment mattered"},
                "dynamics":  {"type": "string", "enum": ["collaborative", "challenging", "tense", "exploratory", "teaching", "debugging"], "description": "Session dynamic"},
                "topic":     {"type": "string", "description": "Short phrase for arc detection"},
            },
            "required": ["project", "shape", "approach", "outcome", "reason", "principle"],
        },
    ),
    Tool(
        name="engram_emit_snapshot",
        description="Emit a cognitive snapshot (working state dump for compaction recovery).",
        inputSchema={
            "type": "object",
            "properties": {
                "project":    {"type": "string", "description": "Project name"},
                "hypothesis": {"type": "string", "description": "Current working theory"},
                "dead_ends":  {"type": "string", "description": "Comma-separated dead ends", "default": ""},
                "next_steps": {"type": "string", "description": "Comma-separated next steps", "default": ""},
            },
            "required": ["project", "hypothesis"],
        },
    ),
    Tool(
        name="pensive_recall",
        description="Query Pensive for relevant past reasoning atoms and context.",
        inputSchema={
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "What to search for"},
                "project": {"type": "string", "description": "Filter by project name", "default": ""},
                "limit":   {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="engram_emit_discovery",
        description="Shorthand: emit a discovery atom (positive insight).",
        inputSchema={
            "type": "object",
            "properties": {
                "project":   {"type": "string", "description": "Project name"},
                "principle": {"type": "string", "description": "What was discovered"},
            },
            "required": ["project", "principle"],
        },
    ),
    Tool(
        name="engram_emit_failure",
        description="Shorthand: emit a failure atom (record what did not work and why).",
        inputSchema={
            "type": "object",
            "properties": {
                "project":   {"type": "string", "description": "Project name"},
                "principle": {"type": "string", "description": "What failed and why"},
            },
            "required": ["project", "principle"],
        },
    ),
    Tool(
        name="engram_emit_narrative",
        description="Emit a standalone narrative fragment (no atom). Use for checkpoint triggers when there's no structured insight to emit.",
        inputSchema={
            "type": "object",
            "properties": {
                "project":   {"type": "string", "description": "Project name"},
                "narrative": {"type": "string", "description": "First-person experiential text (1-4 sentences)"},
                "trigger":   {"type": "string", "enum": ["spontaneous", "checkpoint"], "default": "checkpoint"},
                "dynamics":  {"type": "string", "enum": ["collaborative", "challenging", "tense", "exploratory", "teaching", "debugging"]},
                "stakes":    {"type": "string", "enum": ["high", "medium", "low"], "default": "medium"},
                "topic":     {"type": "string", "description": "Short phrase for arc detection"},
            },
            "required": ["project", "narrative"],
        },
    ),
    Tool(
        name="pensive_analytics",
        description="View Pensive query analytics: latency percentiles, SA/L2 agreement rates, recent misses, and source distribution.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]

# NATIVES: the v3-only tools. Shapes mirror the engine/store contracts.
NATIVE_TOOLS = [
    Tool(
        name="recall",
        description="v3 recall: relevance-ranked, trust-annotated memory as a tiered no-slop payload.",
        inputSchema={
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "What to search for"},
                "project":     {"type": "string", "description": "Restrict to this project's live atoms"},
                "timeScope":   {"type": "array", "items": {"type": "integer"},
                                "minItems": 2, "maxItems": 2,
                                "description": "Inclusive [startUnix, endUnix] effective-time window"},
                "kinds":       {"type": "array", "items": {"type": "string"},
                                "description": "Keep only these atom kinds (atom|narrative|snapshot|document_chunk)"},
                "k":           {"type": "integer", "description": "Max results", "default": 10},
                "tokenBudget": {"type": "integer", "description": "Payload token budget", "default": 1500},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="history",
        description="An atom's Tier-2 neighborhood (body, provenance, live edges) plus its full supersession chain.",
        inputSchema={
            "type": "object",
            "properties": {
                "atomId": {"type": "string", "description": "The atom id (p3:// handle without the scheme)"},
            },
            "required": ["atomId"],
        },
    ),
    Tool(
        name="correct",
        description="Correct an atom: write a new atom and supersede the old one in one flow. The old atom stays readable, chained to its successor. Returns the new atom id.",
        inputSchema={
            "type": "object",
            "properties": {
                "oldAtomId":  {"type": "string", "description": "Id of the atom being corrected"},
                "newText":    {"type": "string", "description": "The corrected atom body"},
                "provenance": {"type": "object",
                               "description": "Optional provenance {source, agent?, sessionId?, sourceRef?}; source defaults to explicit-emit"},
            },
            "required": ["oldAtomId", "newText"],
        },
    ),
    Tool(
        name="pin",
        description="Pin an atom (attach a durable pin facet) so it is marked important. Idempotent.",
        inputSchema={
            "type": "object",
            "properties": {
                "atomId": {"type": "string", "description": "The atom id to pin"},
            },
            "required": ["atomId"],
        },
    ),
    Tool(
        name="recall_records",
        description="Versioned structured recall records for kernel-owned memory adapters.",
        inputSchema=RECALL_RECORDS_INPUT_SCHEMA,
        outputSchema=RECALL_RECORDS_OUTPUT_SCHEMA,
    ),
]

TOOLS = COMPAT_TOOLS + NATIVE_TOOLS

HANDLERS = {
    # compat
    "engram_emit_atom": handle_emit_atom,
    "engram_emit_discovery": handle_emit_discovery,
    "engram_emit_failure": handle_emit_failure,
    "engram_emit_narrative": handle_emit_narrative,
    "engram_emit_snapshot": handle_emit_snapshot,
    "pensive_recall": handle_pensive_recall,
    "pensive_analytics": handle_pensive_analytics,
    # natives
    "recall": handle_recall,
    "recall_records": handle_recall_records,
    "history": handle_history,
    "correct": handle_correct,
    "pin": handle_pin,
}


# --------------------------------------------------------------------------- #
# Dispatch + server assembly                                                    #
# --------------------------------------------------------------------------- #


def dispatch(ctx, name, args):
    """Run one tool and return ``(result, isError)`` -- the server's inner path.

    This is the single place a handler exception is turned into an error
    response, so the daemon NEVER dies on a tool error: an unknown tool, a bad
    argument, a missing atom all come back as ``(message, True)`` and the next
    call is unaffected. A handler that itself returns an ``error:``-prefixed
    string (there are none today, but the legacy contract allowed it) is also
    flagged. ``args`` may be None (treated as empty).
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return f"error: unknown tool '{name}'", True
    try:
        result = handler(ctx, args or {})
    except Exception as exc:  # noqa: BLE001 -- loud-but-contained: report, keep serving
        return f"error: {exc}", True
    isError = isinstance(result, str) and result.startswith("error:")
    return result, isError


def buildServer(ctx):
    """Build the low-level MCP ``Server`` bound to ``ctx``.

    Uses the low-level API (not FastMCP) so the compat tools' ``inputSchema`` is
    the verbatim legacy dict, not one derived from a Python signature. ``call_tool``
    routes through :func:`dispatch`, so it can never raise -- the returned
    ``CallToolResult`` carries ``isError`` and the daemon keeps serving.
    """
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name, arguments):
        result, isError = dispatch(ctx, name, arguments or {})
        if (
            name == "recall_records"
            and isinstance(result, str)
            and len(result.encode("utf-8"))
            > MAX_RECALL_RECORDS_CALL_RESULT_BYTES
        ):
            result = _RECALL_RECORDS_OVERSIZE_ERROR
            isError = True
        response = _callToolResult(result, isError)
        if name != "recall_records":
            return response
        responseBytes = len(response.model_dump_json().encode("utf-8"))
        if responseBytes <= MAX_RECALL_RECORDS_CALL_RESULT_BYTES:
            return response
        bounded = _callToolResult(_RECALL_RECORDS_OVERSIZE_ERROR, True)
        boundedBytes = len(bounded.model_dump_json().encode("utf-8"))
        if boundedBytes > MAX_RECALL_RECORDS_CALL_RESULT_BYTES:
            raise ValueError(_RECALL_RECORDS_UNREPRESENTABLE_CAP_ERROR)
        return bounded

    return server
