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

- **Natives** (``recall``, ``history``, ``correct``, ``pin``). These expose the v3
  capabilities the legacy names cannot: the rich tiered ``recall`` payload, a
  ``history`` view (Tier-2 neighborhood + supersession chain), a one-flow
  ``correct`` (put + supersede, the trust layer's decades rule), and ``pin``.

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

The tool handlers are plain ``(ctx, args) -> str`` functions dispatched by
:func:`dispatch`, which is exactly the path the server's ``call_tool`` takes --
so the handlers are testable directly against a real store + real models without
spawning the daemon.
"""
import json
import uuid

from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from recall.engine import recall
from recall.embedder import embedMissing
from recall.vector_index import FlatIndex, selectIndex
from recall.payload import assembleTier2
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


# --------------------------------------------------------------------------- #
# Resident context: store + models + a rebuildable index                       #
# --------------------------------------------------------------------------- #


class ServeContext:
    """Everything the handlers need, loaded once and reused.

    Holds the canonical ``store``, a resident ``embedder``, the ``modelId`` they
    agree on, and a ``FlatIndex`` rebuilt from the store's embeddings. ``agent``
    is stamped into emit provenance when the caller is known.

    ``reindex`` embeds any not-yet-embedded live atoms and rebuilds the dense
    index, so an atom written by an emit/correct becomes recallable by BOTH the
    lexical (query-time) and dense (index) signals on the next call. It runs at
    construction and after every mutating tool. The concrete index comes from
    ``selectIndex``, the Task 15 size switch: at shadow scale it returns the exact
    ``FlatIndex`` (serving behavior unchanged), and only past ``HNSW_THRESHOLD``
    would it hand back the approximate HNSW index. A full rebuild per emit is the
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
        self.index = FlatIndex()
        self.recallLogErrors = 0
        self.reindex()

    def reindex(self):
        """Embed missing live atoms and rebuild the dense index from the store.

        The index type is chosen by ``selectIndex`` from the store's current size,
        so the daemon rides the flat->HNSW switch automatically as it grows."""
        embedMissing(self.store, self.embedder)
        self.index = selectIndex(self.store, self.modelId)


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


def _logReturnedRecall(ctx, out, query, sourceRef):
    try:
        logRecall(ctx.store, _returnedAtomIds(out), query=query, sourceRef=sourceRef)
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
        ctx.store, ctx.index, ctx.embedder, query,
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
        ctx.store, ctx.index, ctx.embedder, query,
        project=project, timeScope=timeScope, kinds=kinds,
        k=k, tokenBudget=tokenBudget,
    )
    response = out["payload"]
    _logReturnedRecall(ctx, out, query, "mcp.recall")
    return response


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
    "history": handle_history,
    "correct": handle_correct,
    "pin": handle_pin,
}


# --------------------------------------------------------------------------- #
# Dispatch + server assembly                                                    #
# --------------------------------------------------------------------------- #


def dispatch(ctx, name, args):
    """Run one tool and return ``(text, isError)`` -- the server's inner path.

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
        text, isError = dispatch(ctx, name, arguments or {})
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            isError=isError,
        )

    return server
