"""MCP server (compat names + v3 natives) served from the resident daemon.

Risk model (what could silently break, and the test that catches it):

- **Compat schema drift.** A live agent calls ``pensive_recall`` /
  ``engram_emit_*`` with a fixed argument shape; if our tool ``inputSchema`` or
  the emit RESULT string drifts from the production server the agent breaks
  mid-shadow. Caught by ``test_compat_tool_schemas_are_verbatim`` (the exact
  legacy schemas, pinned) and the emit-result-format tests (the exact
  ``atom [..] emitted (emission_id: <uuid>): <principle> (ok)`` shape).
- **Compat answer served from the wrong engine.** ``pensive_recall`` must return
  the legacy listing shape but sourced from the v3 recall engine. Caught by
  ``test_pensive_recall_returns_legacy_shape_from_v3_engine``.
- **Native semantics.** ``recall`` must return the rich v3 payload (not the
  listing); ``history`` the Tier-2 block PLUS the supersession chain; ``correct``
  must create+supersede in one flow and make the old atom surface only chained;
  ``pin`` must be idempotent. One test each, end to end over real models.
- **A tool error must not kill the daemon.** ``correct`` on a missing atom, an
  emit missing a required field, ``pin`` on a missing atom: each returns an MCP
  error response and the NEXT call still succeeds. Caught by the dispatch-level
  sabotage tests.
- **Transport.** The whole thing must actually serve over the real MCP transport
  and shut down cleanly on SIGINT. Caught by the one subprocess smoke test.

Real components end to end: the embedder + reranker load once per session (same
session-fixture pattern as test_engine), a fresh store per test. Schema/shape
tests need no model. The tool handlers are exercised through ``dispatch`` -- the
exact path the server's ``call_tool`` takes -- so the "never dies on a tool
error" contract is tested without spawning the daemon for every case.
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from serve.mcp import (
    ServeContext,
    TOOLS,
    COMPAT_TOOLS,
    NATIVE_TOOLS,
    _NULLABLE_TEXT_SCHEMA,
    dispatch,
)
from recall.embedder import Embedder
from recall.payload import SENTINEL_LOW_CONFIDENCE
from store.store import openStore, putAtom, getAtom, facetsOf, edgesTo, supersede

MODEL_ID = "BAAI/bge-small-en-v1.5"

_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# The GPU stack emits two SwigPy DeprecationWarnings on first import under CPython
# 3.14; filter exactly those, mirroring the recall tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
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
    # A ServeContext over a real store + the session embedder. reindex() runs at
    # construction (empty store -> empty index, cheap).
    return ServeContext(store, embedder, MODEL_ID, agent="heph")


def _put(store, text, project="aegis", kind="atom", occurredAt=None):
    atomInput = {
        "text": text,
        "kind": kind,
        "project": project,
        "provenance": {"source": "bulk-import"},
    }
    if occurredAt is not None:
        atomInput["occurredAt"] = occurredAt
    return putAtom(store, atomInput)


def _tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not registered")


# --------------------------------------------------------------------------- #
# Compat: the tool schemas must mirror the production server VERBATIM           #
# --------------------------------------------------------------------------- #

# The production server's tool schemas, extracted read-only from
# ~/Projects/Engram/tools/pensive-mcp-server (mcp 1.27, live). A live agent's
# call sites are shaped to these; any drift here breaks agents during shadow, so
# they are pinned exactly and the shadow phase (Task 13) diffs against them.
_LEGACY_SCHEMAS = {
    "engram_emit_atom": {
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
    "engram_emit_snapshot": {
        "type": "object",
        "properties": {
            "project":    {"type": "string", "description": "Project name"},
            "hypothesis": {"type": "string", "description": "Current working theory"},
            "dead_ends":  {"type": "string", "description": "Comma-separated dead ends", "default": ""},
            "next_steps": {"type": "string", "description": "Comma-separated next steps", "default": ""},
        },
        "required": ["project", "hypothesis"],
    },
    "pensive_recall": {
        "type": "object",
        "properties": {
            "query":   {"type": "string", "description": "What to search for"},
            "project": {"type": "string", "description": "Filter by project name", "default": ""},
            "limit":   {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["query"],
    },
    "engram_emit_discovery": {
        "type": "object",
        "properties": {
            "project":   {"type": "string", "description": "Project name"},
            "principle": {"type": "string", "description": "What was discovered"},
        },
        "required": ["project", "principle"],
    },
    "engram_emit_failure": {
        "type": "object",
        "properties": {
            "project":   {"type": "string", "description": "Project name"},
            "principle": {"type": "string", "description": "What failed and why"},
        },
        "required": ["project", "principle"],
    },
    "engram_emit_narrative": {
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
    "pensive_analytics": {
        "type": "object",
        "properties": {},
    },
}


# The five emit tools carry one property the production server does not: an
# optional `agent`, so several residents sharing this daemon accumulate distinct
# histories. It is subtracted below rather than folded into _LEGACY_SCHEMAS,
# which keeps the two facts separate: what production shipped, and what we added
# on purpose. The pin therefore still fires on any OTHER drift, including a
# second uninvited property.
_AGENT_STAMPED_TOOLS = frozenset({
    "engram_emit_atom",
    "engram_emit_discovery",
    "engram_emit_failure",
    "engram_emit_narrative",
    "engram_emit_snapshot",
})


def test_compat_tool_schemas_are_verbatim():
    # Every legacy tool is present under its exact name with its exact inputSchema,
    # modulo the one deliberate additive property above.
    # A drift here is a broken agent in shadow, so the whole dict is compared.
    for name, schema in _LEGACY_SCHEMAS.items():
        served = dict(_tool(COMPAT_TOOLS, name).inputSchema)
        if name in _AGENT_STAMPED_TOOLS:
            properties = dict(served["properties"])
            assert properties.pop("agent", None) == _NULLABLE_TEXT_SCHEMA, (
                f"{name}: agent must be the shared nullable-text schema")
            served["properties"] = properties
        assert served == schema, f"{name} inputSchema drifted from legacy"
    # COMPAT_TOOLS is EXACTLY the seven legacy tools, nothing more.
    assert {t.name for t in COMPAT_TOOLS} == set(_LEGACY_SCHEMAS)


def test_native_tools_present_with_required_fields():
    names = {t.name for t in NATIVE_TOOLS}
    assert names == {"recall", "history", "correct", "pin", "recall_records"}, (
        f"native-tool set rule violated: names={sorted(names)!r}"
    )
    expectedRequired = {
        "recall": ["query"],
        "recall_records": ["query"],
        "history": ["atomId"],
        "correct": ["oldAtomId", "newText"],
        "pin": ["atomId"],
    }
    for name, required in expectedRequired.items():
        actual = _tool(NATIVE_TOOLS, name).inputSchema["required"]
        assert actual == required, (
            f"native-required-fields rule violated: tool={name!r} "
            f"actual={actual!r} expected={required!r}"
        )


def test_tools_is_compat_plus_natives_no_overlap():
    assert TOOLS == COMPAT_TOOLS + NATIVE_TOOLS
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))            # no duplicate tool names


# --------------------------------------------------------------------------- #
# Step 1: pensive_recall returns the LEGACY shape, served from the v3 engine    #
# --------------------------------------------------------------------------- #


def test_pensive_recall_returns_legacy_shape_from_v3_engine(ctx, _rerankerWarm):
    for i in range(6):
        _put(ctx.store, f"acoustic modems trade range for data rate at station {i}",
             project="aegis")
    _put(ctx.store, "unrelated note about titanium hull biofouling", project="aegis")
    ctx.reindex()

    text, isError = dispatch(ctx, "pensive_recall",
                             {"query": "acoustic modem range data rate"})

    assert isError is False
    # The exact legacy envelope: "Found N memories:\n" then "- [NN%] (src) summary"
    # lines. (The production server builds precisely this.)
    assert text.startswith("Found ")
    first = text.split("\n", 1)[0]
    assert re.fullmatch(r"Found \d+ memories:", first)
    listing = [ln for ln in text.split("\n") if ln.startswith("- ")]
    assert listing                                     # at least one hit
    for ln in listing:
        assert re.match(r"- \[\d+%\] ", ln)            # legacy per-hit tag shape
    assert "(aegis)" in text                           # project rendered as src
    assert "range for data rate" in text               # served from the v3 store


def test_pensive_recall_empty_uses_legacy_no_memories_string(ctx, _rerankerWarm):
    _put(ctx.store, "a single unrelated atom", project="aegis")
    ctx.reindex()

    text, isError = dispatch(ctx, "pensive_recall",
                             {"query": "quantum chromodynamics lattice gauge"})

    # Either the recall genuinely finds nothing (legacy no-memories string), or it
    # surfaces a weak untrusted hit; both are legacy-shaped. Pin the no-memories
    # branch by using a project filter that matches nothing.
    text2, _ = dispatch(ctx, "pensive_recall",
                        {"query": "anything", "project": "ghost-project"})
    assert text2 == "No memories found for query: anything"


def test_pensive_recall_project_filter_maps_empty_to_no_filter(ctx, _rerankerWarm):
    # project="" (the legacy default) must mean "no project filter", not "the
    # project literally named empty-string" -- otherwise every default call finds
    # nothing.
    _put(ctx.store, "sonar bathymetry swath mapping run", project="aegis")
    ctx.reindex()
    text, isError = dispatch(ctx, "pensive_recall",
                             {"query": "sonar bathymetry swath", "project": ""})
    assert isError is False
    assert text.startswith("Found ")


# --------------------------------------------------------------------------- #
# Compat emits: exact result strings, written to the v3 store                  #
# --------------------------------------------------------------------------- #


def test_emit_atom_writes_v3_store_and_returns_legacy_result(ctx):
    text, isError = dispatch(ctx, "engram_emit_atom", {
        "project": "pensive",
        "shape": "recall must beat BM25 on the eval gate",
        "approach": "RRF fusion + cross-encoder rerank + trust layer",
        "outcome": "succeeded",
        "reason": "reranking pulls paraphrases the dense signal missed",
        "principle": "joint query-document scoring beats bag-of-words at the top",
        "tags": "recall, rerank",
    })

    assert isError is False
    assert re.fullmatch(
        r"atom \[succeeded\] emitted \(emission_id: " + _UUID_RE +
        r"\): joint query-document scoring beats bag-of-words at the top \(ok\)",
        text,
    )
    # The atom really landed in the v3 store: kind='atom', project in the column,
    # provenance source='explicit-emit', importance 0.0.
    rows = ctx.store._conn.execute(
        "SELECT id, kind, project, importance FROM atoms").fetchall()
    assert len(rows) == 1
    atomId, kind, project, importance = rows[0]
    assert kind == "atom"
    assert project == "pensive"
    assert importance == 0.0
    atom = getAtom(ctx.store, atomId)
    assert atom["provenance"][0]["source"] == "explicit-emit"
    assert atom["provenance"][0]["agent"] == "heph"          # from ctx
    # The composed body carries the reasoning fields for recall.
    assert "joint query-document scoring" in atom["text"]
    assert "approach:" in atom["text"]
    # tags became tag facets.
    tags = {f["value"] for f in facetsOf(ctx.store, atomId) if f["key"] == "tag"}
    assert tags == {"recall", "rerank"}


def test_emit_atom_principle_truncated_to_80_in_result(ctx):
    long_principle = "P" * 200
    text, _ = dispatch(ctx, "engram_emit_atom", {
        "project": "p", "shape": "s", "approach": "a", "outcome": "partial",
        "reason": "r", "principle": long_principle,
    })
    # Legacy slices the principle to 80 chars in the result line.
    assert f"): {'P' * 80} (ok)" in text
    assert "P" * 81 not in text
    assert "[partial]" in text


def test_emit_discovery_and_failure_keep_the_legacy_result_shape(ctx):
    """Renamed 2026-08-12: they no longer DELEGATE. Both shorthands and the full
    form now share one `_writeAtom` path instead of rebuilding an argument dict,
    so this gate holds the frozen compat RESULT string still. What they STORE
    changed deliberately and is gated by test_emit_body_shape.py."""
    d, derr = dispatch(ctx, "engram_emit_discovery",
                       {"project": "pensive", "principle": "spreading activation is sub-ms"})
    assert derr is False
    assert re.match(
        r"atom \[succeeded\] emitted \(emission_id: " + _UUID_RE + r"\): ", d)
    assert "spreading activation is sub-ms" in d

    f, ferr = dispatch(ctx, "engram_emit_failure",
                       {"project": "pensive", "principle": "networkx blew the memory budget"})
    assert ferr is False
    assert re.match(
        r"atom \[failed\] emitted \(emission_id: " + _UUID_RE + r"\): ", f)
    assert "networkx blew the memory budget" in f


def test_emit_narrative_writes_narrative_kind_and_legacy_result(ctx):
    text, isError = dispatch(ctx, "engram_emit_narrative", {
        "project": "pensive",
        "narrative": "I finally saw the rerank gap close on the eval; quiet relief.",
    })
    assert isError is False
    assert re.fullmatch(
        r"Narrative fragment emitted \(emission_id: " + _UUID_RE + r"\)", text)
    kinds = [r[0] for r in ctx.store._conn.execute("SELECT kind FROM atoms").fetchall()]
    assert kinds == ["narrative"]


def test_emit_snapshot_writes_snapshot_kind_and_legacy_result(ctx):
    text, isError = dispatch(ctx, "engram_emit_snapshot", {
        "project": "pensive",
        "hypothesis": "the desync is in the stale dense index, not the trust layer",
        "dead_ends": "blamed rerank, blamed fusion",
        "next_steps": "rebuild index after supersede, re-run gate",
    })
    assert isError is False
    assert text == ("snapshot emitted: the desync is in the stale dense index, "
                    "not the trust layer (ok)")
    kinds = [r[0] for r in ctx.store._conn.execute("SELECT kind FROM atoms").fetchall()]
    assert kinds == ["snapshot"]


# --------------------------------------------------------------------------- #
# Per-caller stamping: several residents share one daemon                       #
# --------------------------------------------------------------------------- #

# PENSIVE_V3_AGENT is read once at daemon start and baked into ServeContext, so
# one daemon can stamp exactly one name. The house runs six residents against
# this daemon; without a per-call agent their atoms are indistinguishable from
# each other and from Heph's. A caller-supplied `agent` therefore wins over the
# daemon-wide default, mirroring what `correct` has always done with its
# `provenance` argument.

_MINIMAL_EMIT_ARGS = {
    "engram_emit_atom": {
        "project": "porchlight", "shape": "s", "approach": "a",
        "outcome": "succeeded", "reason": "r", "principle": "p",
    },
    "engram_emit_discovery": {"project": "porchlight", "principle": "p"},
    "engram_emit_failure": {"project": "porchlight", "principle": "p"},
    "engram_emit_narrative": {"project": "porchlight", "narrative": "n"},
    "engram_emit_snapshot": {"project": "porchlight", "hypothesis": "h"},
}


@pytest.fixture
def makeCtx(tmp_path, embedder):
    """Build a ServeContext with a chosen daemon-wide agent, over its own store.

    Stamping is about several callers sharing ONE daemon, so these tests need
    contexts that differ only in ``agent``; the module ``ctx`` fixture is pinned
    to "heph" and its store is shared with the rest of the test.
    """
    opened = []

    def _make(agent):
        s = openStore(tmp_path / f"stamp-{len(opened)}.db")
        opened.append(s)
        return ServeContext(s, embedder, MODEL_ID, agent=agent)

    try:
        yield _make
    finally:
        for s in opened:
            s.close()


def _soleProvenance(ctx):
    """Provenance of the one atom in this store; loud if there is not exactly one."""
    rows = ctx.store._conn.execute("SELECT id FROM atoms").fetchall()
    assert len(rows) == 1, f"expected exactly one atom, found {len(rows)}"
    return getAtom(ctx.store, rows[0][0])["provenance"][0]


def test_emit_atom_stamps_a_caller_supplied_agent(makeCtx):
    """A caller-supplied agent wins over the daemon-wide default.

    Several residents share one Pensive daemon, so a process-wide
    PENSIVE_V3_AGENT can only ever stamp one of them. Without this the atoms of
    six residents are indistinguishable from each other and from Heph's.
    """
    ctx = makeCtx(None)
    text, isError = dispatch(ctx, "engram_emit_atom", {
        **_MINIMAL_EMIT_ARGS["engram_emit_atom"],
        "agent": "sol-agent",
    })
    assert isError is False, text
    assert _soleProvenance(ctx)["agent"] == "sol-agent"


def test_emit_atom_without_an_agent_is_byte_identical_to_today(makeCtx):
    """The compat contract: absent means absent.

    Mid-shadow callers (the tee forward and every existing session) send no
    agent, and their atoms must keep landing exactly as they do now: no agent
    key at all when the daemon has no PENSIVE_V3_AGENT, and ctx.agent when it
    does.
    """
    unset = makeCtx(None)
    _, isError = dispatch(unset, "engram_emit_atom", _MINIMAL_EMIT_ARGS["engram_emit_atom"])
    assert isError is False
    assert _soleProvenance(unset).get("agent") is None

    configured = makeCtx("heph")
    _, isError = dispatch(configured, "engram_emit_atom", _MINIMAL_EMIT_ARGS["engram_emit_atom"])
    assert isError is False
    assert _soleProvenance(configured)["agent"] == "heph"


def test_a_caller_agent_beats_the_daemon_default(makeCtx):
    ctx = makeCtx("heph")
    _, isError = dispatch(ctx, "engram_emit_atom", {
        **_MINIMAL_EMIT_ARGS["engram_emit_atom"],
        "agent": "grok-agent",
    })
    assert isError is False
    assert _soleProvenance(ctx)["agent"] == "grok-agent"


@pytest.mark.parametrize("junk", ["", "   ", 42, None, ["sol-agent"]])
def test_a_blank_or_non_string_agent_falls_back_rather_than_stamping_junk(makeCtx, junk):
    """Fail soft to the default, never stamp an empty or wrong-typed name.

    An atom stamped "" is worse than one stamped NULL: it looks like an answer.
    """
    ctx = makeCtx("heph")
    _, isError = dispatch(ctx, "engram_emit_atom", {
        **_MINIMAL_EMIT_ARGS["engram_emit_atom"],
        "agent": junk,
    })
    assert isError is False
    assert _soleProvenance(ctx)["agent"] == "heph"


@pytest.mark.parametrize("tool", sorted(_MINIMAL_EMIT_ARGS))
def test_every_emit_tool_stamps_the_caller_agent(makeCtx, tool):
    """All five actually stamp, not just the one with the obvious call site.

    ``engram_emit_discovery`` and ``engram_emit_failure`` do not build their own
    provenance: they rebuild an argument dict and delegate to
    ``handle_emit_atom``, copying a fixed list of keys across. A key missing from
    that list is dropped in silence, so those two would accept an agent, return
    success, and stamp nothing. Schema coverage cannot see that; only a real
    emit can.
    """
    ctx = makeCtx(None)
    _, isError = dispatch(ctx, tool, {**_MINIMAL_EMIT_ARGS[tool], "agent": "fable-agent"})
    assert isError is False
    assert _soleProvenance(ctx)["agent"] == "fable-agent"


def test_every_emit_tool_accepts_agent_and_none_requires_it():
    """All five, and none of them changes its required fields.

    Adding agent to a required list would break every existing caller at once.
    """
    for name in _AGENT_STAMPED_TOOLS:
        schema = _tool(COMPAT_TOOLS, name).inputSchema
        assert "agent" in schema["properties"], name
        assert "agent" not in schema.get("required", []), name


def test_the_agent_property_is_scoped_to_the_emit_tools():
    # Reading and analytics have no author to record. A stray agent field there
    # would be a second, unowned stamping surface.
    for name in ("pensive_recall", "pensive_analytics"):
        assert "agent" not in _tool(COMPAT_TOOLS, name).inputSchema["properties"], name


# --------------------------------------------------------------------------- #
# Kind-scoped reindex: a write rebuilds its own class's index and no other       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tool,args", [
    ("engram_emit_atom", {
        "project": "pensive", "shape": "reindex hot loop",
        "approach": "scope the rebuild", "outcome": "succeeded",
        "reason": "emits never write code atoms",
        "principle": "rebuild only the class the write touched",
    }),
    ("engram_emit_narrative", {
        "project": "pensive",
        "narrative": "the per-emit full rebuild was the whole disease",
    }),
    ("engram_emit_snapshot", {
        "project": "pensive",
        "hypothesis": "the burn is the code-class HNSW rebuild, not the embed",
    }),
])
def test_emit_indexes_the_new_atom_in_memory_and_leaves_code_alone(ctx, tool, args):
    # Every emit writes a memory-class atom (atom/narrative/snapshot); the code
    # class (the bulk document_chunk corpus, past the HNSW threshold in
    # production) has no in-daemon write path besides `correct`. Touching its
    # index per emit is pure waste at full-graph-build cost.
    #
    # ASSERTION CHANGED 2026-08-13, and deliberately STRENGTHENED. This used to
    # assert `ctx.indexes["memory"] is not memoryBefore`, i.e. that a NEW index
    # object had been constructed, which pinned the full rebuild as the
    # mechanism. Emits now maintain the index incrementally (one add, 0.115ms,
    # against a 272ms rebuild that blocked the event loop), so the object is
    # mutated in place and identity is preserved. Identity was never the
    # property that mattered: what matters is that the atom became recallable in
    # its own class and that the other class was not disturbed. That is what is
    # asserted now, and it would catch a no-op implementation that the old
    # identity check could not.
    _put(ctx.store, "def spread(): return activation", kind="document_chunk")
    ctx.reindex()
    codeBefore = ctx.indexes["code"]
    codeIdsBefore = list(ctx.indexes["code"]._atomIds)
    memoryIdsBefore = set(ctx.indexes["memory"]._atomIds)

    _, isError = dispatch(ctx, tool, args)

    assert isError is False
    # the code class is untouched, by object AND by contents
    assert ctx.indexes["code"] is codeBefore
    assert ctx.indexes["code"]._atomIds == codeIdsBefore
    # the memory class gained exactly the atom just written
    memoryIdsAfter = set(ctx.indexes["memory"]._atomIds)
    gained = memoryIdsAfter - memoryIdsBefore
    assert len(gained) == 1, f"expected exactly one new memory atom, got {gained}"
    newId = gained.pop()
    assert getAtom(ctx.store, newId)["status"] == "live"


def _retiredIds(index):
    """Atom ids the index has retired, whichever way it tracks them.

    FlatIndex masks row POSITIONS in `_retired`; HnswIndex drops the key from
    the usearch graph and keeps the slot in `_atomIds`. This normalizes both to
    ids so a test can ask the question once.
    """
    retired = getattr(index, "_retired", None)
    if not retired:
        return set()
    return {index._atomIds[pos] for pos in retired}


def test_correct_rebuilds_only_the_corrected_atoms_class(ctx):
    # `correct` inherits the old atom's kind, so it is the one mutating tool
    # that can touch either class: a memory correction must leave the code
    # index alone, and a code correction must rebuild it (the superseded chunk
    # has to drop out of the dense index).
    chunk = _put(ctx.store, "class HnswIndex: ...", kind="document_chunk")
    note = _put(ctx.store, "the chassis fans are slaved to the GPU sensor")
    ctx.reindex()

    codeBefore = ctx.indexes["code"]
    _, err = dispatch(ctx, "correct", {
        "oldAtomId": note,
        "newText": "the chassis fans follow the GPU temp sensor, not the CPU",
    })
    assert err is False
    # ASSERTION CHANGED 2026-08-13: identity-of-index-object pinned the rebuild
    # mechanism, which incremental maintenance deliberately replaced. What is
    # asserted instead is the property that mattered: the corrected atom drops
    # out of its class's index, its replacement appears, and the OTHER class is
    # untouched in both object and contents.
    assert ctx.indexes["code"] is codeBefore
    assert note in _retiredIds(ctx.indexes["memory"]), (
        "the corrected note is still live in the memory index, so it can "
        "surface beside its own correction")

    codeIdsBefore = list(ctx.indexes["code"]._atomIds)
    memoryBefore = ctx.indexes["memory"]
    memoryIdsBefore = list(memoryBefore._atomIds)
    _, err = dispatch(ctx, "correct", {
        "oldAtomId": chunk,
        "newText": "class HnswIndex: pass",
    })
    assert err is False
    # the code class gained the replacement chunk
    assert set(ctx.indexes["code"]._atomIds) - set(codeIdsBefore), \
        "the corrected chunk's replacement never entered the code index"
    # the memory class was not disturbed by a code-class correction
    assert ctx.indexes["memory"] is memoryBefore
    assert ctx.indexes["memory"]._atomIds == memoryIdsBefore


def test_reindex_default_still_rebuilds_every_class(ctx):
    # Pins the kinds=None path (construction and any caller that cannot name
    # what changed): every class rebuilds, same as the original contract.
    _put(ctx.store, "a bulk chunk", kind="document_chunk")
    _put(ctx.store, "a reasoning note")
    before = dict(ctx.indexes)
    ctx.reindex()
    assert ctx.indexes["code"] is not before["code"]
    assert ctx.indexes["memory"] is not before["memory"]


def test_emitted_atom_is_recallable_after_emit(ctx, _rerankerWarm):
    # The living-memory contract: emit then recall must see it (the emit reindexes
    # so both the lexical and dense signals cover the new atom).
    dispatch(ctx, "engram_emit_atom", {
        "project": "pensive",
        "shape": "how do LiFePO4 packs behave in cold deep water",
        "approach": "derate the pack capacity for 4C bottom temperature",
        "outcome": "succeeded",
        "reason": "cold cuts usable capacity ~30 percent",
        "principle": "size the battery for the cold bottom, not the warm surface",
    })
    text, isError = dispatch(ctx, "pensive_recall",
                             {"query": "cold battery capacity derate deep water"})
    assert isError is False
    assert "size the battery for the cold bottom" in text


# --------------------------------------------------------------------------- #
# Native recall: returns the rich v3 payload, not the legacy listing            #
# --------------------------------------------------------------------------- #


def test_native_recall_returns_v3_payload_not_legacy_listing(ctx, _rerankerWarm):
    for i in range(6):
        _put(ctx.store, f"the extended kalman filter fuses INS and DVL at tick {i}",
             project="aegis")
    ctx.reindex()

    text, isError = dispatch(ctx, "recall",
                             {"query": "extended kalman filter INS DVL fusion"})

    assert isError is False
    # The v3 payload uses the p3:// handle grammar, never the legacy "Found N
    # memories" envelope.
    assert not text.startswith("Found ")
    assert "p3://" in text
    assert "extended kalman filter" in text


def test_native_recall_low_confidence_returns_v3_sentinel(ctx, _rerankerWarm):
    _put(ctx.store, "an aegis note", project="aegis")
    ctx.reindex()
    text, isError = dispatch(ctx, "recall",
                             {"query": "anything", "project": "ghost"})
    assert isError is False
    assert text == SENTINEL_LOW_CONFIDENCE


# --------------------------------------------------------------------------- #
# Native history: Tier-2 block + supersession chain                            #
# --------------------------------------------------------------------------- #


def test_history_returns_tier2_and_supersession_chain(ctx, _rerankerWarm):
    old = _put(ctx.store, "the pressure hull rates to 488 meters of depth", project="aegis")
    ctx.reindex()
    _, cerr = dispatch(ctx, "correct", {
        "oldAtomId": old,
        "newText": "the pressure hull now rates to 500 meters of depth",
    })
    assert cerr is False
    new = edgesTo(ctx.store, old, "supersedes")[0]["srcAtom"]

    text, isError = dispatch(ctx, "history", {"atomId": old})

    assert isError is False
    # Tier-2 portion: the atom's own body renders.
    assert "the pressure hull rates to 488 meters of depth" in text
    # The supersession chain lists both ends, oldest to newest, with status.
    assert "supersession chain" in text
    assert f"p3://{old}" in text
    assert f"p3://{new}" in text
    assert "[superseded]" in text                       # old atom's status
    assert "[live]" in text                             # new atom's status


def test_history_missing_atom_errors(ctx):
    text, isError = dispatch(ctx, "history", {"atomId": "01NONEXISTENTATOMIDXXXXXXX"})
    assert isError is True
    assert "error:" in text


# --------------------------------------------------------------------------- #
# Native correct: create + supersede in one flow, old surfaces only chained     #
# --------------------------------------------------------------------------- #


def test_correct_supersedes_and_recall_shows_current_truth(ctx, _rerankerWarm):
    old = _put(ctx.store, "the survey line spacing is 100 meters", project="aegis")
    ctx.reindex()

    text, isError = dispatch(ctx, "correct", {
        "oldAtomId": old,
        "newText": "the survey line spacing is 40 meters after the overlap fix",
        "provenance": {"source": "explicit-emit", "agent": "heph"},
    })
    assert isError is False
    new = edgesTo(ctx.store, old, "supersedes")[0]["srcAtom"]
    assert f"p3://{new}" in text                         # returns the new id

    # The store recorded the supersession: old is superseded, new is live.
    assert getAtom(ctx.store, old)["status"] == "superseded"
    assert getAtom(ctx.store, new)["status"] == "live"
    # The corrected atom inherits the old atom's project (a correction stays put).
    assert getAtom(ctx.store, new)["project"] == "aegis"

    # correct reindexes over LIVE atoms, so recall returns the CURRENT truth: the
    # live successor, trusted. The retired "100 meters" claim is not standalone --
    # it is now history, reachable through history()/the supersession chain.
    rtext, _ = dispatch(ctx, "recall", {"query": "survey line spacing meters"})
    assert "40 meters after the overlap fix" in rtext
    assert "100 meters" not in rtext


def test_superseded_atom_surfaces_only_chained_in_server_recall(ctx, _rerankerWarm):
    # The trust-layer chaining end to end THROUGH the server recall handler: an
    # atom superseded AFTER the index was built (a stale in-flight index, the
    # realistic race) still comes back on the dense signal, and must surface only
    # chained to its live successor -- never as a standalone trusted result.
    old = _put(ctx.store, "the pressure hull rates to 488 meters of depth", project="aegis")
    new = _put(ctx.store, "the pressure hull now rates to 500 meters of depth", project="aegis")
    ctx.reindex()                                        # both live, both indexed
    supersede(ctx.store, old, new, {"source": "explicit-emit"})   # index now stale for old

    text, isError = dispatch(ctx, "recall", {"query": "pressure hull depth rating"})

    assert isError is False
    assert f"superseded by p3://{new}" in text           # old surfaces chained
    assert "500 meters" in text                          # live successor body present


# --------------------------------------------------------------------------- #
# Sabotage: a tool error is an MCP error response; the daemon keeps serving      #
# --------------------------------------------------------------------------- #


def test_correct_nonexistent_atom_errors_then_keeps_serving(ctx, _rerankerWarm):
    _put(ctx.store, "a live aegis note about sonar", project="aegis")
    ctx.reindex()

    bad, isError = dispatch(ctx, "correct",
                            {"oldAtomId": "01MISSINGXXXXXXXXXXXXXXXXXX", "newText": "x"})
    assert isError is True
    assert "error:" in bad
    # No orphan atom was written by the failed correct (old checked before put).
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1

    # The very next call still works -- the error did not wedge the context.
    ok, okErr = dispatch(ctx, "pensive_recall", {"query": "sonar note"})
    assert okErr is False


def test_emit_missing_required_field_errors_cleanly(ctx):
    text, isError = dispatch(ctx, "engram_emit_atom", {
        "project": "p", "shape": "s", "approach": "a",
        "outcome": "succeeded", "reason": "r",   # 'principle' missing
    })
    assert isError is True
    assert "principle" in text                          # names the missing field
    # Nothing was written.
    assert ctx.store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_pin_is_idempotent_and_missing_atom_errors(ctx):
    aid = _put(ctx.store, "a pinnable atom", project="aegis")

    one, e1 = dispatch(ctx, "pin", {"atomId": aid})
    two, e2 = dispatch(ctx, "pin", {"atomId": aid})       # second pin: no-op
    assert e1 is False and e2 is False
    pins = [f for f in facetsOf(ctx.store, aid) if f["key"] == "pin"]
    assert pins == [{"key": "pin", "value": "true"}]      # exactly one, not two

    bad, e3 = dispatch(ctx, "pin", {"atomId": "01MISSINGXXXXXXXXXXXXXXXXXX"})
    assert e3 is True
    assert "error:" in bad
    # Daemon keeps serving after the failed pin.
    ok, e4 = dispatch(ctx, "pin", {"atomId": aid})
    assert e4 is False


def test_unknown_tool_is_an_error_response(ctx):
    text, isError = dispatch(ctx, "nonexistent_tool", {})
    assert isError is True
    assert "unknown tool" in text


def test_pensive_analytics_returns_v3_store_shape_json(ctx):
    import json as _json

    _put(ctx.store, "one atom", project="aegis", kind="atom")
    _put(ctx.store, "a narrative", project="aegis", kind="narrative")
    text, isError = dispatch(ctx, "pensive_analytics", {})

    assert isError is False
    data = _json.loads(text)                             # legacy shape: JSON payload
    assert data["store"]["totalAtoms"] == 2
    assert data["store"]["byKind"] == {"atom": 1, "narrative": 1}
    assert data["store"]["byStatus"]["live"] == 2


def test_serve_context_holds_class_indexes_and_recall_prefers_memory(tmp_path):
    from recall.embedder import Embedder
    from serve.mcp import ServeContext, handle_recall
    from store.store import openStore, putAtom

    store = openStore(tmp_path / "mem.db")
    try:
        for i in range(30):
            putAtom(store, {
                "text": f"def route_{i}(r): return dispatch(r, {i})",
                "kind": "document_chunk", "project": "aegis",
                "importance": 0.0, "provenance": {"source": "bulk-import"},
            })
        putAtom(store, {
            "text": "we chose Authelia forward-auth as the default web perimeter",
            "kind": "atom", "project": "aegis",
            "importance": 0.0, "provenance": {"source": "claude-code"},
        })
        ctx = ServeContext(store, Embedder("BAAI/bge-small-en-v1.5"),
                           "BAAI/bge-small-en-v1.5")
        assert set(ctx.indexes.keys()) == {"memory", "code"}
        out = handle_recall(ctx, {"query": "what did we choose for web auth",
                                  "k": 3})
        # Enrichment on the native handler must not disturb memory results.
        assert "Authelia" in out
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# End-to-end smoke: serve over the real transport, clean SIGINT shutdown         #
# --------------------------------------------------------------------------- #


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_structured_recall_round_trips_over_temporary_http_transport(tmp_path):
    # Seed a tiny store the daemon will serve.
    dbPath = tmp_path / "smoke.db"
    seed = openStore(dbPath)
    _put(seed, "acoustic modems trade range for data rate at the surface buoy",
         project="aegis")
    seed.close()

    port = _free_port()
    srcRoot = str(Path(__file__).resolve().parents[2] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = srcRoot + os.pathsep + env.get("PYTHONPATH", "")
    env["PENSIVE_V3_STORE"] = str(dbPath)
    env["PENSIVE_V3_PORT"] = str(port)
    # Force CPU in the subprocess so it does not contend for VRAM with the
    # session-fixture models already resident in the test process.
    env["HIP_VISIBLE_DEVICES"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""

    logPath = tmp_path / "daemon.log"
    with logPath.open("w+b") as daemonLog:
        proc = subprocess.Popen(
            [sys.executable, "-m", "serve.daemon"],
            cwd=srcRoot, env=env,
            stdout=daemonLog, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_port("127.0.0.1", port, proc, daemonLog, timeout=240)
            result = _run_smoke_client(port)
        finally:
            rc = _shutdown(proc)

    (recall_text, emit_text, records_text, records_value, recall_structured,
     bad_bool_error, bad_unknown_error) = result
    assert recall_text.startswith("Found ") or recall_text == SENTINEL_LOW_CONFIDENCE, (
        f"legacy-recall transport rule violated: text={recall_text!r}"
    )
    assert re.search(
        r"atom \[succeeded\] emitted \(emission_id: " + _UUID_RE + r"\).*\(ok\)",
        emit_text,
    ), f"legacy-emit transport rule violated: text={emit_text!r}"
    assert recall_structured is None, (
        f"existing-string transport rule violated: structuredContent={recall_structured!r}"
    )
    assert records_value["schemaVersion"] == 1, (
        f"structured-version transport rule violated: value={records_value!r}"
    )
    assert json.loads(records_text) == records_value, (
        f"structured-fallback transport rule violated: text={records_text!r} "
        f"value={records_value!r}"
    )
    assert bad_bool_error is True and bad_unknown_error is True, (
        f"transport-input validation rule violated: boolError={bad_bool_error} "
        f"unknownError={bad_unknown_error}"
    )
    # Clean SIGINT shutdown: the process exited on the interrupt, not a kill.
    assert rc == 0, f"clean-SIGINT transport rule violated: returnCode={rc}"


def _subprocess_output(logFile):
    offset = logFile.tell()
    logFile.seek(0)
    output = logFile.read().decode(errors="replace")
    logFile.seek(offset)
    return output


def _wait_for_port(host, port, proc, logFile, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = _subprocess_output(logFile)
            raise AssertionError(f"daemon exited early (rc={proc.returncode}):\n{out}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise AssertionError(
        f"daemon did not bind the port in time; output:\n{_subprocess_output(logFile)}"
    )


def _run_smoke_client(port):
    import anyio
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    async def go():
        url = f"http://127.0.0.1:{port}/mcp"
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                assert {"pensive_recall", "recall", "recall_records"} <= tools, (
                    f"transport-tool-registration rule violated: tools={sorted(tools)!r}"
                )
                r = await session.call_tool(
                    "pensive_recall", {"query": "acoustic modem range data rate"})
                e = await session.call_tool("engram_emit_atom", {
                    "project": "pensive", "shape": "smoke", "approach": "call it",
                    "outcome": "succeeded", "reason": "it served",
                    "principle": "the transport round-trips end to end",
                })
                badBool = await session.call_tool(
                    "recall_records", {"query": "q", "k": True})
                badUnknown = await session.call_tool(
                    "recall_records", {"query": "q", "surprise": 1})
                records = await session.call_tool("recall_records", {
                    "query": "acoustic modem range data rate", "tokenBudget": 8000,
                })
                return (
                    r.content[0].text,
                    e.content[0].text,
                    records.content[0].text,
                    records.structuredContent,
                    r.structuredContent,
                    badBool.isError,
                    badUnknown.isError,
                )

    return anyio.run(go)


def _shutdown(proc):
    # House process rule: SIGINT first, escalate only if it will not stop.
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
        return proc.returncode
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        return proc.returncode
