"""The distiller: turn tailed Claude Code transcripts into house-format atoms.

Phase 4 ambient capture. A transcript is WORK EXHAUST; the distiller reads a
session's newly-tailed deltas, segments them (stage 1, :mod:`ambient.segment`),
and for each candidate span either passes an explicit emit straight through or runs
the span through a local model (stage 2) and dedups the result before it lands in
the v3 store. Dedup lives HERE, not in the tee (the Task 13 tee faithfully mirrors
the old path and never dedups); the distiller is the one place capture is made
idempotent and storm-proof.

Two stages:

  1. **Segmentation** (cheap heuristics) picks candidate spans and flags explicit
     ``engram_emit_*`` calls.
  2. **Distillation** (the local model) rewrites a heuristic span into ONE
     house-format atom. Explicit emits are already curated and BYPASS this stage.

Per-span flow under the default ``"distill"`` policy:

  - explicit emit  -> compose the atom from the emit args, write straight through
                      (source='claude-code'), no model, no dedup. Highest trust.
  - already seen    -> if an atom already carries this exact span's provenance
    (same span)       ``sourceRef``, this is a re-tail: bump its importance, do NOT
                      re-summarize or re-insert. (Anti-storm, exact case.)
  - heuristic span  -> stage-2 model summarize -> dedup against recent atoms ->
                      near-dup? bump the match's importance : insert a new atom.
                      (Anti-storm, near-verbatim case.)

Policy is PER-SOURCE, read off ``transcriptSource["policy"]``, not hardcoded. This
is the deliberate seam for v3.1:

  - ``"distill"``  (this task, the Claude Code source): the flow above -- summarize,
    dedup-merge, bump on near-dupes.
  - ``"verbatim"`` (THE V3.1 SEAM, not wired to any real source in v3.0): insert
    EVERY span exactly as-is -- no stage-2 model, no dedup-merge, duplicates
    preserved. This is the branch bulk PERSONAL imports (chat/email exports, a
    person's own words) will fill, where preserving the exact phrasing and every
    repetition is the whole point. It is exercised by a unit test so it cannot rot,
    but no v3.0 source sets it.

The transcript / delta shapes are documented in :mod:`ambient.segment`. A
``transcriptSource`` is::

    {
      "sessionId": "<id>",
      "policy": "distill" | "verbatim",     # default "distill"
      "source": "claude-code",              # provenance source (default)
      "project": "<name>" | None,           # optional, from session-replay project
      "agent": "<name>" | None,             # optional
      "deltas": [ <transcriptDelta>, ... ], # the new slices to distill
    }

The stage-2 model is reached through a narrow :class:`ModelClient` duck type
(``summarizeSpan(spanText) -> {"text", "kind"?} | str``). :class:`OrnithModelClient`
is the real implementation against the local llama-server OpenAI-compatible
endpoint that backs the Hermes/ornith stack; tests inject a deterministic fake, so
nothing here depends on a live 35B.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from store.store import putAtom, addFacet
from ambient.segment import segment, KIND_EXPLICIT_EMIT
from ambient.dedup import dedup

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from pensive.mega_extract import MegaExtractor  # noqa: E402  (path set above)
from pensive.patterns import REAL_DATA_PATTERNS, build_pattern_set  # noqa: E402

__all__ = [
    "distill",
    "OrnithModelClient",
    "DISTILL_POLICY",
    "VERBATIM_POLICY",
    "SOURCE_CLAUDE_CODE",
    "IMPORTANCE_BUMP",
    "IMPORTANCE_CAP",
    "ORNITH_BASE_URL",
    "ORNITH_MODEL",
]

# --- policies + source ------------------------------------------------------ #

DISTILL_POLICY = "distill"
VERBATIM_POLICY = "verbatim"

# Provenance source for atoms captured from Claude Code transcripts (schema enum).
# Both distilled atoms and explicit-emit passthroughs use it -- they share an
# origin (the CC transcript); the passthrough is distinguished by carrying the
# agent's verbatim-composed text, not a model summary.
SOURCE_CLAUDE_CODE = "claude-code"

# --- importance accrual ----------------------------------------------------- #

# How much a near-dup / re-tail raises the existing atom's importance. Small: a
# restatement is weak evidence, but repeated restatement adds up.
IMPORTANCE_BUMP = 0.05

# The ceiling importance accrues to. Matches recall.fusion.importanceFactor's own
# min(importance, 1.0) clamp -- importance above 1.0 has ZERO ranking effect, so
# capping here keeps a hot span from running importance to infinity while losing
# nothing. Feeding one span forever tops out at IMPORTANCE_CAP, never beyond.
IMPORTANCE_CAP = 1.0

# --- the real local-model endpoint (discovered read-only, see report) ------- #
#
# From ~/Projects/hermes-agent/AEGIS-INTEGRATION.md: llama-server hosts
# ornith-1.0-35b Q4_K_M on 127.0.0.1:8000, OpenAI-compatible /v1. This is the same
# endpoint the Hermes stack uses. UNTESTED-LIVE here: the tests never hit it (they
# inject a fake); confirm against a running server before relying on it.
ORNITH_BASE_URL = "http://127.0.0.1:8000/v1"
ORNITH_MODEL = "ornith-1.0-35b-Q4_K_M.gguf"

_HOUSE_SYSTEM_PROMPT = (
    "You distill a slice of an engineering work transcript into ONE memory atom.\n"
    "Write a compact house-format atom: the problem shape, what was tried, the "
    "outcome and why, and the transferable principle. Plain and concrete, no "
    "filler, no rule-of-three padding, no marketing cadence. If the slice carries "
    "no durable insight, reply with exactly: SKIP."
)


class OrnithModelClient:
    """Stage-2 summarizer against the local llama-server OpenAI-compatible endpoint.

    UNTESTED-LIVE: implemented against the documented endpoint
    (:data:`ORNITH_BASE_URL`, model :data:`ORNITH_MODEL`) discovered read-only from
    the Hermes integration notes, but the test suite never instantiates it -- it
    injects a fake. Stdlib ``urllib`` only, so importing the distiller pulls in no
    HTTP dependency.
    """

    def __init__(self, baseUrl=ORNITH_BASE_URL, model=ORNITH_MODEL,
                 timeout=120, maxTokens=512):
        self.baseUrl = baseUrl.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.maxTokens = maxTokens

    def summarizeSpan(self, spanText):
        """Summarize one span -> ``{"text": <house-format atom>, "kind": "atom"}``.

        Returns ``{"text": None}`` when the model replies ``SKIP`` (the slice has no
        durable insight); the distiller drops such spans. Raises on a transport or
        protocol error -- the ambient loop, not this client, decides how to survive
        a down model (the distiller skips a span whose summarization raises)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _HOUSE_SYSTEM_PROMPT},
                {"role": "user", "content": spanText},
            ],
            "max_tokens": self.maxTokens,
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.baseUrl}/chat/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        if content == "SKIP" or not content:
            return {"text": None, "kind": "atom"}
        return {"text": content, "kind": "atom"}


# --------------------------------------------------------------------------- #
# Store-write helpers                                                          #
# --------------------------------------------------------------------------- #


def _spanTextDigest(span):
    return hashlib.blake2b(
        str(span.get("text", "")).encode("utf-8"),
        digest_size=8,
    ).hexdigest()


def _spanRef(span):
    """The provenance ``sourceRef`` for a span.

    Trusted offsets use ``<sessionId>#<offset>.<textDigest>``; synthetic refs use
    ``<sessionId>#synthetic.<offset>.<textDigest>``. The content digest keeps exact
    idempotency byte-stable: a re-tail of the same span maps to the same ref, while
    different text at a reused offset lands as a distinct atom."""
    if span.get("_sourceRef") is not None:
        return span["_sourceRef"]
    sessionId = span.get("sessionId")
    offset = span.get("offset")
    if sessionId is None or offset is None:
        return None
    if str(offset) == "None" or str(offset).startswith("None."):
        return None
    return f"{sessionId}#{offset}.{_spanTextDigest(span)}"


def _syntheticSourceRef(span):
    sessionId = span.get("sessionId")
    if sessionId is None:
        return None
    return f"{sessionId}#synthetic.{span.get('offset')}.{_spanTextDigest(span)}"


def _existingBySourceRef(store, sourceRef):
    """The id of an existing atom whose provenance carries ``sourceRef``, or None.

    This is the exact-span idempotency probe: if a span was already captured, its
    provenance row already points at this ref. Returns the atom id so the caller can
    bump (distilled) or skip (explicit) rather than insert a second copy."""
    if sourceRef is None:
        return None
    row = store._conn.execute(
        "SELECT atom_id FROM provenance WHERE source_ref = ? LIMIT 1",
        (sourceRef,),
    ).fetchone()
    return row[0] if row else None


def _accrueImportance(store, atomId, increment=IMPORTANCE_BUMP, cap=IMPORTANCE_CAP):
    """Raise a live atom's importance by ``increment``, clamped to ``cap``.

    The anti-storm bump: a re-tail or near-dup is reinforcement, not a new memory,
    so the existing atom gets more important instead of a duplicate being written.
    ``MIN(cap, importance + increment)`` is a scalar min in SQLite, so importance
    can never exceed ``cap`` no matter how many times a span recurs. Only touches
    live atoms (a retired atom is not reinforced). One committed UPDATE."""
    conn = store._conn
    try:
        conn.execute(
            "UPDATE atoms SET importance = MIN(?, importance + ?) "
            "WHERE id = ? AND status = 'live'",
            (cap, increment, atomId),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _provenance(source, span):
    """Provenance dict for a span-derived atom: the source, the session, and the
    span pointer. ``agent`` is added by the caller when the source knows it."""
    return {
        "source": source.get("source", SOURCE_CLAUDE_CODE),
        "sessionId": span.get("sessionId"),
        "sourceRef": _spanRef(span),
    }


def _newExtractor():
    return MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))


def _addEntityFacets(store, atomId, extractor, text):
    labels = {label for label, _etype in extractor.extract(text)}
    for label in labels:
        addFacet(store, atomId, "entity", label)


def _insertAtom(store, source, span, text, kind, extractor):
    """Insert one atom for ``span`` with body ``text`` and return its id.

    Importance starts at 0.0 (earned, not assigned) -- accrual raises it later. The
    source's ``project``/``agent`` ride along when present. Store errors propagate
    loud (the decades rule: a failed write is real, not swallowed)."""
    prov = _provenance(source, span)
    agent = source.get("agent")
    if agent:
        prov["agent"] = agent
    atomId = putAtom(store, {
        "text": text,
        "kind": kind,
        "project": source.get("project"),
        "importance": 0.0,
        "provenance": prov,
    })
    _addEntityFacets(store, atomId, extractor, text)
    return atomId


# --- explicit-emit composition (mirrors the production emit shapes) --------- #
#
# An explicit emit is the agent's own curated atom. We reproduce the essential
# text of each emit type locally (rather than routing through serve.mcp, which
# would couple ambient->serve and stamp source='explicit-emit'). The shapes are
# stable -- they mirror the production pensive-mcp-server tools.

_EMIT_KIND = {
    "engram_emit_atom": "atom",
    "engram_emit_discovery": "atom",
    "engram_emit_failure": "atom",
    "engram_emit_narrative": "narrative",
    "engram_emit_snapshot": "snapshot",
}


def _composeEmitText(tool, args):
    """Compose the atom body for an explicit ``engram_emit_*`` call from its args.

    Faithful-but-minimal: keeps the field the emit is ABOUT prominent (principle /
    narrative / hypothesis) so the passthrough atom reads like what the agent meant
    to record. Returns ``(text, kind)``; an unknown tool falls back to a generic
    atom body."""
    kind = _EMIT_KIND.get(tool, "atom")
    if tool == "engram_emit_narrative":
        text = (args.get("narrative") or "").strip()
    elif tool == "engram_emit_snapshot":
        lines = [f"hypothesis: {(args.get('hypothesis') or '').strip()}"]
        if args.get("dead_ends"):
            lines.append(f"dead ends: {args['dead_ends']}")
        if args.get("next_steps"):
            lines.append(f"next steps: {args['next_steps']}")
        text = "\n".join(lines)
    elif tool == "engram_emit_atom":
        parts = [(args.get("shape") or "").strip()]
        if args.get("approach"):
            parts.append(f"approach: {args['approach'].strip()}")
        if args.get("outcome") or args.get("reason"):
            parts.append(
                f"outcome: {(args.get('outcome') or '').strip()}. "
                f"{(args.get('reason') or '').strip()}")
        if args.get("principle"):
            parts.append(f"principle: {args['principle'].strip()}")
        text = "\n".join(p for p in parts if p.strip())
    else:  # discovery / failure / unknown -> principle-centered
        label = "failed approach" if tool == "engram_emit_failure" else "discovery"
        principle = (args.get("principle") or "").strip()
        text = f"{label}: {principle}" if principle else label
    narrative = args.get("narrative")
    if tool != "engram_emit_narrative" and isinstance(narrative, str) and narrative.strip():
        text = f"{text}\n\n{narrative.strip()}"
    return text, kind


def _emitTags(args):
    tags = args.get("tags")
    if isinstance(tags, str):
        candidates = tags.split(",")
    elif isinstance(tags, list):
        candidates = tags
    else:
        return []
    return list(dict.fromkeys(
        tag.strip()
        for tag in candidates
        if isinstance(tag, str) and tag.strip()
    ))


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #


def _summaryText(summary):
    """Normalize a ModelClient result to ``(text, kind)``. The client may return a
    dict ``{"text", "kind"?}`` or a bare string; a ``None``/blank text means the
    model declined (SKIP) and the span is dropped."""
    if isinstance(summary, dict):
        text = summary.get("text")
        kind = summary.get("kind", "atom")
    else:
        text = summary
        kind = "atom"
    if not isinstance(text, str) or not text.strip():
        return None, None
    return text.strip(), kind


def _hasTrustedSourceRef(span):
    return bool(span.get("_sourceRefTrusted")) and _spanRef(span) is not None


def _distillSpan(store, source, span, model, embedder, extractor, result):
    """Process one heuristic span under the ``distill`` policy (mutates ``result``).

    Order matters for storm-proofing and cost: the exact-span idempotency check runs
    BEFORE the model, so a re-tail of a span already captured bumps importance
    without paying for another summarization; only a genuinely new span is
    summarized, then dedup'd against recent atoms."""
    ref = _spanRef(span)
    existing = _existingBySourceRef(store, ref) if _hasTrustedSourceRef(span) else None
    if existing is not None:
        # Exact same span, seen again (a re-tail / overlapping delta): reinforce the
        # atom we already wrote, never a second copy.
        _accrueImportance(store, existing)
        result["bumped"] += 1
        return

    try:
        summary = model.summarizeSpan(span["text"])
    except Exception:  # noqa: BLE001 -- a down/erroring model must not kill the batch
        # The transcript is durable; the span can be re-tailed once the model is
        # back. Skip it now rather than aborting every later span in this delta.
        result["skipped"] += 1
        return

    text, kind = _summaryText(summary)
    if text is None:
        result["skipped"] += 1   # model declined (SKIP) -- no durable insight
        return

    verdict = dedup(store, embedder, text)
    if verdict["isDup"]:
        _accrueImportance(store, verdict["nearId"])
        result["bumped"] += 1
        return

    atomId = _insertAtom(store, source, span, text, kind, extractor)
    result["inserted"] += 1
    result["atomIds"].append(atomId)


def _passthroughSpan(store, source, span, extractor, result):
    """Write an explicit-emit span straight through (bypass stage 2 and dedup).

    Curated by the agent, highest trust: compose the atom from the emit args and
    insert with source='claude-code'. Idempotent on re-tail -- if this exact span's
    provenance is already present, skip silently rather than duplicate a deliberate
    emit (an explicit emit does not accrue importance from a mechanical re-tail)."""
    ref = _spanRef(span)
    if _hasTrustedSourceRef(span) and _existingBySourceRef(store, ref) is not None:
        result["skipped"] += 1
        return
    emit = span["explicitEmit"]
    text, kind = _composeEmitText(emit["tool"], emit["args"])
    if text is None or text == "":
        result["skipped"] += 1
        return
    atomId = _insertAtom(store, source, span, text, kind, extractor)
    for tag in _emitTags(emit["args"]):
        addFacet(store, atomId, "tag", tag)
    result["passthrough"] += 1
    result["atomIds"].append(atomId)


def _verbatimSpan(store, source, span, extractor, result):
    """Insert a span exactly as-is -- THE V3.1 SEAM.

    No stage-2 model, no dedup-merge: the span's own text becomes the atom, and a
    repeated span becomes a SECOND atom (repetition preserved). This is the branch
    person-sourced bulk imports will fill, where the exact words and every recurrence
    are the record. Not reached by any v3.0 source; proven by a unit test so it
    cannot silently rot. An explicit emit under a verbatim source is still composed
    from its args (it is structured, not prose); everything else is inserted raw."""
    if span["kind"] == KIND_EXPLICIT_EMIT and span.get("explicitEmit"):
        emit = span["explicitEmit"]
        text, kind = _composeEmitText(emit["tool"], emit["args"])
    else:
        text, kind = span["text"], "atom"
    if text is None or text == "":
        result["skipped"] += 1
        return
    atomId = _insertAtom(store, source, span, text, kind, extractor)
    result["inserted"] += 1
    result["atomIds"].append(atomId)


def _verbatimTextBlocks(content):
    if isinstance(content, str):
        if content != "":
            yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text != "":
            yield text


def _verbatimSpans(transcriptSource, delta, deltaIndex):
    sessionId = delta.get("sessionId", transcriptSource.get("sessionId"))
    base = delta.get("offset", deltaIndex)
    if base is None:
        base = deltaIndex
    trusted = sessionId is not None and "offset" in delta and delta.get("offset") is not None
    events = delta.get("events")
    if not isinstance(events, list):
        return []
    spans = []
    for eventIndex, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        for local, text in enumerate(_verbatimTextBlocks(event.get("content"))):
            span = {
                "sessionId": sessionId,
                "offset": f"{base}.{eventIndex}.{local}",
                "text": text,
                "kind": "atom",
                "explicitEmit": None,
                "_sourceRefTrusted": trusted,
            }
            if not trusted:
                span["_sourceRef"] = _syntheticSourceRef(span)
            spans.append(span)
    return spans


def distill(store, transcriptSource, model, embedder):
    """Distill a transcript source's new deltas into atoms -> a result summary.

    ``store`` and ``transcriptSource`` are the fixed contract; ``model`` (the
    stage-2 :class:`ModelClient` duck type) and ``embedder`` (the dedup embedder,
    per :func:`ambient.dedup.dedup`'s own fixed signature) are the two collaborators
    the two-stage pipeline needs.

    Segments every delta and processes each span by the source's policy:

    - ``"distill"`` (default): explicit emits pass through untouched; heuristic spans
      are summarized by the model, dedup'd, and either inserted or (on a re-tail or
      near-dup) bumped. Idempotent and storm-proof -- the same span never becomes two
      atoms.
    - ``"verbatim"`` (the v3.1 seam): every span inserted as-is, no model, no
      dedup-merge, repetition preserved.

    Returns ``{"inserted", "bumped", "passthrough", "skipped", "atomIds"}``:
    ``inserted`` new atoms, ``bumped`` importance-accruals on existing atoms,
    ``passthrough`` explicit emits written straight through, ``skipped`` spans
    dropped (blank, model-declined, model-errored, or an idempotent re-tail), and
    the ids of the atoms this call created."""
    result = {"inserted": 0, "bumped": 0, "passthrough": 0, "skipped": 0,
              "atomIds": []}
    if not isinstance(transcriptSource, dict):
        return result

    policy = transcriptSource.get("policy", DISTILL_POLICY)
    if policy not in {DISTILL_POLICY, VERBATIM_POLICY}:
        raise ValueError(f"unknown ambient distiller policy: {policy!r}")
    sessionId = transcriptSource.get("sessionId")
    deltas = transcriptSource.get("deltas") or []
    extractor = _newExtractor()

    for deltaIndex, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            continue
        if policy == VERBATIM_POLICY:
            spans = _verbatimSpans(transcriptSource, delta, deltaIndex)
        else:
            # segment() reads sessionId off the delta; inject the source's so a delta
            # need not repeat it. Missing offsets get the delta index so refs cannot
            # collide across separate tailed slices.
            deltaForSeg = dict(delta)
            if deltaForSeg.get("offset") is None:
                deltaForSeg.pop("offset", None)
            deltaForSeg.setdefault("sessionId", sessionId)
            deltaForSeg.setdefault("offset", deltaIndex)
            trusted = (
                deltaForSeg.get("sessionId") is not None
                and "offset" in delta
                and delta.get("offset") is not None
            )
            spans = segment(deltaForSeg)
            for span in spans:
                span["_sourceRefTrusted"] = trusted
                if not trusted:
                    span["_sourceRef"] = _syntheticSourceRef(span)
        for span in spans:
            if not span.get("text") or (
                policy != VERBATIM_POLICY and not str(span["text"]).strip()
            ):
                result["skipped"] += 1     # malformed / empty span -> skip, never raise
                continue
            if policy == VERBATIM_POLICY:
                _verbatimSpan(store, transcriptSource, span, extractor, result)
            elif span["kind"] == KIND_EXPLICIT_EMIT and span.get("explicitEmit"):
                _passthroughSpan(store, transcriptSource, span, extractor, result)
            else:
                _distillSpan(store, transcriptSource, span, model, embedder, extractor, result)

    return result
