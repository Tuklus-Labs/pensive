"""Stage 1: cheap heuristic segmentation of a transcript delta into spans.

The distiller is two-stage. This is the FIRST, deliberately dumb stage: turn a
newly-tailed slice of a Claude Code transcript into a handful of candidate spans
that MIGHT be worth remembering. The real judgement -- what an atom actually says
-- is stage 2's local-model pass (see :mod:`ambient.distiller`). So the bar here
is recall, not precision: keyword/structural cues catch decisions, discoveries,
failures and corrections, and the model downstream throws back the noise.

The one exception that BYPASSES stage 2 entirely is an explicit emit: an
``engram_emit_*`` tool call the agent already made. That is a curated atom, the
highest-trust signal in the pipeline, so it is captured as its own span carrying
the tool name + args and marked :data:`KIND_EXPLICIT_EMIT`; the distiller writes
it straight through without asking a model to "improve" it.

Transcript / delta shape (grounded in a read-only look at the real session-replay
store, ``~/.local/share/session-replay/sessions.db``, and the Claude Code JSONL
event format it records). A ``transcriptDelta`` is a newly-appended slice::

    {
      "sessionId": "<session id>",      # which session this came from
      "offset": <int|str>,              # base position of the slice in the session
      "events": [ <event>, ... ],       # the new turns/events since the last tail
    }

where an ``event`` is a Claude Code message::

    {"role": "user" | "assistant",
     "content": <str> | [ <block>, ... ]}

and a ``block`` is one of::

    {"type": "text",      "text": "<assistant reasoning>"}
    {"type": "tool_use",  "name": "engram_emit_discovery", "input": { ... }}
    {"type": "tool_result", ...}                 # tool output -- ignored here

Only ASSISTANT reasoning is distilled. A user turn is a PROMPT and a tool_result
is EXHAUST -- neither is the agent's own insight, so capturing them as memories
would flood the store with input echoes. (v3.1's verbatim personal-import path is
different, but that is the distiller's policy branch, not this segmenter's job.)

Robustness: this runs on a live tail, so a torn line or a shape we did not model
must not kill the loop. Every event/block is parsed defensively -- a malformed one
is SKIPPED, never raised. The segmenter's contract is best-effort capture.

Stdlib only.
"""

__all__ = [
    "segment",
    "KIND_DECISION",
    "KIND_DISCOVERY",
    "KIND_FAILURE",
    "KIND_CORRECTION",
    "KIND_EXPLICIT_EMIT",
    "HEURISTIC_KINDS",
    "EMIT_TOOL_PREFIX",
]

# --- span kinds ------------------------------------------------------------- #

KIND_DECISION = "decision"
KIND_DISCOVERY = "discovery"
KIND_FAILURE = "failure"
KIND_CORRECTION = "correction"
KIND_EXPLICIT_EMIT = "explicit-emit"

# The four heuristic kinds, in the precedence a paragraph is classified by (first
# match wins). A correction overrides a failure it mentions ("actually, that
# failed approach was wrong"); a failure overrides the discovery/decision framing
# it may share; a discovery ("found that X") outranks the weaker decision cue an
# incidental "I'll" would trip. The order encodes which signal is most load-bearing.
HEURISTIC_KINDS = (KIND_CORRECTION, KIND_FAILURE, KIND_DISCOVERY, KIND_DECISION)

# The MCP emit tool family. Any tool_use whose name starts with this is an
# explicit, already-curated emit and becomes an explicit-emit span.
EMIT_TOOL_PREFIX = "engram_emit"

# --- cue phrases (lowercased substring match; simple on purpose) ------------ #
#
# These are intentionally small and blunt. Stage 2 does the real work, so a cue
# only has to be a plausible signal that a paragraph carries a decision/finding;
# false positives cost a cheap model call, not a bad memory.

_CUES = {
    KIND_CORRECTION: (
        "actually,", "actually i", "actually the", "correction:", "i was wrong",
        "we were wrong", "scratch that", "on second thought", "let me fix",
        "that was wrong", "to correct", "revert",
    ),
    KIND_FAILURE: (
        "failed", "doesn't work", "does not work", "didn't work", "did not work",
        "that broke", "broke the", "dead end", "no luck", "regression",
        "the error was", "still failing", "not working",
    ),
    KIND_DISCOVERY: (
        "turns out", "it turns out", "found that", "i found ", "we found ",
        "the root cause", "root cause:", "discovered", "realized", "the bug is",
        "the bug was", "the issue is", "the issue was", "the problem is",
        "it works because", "the reason is", "the key insight",
    ),
    KIND_DECISION: (
        "i'll ", "we'll ", "let's ", "let us ", "i will ", "we will ",
        "going with", "decided to", "decision:", "the plan is", "the approach is",
        "we should ", "i should ", "plan:", "going to use",
    ),
}


def _classify(paragraph):
    """The heuristic kind of a paragraph, or None if no cue matches.

    Checks the cue lists in :data:`HEURISTIC_KINDS` precedence and returns the
    first kind whose cue appears (case-insensitive substring). None means "not a
    candidate" -- the paragraph is dropped."""
    low = paragraph.lower()
    for kind in HEURISTIC_KINDS:
        for cue in _CUES[kind]:
            if cue in low:
                return kind
    return None


def _textBlocks(content):
    """Yield the assistant TEXT strings in an event's content, defensively.

    ``content`` may be a bare string (older transcript rows) or a list of Claude
    Code content blocks. A non-text block (tool_use/tool_result) or a malformed
    entry is skipped -- best-effort, never raise."""
    if isinstance(content, str):
        if content.strip():
            yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                yield text


def _emitBlocks(content):
    """Yield ``(tool, args)`` for every explicit-emit tool_use in an event.

    A tool_use block whose ``name`` starts with :data:`EMIT_TOOL_PREFIX` is an
    explicit emit. ``args`` is the block's ``input`` dict (empty dict if absent).
    Malformed blocks are skipped."""
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if isinstance(name, str) and name.startswith(EMIT_TOOL_PREFIX):
            args = block.get("input")
            yield name, args if isinstance(args, dict) else {}


def _paragraphs(text):
    """Split assistant text into paragraph candidates on blank lines, trimmed,
    empties dropped. Paragraph granularity keeps a span coherent (a whole thought)
    without the model having to reassemble sentence fragments."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def segment(transcriptDelta):
    """Segment one transcript delta into candidate spans -> ``[candidateSpan]``.

    A ``candidateSpan`` is::

        {
          "sessionId": "<id>",
          "offset": "<delta-offset>.<event-index>.<local-index>",  # locates the span
          "text": "<span text>",              # assistant reasoning, or the emit gist
          "kind": "<decision|discovery|failure|correction|explicit-emit>",
          "explicitEmit": None | {"tool": "<name>", "args": { ... }},
        }

    Two sources of spans, in event order:

    - **Explicit emits** (highest trust): every ``engram_emit_*`` tool_use becomes
      a span with ``kind='explicit-emit'`` and ``explicitEmit`` set. The distiller
      writes these straight through, bypassing the stage-2 model.

    - **Heuristic spans**: each assistant text paragraph that trips a decision /
      discovery / failure / correction cue becomes a span tagged with that kind.
      User turns and tool results are ignored (prompt/exhaust, not insight).

    ``offset`` composes the delta's base offset, the event index, and a within-event
    index so every span has a stable, unique pointer back into the session -- the
    distiller turns it into provenance ``sourceRef`` (``<sessionId>#<offset>``) and
    uses it for idempotency. An empty delta, or one with no cue-matching content,
    yields ``[]``. Malformed events/blocks are skipped, never raised.
    """
    if not isinstance(transcriptDelta, dict):
        return []
    sessionId = transcriptDelta.get("sessionId")
    base = transcriptDelta.get("offset", 0)
    events = transcriptDelta.get("events")
    if not isinstance(events, list):
        return []

    spans = []
    for eventIndex, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        content = event.get("content")
        local = 0

        # Explicit emits first -- they are the strongest signal and their order
        # relative to surrounding text does not matter (they bypass stage 2).
        for tool, args in _emitBlocks(content):
            spans.append({
                "sessionId": sessionId,
                "offset": f"{base}.{eventIndex}.{local}",
                "text": _emitGist(tool, args),
                "kind": KIND_EXPLICIT_EMIT,
                "explicitEmit": {"tool": tool, "args": args},
            })
            local += 1

        # Heuristic spans come only from assistant reasoning text.
        if event.get("role") == "assistant":
            for text in _textBlocks(content):
                for paragraph in _paragraphs(text):
                    kind = _classify(paragraph)
                    if kind is None:
                        continue
                    spans.append({
                        "sessionId": sessionId,
                        "offset": f"{base}.{eventIndex}.{local}",
                        "text": paragraph,
                        "kind": kind,
                        "explicitEmit": None,
                    })
                    local += 1

    return spans


def _emitGist(tool, args):
    """A short readable label for an explicit-emit span's ``text``. The distiller
    composes the real atom body from ``args``; this is only what segment carries so
    the span is legible in isolation."""
    principle = args.get("principle") or args.get("hypothesis") or args.get("narrative")
    if isinstance(principle, str) and principle.strip():
        return f"{tool}: {principle.strip()}"
    return tool
