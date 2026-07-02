"""Shadow A/B logging: log v3's recall answer beside the old one, per query.

Phase 3's evidence-gathering half. Behind the ``PENSIVE_V3_TEE`` flag the OLD
production server, after answering a ``pensive_recall``, fire-and-forgets the
query plus its own answer to this daemon's ``POST /shadow/recall``. Here we run
the SAME query through the full v3 recall pipeline and append one JSON line to
``daemon/eval/shadow.jsonl`` pairing the two answers. That file is the corpus an
A/B judge reads to decide the cutover: does v3 answer at least as well as the
production stack on the real live query stream?

The line schema (append-only, one object per line)::

    {
      "ts": "<ISO-8601 UTC>",
      "query": "<the query>",
      "old": {"resultText": "<the OLD server's full answer, NOT trimmed>"},
      "v3":  {"ids": [<up to 10 result atomIds>],
              "payload": "<the v3 tiered payload>",
              "tokensUsed": <int>,
              "lowConfidence": <bool>,
              "latencyMs": <float>}
    }

The old text is stored WHOLE -- the judge needs both answers verbatim, so we
never truncate it. Like the tee, this is a CONTAINED boundary: a recall error or
an unwritable log path is counted and logged, never raised into the serving path
(the old server already answered and does not wait on us).
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from recall.engine import recall

__all__ = [
    "defaultShadowLogPath",
    "buildShadowRecord",
    "appendShadowLine",
    "runShadow",
    "SHADOW_LOG_ENV",
    "MAX_SHADOW_IDS",
]

# Env override for the shadow log path (tests point it at a tmp file; the live
# daemon leaves it unset and writes the repo default below).
SHADOW_LOG_ENV = "PENSIVE_V3_SHADOW_LOG"

# The shadow line records at most this many v3 result atomIds (the plan's top-10).
MAX_SHADOW_IDS = 10


def defaultShadowLogPath():
    """Resolve the shadow log path: ``$PENSIVE_V3_SHADOW_LOG`` or the repo default.

    The default is ``daemon/eval/shadow.jsonl`` (already gitignored via
    ``daemon/eval/.gitignore``'s ``*.jsonl`` -- it holds real atom text and must
    never enter git). Resolved from this file's location so it is stable
    regardless of the daemon's cwd: parents[2] is ``daemon/`` (serve -> src ->
    daemon).
    """
    override = os.environ.get(SHADOW_LOG_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "eval" / "shadow.jsonl"


def buildShadowRecord(query, oldResultText, v3Out, latencyMs):
    """Assemble one shadow log record from a v3 ``recall`` result dict.

    ``v3Out`` is the ``recall`` return ``{results, payload, tokensUsed,
    lowConfidence}``. The old answer is stored WHOLE (never trimmed); the v3 block
    records the top-``MAX_SHADOW_IDS`` result atomIds plus the payload and the
    trust/budget/latency metadata the A/B judge scores on.
    """
    results = v3Out.get("results") or []
    ids = [r["atomId"] for r in results[:MAX_SHADOW_IDS]]
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "old": {"resultText": oldResultText},
        "v3": {
            "ids": ids,
            "payload": v3Out.get("payload"),
            "tokensUsed": v3Out.get("tokensUsed"),
            "lowConfidence": v3Out.get("lowConfidence"),
            "latencyMs": latencyMs,
        },
    }


def appendShadowLine(logPath, record):
    """Append one JSON record as a line to ``logPath`` (append-only).

    Creates the parent directory if needed and opens in append mode, so lines
    accrete and are never overwritten. ``ensure_ascii=False`` keeps real atom text
    readable in the log. Raises on any filesystem error; :func:`runShadow` is the
    boundary that contains it.
    """
    logPath = Path(logPath)
    logPath.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(logPath, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def runShadow(ctx, counters, rawBody, logPath):
    """Run the v3 recall for a shadow request and log it -> ``(httpStatus, bodyDict)``.

    ``rawBody`` is the request body bytes: ``{"query": <str>, "oldResultText":
    <str>}`` -- the shape the Engram-side forward sends. On success appends one
    line and increments ``shadowLogged``; on any failure increments
    ``shadowFailed`` and returns an error status, never raising (the boundary
    contains it so the old server's fire-and-forget POST is undisturbed).
    """
    try:
        parsed = json.loads(rawBody)
        if not isinstance(parsed, dict):
            raise ValueError("shadow payload must be a JSON object")
        query = parsed.get("query", "")
        oldResultText = parsed.get("oldResultText", "")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        counters._inc("shadowFailed")
        return 400, {"error": f"malformed shadow payload: {exc}"}

    # From here on every failure -- a recall desync raised by the v3 pipeline, or
    # an unwritable log path -- is contained at this ONE boundary: counted, logged
    # to the daemon's stderr by the caller if it wants, and returned as a normal
    # 500 so the old server's fire-and-forget POST never sees an exception or a
    # hang. The old path already answered; the shadow copy is best-effort.
    try:
        t0 = time.perf_counter()
        out = recall(
            ctx.store, ctx.index, ctx.embedder, query,
            k=ctx.defaultK, tokenBudget=ctx.defaultTokenBudget,
        )
        latencyMs = (time.perf_counter() - t0) * 1000.0
        record = buildShadowRecord(query, oldResultText, out, latencyMs)
        appendShadowLine(logPath, record)
    except Exception as exc:  # noqa: BLE001 -- contained boundary, must not bleed
        counters._inc("shadowFailed")
        return 500, {"error": f"shadow logging failed: {exc}"}

    counters._inc("shadowLogged")
    return 200, {"ok": True}
