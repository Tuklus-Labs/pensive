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
      "project": "<project filter, or null when the query was unscoped>",
      "old": {"resultText": "<the OLD server's full answer, NOT trimmed>"},
      "v3":  {"ids": [<up to 10 result atomIds>],
              "payload": "<the v3 tiered payload>",
              "tokensUsed": <int>,
              "lowConfidence": <bool>,
              "latencyMs": <float>}
    }

``project`` carries the recall's project filter so the A/B is apples-to-apples:
the old server scopes ``pensive_recall`` by project, so v3's shadow recall is
scoped the same way and the line records which scope both answers were computed
under (null = whole store, matching the old server's ``if project:`` no-filter
default; the empty-string legacy default normalizes to null here too).

The old text is stored WHOLE -- the judge needs both answers verbatim, so we
never truncate it. Like the tee, this is a CONTAINED boundary: a recall error or
an unwritable log path is counted and logged, never raised into the serving path
(the old server already answered and does not wait on us).

Readers of the log MUST tolerate a torn final line: a crash mid-write can leave
one incomplete (unparseable) JSON line at the tail. Each line is emitted with a
single ``os.write`` to an ``O_APPEND`` fd to shrink that window, but a reader
should still skip any line that does not ``json.loads`` cleanly rather than abort.
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


def buildShadowRecord(query, oldResultText, v3Out, latencyMs, project=None):
    """Assemble one shadow log record from a v3 ``recall`` result dict.

    ``v3Out`` is the ``recall`` return ``{results, payload, tokensUsed,
    lowConfidence}``. ``project`` is the recall's scope (null when unscoped). The
    old answer is stored WHOLE (never trimmed); the v3 block records the
    top-``MAX_SHADOW_IDS`` result atomIds plus the payload and the
    trust/budget/latency metadata the A/B judge scores on.
    """
    results = v3Out.get("results") or []
    ids = [r["atomId"] for r in results[:MAX_SHADOW_IDS]]
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "project": project,
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

    Creates the parent directory if needed, then emits the whole line (JSON +
    newline) in a SINGLE ``os.write`` to an ``O_APPEND`` fd. ``O_APPEND`` makes
    each write seek-to-end atomically, and packing the newline into the one write
    shrinks the torn-line window a crash could leave (a partial byte range instead
    of, say, the record without its terminator). ``ensure_ascii=False`` keeps real
    atom text readable. Raises on any filesystem error; :func:`runShadow` is the
    boundary that contains it.
    """
    logPath = Path(logPath)
    logPath.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(logPath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def runShadow(ctx, counters, rawBody, logPath):
    """Run the v3 recall for a shadow request and log it -> ``(httpStatus, bodyDict)``.

    ``rawBody`` is the request body bytes: ``{"query": <str>, "oldResultText":
    <str>, "project"?: <str|null>}`` -- the shape the Engram-side forward sends.
    ``project`` scopes the v3 recall to match the old server's project filter
    (missing/empty/null all normalize to "whole store", mirroring the old
    server's ``if project:`` default). On success appends one line and increments
    ``shadowLogged``; on any failure increments ``shadowFailed`` and returns an
    error status, never raising (the boundary contains it so the old server's
    fire-and-forget POST is undisturbed).
    """
    try:
        parsed = json.loads(rawBody)
        if not isinstance(parsed, dict):
            raise ValueError("shadow payload must be a JSON object")
        query = parsed.get("query", "")
        oldResultText = parsed.get("oldResultText", "")
        # "" (the legacy default) and null both mean "no filter", matching the old
        # server; a real slug scopes the recall. recall() takes project=None.
        project = parsed.get("project") or None
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
            ctx.store, ctx.indexes, ctx.embedder, query,
            project=project, k=ctx.defaultK, tokenBudget=ctx.defaultTokenBudget,
        )
        latencyMs = (time.perf_counter() - t0) * 1000.0
        record = buildShadowRecord(query, oldResultText, out, latencyMs, project=project)
        appendShadowLine(logPath, record)
    except Exception as exc:  # noqa: BLE001 -- contained boundary, must not bleed
        counters._inc("shadowFailed")
        return 500, {"error": f"shadow logging failed: {exc}"}

    counters._inc("shadowLogged")
    return 200, {"ok": True}
