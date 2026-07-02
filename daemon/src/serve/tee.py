"""Double-write tee receiver: land every OLD-path emit in the v3 store too.

Phase 3's "make the parallel run real." The production MCP server
(``~/Projects/Engram/tools/pensive-mcp-server``) stays authoritative: it writes
to the old store first and always completes its own response. Behind the
``PENSIVE_V3_TEE`` flag it then fire-and-forgets the same emit payload to this
daemon's ``POST /tee/emit`` endpoint, which replays it through the Task 12 emit
handlers into the v3 store. The old write is authoritative; this is a shadow copy
for the A/B cutover evidence, nothing depends on it, and its failure must never
bleed back into the old path.

This module is the ONE place in v3 where failures are contained by design rather
than raised: the v3 components fail LOUD, but the tee boundary contains, counts,
and logs, because the old serving path must stay byte-identical whether or not the
tee succeeds. Every outcome updates a :class:`Counters` the daemon exposes at
``GET /status`` for the Phase 3 gate check.

Contract notes:

- Only the legacy EMIT tools are accepted (``_TEE_EMIT_TOOLS``); a recall belongs
  on ``/shadow/recall``, not here, so a non-emit tool name is a 400.
- A duplicate emit tee'd twice writes TWO rows, and that is CORRECT: the old path
  is authoritative and the distiller (a later task) owns dedup. The tee never
  dedups -- it faithfully mirrors whatever the old path accepted.
- The tee holds only the v3 ``store`` (through ``ctx``); it has no handle to the
  legacy socket/store, so by construction a v3 write failure cannot touch the old
  path. Containment here just means the old server's fire-and-forget POST never
  sees an exception or a hang.
"""
import json
import threading

from serve.mcp import dispatch

__all__ = ["Counters", "handleTeeEmit", "TEE_EMIT_TOOLS"]

# The legacy emit tool names the tee accepts. These map 1:1 to the v3 compat emit
# handlers (serve.mcp.HANDLERS). pensive_recall / pensive_analytics / the v3
# natives are deliberately absent -- a recall is shadowed via /shadow/recall.
TEE_EMIT_TOOLS = frozenset({
    "engram_emit_atom",
    "engram_emit_discovery",
    "engram_emit_failure",
    "engram_emit_narrative",
    "engram_emit_snapshot",
})


class Counters:
    """Thread-safe tallies for the tee/shadow boundary, read at ``GET /status``.

    Four counters back the Phase 3 gate check:

    - ``teeReceived``  -- every POST that reached ``/tee/emit`` (arrivals).
    - ``teeFailed``    -- of those, the ones that failed (malformed payload or a
      v3 store error). Successful writes = ``teeReceived - teeFailed``.
    - ``shadowLogged`` -- shadow recalls whose JSONL line was appended.
    - ``shadowFailed`` -- shadow recalls that failed (bad payload, recall error,
      or log-write error).

    Increments take a lock because the daemon may serve concurrent requests;
    reads of a single int are atomic enough for assertions/reporting, and
    :meth:`snapshot` takes the lock for a consistent four-tuple.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.teeReceived = 0
        self.teeFailed = 0
        self.shadowLogged = 0
        self.shadowFailed = 0

    def _inc(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def snapshot(self):
        with self._lock:
            return {
                "teeReceived": self.teeReceived,
                "teeFailed": self.teeFailed,
                "shadowLogged": self.shadowLogged,
                "shadowFailed": self.shadowFailed,
            }


def handleTeeEmit(ctx, counters, rawBody):
    """Replay one tee'd emit into the v3 store -> ``(httpStatus, bodyDict)``.

    ``rawBody`` is the request body bytes: a JSON object ``{"tool": <emit name>,
    "args": <the legacy tool arguments>}`` -- exactly the shape the Engram-side
    forward sends. Every call increments ``teeReceived``; any failure also
    increments ``teeFailed``. Never raises: a malformed payload is a 400, a v3
    store error is a 500, and both come back as a normal return so the old
    server's fire-and-forget POST is never disturbed.
    """
    counters._inc("teeReceived")

    try:
        parsed = json.loads(rawBody)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        counters._inc("teeFailed")
        return 400, {"error": f"malformed tee payload: {exc}"}
    if not isinstance(parsed, dict):
        counters._inc("teeFailed")
        return 400, {"error": "tee payload must be a JSON object"}

    tool = parsed.get("tool")
    args = parsed.get("args")
    if tool not in TEE_EMIT_TOOLS:
        counters._inc("teeFailed")
        return 400, {"error": f"tee: unknown or non-emit tool {tool!r}"}
    if not isinstance(args, dict):
        counters._inc("teeFailed")
        return 400, {"error": "tee: 'args' must be a JSON object"}

    # dispatch() already contains handler exceptions and returns (text, isError),
    # so a v3 store error surfaces as isError. The try/except is belt-and-suspenders
    # for an UNMODELED raise (a bug in dispatch, an error before it returns): the
    # boundary must still count it as teeFailed and return a clean 500, never let a
    # raw exception escape to Starlette (which would 500 with the counter stuck at
    # "received but not failed").
    try:
        text, isError = dispatch(ctx, tool, args)
    except Exception as exc:  # noqa: BLE001 -- contained boundary, must not bleed
        counters._inc("teeFailed")
        return 500, {"error": f"tee dispatch failed: {exc}"}
    if isError:
        counters._inc("teeFailed")
        return 500, {"error": text}
    return 200, {"ok": True, "result": text}
