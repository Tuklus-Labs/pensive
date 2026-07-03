"""Conservative ambient drift watcher.

``onTail(store, index, embedder, tailText, ctx)`` watches one conversation tail
and returns either ``None`` or one compact injection payload:

``{"kind": "memory", "atomId": str, "text": str, "score": float}``

The caller owns ``ctx`` and therefore owns session isolation. This module reads
and writes only these keys:

- ``now``: optional numeric unix timestamp used by tests/callers; defaults to
  ``time.time()``.
- ``driftLastInjectedAt``: numeric unix timestamp written after an injection and
  read to enforce cooldown.
- ``recentContext``: optional recent-context view. Strings are scanned for the
  matched atom id or text. Dict items may expose ``atomId``/``id`` and ``text``.

Malformed caller input returns ``None``. Store/index/embedder operational errors
are allowed to propagate because those are real faults, not conversation drift.
"""
import time

from store.store import getAtom

__all__ = ["CONFIDENCE_FLOOR", "COOLDOWN_SECONDS", "SEARCH_K", "onTail"]

# Task 16 calibration for this embedding space put near-verbatim pairs at
# >= 0.9863 and related-but-distinct pairs at <= 0.8069. The watcher values
# precision over recall, so 0.97 admits near-verbatim drift while leaving a wide
# gap above merely-related memories. The recall harness owns final tuning.
CONFIDENCE_FLOOR = 0.97

# One injection per short ambient window. Fifteen minutes is long enough to avoid
# chattering in an active session, short enough that a genuinely recurring topic
# can surface again later. The recall harness owns final tuning.
COOLDOWN_SECONDS = 15 * 60

# Search a small candidate set but inject only the best hit. Extra candidates let
# future callers inspect near misses without making the watcher chatty today. The
# recall harness owns final tuning.
SEARCH_K = 3


def onTail(store, index, embedder, tailText, ctx):
    """Return one memory injection for a high-confidence drift hit, else None."""
    if not isinstance(ctx, dict) or not isinstance(tailText, str):
        return None
    if not tailText.strip():
        return None

    now = _now(ctx)
    if now is None or _coolingDown(ctx, now):
        return None

    vecs = embedder.embed([tailText])
    if not vecs:
        return None

    hits = index.search(vecs[0], SEARCH_K)
    if not hits:
        return None

    atomId, score = hits[0]
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if score < CONFIDENCE_FLOOR:
        return None

    atom = getAtom(store, atomId)
    if atom is None or not isinstance(atom.get("text"), str):
        return None
    if _alreadyInContext(atom["id"], atom["text"], ctx.get("recentContext")):
        return None

    ctx["driftLastInjectedAt"] = now
    return {
        "kind": "memory",
        "atomId": atom["id"],
        "text": atom["text"],
        "score": score,
    }


def _now(ctx):
    if "now" not in ctx:
        return time.time()
    try:
        return float(ctx["now"])
    except (TypeError, ValueError):
        return None


def _coolingDown(ctx, now):
    if "driftLastInjectedAt" not in ctx:
        return False
    try:
        last = float(ctx["driftLastInjectedAt"])
    except (TypeError, ValueError):
        return True
    return now - last < COOLDOWN_SECONDS


def _alreadyInContext(atomId, text, recentContext):
    if recentContext is None:
        return False
    if isinstance(recentContext, str):
        return atomId in recentContext or text in recentContext
    if not isinstance(recentContext, (list, tuple)):
        return True

    for item in recentContext:
        if isinstance(item, str):
            if atomId in item or text in item:
                return True
        elif isinstance(item, dict):
            seenId = item.get("atomId", item.get("id"))
            seenText = item.get("text")
            if seenId == atomId or seenText == text:
                return True
            if isinstance(seenText, str) and (atomId in seenText or text in seenText):
                return True
        else:
            return True
    return False
