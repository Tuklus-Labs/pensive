"""Tiered plain-text payload assembly -- the spec section 8 "no-slop" contract.

Recall's output is read by a model in a system prompt, so the payload is plain
text with NO markdown furniture: no headers, no bullet lists, no bold/italic, no
emoji. The only structure is a fixed three-tier line grammar the reader learns
once. Bodies render VERBATIM (a stored atom whose text happens to contain
markdown is data, not formatting -- we neither strip nor "fix" it); the furniture
WE add is what must stay markdown-free, and every furniture line is prefixed so
adversarial body characters can only ever appear mid-line, never at a line start.

Three tiers:

- **Tier 0 -- handle line.** One line, machine-parseable, the densest useful form::

      p3://<atomId> | <YYYY-MM-DD> | <confidence> | <gist>

  The date is ``COALESCE(occurred_at, created_at)`` (when the thing happened wins,
  the record time is the fallback -- the store-wide convention). ``confidence`` is
  the trust score to two decimals. ``gist`` is the first ~80 chars of the body with
  all whitespace collapsed to single spaces, so the handle is always exactly one
  line even when the body is multi-paragraph.

- **Tier 1 -- handle + body + provenance.** The handle line, then the full stored
  text verbatim, then one provenance line::

      source <source>, <agent or session or 'unknown'>, recorded <YYYY-MM-DD>

  A superseded atom appends ``, superseded by p3://<successor>`` to that line, so a
  historical fact always carries a pointer to what replaced it.

- **Tier 2 -- neighborhood.** Tier 1 plus one line per LIVE outgoing edge::

      <type> -> p3://<dst> <gist>

  Only edges whose destination atom is live are rendered (a relation to a
  superseded/tombstoned/missing atom is not surfaced). recall() itself assembles
  Tiers 0+1; Tier 2 is exposed for the Task 12 history/neighborhood view via
  :func:`assembleTier2`.

Budget discipline: tokens are approximated by a deliberately CONSERVATIVE
heuristic, ``ceil(len(text) / 3)``, which OVERestimates for English -- a violated
budget (a payload that blows the model's context) is strictly worse than an
underfilled one, so we round the wrong way on purpose. Entries are atomic: whole
Tier-1 entries are fit top-down and the tail is dropped; an atom body is NEVER cut
mid-sentence. If not even the top entry fits, its Tier-0 handle alone is emitted
(if THAT fits), else the budget sentinel.

Stdlib only; ``store`` reads go through the store's public accessors.
"""
import math
from datetime import datetime, timezone

from store.store import getAtom, edgesFrom

__all__ = [
    "estimateTokens",
    "tier0Handle",
    "tier1Entry",
    "assembleTier2",
    "assemblePayload",
    "SENTINEL_LOW_CONFIDENCE",
    "SENTINEL_BUDGET_TOO_SMALL",
    "CHARS_PER_TOKEN",
    "GIST_CHARS",
    "MAX_LOW_CONF_HANDLES",
    "HANDLE_SCHEME",
]

# --- named constants (one place each) -------------------------------------- #

# Conservative token heuristic: ceil(len / CHARS_PER_TOKEN). 3 chars/token
# OVERestimates real English (~4 chars/token), so the payload can only come in
# UNDER a true tokenizer's count -- a busted budget is worse than a short one.
CHARS_PER_TOKEN = 3

# Tier-0 gist length: the leading slice of the (whitespace-collapsed) body.
GIST_CHARS = 80

# The low-confidence payload lists at most this many best-untrusted Tier-0 handles
# after the sentinel -- enough to orient the reader, never enough to pad.
MAX_LOW_CONF_HANDLES = 3

# The handle URI scheme. Every furniture line that names an atom uses it, so the
# reader (and a downstream parser) has one stable token to anchor on.
HANDLE_SCHEME = "p3://"

# Sentinels, verbatim per the spec.
SENTINEL_LOW_CONFIDENCE = "low confidence: no trusted match"
SENTINEL_BUDGET_TOO_SMALL = "recall: token budget too small"

# Blank line between Tier-1 entries; single newline within an entry and between
# low-confidence handle lines.
_ENTRY_SEPARATOR = "\n\n"
_LINE_SEPARATOR = "\n"


def estimateTokens(text):
    """Conservative token estimate: ``ceil(len(text) / CHARS_PER_TOKEN)``.

    The single place the heuristic lives. OVERestimates for English on purpose
    (see the module docstring): every budget decision rounds toward a shorter
    payload. ``""`` -> 0.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _fmtDate(epochSeconds):
    """Unix seconds -> ``YYYY-MM-DD`` in UTC (epoch seconds are inherently UTC).

    ``None`` -> ``"unknown"`` (a provenance row with no recorded time cannot
    happen via putAtom, but the payload never crashes on missing data)."""
    if epochSeconds is None:
        return "unknown"
    return datetime.fromtimestamp(epochSeconds, timezone.utc).strftime("%Y-%m-%d")


def _gist(text):
    """First ``GIST_CHARS`` chars of ``text`` with all whitespace collapsed.

    ``str.split()`` + ``" ".join`` collapses every run of whitespace -- crucially
    newlines -- to a single space, so the gist is always ONE line regardless of
    the body. Adversarial body text (markdown, control chars) passes through as
    literal characters; only the newlines are removed, because a newline would
    break the single-line handle grammar. Truncation is a plain slice: the gist is
    a preview, not the body, so a mid-word cut is fine here (unlike a Tier-1 body,
    which is never cut)."""
    return " ".join(text.split())[:GIST_CHARS]


def _effectiveTime(atom):
    """``COALESCE(occurred_at, created_at)`` for a getAtom() dict."""
    return atom["occurredAt"] if atom["occurredAt"] is not None else atom["createdAt"]


def _handleLine(atom, confidence):
    """Tier-0 handle line for a getAtom() dict at the given confidence."""
    date = _fmtDate(_effectiveTime(atom))
    return (
        f"{HANDLE_SCHEME}{atom['id']} | {date} | "
        f"{confidence:.2f} | {_gist(atom['text'])}"
    )


def _provenanceLine(atom, supersededBy=None):
    """The one provenance line for a Tier-1 entry.

    Uses the FIRST provenance row (write order); distiller bump/attach rows also
    land on atoms, so only ULID insertion ordering via getAtom()'s ``ORDER BY id``
    keeps the creation row at index 0. ``<agent or session or 'unknown'>`` prefers
    the agent, falls back to the session id, then the literal ``unknown``. A
    superseded atom appends its live-successor pointer."""
    prov = atom["provenance"][0] if atom["provenance"] else None
    if prov is None:
        source, who, recorded = "unknown", "unknown", "unknown"
    else:
        source = prov["source"]
        who = prov["agent"] or prov["sessionId"] or "unknown"
        recorded = _fmtDate(prov["recordedAt"])
    line = f"source {source}, {who}, recorded {recorded}"
    if supersededBy:
        line += f", superseded by {HANDLE_SCHEME}{supersededBy}"
    return line


def _tier1Entry(atom, confidence, supersededBy=None):
    """Handle line + verbatim body + provenance line, joined by single newlines."""
    return (
        f"{_handleLine(atom, confidence)}\n"
        f"{atom['text']}\n"
        f"{_provenanceLine(atom, supersededBy)}"
    )


def _fetch(store, atomId):
    """getAtom() or a loud error. By the time payload runs, the trust layer has
    already proven every result id lives in the store (it raises on desync), so a
    None here is a real invariant breach, not an expected miss -- surface it."""
    atom = getAtom(store, atomId)
    if atom is None:
        raise ValueError(f"payload: atom {atomId!r} absent from the store")
    return atom


def tier0Handle(store, result):
    """Tier-0 handle line for a recall result dict (``{atomId, confidence, ...}``)."""
    return _handleLine(_fetch(store, result["atomId"]), result["confidence"])


def tier1Entry(store, result):
    """Tier-1 entry for a recall result dict. Honors ``supersededBy`` if present."""
    atom = _fetch(store, result["atomId"])
    return _tier1Entry(atom, result["confidence"], result.get("supersededBy"))


def assembleTier2(store, atomId, confidence=0.0, supersededBy=None):
    """Tier-2 neighborhood block for ``atomId``: Tier 1 + one line per live edge.

    The Tier-1 portion needs a confidence; the history/neighborhood view that
    consumes Tier 2 (Task 12) is not a ranked recall, so ``confidence`` defaults
    to 0.0 and callers with a real score may pass it. The two-argument call form
    ``assembleTier2(store, atomId)`` is the interface contract; the extra
    parameters are optional.

    Only LIVE outgoing edges render: an edge whose destination is superseded,
    tombstoned, or missing is dropped, so the neighborhood never dangles a pointer
    at a non-recallable atom. Edges keep :func:`store.store.edgesFrom` order
    (oldest first)."""
    atom = _fetch(store, atomId)
    entry = _tier1Entry(atom, confidence, supersededBy)
    edgeLines = []
    for edge in edgesFrom(store, atomId):
        dst = getAtom(store, edge["dstAtom"])
        if dst is None or dst["status"] != "live":
            continue
        edgeLines.append(
            f"{edge['type']} -> {HANDLE_SCHEME}{edge['dstAtom']} {_gist(dst['text'])}"
        )
    if edgeLines:
        return entry + _LINE_SEPARATOR + _LINE_SEPARATOR.join(edgeLines)
    return entry


def assemblePayload(store, results, tokenBudget):
    """Assemble the recall payload from trust results -> ``(payload, tokens, lowConf)``.

    ``results`` is the trust-annotated, best-first (reranked-order) list. Two modes:

    - **At least one trusted result** (some ``shouldTrust`` is True): emit Tier-1
      entries top-down while they fit ``tokenBudget`` by the module heuristic,
      dropping the tail atomically -- a body is never cut. If not even the top
      entry fits, fall back to its Tier-0 handle alone, and failing that the budget
      sentinel. ``lowConf`` is False.

    - **No trusted result** (empty, or every ``shouldTrust`` False): emit the
      low-confidence sentinel followed by up to ``MAX_LOW_CONF_HANDLES`` Tier-0
      handles of the best untrusted hits, budget permitting. Never a Tier-1 body --
      weak matches are oriented, not padded. ``lowConf`` is True.

    ``tokens`` is ``estimateTokens`` of the FINAL payload string (furniture and
    separators included)."""
    if not any(r.get("shouldTrust") for r in results):
        return _lowConfidencePayload(store, results, tokenBudget)

    entries = []
    for result in results:
        entry = tier1Entry(store, result)
        trial = _ENTRY_SEPARATOR.join(entries + [entry])
        if estimateTokens(trial) <= tokenBudget:
            entries.append(entry)
        else:
            # Atomic drop: the first entry that does not fit takes the whole tail
            # with it. Trying to squeeze a smaller later entry in would reorder the
            # payload away from best-first and is not the contract.
            break

    if entries:
        payload = _ENTRY_SEPARATOR.join(entries)
    else:
        # Not even the top entry fit whole. Degrade to its handle, then the
        # sentinel -- never a truncated body.
        handle = tier0Handle(store, results[0])
        payload = handle if estimateTokens(handle) <= tokenBudget else SENTINEL_BUDGET_TOO_SMALL

    return payload, estimateTokens(payload), False


def _lowConfidencePayload(store, results, tokenBudget):
    """Sentinel + up to MAX_LOW_CONF_HANDLES Tier-0 handles (budget permitting)."""
    lines = [SENTINEL_LOW_CONFIDENCE]
    for result in results[:MAX_LOW_CONF_HANDLES]:
        handle = tier0Handle(store, result)
        trial = _LINE_SEPARATOR.join(lines + [handle])
        if estimateTokens(trial) <= tokenBudget:
            lines.append(handle)
        else:
            break
    payload = _LINE_SEPARATOR.join(lines)
    return payload, estimateTokens(payload), True
