"""Tiered plain-text payload assembly -- the spec section 8 "no-slop" contract.

Recall's output is read by a model in a system prompt, so the payload is plain
text with NO markdown furniture: no headers, no bullet lists, no bold/italic, no
emoji. The only structure is a fixed three-tier line grammar the reader learns
once. Body CONTENT renders verbatim (a stored atom whose text happens to contain
markdown is data, not formatting -- we neither strip nor "fix" it); the furniture
WE add is what must stay markdown-free.

Column 0 belongs to the furniture, and that is a security boundary rather than a
layout choice. In this grammar the ONLY thing separating "the daemon says this
atom is p3://X at confidence 0.99" from "some atom's body contains that sentence"
is which column the text starts at, and atom bodies are written by any connected
agent. A body is free to contain a line shaped exactly like a handle or a
provenance line, so rendered flush left it would be indistinguishable from this
module's own framing, and retrieved memory would become a prompt-injection
channel into every agent that recalls. Two guards make the boundary structural:

- Every body line is prefixed with :data:`BODY_INDENT` (:func:`_indentBody`), so
  stored text cannot reach column 0 no matter what it contains.
- Every furniture line is collapsed to one physical line (:func:`_oneLine`),
  because furniture interpolates store-controlled fields too -- the Tier-0 gist,
  the provenance source and agent -- and a line break in one of those would split
  a furniture line in two and hand the tail column 0.

Both use ``str.splitlines``, which knows the whole Unicode line-boundary set
(``\\r``, ``\\x85``, ``\\u2028``, ...) rather than just ``\\n``, so a consumer
that splits differently than we joined cannot be shown a different payload than
we rendered. The cost is that a body's line SEPARATORS are normalized to ``\\n``;
its characters are otherwise untouched, and nothing is ever dropped.

Three tiers:

- **Tier 0 -- handle line.** One line, machine-parseable, the densest useful form::

      p3://<atomId> | <YYYY-MM-DD> | <confidence> | <gist>

  The date is ``COALESCE(occurred_at, created_at)`` (when the thing happened wins,
  the record time is the fallback -- the store-wide convention). ``confidence`` is
  the trust score to two decimals. ``gist`` is the first ~80 chars of the body with
  all whitespace collapsed to single spaces, so the handle is always exactly one
  line even when the body is multi-paragraph.

- **Tier 1 -- handle + body + provenance.** The handle line, then the full stored
  text with every line indented by :data:`BODY_INDENT`, then one provenance line::

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
import re
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
    "gistOf",
    "MAX_LOW_CONF_HANDLES",
    "HANDLE_SCHEME",
    "BODY_INDENT",
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

# Every rendered body line starts with this, which is what keeps column 0 for the
# furniture (see the module docstring). Exported so a parser can strip it back off
# instead of guessing. Two spaces: enough to read as nested, and short of the four
# that would make a body look like an indented markdown code block.
BODY_INDENT = "  "

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


# Emit-template scaffolding. A body opening with these is the composed-atom
# shape: `_composeAtomText` writes shape / approach: / outcome: / principle: as
# separate lines, so position 0 is a field label rather than a claim.
_SCAFFOLD_PREFIXES = ("approach:", "outcome:", "domain:", "dead ends:", "next steps:")

# The transferable claim. When a body has one, it is the single most useful
# sentence in it, which is exactly what an 80-character preview should spend
# itself on.
_PRINCIPLE_PREFIX = "principle:"

# Shape values the one-argument emit shorthands used to synthesize. They are
# labels, never content. Kept as literals rather than a pattern because a real
# author-supplied shape IS content and must not be skipped.
_SYNTHETIC_SHAPES = ("discovery", "failed approach")

# A line carrying only markup: bullet markers, rules, fence and quote
# characters. A chunk cut mid-document routinely opens on one of these.
_MARKUP_ONLY_RE = re.compile(r"^[\s\-\*\+#>|`_=~.:]+$")


def gistOf(text):
    """The most informative ~``GIST_CHARS`` of a body, as ONE physical line.

    Selection, not position. The previous rule was ``" ".join(text.split())[:80]``
    and for the three body shapes that dominate this store that is close to the
    worst available window: emit-template atoms open with up to 53 characters of
    scaffolding, distiller atoms open with a question, and document_chunks open
    wherever the chunker happened to cut. Working retrieval was therefore being
    read as broken ranking -- measured 2026-08-12, where four of five real probes
    looked like misses from the rendered line and three of those four had hit.

    Order, best signal first:

    1. a ``principle:`` line, which is the body's transferable claim
    2. else the first line that is not scaffolding, not a bare synthesized shape,
       not a question, and not markup-only
    3. else the old leading-slice rule, because an empty gist is a dead sensor
       and a preview of something beats a preview of nothing

    This is a RENDER decision and touches no score: ranking must be unaffected,
    so the two can be evaluated apart. Adversarial characters still pass through
    as literals; the forgery guard lives in the body-indent path and duplicating
    it here would leave two mechanisms answering one question, which is how one
    of them quietly stops being load-bearing.
    """
    if not text:
        return ""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    for line in lines:
        if line.lower().startswith(_PRINCIPLE_PREFIX):
            claim = line[len(_PRINCIPLE_PREFIX):].strip()
            if claim:
                return claim[:GIST_CHARS]

    for i, line in enumerate(lines):
        low = line.lower()
        if low.startswith(_SCAFFOLD_PREFIXES):
            continue
        if low in _SYNTHETIC_SHAPES:
            continue
        if line.endswith("?"):
            continue
        if _MARKUP_ONLY_RE.match(line):
            continue
        # Skip TO the first line worth reading, then keep filling from there.
        # Returning that line alone would be a regression for chunks: a body
        # opening on a short markdown heading would spend 14 of 80 characters
        # and waste the rest, where the old flatten-and-slice at least filled
        # the window. Selection decides where to START, not how much to show.
        return " ".join(lines[i:])[:GIST_CHARS]

    return " ".join(text.split())[:GIST_CHARS]


def _gist(text):
    """Module-internal alias for :func:`gistOf`; see it for the selection rule."""
    return gistOf(text)


def _oneLine(line):
    """A furniture line, guaranteed to be ONE physical line.

    Applied to the whole assembled line rather than per field, so a future field
    cannot be added and forgotten. Furniture interpolates store data an emitting
    agent controls end to end (the gist, the provenance source/agent/session, an
    edge type): a line break in any of those splits one furniture line in two and
    lands the tail at column 0, forging a handle or a provenance line without ever
    touching a body. Line breaks become spaces because furniture is a single
    sentence per line -- the field it came from was never meant to be multi-line."""
    return " ".join(line.splitlines())


def _indentBody(text):
    """Stored body text with every line prefixed by :data:`BODY_INDENT`.

    This is what makes the column-0 boundary structural instead of aspirational.
    A body containing a literal ``p3://... | ...`` or ``source ...`` line renders
    indented, so it reads as data rather than as the framing this module emits.
    Nothing is stripped or rewritten: each line's characters survive verbatim and
    only the indent is added. An empty body still renders as one indented line, so
    an EMPTY line in the payload always means an entry boundary and never a body,
    which keeps the entry grammar parseable by splitting on a blank line."""
    lines = text.splitlines() or [""]
    return _LINE_SEPARATOR.join(BODY_INDENT + line for line in lines)


def _effectiveTime(atom):
    """``COALESCE(occurred_at, created_at)`` for a getAtom() dict."""
    return atom["occurredAt"] if atom["occurredAt"] is not None else atom["createdAt"]


def _handleLine(atom, confidence):
    """Tier-0 handle line for a getAtom() dict at the given confidence."""
    date = _fmtDate(_effectiveTime(atom))
    return _oneLine(
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
    return _oneLine(line)


def _tier1Entry(atom, confidence, supersededBy=None):
    """Handle line + indented body + provenance line, joined by single newlines.

    The body goes through :func:`_indentBody` rather than in raw: the furniture
    owns column 0, a body never does."""
    return (
        f"{_handleLine(atom, confidence)}\n"
        f"{_indentBody(atom['text'])}\n"
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
        edgeLines.append(_oneLine(
            f"{edge['type']} -> {HANDLE_SCHEME}{edge['dstAtom']} {_gist(dst['text'])}"
        ))
    if edgeLines:
        return entry + _LINE_SEPARATOR + _LINE_SEPARATOR.join(edgeLines)
    return entry


def _enrichedEntry(store, result, enricher, remainingBudget):
    """Tier-1 entry plus enricher lines, degraded to fit remainingBudget.

    Degrade order (spec: attachments go first, bodies are never touched):
    drop ``relates`` lines from the end one at a time, then the ``at`` line,
    then the bare entry. Returns the best-fitting string, which may still
    exceed remainingBudget (the caller's atomic-drop rule then applies to the
    WHOLE entry, exactly as for a bare oversized entry). An enricher that
    raises is treated as no enricher for this entry: serve-time enrichment is
    best-effort by contract and must never break recall.
    """
    entry = tier1Entry(store, result)
    try:
        extra = list(enricher.lines(result))
    except Exception:
        extra = []
    while extra:
        candidate = entry + _LINE_SEPARATOR + _LINE_SEPARATOR.join(extra)
        if estimateTokens(candidate) <= remainingBudget:
            return candidate
        extra.pop()  # relates lines shed from the end; the at line goes last
    return entry


def assemblePayload(store, results, tokenBudget, enricher=None):
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

    ``enricher`` (optional) appends per-result furniture lines that degrade before
    anything else under budget; None preserves the exact legacy payload.

    ``tokens`` is ``estimateTokens`` of the FINAL payload string (furniture and
    separators included)."""
    if not any(r.get("shouldTrust") for r in results):
        return _lowConfidencePayload(store, results, tokenBudget)

    entries = []
    for result in results:
        if enricher is not None:
            joined = _ENTRY_SEPARATOR.join(entries) if entries else ""
            used = estimateTokens(joined) if entries else 0
            sep = estimateTokens(_ENTRY_SEPARATOR) if entries else 0
            entry = _enrichedEntry(store, result, enricher,
                                   tokenBudget - used - sep)
        else:
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
