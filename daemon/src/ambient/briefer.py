"""Session briefer: the working-set VIEW assembled at session start (Phase 4).

What an agent should know before its first exchange, computed on demand and never
written back. The spec's load-bearing sentence: "The working set is a VIEW: pins,
recency, importance, and usage compute it; nothing moves or mutates in the store
to produce it." So ``brief()`` performs ZERO writes -- it only reads the store and
formats what it finds.

This SUPPLEMENTS the existing Kairos SessionStart briefing; it does not replace
it. Kairos/forecast integration is out of scope here -- the current SessionStart
hook already emits the Kairos briefing, and the v3 brief is a second, additive
block. (See ``daemon/hooks/session-brief-v3.sh`` for the additive wiring.)

Three sections, in fixed display order, each budget-tiered:

1. ``pinned:`` -- atoms carrying the ``pin`` facet (Task 12's pin native, facet
   ``key='pin'``). The standing-principles tier: ALWAYS included first, as whole
   Tier-1 bodies, and NEVER dropped for budget. If the pins alone exceed the
   budget they degrade to handle-only lines and a ``pins exceed budget`` warning
   line is written into the brief -- but every pin is still present. A pin is
   never silently omitted.

2. ``active:`` -- live atoms ranked by ``importanceFactor * timeFactor`` reusing
   fusion.py's exact constants (no new decay math), rendered as Tier-0 handle
   lines grouped by project (best-scoring group first). Filled top-down until the
   budget is reached; whole entries are dropped, never cut.

3. ``loose ends for <agent>:`` -- atoms addressed to the starting agent via the
   ``for:<agent>`` tag convention (facet ``key='tag'``, ``value='for:<agent>'``),
   rendered as Tier-1 bodies. Filled until budget; whole entries dropped. Absent
   entirely when no agent is given (a null agent cannot be addressed).

The addressing convention, established here: to leave a note for a specific agent's
next session, attach a facet ``(key='tag', value='for:<agent>')`` to the atom. The
structured facet is the mechanism -- not a free-text marker -- so addressing is
exact and never false-positives on body text that merely mentions an agent.

VIEW semantics and no-slop discipline:

- Every section is LIVE-only. A superseded (or tombstoned) atom never appears in
  any section, so a retired fact is never surfaced as current.
- Budget is measured by :func:`recall.payload.estimateTokens` over the FINAL
  assembled string (headers and separators included); entries are atomic per the
  payload discipline (drop a whole entry, never truncate a body).
- Pins and loose ends reuse :func:`recall.payload.tier1Entry`; active reuses
  :func:`recall.payload.tier0Handle`. This is a VIEW, not a trust-ranked recall,
  so the handle's confidence slot is a fixed 0.00 -- the same convention
  :func:`recall.payload.assembleTier2` uses for a non-ranked view. It is a
  structural placeholder, not a low-trust signal.
- The furniture WE add (section headers, project labels, the warning line) is
  plain lowercase text: no markdown, no bullets, no emoji. Bodies render verbatim.

Empty store (or every section empty) yields the single line ``no memory yet``
rather than a bare stack of empty headers.

Stdlib only; store access is read-only -- plain SELECTs here plus the payload
tier renderers, which fetch through the store's public :func:`store.store.getAtom`.
No :func:`recall` model dependency -- the brief is pure store math.
"""
import time

from recall.payload import estimateTokens, tier0Handle, tier1Entry
from recall.fusion import importanceFactor, timeFactor

__all__ = [
    "brief",
    "looseHeader",
    "SECTION_PINNED",
    "SECTION_ACTIVE",
    "PIN_FACET_KEY",
    "LOOSE_TAG_KEY",
    "FOR_TAG_PREFIX",
    "PINS_EXCEED_BUDGET",
    "EMPTY_BRIEF",
    "DEFAULT_BUDGET",
    "ACTIVE_CANDIDATE_CAP",
]

# --- named constants (one place each) -------------------------------------- #

# Default token budget for the whole brief (spec: 1500 by estimateTokens).
DEFAULT_BUDGET = 1500

# The facet key Task 12's pin native writes. A pinned atom has a facet whose key
# is this (value unconstrained -- membership is the fact).
PIN_FACET_KEY = "pin"

# Loose-ends addressing: a facet (key=LOOSE_TAG_KEY, value=FOR_TAG_PREFIX+agent).
LOOSE_TAG_KEY = "tag"
FOR_TAG_PREFIX = "for:"

# Section headers -- lowercase, minimal, the only structure the reader learns.
SECTION_PINNED = "pinned:"
SECTION_ACTIVE = "active:"

# The warning line written into the brief when the pins alone blow the budget.
PINS_EXCEED_BUDGET = "pins exceed budget"

# The whole-brief fallback when nothing is recallable (empty or all-empty store).
EMPTY_BRIEF = "no memory yet"

# Active-thread candidate pool: the N most-recent live atoms are scored. "Active"
# is inherently recency-biased, so drawing candidates from the recent tail bounds
# the scan on a decades-scale store while still surfacing what is live now. Pins
# (standing principles) are the tier that carries important-but-old memory, so an
# old atom outside this window is not lost to the reader -- it is just not "active".
ACTIVE_CANDIDATE_CAP = 500

# A VIEW has no trust score; the handle confidence slot is a fixed placeholder,
# mirroring assembleTier2's non-ranked default.
_VIEW_CONFIDENCE = 0.0

# Blank line between sections and between Tier-1 entries; single newline between
# Tier-0 handle lines and within a section's header/lines.
_SECTION_SEP = "\n\n"
_LINE_SEP = "\n"


def looseHeader(agent):
    """The loose-ends section header for ``agent`` (e.g. ``loose ends for heph:``)."""
    return f"loose ends for {agent}:"


def _projectLabel(project):
    """The active-section per-project sub-label. A NULL project renders as
    ``unscoped:`` so an atom with no project still groups somewhere legible."""
    return f"{project if project is not None else 'unscoped'}:"


def _result(atomId):
    """The minimal recall-result shape payload's tier renderers consume. A VIEW
    carries no trust score, so confidence is the fixed placeholder."""
    return {"atomId": atomId, "confidence": _VIEW_CONFIDENCE}


# --- section source queries (all read-only) -------------------------------- #


def _pinnedIds(store):
    """Live pinned atom ids, standing-principles order: importance desc, then most
    recent effective time, then id. The ``IN (SELECT ...)`` avoids the row
    multiplication a JOIN would cause when an atom carries several pin facets."""
    rows = store._conn.execute(
        "SELECT id FROM atoms "
        "WHERE status = 'live' "
        "AND id IN (SELECT atom_id FROM facets WHERE key = ?) "
        "ORDER BY importance DESC, COALESCE(occurred_at, created_at) DESC, id",
        (PIN_FACET_KEY,),
    ).fetchall()
    return [r[0] for r in rows]


def _looseEndIds(store, agent):
    """Live atoms addressed to ``agent`` via the ``for:<agent>`` tag, newest first.
    A null agent cannot be addressed, so it has no loose ends."""
    if agent is None:
        return []
    value = f"{FOR_TAG_PREFIX}{agent}"
    rows = store._conn.execute(
        "SELECT id FROM atoms "
        "WHERE status = 'live' "
        "AND id IN (SELECT atom_id FROM facets WHERE key = ? AND value = ?) "
        "ORDER BY COALESCE(occurred_at, created_at) DESC, id",
        (LOOSE_TAG_KEY, value),
    ).fetchall()
    return [r[0] for r in rows]


def _addressedIds(store):
    """Live-or-not atom ids carrying any ``for:<agent>`` tag -- directed notes that
    belong in an agent's loose-ends section, never the ambient active threads. A
    note addressed to one agent must not clutter another agent's active view, so
    every addressed atom is excluded from active regardless of which agent it names.
    Status is irrelevant here (this only ever feeds a set-subtraction)."""
    rows = store._conn.execute(
        "SELECT DISTINCT atom_id FROM facets WHERE key = ? AND value LIKE ?",
        (LOOSE_TAG_KEY, f"{FOR_TAG_PREFIX}%"),
    ).fetchall()
    return {r[0] for r in rows}


def _activeRanked(store, now, excludeIds):
    """Active-thread candidates ranked by ``importanceFactor * timeFactor``.

    Draws the ``ACTIVE_CANDIDATE_CAP`` most-recent live atoms, drops the ones
    already shown elsewhere (pins, loose ends), scores each with fusion's own math
    -- :func:`recall.fusion.importanceFactor` times
    :func:`recall.fusion.timeFactor`, the exact prior
    :func:`recall.fusion.applyPriors` uses -- and returns ``[(atomId, project,
    score)]`` best-first. The sort is stable, so equal scores keep the recency
    order the SQL imposed."""
    rows = store._conn.execute(
        "SELECT id, project, importance, COALESCE(occurred_at, created_at) "
        "FROM atoms WHERE status = 'live' "
        "ORDER BY COALESCE(occurred_at, created_at) DESC, id "
        "LIMIT ?",
        (ACTIVE_CANDIDATE_CAP,),
    ).fetchall()
    scored = []
    for atomId, project, importance, effectiveTime in rows:
        if atomId in excludeIds:
            continue
        score = importanceFactor(importance) * timeFactor(now - effectiveTime)
        scored.append((atomId, project, score))
    scored.sort(key=lambda t: -t[2])
    return scored


# --- section rendering ----------------------------------------------------- #


def _renderPinned(store, pinnedIds, budget):
    """The pinned section, or None when there are no pins.

    Whole Tier-1 bodies when they fit the budget. When they do not, EVERY pin is
    kept as a Tier-0 handle and the ``pins exceed budget`` warning is appended --
    pins are the never-dropped tier, so this block may itself exceed ``budget``;
    that is the deliberate cost of never losing a standing principle."""
    if not pinnedIds:
        return None
    results = [_result(pid) for pid in pinnedIds]
    entries = [tier1Entry(store, r) for r in results]
    block = SECTION_PINNED + _LINE_SEP + _SECTION_SEP.join(entries)
    if estimateTokens(block) <= budget:
        return block
    handles = [tier0Handle(store, r) for r in results]
    return (
        SECTION_PINNED + _LINE_SEP
        + _LINE_SEP.join(handles) + _LINE_SEP
        + PINS_EXCEED_BUDGET
    )


def _renderActiveBlock(store, selected):
    """Render selected ``[(atomId, project)]`` as the active section: a header, then
    per-project sub-labels each followed by that project's Tier-0 handles.

    ``selected`` is in score order, so the first appearance of each project is its
    best-scored atom -- iterating an insertion-ordered dict therefore lists the
    strongest group first."""
    groups = {}
    for atomId, project in selected:
        groups.setdefault(project, []).append(atomId)
    lines = [SECTION_ACTIVE]
    for project, ids in groups.items():
        lines.append(_projectLabel(project))
        for atomId in ids:
            lines.append(tier0Handle(store, _result(atomId)))
    return _LINE_SEP.join(lines)


def _fitActive(store, scored, committed, budget):
    """Greedily fit active handles under the remaining budget; return the block or
    None. Adds atoms in score order and stops at the first that would push the whole
    brief-so-far over ``budget`` (atomic drop, best-first -- the payload rule)."""
    selected = []
    for atomId, project, _score in scored:
        trial = selected + [(atomId, project)]
        whole = _SECTION_SEP.join(committed + [_renderActiveBlock(store, trial)])
        if estimateTokens(whole) <= budget:
            selected = trial
        else:
            break
    if not selected:
        return None
    return _renderActiveBlock(store, selected)


def _fitLoose(store, looseIds, committed, budget, agent):
    """Greedily fit loose-end Tier-1 entries under the remaining budget; return the
    block or None. Same atomic-drop discipline as :func:`_fitActive`."""
    if not looseIds:
        return None
    header = looseHeader(agent)
    selected = []
    for atomId in looseIds:
        entry = tier1Entry(store, _result(atomId))
        trial = selected + [entry]
        block = header + _LINE_SEP + _SECTION_SEP.join(trial)
        whole = _SECTION_SEP.join(committed + [block])
        if estimateTokens(whole) <= budget:
            selected = trial
        else:
            break
    if not selected:
        return None
    return header + _LINE_SEP + _SECTION_SEP.join(selected)


def brief(store, opts=None):
    """Assemble the session-start working set for ``store`` -> plain-text string.

    ``opts`` is ``{agent, budget=1500}`` (the fixed interface), plus an optional
    ``now`` (unix seconds) that lets a caller brief "as of" a fixed clock -- it
    defaults to the wall clock and is the hook tests use for determinism.

    - ``agent``: the starting agent; drives the loose-ends section. ``None`` (or
      absent) yields no loose-ends section.
    - ``budget``: token budget for the whole brief by
      :func:`recall.payload.estimateTokens` (default :data:`DEFAULT_BUDGET`).

    Returns the composed brief: ``pinned`` (never dropped), then ``active``
    (recency+importance, budget-limited), then ``loose ends for <agent>``
    (budget-limited), each section omitted when it has no content. When every
    section is empty the result is :data:`EMPTY_BRIEF`. Performs ZERO writes.
    """
    opts = opts or {}
    agent = opts.get("agent")
    budget = opts.get("budget", DEFAULT_BUDGET)
    now = opts.get("now")
    if now is None:
        now = int(time.time())

    pinnedIds = _pinnedIds(store)
    looseIds = _looseEndIds(store, agent)
    # Active excludes pins (shown above) and every addressed note (it belongs in
    # some agent's loose-ends section, not the ambient active threads). looseIds is
    # a subset of the addressed set, so this also de-dups the current agent's notes.
    exclude = set(pinnedIds) | _addressedIds(store)
    activeScored = _activeRanked(store, now, exclude)

    sections = []
    pinnedBlock = _renderPinned(store, pinnedIds, budget)
    if pinnedBlock:
        sections.append(pinnedBlock)

    activeBlock = _fitActive(store, activeScored, sections, budget)
    if activeBlock:
        sections.append(activeBlock)

    looseBlock = _fitLoose(store, looseIds, sections, budget, agent)
    if looseBlock:
        sections.append(looseBlock)

    if not sections:
        return EMPTY_BRIEF
    return _SECTION_SEP.join(sections)
