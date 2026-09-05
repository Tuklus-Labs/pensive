"""Trust layer: an honest confidence, a shouldTrust boolean, and a one-clause
plain-text ``why`` on every reranked result -- plus the supersession invariant.

This is v2's paradigm-independent crown jewel ported onto fusion evidence. v2's
``boundary.py`` computed trust for spreading-activation retrieval; the MATH it
used -- the top1-top2 disambiguation gap, the decisive/tight gap thresholds, and
the should-trust gate -- transfers even though the surrounding envelope
(spreading-activation score arrays, frequency bands) does not. The specific
formulas are ported here with ``boundary.py:<line>`` attribution rather than
imported, because the input here is a reranked ``[(atomId, score)]`` list from a
cross-encoder, not an activation vector.

``confidence`` blends three sub-scores, each in [0,1], with weights that sum to
1.0 (so the blend is in [0,1] with no separate clamp needed):

  * signal agreement -- an atom hit by BOTH bm25 and dense is stronger than a
    single-signal atom; a facet hit adds a smaller increment. This is the
    fusion-world analogue of RRF's agreement reward.
  * the top1-top2 rerank gap -- a decisive lead of the top result over the
    runner-up raises the top result's confidence (ported from boundary.py).
  * a temporal term -- confidence decays with age toward, but never below, a
    floor (mirrors ``fusion.timeFactor`` but with trust-specific constants so
    Task 11's harness tunes trust independently of the fusion prior).

THE SUPERSESSION INVARIANT (critical): a ``superseded`` atom NEVER surfaces
alone. If it appears at all it carries a ``supersededBy`` key pointing at the
LIVE end of its supersession chain, with confidence capped below ``TRUST_FLOOR``
(a superseded fact is historical -- never trusted). A chain that ends in a
tombstone or a missing atom drops the entry entirely. A supersession cycle is
canonical corruption and raises. No output row is ever ``status='superseded'``
without a ``supersededBy`` key pointing at a live atom.

signalHits contract (defined here; Task 10 threads it through): a dict
``atomId -> set`` of signal names drawn from ``{"bm25", "dense", "facet"}`` --
which Task 6 signals hit each atom. A reranked atom ABSENT from ``signalHits`` is
treated as having hit no known signals (an empty set) -> minimal agreement and an
honest low confidence; this is explanation data, so a gap degrades the reason
rather than corrupting the result. That is distinct from the store desync below,
which IS correctness-critical (no status -> the invariant cannot be enforced) and
so fails loudly.

Stdlib + ``math`` only; ``store._conn`` reads are sanctioned.
"""
import math

__all__ = [
    "assessTrust",
    "TRUST_FLOOR",
    "SUPERSEDED_CONF_CAP",
    "UNCORROBORATED_CORPUS_CONF_CAP",
    "CORPUS_KINDS",
    "IMPORT_SOURCES",
    "CORROBORATING_SIGNALS",
    "W_AGREE",
    "W_GAP",
    "W_TIME",
    "AGREE_PRIMARY",
    "AGREE_DUAL_BONUS",
    "AGREE_FACET",
    "GAP_TAU",
    "GAP_NEUTRAL",
    "GAP_DECISIVE",
    "GAP_SINGLE",
    "TRUST_TAU",
    "TRUST_TIME_FLOOR",
    "TIME_RECENT",
    "TIME_AGED",
]

# --- blend weights --------------------------------------------------------- #
# Sum to 1.0 so that with every sub-score in [0,1] the blend is in [0,1] with no
# clamp required. Agreement dominates (independent signals agreeing is the core
# "is this really relevant" evidence); the gap is disambiguation; time modulates.
W_AGREE = 0.5
W_GAP = 0.3
W_TIME = 0.2

# The no-clamp-needed contract, documented as an assertion: the blend can only
# stay in [0,1] for every sub-score in [0,1] because the weights sum to exactly
# 1.0. The clamp in assessTrust stays as a defensive backstop; this guards the
# invariant at import so a future weight edit that breaks it fails loudly here.
assert W_AGREE + W_GAP + W_TIME == 1.0, "blend weights must sum to 1.0"

# --- signal-agreement sub-score -------------------------------------------- #
# The fusion-world analogue of RRF's agreement reward (fusion.py rrf docstring):
# one primary signal earns the base; bm25 AND dense together earn the bonus on
# top; a facet hit adds a smaller increment. Capped at 1.0.
AGREE_PRIMARY = 0.5       # any one of {bm25, dense} present
AGREE_DUAL_BONUS = 0.4    # both bm25 AND dense present (0.5 + 0.4 = 0.9)
AGREE_FACET = 0.1         # facet hit -- the smaller increment

# --- disambiguation sub-score (ported from boundary.py) -------------------- #
# v2 computed ``disambiguation_gap = top_scores[0] - top_scores[1]``
# (boundary.py:307-309) and treated a gap >= 0.10 as a clear winner
# (boundary.py:58-61, the "high" band) and a gap < 0.05 as tied/ambiguous
# (boundary.py:62-64 and :324-326, which set context_needed). Those absolute
# thresholds lived on the spreading-activation score scale; rerank scores are
# cross-encoder logits, so the gap is mapped through a saturating
# ``1 - exp(-gap / GAP_TAU)`` (structural mirror of fusion.timeFactor's exp) that
# rises from 0 toward 1 as the lead grows. The top-1 sub-score is FLOORED at
# GAP_NEUTRAL so the gap term only ever ADDS confidence to a decisive leader and
# never inverts a near-tie below a mid-list atom. Ported as a continuous score
# rather than v2's discrete high/medium/low band (boundary.py:49-66), because the
# brief's shouldTrust is a continuous floor, not a band.
GAP_TAU = 1.5             # separation e-folding scale, in rerank-logit units
GAP_NEUTRAL = 0.5         # floor for the top-1 gap sub-score; also non-top atoms
GAP_DECISIVE = 0.75       # gapScore >= this reads as "decisive gap" in why
GAP_SINGLE = 1.0          # a lone result has no competitor -> nothing to resolve

# --- temporal sub-score (mirrors fusion.timeFactor, trust-tunable) --------- #
# Same shape as fusion.timeFactor (FLOOR + (1-FLOOR)*exp(-age/TAU)) but with
# distinct constants so the harness tunes trust decay independently of the
# fusion time prior. Floored above zero: an old atom loses freshness, never all
# confidence. Age is clamped to >= 0 (a future-dated atom reads as freshest, and
# the clamp keeps exp from overflowing on an absurd far-future timestamp).
TRUST_TAU = 180 * 86_400  # ~6 months; the "recent" scale for trust
TRUST_TIME_FLOOR = 0.5    # temporal sub-score never decays below this
TIME_RECENT = 0.9         # temporalScore >= this reads as "recent" in why
TIME_AGED = 0.6           # temporalScore <= this reads as "aged" in why

# --- trust decision + supersession ----------------------------------------- #
TRUST_FLOOR = 0.6         # shouldTrust = confidence >= TRUST_FLOOR
# A superseded atom that survives (its chain ends live) is historical: its
# confidence is capped strictly below TRUST_FLOOR so shouldTrust is always False.
SUPERSEDED_CONF_CAP = 0.4

# A bulk-imported corpus row found by one signal family lacks corroboration.
# Cap it strictly below TRUST_FLOOR so `shouldTrust` and
# `confidence` cannot disagree, mirroring the supersession cap rather than
# inventing a second mechanism.
#
# Measured 2026-08-12: 93% of the live store is bulk-imported `document_chunk`,
# and a keyword-only chunk ranked top with a decisive gap scores 0.749 against a
# 0.6 floor. The blend cannot see this by itself -- it reads signal agreement,
# the top1-top2 gap and recency, and a top-ranked recent chunk scores well on all
# three while resting on exactly one signal. So the trust bit was carrying almost
# no information across the majority of the store.
UNCORROBORATED_CORPUS_CONF_CAP = 0.4

# Kinds that are IMPORTED MATERIAL rather than authored memory.
CORPUS_KINDS = frozenset({"document_chunk"})

# Provenance sources that mean "this row came from an import, not an agent".
IMPORT_SOURCES = frozenset({"bulk-import"})

# Non-lexical signal families. Two different families must agree; a dense hit
# alone does not corroborate itself. The auxiliary 'openai' tag aliases dense.
CORROBORATING_SIGNALS = frozenset({"dense", "facet"})


def _agreementScore(signals):
    """Signal-agreement sub-score in [0,1] from a set of hit signal names."""
    hasBm25 = "bm25" in signals
    hasDense = "dense" in signals
    hasFacet = "facet" in signals
    score = 0.0
    if hasBm25 or hasDense:
        score += AGREE_PRIMARY
    if hasBm25 and hasDense:
        score += AGREE_DUAL_BONUS
    if hasFacet:
        score += AGREE_FACET
    return min(1.0, score)


def _gapScore(isTop, gap):
    """Disambiguation sub-score in [0,1].

    ``gap`` is ``score(top1) - score(top2)`` for the top-ranked atom, or None
    when there is no runner-up (a lone result). Only the top-1 atom carries a
    real gap; every other atom gets GAP_NEUTRAL because disambiguation is a
    property of the leader (boundary.py computes it solely from the top two).
    """
    if not isTop:
        return GAP_NEUTRAL
    if gap is None:
        return GAP_SINGLE
    saturating = 1.0 - math.exp(-max(0.0, gap) / GAP_TAU)
    # Floor at neutral: a tight gap yields no bonus, never a penalty below a
    # mid-list atom. Ported thresholds (boundary.py:59/63) decide the why label.
    return max(GAP_NEUTRAL, saturating)


def _temporalScore(ageSeconds):
    """Freshness sub-score in [TRUST_TIME_FLOOR, 1.0]; never zero.

    Mirrors ``fusion.timeFactor`` with trust-specific constants. Negative age
    (future-dated atom) clamps to 0 -> the freshest boost, which also stops exp
    from overflowing on a far-future timestamp.
    """
    age = max(0, ageSeconds)
    return TRUST_TIME_FLOOR + (1.0 - TRUST_TIME_FLOOR) * math.exp(-age / TRUST_TAU)


def _buildWhy(signals, gapScore, temporalScore, isTop):
    """One plain-text clause of comma-separated evidence tokens (no markdown,
    no newline). Always non-empty: the signal token is always emitted."""
    hasBm25 = "bm25" in signals
    hasDense = "dense" in signals
    hasFacet = "facet" in signals

    parts = []
    if hasBm25 and hasDense:
        parts.append("both signals agree")
    elif hasBm25:
        parts.append("keyword signal only")
    elif hasDense:
        parts.append("semantic signal only")
    if hasFacet:
        parts.append("facet match")
    if not parts:
        parts.append("no signal evidence")

    # Gap label only for the top-1 (the only atom with a real gap sub-score).
    if isTop:
        if gapScore >= GAP_DECISIVE:
            parts.append("decisive gap")
        elif gapScore <= GAP_NEUTRAL:
            parts.append("tight gap")

    if temporalScore >= TIME_RECENT:
        parts.append("recent")
    elif temporalScore <= TIME_AGED:
        parts.append("aged")

    return ", ".join(parts)


def _successorMap(store, rerankedIds):
    """One SELECT for the relevant supersedes edges -> (successorMap, chainIds,
    forkedOldIds).

    A supersedes edge points new->old (src=new successor, dst=old superseded),
    so ``successorMap[old] = new``. A recursive CTE walks the successor chains
    reachable from the reranked atoms in a single query; UNION dedups, so a
    supersession cycle yields a finite edge set (the Python walk that consumes
    this map detects the cycle and raises). ``chainIds`` is every atom named by
    those edges, so the caller can fold them into the one status/time SELECT.
    ``forkedOldIds`` is the set of old atoms with MORE THAN ONE distinct
    successor edge -- a fork the deterministic pin resolves but downstream should
    still be told about (via the ``why`` marker).
    """
    if not rerankedIds:
        return {}, set(), set()
    placeholders = ",".join("?" for _ in rerankedIds)
    rows = store._conn.execute(
        f"""
        WITH RECURSIVE walk(old_id, new_id) AS (
            SELECT dst_atom, src_atom FROM edges
              WHERE type = 'supersedes' AND dst_atom IN ({placeholders})
            UNION
            SELECT e.dst_atom, e.src_atom FROM edges e
              JOIN walk w ON e.dst_atom = w.new_id
              WHERE e.type = 'supersedes'
        )
        SELECT old_id, new_id FROM walk ORDER BY old_id, new_id
        """,
        tuple(rerankedIds),
    ).fetchall()

    successorMap = {}
    chainIds = set()
    successorsByOld = {}
    for oldId, newId in rows:
        # First successor wins (rows are ORDER BY old_id,new_id, so the pinned
        # branch is the smallest-ULID successor -- stable across runs). A fork --
        # two atoms both claiming to supersede one old atom -- is unexpected, so
        # pin it deterministically rather than let dict insertion order decide.
        successorMap.setdefault(oldId, newId)
        chainIds.add(oldId)
        chainIds.add(newId)
        successorsByOld.setdefault(oldId, set()).add(newId)
    forkedOldIds = {old for old, succ in successorsByOld.items() if len(succ) > 1}
    return successorMap, chainIds, forkedOldIds


def _liveEnd(atomId, successorMap, forkedOldIds):
    """Walk the supersession chain from a superseded atom to its terminal node.

    Returns ``(terminal, forked)``: the id of the atom the chain ends at (the one
    nobody supersedes -- may be live, tombstoned, missing, or under corruption
    still superseded), and whether the walked path crossed a fork point (a node
    with more than one successor edge, where the pin chose one branch). Raises
    ValueError naming the atoms on a cycle (a superseded atom revisited while
    walking).
    """
    seen = []
    forked = False
    cur = atomId
    while cur in successorMap:
        if cur in seen:
            names = ", ".join(sorted(set(seen + [cur])))
            raise ValueError(
                f"supersession cycle detected among atoms: {names}"
            )
        if cur in forkedOldIds:
            forked = True
        seen.append(cur)
        cur = successorMap[cur]
    return cur, forked


def _isUncorroboratedCorpus(kind, sources, signals):
    """True when imported material has fewer than two signal families.

    All three must hold, and each is doing work:

    * ``kind`` is corpus -- an authored atom is a memory whatever found it.
    * EVERY provenance source is an import -- a row an agent also wrote through
      the emit path is not the 283k-row bulk load, and "one source is an import"
      would condemn those.
    * fewer than two independent signal families -- aliases of the same dense
      signal do not turn one match into corroboration.

    A row with NO provenance at all is not treated as corpus: absent provenance
    is unknown, and unknown is not a licence to downgrade something an agent may
    have written. That fails toward trusting too much, which is the direction the
    supersession invariant already guards.
    """
    if kind not in CORPUS_KINDS:
        return False
    if not sources or not sources <= IMPORT_SOURCES:
        return False
    families = signals & ({"bm25"} | CORROBORATING_SIGNALS)
    if "openai" in signals:
        families = families | {"dense"}
    return len(families) < 2


def assessTrust(reranked, signalHits, Store, now):
    """Annotate reranked results with confidence, shouldTrust, why, supersession.

    ``reranked`` is the ``[(atomId, score)]`` list from :func:`recall.rerank.rerank`,
    best-first. ``signalHits`` maps ``atomId -> set`` of hit signal names (see the
    module docstring). ``Store`` is the canonical store; ``now`` is unix seconds.

    Returns a list, in reranked order minus drops, of dicts:
    ``{atomId, score, confidence, shouldTrust, why}`` for a live atom, plus a
    ``supersededBy`` key for a surviving superseded atom. ``confidence`` is in
    [0,1]; ``shouldTrust = confidence >= TRUST_FLOOR``; ``why`` is one plain-text
    clause.

    Supersession: a ``superseded`` atom is dropped unless its chain resolves to a
    LIVE end, in which case it survives with ``supersededBy`` set to that live id
    and confidence capped below TRUST_FLOOR. A chain ending in a tombstone or a
    missing atom drops the entry; a cycle raises. A ``tombstone`` atom in the
    input is dropped (it has been retracted). An unrecognized status is
    corruption and raises.

    Reads are batched: one recursive-CTE SELECT for the relevant supersedes
    edges, one SELECT for statuses/effective-times over the reranked atoms and
    every atom named by those edges. A reranked atom absent from the store is an
    index/store desync and raises ValueError naming it (parity with
    :func:`recall.fusion.applyPriors`).
    """
    if not reranked:
        return []

    rerankedIds = [atomId for atomId, _ in reranked]

    # One SELECT for the relevant supersedes edges (successor chains).
    successorMap, chainIds, forkedOldIds = _successorMap(Store, rerankedIds)

    # One SELECT for statuses + effective times over the reranked atoms and every
    # atom named by a chain edge (so a chain's live end can be resolved without a
    # second round-trip). Time convention matches fusion.py: COALESCE(occurred_at,
    # created_at).
    allIds = list(set(rerankedIds) | chainIds)
    placeholders = ",".join("?" for _ in allIds)
    rows = Store._conn.execute(
        f"SELECT id, status, COALESCE(occurred_at, created_at), kind "
        f"FROM atoms WHERE id IN ({placeholders})",
        tuple(allIds),
    ).fetchall()
    statusInfo = {r[0]: (r[1], r[2], r[3]) for r in rows}

    # One SELECT for provenance sources over the same ids. An atom can carry
    # several provenance rows, so this is a SET per atom: "every source is an
    # import" is a different claim from "one source is an import", and only the
    # first justifies withholding trust.
    sourcesById = {}
    for atomId, source in Store._conn.execute(
        f"SELECT atom_id, source FROM provenance WHERE atom_id IN ({placeholders})",
        tuple(allIds),
    ).fetchall():
        sourcesById.setdefault(atomId, set()).add(source)

    # The top-1's lead over the runner-up drives disambiguation for the top atom.
    topGap = None
    if len(reranked) >= 2:
        topGap = reranked[0][1] - reranked[1][1]

    out = []
    for index, (atomId, score) in enumerate(reranked):
        if atomId not in statusInfo:
            # A reranked id absent from the store means the index and store have
            # desynced -- surface it loudly (parity with applyPriors/rerank).
            raise ValueError(
                f"trust candidate {atomId!r} is absent from the store "
                "(index/store desync)"
            )
        status, effectiveTime, kind = statusInfo[atomId]
        isTop = index == 0

        signals = signalHits.get(atomId, set())
        agreement = _agreementScore(signals)
        gapScore = _gapScore(isTop, topGap if isTop else None)
        temporalScore = _temporalScore(now - effectiveTime)
        blend = W_AGREE * agreement + W_GAP * gapScore + W_TIME * temporalScore
        confidence = max(0.0, min(1.0, blend))

        if status == "live":
            why = _buildWhy(signals, gapScore, temporalScore, isTop)
            if _isUncorroboratedCorpus(kind, sourcesById.get(atomId, set()), signals):
                confidence = min(confidence, UNCORROBORATED_CORPUS_CONF_CAP)
                why = "imported corpus, insufficient independent signals, uncorroborated"
            out.append({
                "atomId": atomId,
                "score": score,
                "confidence": confidence,
                "shouldTrust": confidence >= TRUST_FLOOR,
                "why": why,
            })
        elif status == "tombstone":
            # A retracted atom must not surface in recall.
            continue
        elif status == "superseded":
            end, forked = _liveEnd(atomId, successorMap, forkedOldIds)
            endStatus = statusInfo.get(end, (None, None, None))[0]
            if endStatus != "live":
                # Chain ends in a tombstone, a missing atom, or (corruption) a
                # superseded terminal with no successor edge: never surface a
                # superseded atom without a LIVE successor -- drop it entirely.
                # (A fork whose PINNED branch ends dead also drops here; walking
                # the other branches is deferred to the harness era.)
                continue
            # Historical fact: keep it discoverable but never trusted. Decay the
            # earned confidence, then cap strictly below TRUST_FLOOR. A fork on
            # the resolved path is surfaced so the ambiguity is visible: the pin
            # is deterministic, but it did choose one of several successors.
            capped = min(confidence, SUPERSEDED_CONF_CAP)
            why = (
                "superseded by newer atom (forked)"
                if forked
                else "superseded by newer atom"
            )
            out.append({
                "atomId": atomId,
                "score": score,
                "confidence": capped,
                "shouldTrust": capped >= TRUST_FLOOR,   # always False by the cap
                "why": why,
                "supersededBy": end,
            })
        else:
            raise ValueError(
                f"atom {atomId!r} has unrecognized status {status!r} "
                "(store corruption)"
            )

    return out
