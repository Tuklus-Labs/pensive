"""Fusion + priors: combine the Task 6 signals, then bias by importance and time.

Two stages, kept separate so Task 11's eval harness can tune each independently:

- :func:`rrf` -- Reciprocal Rank Fusion. Merges several best-first ranked lists
  into one. It reads ONLY rank positions, never the per-signal scores (BM25's
  negated cost and cosine similarity are not on a shared scale, so their raw
  magnitudes are not comparable -- rank is). An atom's fused score is the sum
  over the lists it appears in of ``weight / (k + rank)`` (rank 1-based). Because
  the term shrinks with rank, a high placement contributes more; because it
  SUMS across lists, an atom several signals agree on outranks one a single
  signal placed higher. That agreement reward is the whole point of RRF.

- :func:`applyPriors` -- the two-sided time prior plus importance. Multiplies
  each fused score by ``importanceFactor * timeFactor`` (or, under a
  ``timeScope``, restricts to a window instead of decaying). The load-bearing
  invariant is the FLOOR on ``timeFactor``: a fresh atom gets at most a 2x boost
  over the oldest possible atom, so importance (also up to 2x) can always offset
  age. "Decades" can outrank "freshness"; age alone never buries an important
  old memory. See :func:`timeFactor`.

Time semantics match Task 6's facet filter exactly: an atom's effective time is
``COALESCE(occurred_at, created_at)`` -- when the remembered thing happened wins,
the record time is the fallback.

Stdlib + ``math`` only; the fusion stage is dependency-free by design.
"""
import math

__all__ = ["rrf", "applyPriors", "timeFactor", "importanceFactor", "FLOOR", "TAU"]

# --- time-prior constants (named so the floor property is legible) --------- #

# timeFactor never decays below this. With importanceFactor topping out at 2.0
# and timeFactor bounded in [FLOOR, 1.0], the freshest-trivial atom scores
# 1.0 * 1.0 * 1.0 = 1.0 while an oldest-but-maximally-important atom scores
# 1.0 * 2.0 * 0.5 = 1.0 -- a tie, never a burial. Lower FLOOR would let age win.
FLOOR = 0.5

_DAY_SECONDS = 86_400
# e-folding time of the freshness boost: after TAU seconds the above-floor part
# of timeFactor has decayed to 1/e. 90 days -- a season -- is the "recent" scale.
TAU = 90 * _DAY_SECONDS


def rrf(lists, k=60, weights=None):
    """Reciprocal-rank-fuse several best-first ranked lists -> ``[(atomId, fused)]``.

    ``lists`` is a list of signal outputs, each ``[(atomId, score), ...]`` ranked
    best-first (the Task 6 convention). The per-tuple ``score`` is IGNORED -- only
    an atom's 1-based rank position within each list is used. An atom's fused
    score is::

        sum over the lists it appears in of  weight_list / (k + rank)

    ``k`` (default 60, the canonical RRF constant) damps the influence of exact
    rank so lower placements still contribute. ``weights`` is the per-list tuning
    lever Task 11's harness owns; it defaults to 1.0 for every list (uniform
    fusion). When supplied it must have one entry per list, else ``ValueError``.

    The result is sorted best-first by fused score; ties keep first-seen order
    (a stable sort over insertion order), so fusion is deterministic. Empty input
    (or all-empty lists) yields ``[]``.
    """
    if weights is None:
        weights = [1.0] * len(lists)
    elif len(weights) != len(lists):
        raise ValueError(
            f"weights length {len(weights)} does not match lists length {len(lists)}"
        )

    # dict preserves first-seen order (Python 3.7+); a later stable sort by score
    # therefore breaks ties by first appearance, keeping fusion deterministic.
    scores = {}
    for ranked, weight in zip(lists, weights):
        for rank, (atomId, _score) in enumerate(ranked, start=1):
            scores[atomId] = scores.get(atomId, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: -kv[1])


def timeFactor(ageSeconds):
    """Freshness multiplier for an atom ``ageSeconds`` old -> a float in [FLOOR, 1.0].

    ``FLOOR + (1 - FLOOR) * exp(-age / TAU)`` with ``age = max(0, ageSeconds)``:
    1.0 at age 0, decaying toward -- but never below -- ``FLOOR`` as age grows,
    and never ABOVE 1.0. Clamping negative age to 0 is load-bearing on the upper
    side: a future-dated atom (``occurred_at`` ahead of ``now`` -- a skewed clock
    or a bulk import of misdated archives) is treated as freshest instead of
    earning a boost > 1.0 that would break this postcondition, and the clamp also
    keeps ``exp`` from overflowing on an absurd far-future timestamp (which would
    otherwise raise and crash the whole recall call). The FLOOR is the lower-side
    design: it caps how much freshness can matter, so a maximally-important old
    atom (importanceFactor 2.0 * FLOOR 0.5 = 1.0) is never buried beneath a fresh
    trivial one (importanceFactor 1.0 * timeFactor 1.0 = 1.0). Age alone cannot
    win. For a very large age the exponential underflows to 0.0, so the factor
    equals ``FLOOR`` exactly. Bounded in [FLOOR, 1.0] for ANY input, positive or
    negative.
    """
    age = max(0, ageSeconds)
    return FLOOR + (1.0 - FLOOR) * math.exp(-age / TAU)


def importanceFactor(importance):
    """Importance multiplier -> a float in [1.0, 2.0].

    ``1.0 + min(importance, 1.0)``: importance is earned (0.0 by default) and
    capped at 1.0, so the factor is bounded in [1.0, 2.0] and a runaway importance
    value cannot dominate. It is the importance half of the ``importanceFactor *
    timeFactor`` prior; paired with :func:`timeFactor` (bounded [FLOOR, 1.0]) the
    two guarantee a maximally-important old atom (2.0 * 0.5 = 1.0) is never buried
    beneath a fresh trivial one (1.0 * 1.0 = 1.0). The single place this formula
    lives -- :func:`applyPriors` and the ambient briefer both call it, so the
    ranking math never forks."""
    return 1.0 + min(importance, 1.0)


def applyPriors(fused, store, hints):
    """Bias fused scores by importance and time -> ``[(atomId, score)]`` best-first.

    Each score becomes ``fusedScore * importanceFactor * timeFactor`` where:

    - ``importanceFactor`` (:func:`importanceFactor`) -- ``1.0 + min(importance,
      1.0)``: importance is earned (0.0 by default) and capped at 1.0, so the
      factor is bounded in [1.0, 2.0] and a runaway importance value cannot dominate.
    - ``timeFactor`` is :func:`timeFactor` of the atom's age, measured from
      ``hints["now"]`` against its effective time ``COALESCE(occurred_at,
      created_at)``. Bounded in [FLOOR, 1.0]; see the module docstring for why
      the floor keeps old important atoms from being buried.

    ``hints["now"]`` is REQUIRED whenever the time prior runs (no ``timeScope``);
    a missing ``now`` raises ``ValueError`` naming the contract rather than
    surfacing a bare ``KeyError`` -- the time prior has no sensible default clock.

    When ``hints["timeScope"] = (startUnix, endUnix)`` is present, the time prior
    is NOT applied. Instead results are RESTRICTED to atoms whose effective time
    falls in the inclusive window; the importance factor still applies. A
    degenerate window (``start > end``) matches nothing and yields ``[]`` --
    consistent with the Task 6 facet time filter. ``hints["now"]`` is not read in
    this mode.

    Importance and timestamps are fetched in ONE batched ``SELECT ... IN`` over
    the fused atom ids (never per-atom). A fused candidate absent from the store
    means the index and store have desynced; that is an inconsistency the decades
    rule says to surface loudly, so it raises ``ValueError`` naming the missing
    id rather than silently dropping it. Empty ``fused`` -> ``[]``.
    """
    hints = hints or {}
    if not fused:
        return []

    atomIds = [atomId for atomId, _ in fused]
    placeholders = ",".join("?" for _ in atomIds)
    rows = store._conn.execute(
        f"SELECT id, importance, COALESCE(occurred_at, created_at) "
        f"FROM atoms WHERE id IN ({placeholders})",
        tuple(atomIds),
    ).fetchall()
    info = {r[0]: (r[1], r[2]) for r in rows}

    timeScope = hints.get("timeScope")
    scoped = timeScope is not None
    if scoped:
        start, end = timeScope
    elif "now" not in hints:
        # The time prior needs a reference clock and there is no sensible
        # default; name the contract loudly, like the desync error below, rather
        # than letting a bare KeyError surface from deep in the loop.
        raise ValueError(
            "applyPriors requires hints['now'] when timeScope is absent"
        )

    out = []
    for atomId, fusedScore in fused:
        if atomId not in info:
            raise ValueError(
                f"fused candidate {atomId!r} is absent from the store "
                "(index/store desync)"
            )
        importance, effectiveTime = info[atomId]
        impFactor = importanceFactor(importance)
        if scoped:
            # Window restriction replaces the time prior. start > end satisfies
            # nothing, so a degenerate window naturally yields [].
            if start <= effectiveTime <= end:
                out.append((atomId, fusedScore * impFactor))
        else:
            # timeFactor clamps a negative age (future-dated atom) to the
            # freshest boost, so a skewed occurred_at never scores above 1.0 nor
            # overflows exp -- the raw difference is safe to hand it.
            age = hints["now"] - effectiveTime
            out.append((atomId, fusedScore * impFactor * timeFactor(age)))

    out.sort(key=lambda kv: -kv[1])
    return out
