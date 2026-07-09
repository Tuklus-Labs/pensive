"""Recall orchestrator: the whole Phase 2 pipeline as one function.

Everything Tasks 6-9 built becomes a single call here. ``recall`` threads three
parallel signals through fusion, priors, rerank, and trust, then hands the
annotated results to the payload assembler:

    per-class signals  ->  RRF fusion  ->  facet boost  ->  priors  ->  kinds
        filter  ->  round-robin rerank head  ->  cross-encoder rerank  ->  trust
        ->  tiered payload

The ordering resolves ambiguities that span the parts, so it is fixed and worth
stating in one place:

- The project hint drives ``facetSignal``; its ``filterSet`` (when not None)
  restricts the bm25 and dense candidate lists BEFORE fusion. A degenerate
  window -- a real project with no live atoms -- yields an empty ``filterSet``,
  which short-circuits to the low-confidence sentinel rather than an error.
- The time window (``timeScope``) is applied EXACTLY ONCE, inside ``applyPriors``.
  It is deliberately NOT also passed to ``facetSignal`` as a ``timeRange``; the
  window must not be double-applied.
- The facet ``boostSet`` is a post-fusion MULTIPLICATIVE boost (``* FACET_BOOST``)
  applied before priors, and its members count as a ``facet`` signal hit.
- ``kinds`` filters the fused list in one batched SELECT BEFORE rerank, so the
  rerank head is spent on eligible atoms.
- ``now`` is sampled ONCE at entry and threaded to both the time prior and the
  trust layer, so every clock-dependent decision in a single call agrees.

Loud failures propagate: a desync ``ValueError`` from ``applyPriors``/``rerank``/
``assessTrust`` or a supersession-cycle error bubbles up unswallowed -- the
decades rule says surface inconsistency, never paper over it.

Daemon-internal modules may read through ``store._conn`` by sanctioned convention
for batched SELECTs the public store API does not expose; the ``kinds`` SELECT
below is one of those intentional internal reads.

The public shape is the ``RecallResult`` dict Tasks 11 (eval) and 12 (MCP) build
on: ``{results, payload, tokensUsed, lowConfidence}``.
"""
import time

from recall.strata import classesForKinds, interleave, splitByClass
from recall.signals import bm25, dense, facetSignal
from recall.fusion import rrf, applyPriors
from recall.rerank import rerank
from recall.trust import assessTrust
from recall.payload import assemblePayload

__all__ = ["recall", "FACET_BOOST", "RERANK_HEAD"]

# Post-fusion multiplicative boost for atoms carrying a query-matching entity
# facet. Small on purpose (a nudge, not an override) and harness-tunable: Task
# 11's eval owns the final value. 1.15 = a 15% lift to a boosted atom's fused
# score before priors.
FACET_BOOST = 1.15

# Cross-encoder rerank head. The Task 8 warm budget was benched at ~31ms for 50
# pairs; scoring only the top 64 fused candidates keeps margin while bounding the
# pool, and the harness owns future tuning. Candidates below this head keep their
# fused RRF order and are appended after the reranked head, never dropped.
RERANK_HEAD = 64

# Recall breadth per signal: the plan's top-200 candidate generation. Matches the
# signals-module default; named here so the orchestrator states its own contract.
_SIGNAL_K = 200


def recall(store, indexes, embedder, query, project=None, timeScope=None,
           kinds=None, k=10, tokenBudget=1500):
    """Run the full recall pipeline and assemble a tiered payload.

    ``store`` is the canonical store, ``indexes`` a ``{className: VectorIndex}``
    map (one built dense index per kind-class, from ``buildClassIndexes``), and
    ``embedder`` a loaded ``Embedder``. Candidates are generated PER CLASS (a
    kind-scoped bm25 list plus that class's own dense index), so the large code
    corpus can no longer starve the smaller reasoning memory out of the pool.
    Fusion, the facet boost, and priors run globally as before; the rerank head is
    then filled by round-robin across the per-class fused rankings, so the
    cross-encoder scores a fair mix and its scores decide the final order. Options:

    - ``project``: restrict recall to that project's live atoms (facet filter).
    - ``timeScope`` ``(startUnix, endUnix)``: restrict to atoms whose effective
      time falls in the inclusive window (applied once, in the priors stage).
    - ``kinds``: restrict to these atom kinds. Stratification runs over only the
      classes overlapping ``kinds`` (a single-class request degrades to today's
      non-stratified behavior).
    - ``k``: number of results to return (default 10).
    - ``tokenBudget``: payload budget by the conservative heuristic (default 1500).

    Returns the ``RecallResult`` dict ``{results, payload, tokensUsed,
    lowConfidence}``.
    """
    now = int(time.time())

    # Empty/whitespace query: no lexical signal (bm25 sanitizes to nothing) and no
    # MEANINGFUL dense signal -- but dense() still embeds whitespace to a valid
    # unit vector and would return the whole index, so the pipeline alone would
    # surface junk. Guard it: the contract is empty results + the low-confidence
    # sentinel. (Verified empirically that dense("") / dense("   ") return the full
    # index, so this guard is load-bearing, not decorative.)
    if not query or not query.strip():
        return _emptyResult(store, tokenBudget)

    # 1. Facet signal: project filterSet + entity boostSet from the query text.
    #    timeScope is intentionally NOT forwarded here (it belongs to the priors
    #    stage, applied exactly once).
    facetHints = {"query": query}
    if project is not None:
        facetHints["project"] = project
    facet = facetSignal(store, facetHints)
    boostSet = facet["boostSet"]
    filterSet = facet["filterSet"]

    # A present-but-empty filterSet means the project/window matched no live atom.
    # Nothing is recallable; return the sentinel, not an error.
    if filterSet is not None and not filterSet:
        return _emptyResult(store, tokenBudget)

    # 2. Per-class candidate generation. classesForKinds(None) is every class;
    #    a kinds subset narrows to the overlapping classes. No class -> sentinel.
    classes = classesForKinds(kinds)
    if not classes:
        return _emptyResult(store, tokenBudget)
    classNames = [name for name, _ in classes]

    bmHits = []
    dnHits = []
    for name, classKinds in classes:
        bmHits.extend(bm25(store, query, _SIGNAL_K, kinds=classKinds))
        classIndex = indexes.get(name)
        if classIndex is not None:
            dnHits.extend(dense(classIndex, embedder, query, _SIGNAL_K))

    # signalHits: which of {bm25, dense, facet} hit each atom, drawn from the raw
    # top-200 lists and the boostSet. Trust reads this as explanation evidence.
    signalHits = {}
    for atomId, _ in bmHits:
        signalHits.setdefault(atomId, set()).add("bm25")
    for atomId, _ in dnHits:
        signalHits.setdefault(atomId, set()).add("dense")
    for atomId in boostSet:
        signalHits.setdefault(atomId, set()).add("facet")

    # 3. The project filterSet restricts the candidate universe BEFORE fusion.
    if filterSet is not None:
        bmHits = [pair for pair in bmHits if pair[0] in filterSet]
        dnHits = [pair for pair in dnHits if pair[0] in filterSet]

    # 4. Reciprocal-rank fusion (rank-only; per-signal scores are not comparable).
    fused = rrf([bmHits, dnHits])

    # 5. Post-fusion facet boost, applied before priors. applyPriors re-sorts, so
    #    the boosted scores feed the ranking that decides the rerank head.
    if boostSet:
        fused = [
            (atomId, score * FACET_BOOST if atomId in boostSet else score)
            for atomId, score in fused
        ]

    # 6. Priors: importance * time decay, OR (under timeScope) a window
    #    restriction. The window is applied here and ONLY here.
    priorHints = {"now": now}
    if timeScope is not None:
        priorHints["timeScope"] = timeScope
    fused = applyPriors(fused, store, priorHints)

    # 7. kinds filter before the head is built: the per-class dense index for a
    #    class returns every kind in that class, so an explicit narrow kinds
    #    request (e.g. only 'narrative') still needs this to drop the other
    #    in-class kinds. An empty result here short-circuits.
    if kinds is not None:
        fused = _filterKinds(store, fused, kinds)

    if not fused:
        return _emptyResult(store, tokenBudget)

    # 8. Stratified rerank head: round-robin across the per-class fused rankings so
    #    the cross-encoder scores a fair mix of both classes, then let its scores
    #    decide. The untouched fused tail keeps global RRF order below the head.
    perClass = splitByClass(fused, store, classNames)
    rerankHead = interleave([perClass[name] for name in classNames], RERANK_HEAD)
    rerankedHead = rerank(query, rerankHead, store)
    rerankedIds = {atomId for atomId, _ in rerankedHead}
    headIds = {atomId for atomId, _ in rerankHead}
    reranked = (
        rerankedHead
        + [pair for pair in rerankHead if pair[0] not in rerankedIds]
        + [pair for pair in fused if pair[0] not in headIds]
    )

    # 9. Trust: confidence, shouldTrust, why, supersession -- same ``now``.
    assessed = assessTrust(reranked, signalHits, store, now)

    # 10. Trim to k and assemble the tiered payload within budget.
    results = assessed[:k]
    payload, tokensUsed, lowConfidence = assemblePayload(store, results, tokenBudget)
    return {
        "results": results,
        "payload": payload,
        "tokensUsed": tokensUsed,
        "lowConfidence": lowConfidence,
    }


def _emptyResult(store, tokenBudget):
    """The no-candidates RecallResult: empty results, the low-confidence sentinel,
    lowConfidence True. Routed through ``assemblePayload`` so the sentinel and its
    token count come from the one payload authority, never a second copy."""
    payload, tokensUsed, lowConfidence = assemblePayload(store, [], tokenBudget)
    return {
        "results": [],
        "payload": payload,
        "tokensUsed": tokensUsed,
        "lowConfidence": lowConfidence,
    }


def _filterKinds(store, fused, kinds):
    """Keep only fused atoms whose ``kind`` is in ``kinds`` -- one batched SELECT.

    An empty ``kinds`` (or one naming no present kind) filters everything out and
    returns ``[]``, which the caller reads as "nothing eligible" -> sentinel."""
    if not fused:
        return []
    kindSet = set(kinds)
    if not kindSet:
        return []
    atomIds = [atomId for atomId, _ in fused]
    placeholders = ",".join("?" for _ in atomIds)
    rows = store._conn.execute(
        f"SELECT id, kind FROM atoms WHERE id IN ({placeholders})",
        tuple(atomIds),
    ).fetchall()
    kindById = {r[0]: r[1] for r in rows}
    return [pair for pair in fused if kindById.get(pair[0]) in kindSet]
