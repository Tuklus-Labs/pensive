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
import sys
import time

from recall.strata import classesForKinds, interleave, splitByClass
from recall.signals import bm25, dense, facetSignal
from recall.fusion import rrf, applyPriors
from recall.rerank import rerank
from recall.trust import assessTrust
from recall.payload import assemblePayload
from recall.enrich import Enricher

__all__ = ["recall", "FACET_BOOST", "RERANK_HEAD", "TIERS", "MEMORY_KINDS",
           "tierDefaults"]

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

# --------------------------------------------------------------------------- #
# The tier contract                                                            #
# --------------------------------------------------------------------------- #
#
# Aegis/CLAUDE.md:113 specifies an L1/L2/L3 memory hierarchy and ends "Do not
# break the tier structure". The v3 rewrite broke it: one monolithic path where
# every caller paid for a cross-encoder whether or not its question needed one.
# Budgets (Gary, 2026-08-12): L1 <= 1ms, L2 <= 20ms, L3 <= 125ms, client-observed.
#
# L1 is not here. Identity retrieval has no ranking stage at all and lives in
# serve/l1.py behind a lean HTTP route, because the MCP tools/call envelope alone
# measured 2.486ms P95 against a 1ms budget -- the transport choice IS the design.
#
# NEITHER L2 NOR L3 RUNS THE CROSS-ENCODER. House rule (Gary, 2026-08-13): batch
# encodes on the GPU, stream encodes on the CPU. A cross-encoder pass is a stream
# encode, and on CPU it measured 3164ms for 50 real pairs -- 25x the entire L3
# budget on its own. It belongs out of band, refining results after they are
# served, never blocking first token. Ablation, curated probes, in-process:
#
#     L2 (memory kinds, enrich)   R@10 5/6   p50  18.62ms   p95  24.36ms
#     L3 (all kinds, enrich)      R@10 5/6   p50  64.09ms   p95 111.71ms
#
# The difference between the tiers is which CORPUS they will read, not how hard
# they think about it. L2 answers from authored memory. L3 also reads the
# 283k-row imported corpus, which is where a document_chunk answer lives -- some
# answers exist ONLY in chunk form, so this is a boundary, never a deletion.
MEMORY_KINDS = ("atom", "narrative", "snapshot")

TIERS = ("L2", "L3")

# L2 is the DEFAULT for agent retrieve. Filed by Grok in V3.1-AGENT-GRIPES.md:
# 93% of the store is bulk-imported document_chunk, and walking it by default put
# `def create_agent` in the trusted set of a 1500-token envelope while the atom
# that actually answered had to share the budget with it.
DEFAULT_TIER = "L2"


def tierDefaults(tier):
    """``tier`` -> ``{kinds, rerank, enrich}``. Unknown tier raises.

    Raises rather than falling back to a default: a caller that names a tier this
    engine does not serve has a wrong belief about the contract, and silently
    serving it something else is how that belief survives.
    """
    if tier == "L2":
        return {"kinds": list(MEMORY_KINDS), "rerank": False, "enrich": True}
    if tier == "L3":
        return {"kinds": None, "rerank": False, "enrich": True}
    raise ValueError(
        f"unknown tier {tier!r}: this engine serves {TIERS}; "
        f"L1 is identity retrieval and lives on the lean route in serve/l1.py"
    )

# Recall breadth per signal: the plan's top-200 candidate generation. Matches the
# signals-module default; named here so the orchestrator states its own contract.
_SIGNAL_K = 200

# Aux-signal failure counter for log throttling (powers of two), module-level so
# a dead API logs a handful of lines across thousands of recalls, not thousands.
_auxFailures = 0


def _auxHits(aux, query, k):
    """Embed the query on the aux model and search its class indexes.

    Any failure returns ``[]``: the aux signal is optional evidence and recall
    must never fail because a remote embedder did. Failures log to stderr,
    throttled to power-of-two occurrences so a dead API cannot flood the journal.
    """
    global _auxFailures
    try:
        vecs = aux.embedder.embed([query])
    except Exception as exc:
        _auxFailures += 1
        if _auxFailures & (_auxFailures - 1) == 0:
            print(
                f"[recall] aux dense signal failed ({_auxFailures}x): {exc}",
                file=sys.stderr, flush=True,
            )
        return []
    if not vecs:
        return []
    hits = []
    for index in aux.indexes.values():
        hits.extend(index.search(vecs[0], k))
    return hits


def recall(store, indexes, embedder, query, project=None, timeScope=None,
           kinds=None, k=10, tokenBudget=1500, enrich=False, aux=None,
           tier=None, rerankEnabled=True):
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
    - ``enrich``: attach serve-time enrichment lines (chunk file location and
      related memory) to document_chunk results. Off by default; the eval gate
      and legacy callers measure the bare pipeline.
    - ``aux``: an optional ``recall.aux_dense.AuxDense`` (a second embedder plus
      its own per-class indexes). When present its hits fuse as one more RRF
      list and count as dense-family evidence (tagged both ``dense`` and
      ``openai`` in signalHits). When None -- or when the aux embed fails at
      query time -- the pipeline is byte-identical to the two-signal engine.

    Returns the ``RecallResult`` dict ``{results, payload, tokensUsed,
    lowConfidence}``.
    """
    # Tier resolution happens FIRST and overrides the individual knobs, so a
    # caller cannot ask for "L2" and separately pass kinds that contradict it.
    if tier is not None:
        d = tierDefaults(tier)
        kinds = d["kinds"]
        enrich = d["enrich"]
        rerankEnabled = d["rerank"]

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

    # Aux dense hits (optional third list). One aux embed per call, guarded:
    # a query-time failure yields [] and the pipeline continues on base signals.
    oaHits = _auxHits(aux, query, _SIGNAL_K) if aux is not None else []

    # signalHits: which of {bm25, dense, facet} hit each atom, drawn from the raw
    # top-200 lists and the boostSet. Trust reads this as explanation evidence.
    # Aux hits tag BOTH "dense" (they are dense-cosine evidence, and trust's
    # agreement math keys on the dense family) and "openai" (kept distinct for
    # debugging and future trust tuning; trust ignores unknown names).
    signalHits = {}
    for atomId, _ in bmHits:
        signalHits.setdefault(atomId, set()).add("bm25")
    for atomId, _ in dnHits:
        signalHits.setdefault(atomId, set()).add("dense")
    for atomId, _ in oaHits:
        signalHits.setdefault(atomId, set()).update(("dense", "openai"))
    for atomId in boostSet:
        signalHits.setdefault(atomId, set()).add("facet")

    # 3. The project filterSet restricts the candidate universe BEFORE fusion.
    if filterSet is not None:
        bmHits = [pair for pair in bmHits if pair[0] in filterSet]
        dnHits = [pair for pair in dnHits if pair[0] in filterSet]
        oaHits = [pair for pair in oaHits if pair[0] in filterSet]

    # 4. Reciprocal-rank fusion (rank-only; per-signal scores are not comparable).
    #    The aux list rides along only when the feature is on, so aux=None keeps
    #    fusion arithmetic byte-identical to the two-signal engine.
    signalLists = [bmHits, dnHits]
    if aux is not None:
        signalLists.append(oaHits)
    fused = rrf(signalLists)

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
    if rerankEnabled:
        rerankedHead = rerank(query, rerankHead, store)
    else:
        # Tiered serve path: keep the fused RRF order. The cross-encoder is a
        # stream encode and costs 3164ms for 50 real pairs on CPU, which is 25x
        # the whole L3 budget; it refines out of band instead of blocking a
        # caller. Measured on the curated probes, dropping it left R@10 at 5/6.
        rerankedHead = list(rerankHead)
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
    enricher = Enricher(store) if enrich else None
    payload, tokensUsed, lowConfidence = assemblePayload(
        store, results, tokenBudget, enricher=enricher)
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
