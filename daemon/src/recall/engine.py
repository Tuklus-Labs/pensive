"""Recall orchestrator: the whole Phase 2 pipeline as one function.

Everything Tasks 6-9 built becomes a single call here. ``recall`` threads three
parallel signals through fusion, priors, rerank, and trust, then hands the
annotated results to the payload assembler:

    per-class signals  ->  RRF fusion  ->  facet boost  ->  priors  ->  kinds
        filter  ->  round-robin rerank head  ->  cross-encoder rerank  ->  trust
        ->  tiered payload

The ordering resolves ambiguities that span the parts, so it is fixed and worth
stating in one place:

- Explicit project, agent, narrow-kind, and effective-time constraints are
  intersected before candidate generation. BM25 applies them in SQL before its
  LIMIT; dense indexes apply the same allowed-ID set before their top-k.
- ``timeScope`` also tells ``applyPriors`` to skip normal recency decay. Its
  inclusive predicate is repeated there as a defensive check over the already
  scoped candidates.
- The facet ``boostSet`` is a post-fusion MULTIPLICATIVE boost (``* FACET_BOOST``)
  applied before priors, and its members count as a ``facet`` signal hit.
- A full kind class uses its prebuilt class index. A subset such as only
  ``narrative`` also supplies an allowed-ID set before dense top-k.
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

from recall.strata import KIND_CLASSES, classesForKinds, interleave, splitByClass
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


class _OnceQueryEmbedder:
    """Share one query embedding across per-class dense searches."""

    def __init__(self, embedder):
        self._embedder = embedder
        self._vectors = None

    def embed(self, texts):
        if self._vectors is None:
            self._vectors = self._embedder.embed(texts)
        return self._vectors


def _auxHits(aux, query, k, allowedIds=None, classNames=None):
    """Embed the query on the aux model and search its class indexes.

    ``classNames`` selects the active kind classes before any per-index top-k;
    ``allowedIds`` narrows each selected index when metadata adds a finer scope.
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
    selected = set(classNames) if classNames is not None else None
    for name, index in aux.indexes.items():
        if selected is not None and name not in selected:
            continue
        if allowedIds is None:
            hits.extend(index.search(vecs[0], k))
        else:
            hits.extend(index.search(vecs[0], k, allowedIds=allowedIds))
    return hits


def _scopeAtomSet(store, project, timeScope, kinds, agent):
    """Live atom IDs satisfying every explicit recall constraint, or None."""
    clauses = ["a.status = 'live'"]
    params = []
    scoped = False

    if project is not None:
        scoped = True
        clauses.append("a.project = ?")
        params.append(project)
    if timeScope is not None:
        scoped = True
        start, end = timeScope
        clauses.append("COALESCE(a.occurred_at, a.created_at) BETWEEN ? AND ?")
        params.extend((start, end))
    if kinds is not None:
        wantedKinds = tuple(kinds)
        if not wantedKinds:
            return set()
        wantedSet = set(wantedKinds)
        selectedClassKinds = {
            kind
            for _name, classKinds in KIND_CLASSES
            if wantedSet.intersection(classKinds)
            for kind in classKinds
        }
        # A whole class is already the index's native candidate universe. Only a
        # narrower or partly unknown set needs an additional allowed-ID mask.
        if wantedSet != selectedClassKinds:
            scoped = True
            ph = ",".join("?" for _ in wantedKinds)
            clauses.append(f"a.kind IN ({ph})")
            params.extend(wantedKinds)
    if agent:
        wantedAgents = (agent,) if isinstance(agent, str) else tuple(
            value for value in agent if value)
        if wantedAgents:
            scoped = True
            ph = ",".join("?" for _ in wantedAgents)
            clauses.append(
                "a.id IN (SELECT atom_id FROM provenance "
                f"WHERE agent IN ({ph}))"
            )
            params.extend(wantedAgents)

    if not scoped:
        return None
    rows = store._conn.execute(
        "SELECT a.id FROM atoms a WHERE " + " AND ".join(clauses),
        tuple(params),
    ).fetchall()
    return {row[0] for row in rows}


def recall(store, indexes, embedder, query, project=None, timeScope=None,
           kinds=None, k=10, tokenBudget=1500, enrich=False, aux=None,
           tier=None, rerankEnabled=True, agent=None):
    """Run the full recall pipeline and assemble a tiered payload.

    ``store`` is the canonical store, ``indexes`` a ``{className: VectorIndex}``
    map (one built dense index per kind-class, from ``buildClassIndexes``), and
    ``embedder`` a loaded ``Embedder``. Candidates are generated PER CLASS (a
    kind-scoped bm25 list plus that class's own dense index), so the large code
    corpus can no longer starve the smaller reasoning memory out of the pool.
    Fusion, the facet boost, and priors run globally as before; the rerank head is
    then filled by round-robin across the per-class fused rankings, so the
    cross-encoder scores a fair mix and its scores decide the final order. Options:

    - ``project``: restrict recall to that project's live atoms before each
      signal chooses its candidates.
    - ``timeScope`` ``(startUnix, endUnix)``: restrict to atoms whose effective
      time falls in the inclusive window before candidate selection. The priors
      stage repeats the predicate defensively and disables normal recency decay.
    - ``kinds``: restrict to these atom kinds. Stratification runs over only the
      classes overlapping ``kinds`` (a single-class request degrades to today's
      non-stratified behavior).
    - ``agent``: restrict to atoms whose provenance.agent matches. A string or
      a sequence of strings. Unset (None/empty) is unscoped: today's behavior.
      NULL-agent rows are excluded when this is set. Connection ``?agent=`` is
      a write stamp, not this filter.
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

    # Resolve the complete eligible universe before any signal takes its top-k.
    # None preserves the unscoped fast paths; an empty set short-circuits before
    # either embedder runs.
    filterSet = _scopeAtomSet(store, project, timeScope, kinds, agent)
    if filterSet is not None and not filterSet:
        return _emptyResult(store, tokenBudget)

    # Entity facets are a boost signal. Scope membership comes from the combined
    # query above so project, time, kinds, and agent share one AND intersection.
    facet = facetSignal(store, {"query": query})
    boostSet = facet["boostSet"]

    # 2. Per-class candidate generation. classesForKinds(None) is every class;
    #    a kinds subset narrows to the overlapping classes. No class -> sentinel.
    classes = classesForKinds(kinds)
    if not classes:
        return _emptyResult(store, tokenBudget)
    classNames = [name for name, _ in classes]

    bmHits = []
    dnHits = []
    queryEmbedder = _OnceQueryEmbedder(embedder)
    for name, classKinds in classes:
        bmHits.extend(bm25(
            store, query, _SIGNAL_K, kinds=classKinds, agent=agent,
            project=project, timeScope=timeScope,
        ))
        classIndex = indexes.get(name)
        if classIndex is not None:
            if filterSet is None:
                dnHits.extend(dense(classIndex, queryEmbedder, query, _SIGNAL_K))
            else:
                dnHits.extend(dense(
                    classIndex, queryEmbedder, query, _SIGNAL_K,
                    allowedIds=filterSet,
                ))

    # Aux dense hits (optional third list). One aux embed per call, guarded:
    # a query-time failure yields [] and the pipeline continues on base signals.
    if aux is None:
        oaHits = []
    else:
        oaHits = _auxHits(
            aux, query, _SIGNAL_K, allowedIds=filterSet,
            classNames=classNames,
        )

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

    # 3. The combined filterSet is also a defensive pre-fusion boundary.
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

    # 6. Priors: importance * time decay, OR (under timeScope) a defensive
    #    repeat of the already-applied eligibility window without recency decay.
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

    if agent:
        fused = _filterAgent(store, fused, agent)

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


def _filterAgent(store, fused, agent):
    """Keep fused atoms that carry at least one provenance.agent in ``agent``.

    One batched SELECT, same shape as :func:`_filterKinds`. An atom with two
    provenance rows (heph and grok) is eligible for both. NULL-agent rows are
    not a silent majority: they drop unless the caller left the filter unset.
    """
    if not fused:
        return []
    if isinstance(agent, str):
        wanted = (agent,)
    else:
        wanted = tuple(a for a in agent if a)
    if not wanted:
        return []
    atomIds = [atomId for atomId, _ in fused]
    idPh = ",".join("?" for _ in atomIds)
    agPh = ",".join("?" for _ in wanted)
    rows = store._conn.execute(
        f"SELECT DISTINCT atom_id FROM provenance "
        f"WHERE atom_id IN ({idPh}) AND agent IN ({agPh})",
        atomIds + list(wanted),
    ).fetchall()
    keep = {r[0] for r in rows}
    return [pair for pair in fused if pair[0] in keep]
