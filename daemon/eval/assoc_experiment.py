"""Association-signal experiment harness (Task 21).

This file is deliberately eval-only. It builds a fourth fusion signal for the
gate harness and leaves the serving recall path untouched until the controller
decides measured lift justifies adoption.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_DAEMON = Path(__file__).resolve().parents[1]
_DAEMON_SRC = _DAEMON / "src"
for _path in (_DAEMON, _DAEMON_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval import gate as gate_mod  # noqa: E402
from ingest.backfill import backfill  # noqa: E402
from recall.embedder import Embedder, embedMissing  # noqa: E402
from recall.engine import FACET_BOOST  # noqa: E402
from recall.engine import _emptyResult, _filterKinds  # noqa: E402
from recall.fusion import applyPriors, rrf  # noqa: E402
from recall.payload import assemblePayload  # noqa: E402
from recall.rerank import rerank  # noqa: E402
from recall.signals import bm25, dense, facetSignal  # noqa: E402
from recall.trust import assessTrust  # noqa: E402
from recall.vector_index import FlatIndex  # noqa: E402
from store.store import edgesFrom, edgesTo, openStore  # noqa: E402

MODEL_ID = gate_mod.MODEL_ID
ASSOC_SEED_K = 20
ASSOC_SIGNAL_K = 200
SPEC_POWER = 1.0
FACET_ASSOC_KEYS = ("entity", "tag")
HUB_CAP = 128
MAX_SEED_FACETS = 32
_PARITY_KEYS = (
    "r_at_1",
    "r_at_5",
    "r_at_10",
    "r_at_20",
    "mrr_at_10",
    "n",
    "low_confidence_rate",
)


def assocSignal(store, seedAtomIds, k):
    """Specificity-damped association walk -> ``[(atomId, score)]``.

    Facet co-occurrence is the primary adjacency because the v3 store derives
    facet back-edges instead of materializing them. For each seed ``s``, facet
    ``f``, and candidate ``c`` sharing ``f``:

        contribution(s,f,c) = 1 / degree(f)^SPEC_POWER

    ``FACET_ASSOC_KEYS`` is intentionally limited to ``entity`` and ``tag``:
    entities are the most specific semantic anchors, while tags carry curated
    scopes such as source/session labels. Project/era facets are broader
    filters elsewhere in recall, so treating them as assoc edges would mostly
    create hubs. Facets whose live-atom degree exceeds ``HUB_CAP`` are skipped
    because their damped contribution is negligible and their fanout is costly.
    The per-seed facet scan is also capped at ``MAX_SEED_FACETS`` ordered
    key/value rows, bounding derived neighbor work to
    ``MAX_SEED_FACETS * HUB_CAP`` candidates per seed.

    Stored edge rows, when present, are a bonus signal and keep the original
    two-hop formula. For every seed ``s``, through-atom ``t`` adjacent to ``s``,
    and second-hop candidate ``c`` adjacent to ``t``:

        contribution(s,t,c) = weight(s,t) * weight(t,c) / degree(t)^SPEC_POWER

    with ``SPEC_POWER = 1.0`` and ``degree(t) = count(unique incident neighbors)``.
    The linear inverse-degree penalty adapts the v2 specificity precedent:
    a rare shared facet/thread node is discriminative, while a high-degree hub is
    usually popularity. Linear damping is intentionally conservative for this
    build-phase experiment: strong edge weights can still matter, but hub paths
    must pay for every extra neighbor. The walk is deterministic: no sampling,
    score-descending order, atom-id tie-break, and seeds/through nodes excluded
    from the stored-edge two-hop returned candidates.

    Edges use logical undirected semantics for this eval walk. Mirrored physical
    rows between the same endpoints and type are one logical edge at max weight;
    genuinely different types between the same endpoints remain separate
    contributions. Dead atoms conduct but never emit: walking through a
    tombstoned/superseded node preserves neighborhood structure, while returning
    it as a candidate would violate live-only recall.
    """
    if k <= 0:
        return []
    seeds = list(dict.fromkeys(seedAtomIds))
    seedSet = set(seeds)
    if not seeds:
        return []

    scores = defaultdict(float)
    for seed in seeds:
        for candidate, contribution in _facetCooccurrenceNeighbors(store, seed):
            if candidate in seedSet:
                continue
            scores[candidate] += contribution

        for through, firstWeight, _firstType, _throughLive in _neighbors(store, seed):
            if through in seedSet:
                continue
            throughNeighbors = _neighbors(store, through)
            degree = len({
                neighbor for neighbor, _weight, _edgeType, _neighborLive
                in throughNeighbors
            })
            if degree == 0:
                continue
            damping = degree ** SPEC_POWER
            for candidate, secondWeight, _secondType, candidateLive in throughNeighbors:
                if candidate in seedSet or candidate == through:
                    continue
                if not candidateLive:
                    continue
                contribution = (firstWeight * secondWeight) / damping
                scores[candidate] += contribution

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def compareMetrics(queries, baselineRankings, assocRankings):
    """Compare two sourceRef rankings with gate.py's metric helpers."""
    baseline = _metricsForRankings(queries, baselineRankings)
    assoc = _metricsForRankings(queries, assocRankings)
    report = {
        "baseline": baseline,
        "assoc": assoc,
        "delta": _delta(assoc, baseline),
    }

    subset = [
        i for i, query in enumerate(queries)
        if _isTemporalNeighborhoodQuery(query)
    ]
    if subset:
        subQueries = [queries[i] for i in subset]
        subBaseline = [baselineRankings[i] for i in subset]
        subAssoc = [assocRankings[i] for i in subset]
        baselineSubset = _metricsForRankings(subQueries, subBaseline)
        assocSubset = _metricsForRankings(subQueries, subAssoc)
        report["temporal_neighborhood"] = {
            "assumption": (
                "identified by query metadata fields whose normalized value "
                "contains temporal-neighborhood, in-flight, in_flight, or "
                "what-else-was-in-flight"
            ),
            "baseline": baselineSubset,
            "assoc": assocSubset,
            "delta": _delta(assocSubset, baselineSubset),
        }
    else:
        report["temporal_neighborhood"] = {
            "status": "UNAVAILABLE",
            "assumption": (
                "unavailable: gate query records expose no metadata marking "
                "what-else-was-in-flight temporal-neighborhood queries"
            ),
            "baseline": None,
            "assoc": None,
            "delta": None,
        }
    return report


def gateWithAssoc(
    store,
    index,
    embedder,
    queries,
    recallK=gate_mod._RECALL_K,
    gateBaseline=None,
    returnBaselineRankings=False,
):
    """Run the assoc arm after proving the cloned no-assoc gate is in parity."""
    baseline = gateBaseline
    if baseline is None:
        baseline = gate_mod.gate(store, index, embedder, queries)
    baselineClone, baselineRankings = _baselineWithRankings(
        store, index, embedder, queries, recallK
    )
    _assertGateParity(baseline, baselineClone)
    assoc, assocRankings = _assocWithRankings(
        store, index, embedder, queries, recallK
    )
    _assertAssocNotInert(store, baselineRankings, assocRankings)
    if returnBaselineRankings:
        return assoc, baselineRankings, assocRankings
    return assoc, assocRankings


def runChat(exportDir, dbPath, nQueries, embedder, log):
    records, queries, corpusRefs, corpusTexts = gate_mod.buildChatExportBenchmark(
        exportDir, nQueries=nQueries
    )
    log(f"chat corpus: {len(records)} chunk docs, {len(queries)} queries")
    store, index, stats = _prepareStore(dbPath, records, embedder, log)
    try:
        baseline = gate_mod.gate(store, index, embedder, queries)
        assoc, baselineRankings, assocRankings = gateWithAssoc(
            store, index, embedder, queries,
            gateBaseline=baseline,
            returnBaselineRankings=True,
        )
        bm25 = gate_mod.bm25Baseline(corpusRefs, corpusTexts, queries)
        return _experimentReport(
            len(records), stats, baseline, assoc, bm25, queries,
            baselineRankings, assocRankings, _derivedGraphStats(store),
        )
    finally:
        store.close()


def runAtoms(exportPath, dbPath, clusterKey, nQueries, embedder, log,
             excludeKeys=frozenset()):
    with open(exportPath, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    log(f"atom corpus: {len(records)} records from {exportPath}")
    queries, corpusRefs, corpusTexts, eligible = gate_mod.buildSiblingBenchmark(
        records, clusterKey=clusterKey, nQueries=nQueries, excludeKeys=excludeKeys
    )
    log(f"sibling benchmark ({clusterKey}): {eligible} eligible, "
        f"{len(queries)} sampled")
    store, index, stats = _prepareStore(dbPath, records, embedder, log)
    try:
        baseline = gate_mod.gate(store, index, embedder, queries)
        assoc, baselineRankings, assocRankings = gateWithAssoc(
            store, index, embedder, queries,
            gateBaseline=baseline,
            returnBaselineRankings=True,
        )
        bm25 = gate_mod.bm25Baseline(corpusRefs, corpusTexts, queries)
        report = _experimentReport(
            len(records), stats, baseline, assoc, bm25, queries,
            baselineRankings, assocRankings, _derivedGraphStats(store),
        )
        report["cluster_key"] = clusterKey
        report["eligible_queries"] = eligible
        return report
    finally:
        store.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["chat", "atoms"], required=True)
    ap.add_argument("--export", help="atom export jsonl (for --corpus atoms)")
    ap.add_argument(
        "--export-dir",
        default=str(Path.home() / "Projects" / "chatgpt-export"),
    )
    ap.add_argument("--db", required=True, help="dev store path (untracked)")
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument(
        "--cluster-key", default="project", choices=["project", "session", "tag"]
    )
    ap.add_argument(
        "--exclude-keys", default="",
        help="comma-separated cluster values to treat as non-clusters",
    )
    ap.add_argument("--out", help="write result json here")
    args = ap.parse_args()

    embedder = Embedder(MODEL_ID)
    _log(f"embedder loaded on {embedder.device}")
    if args.corpus == "chat":
        res = runChat(args.export_dir, args.db, args.queries, embedder, _log)
    else:
        if not args.export:
            ap.error("--export is required for --corpus atoms")
        excludeKeys = frozenset(
            k.strip() for k in args.exclude_keys.split(",") if k.strip()
        )
        res = runAtoms(
            args.export, args.db, args.cluster_key, args.queries,
            embedder, _log, excludeKeys=excludeKeys,
        )
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        _log(f"wrote {args.out}")


def _neighbors(store, atomId):
    byLogicalEdge = {}
    for edge in edgesFrom(store, atomId):
        neighbor = edge["dstAtom"]
        key = _logicalEdgeKey(atomId, neighbor, edge["type"])
        current = byLogicalEdge.get(key)
        weight = edge["weight"] if current is None else max(current[1], edge["weight"])
        byLogicalEdge[key] = (neighbor, weight, edge["type"])
    for edge in edgesTo(store, atomId):
        neighbor = edge["srcAtom"]
        key = _logicalEdgeKey(atomId, neighbor, edge["type"])
        current = byLogicalEdge.get(key)
        weight = edge["weight"] if current is None else max(current[1], edge["weight"])
        byLogicalEdge[key] = (neighbor, weight, edge["type"])

    statuses = _atomStatuses(
        store,
        [neighbor for neighbor, _weight, _type in byLogicalEdge.values()],
    )
    out = []
    for neighbor, weight, edgeType in byLogicalEdge.values():
        out.append((neighbor, weight, edgeType, statuses.get(neighbor) == "live"))
    return sorted(out, key=lambda item: (item[0], item[2]))


def _facetCooccurrenceNeighbors(store, atomId):
    rows = store._conn.execute(
        """
        WITH seed_facets AS (
            SELECT key, value
            FROM facets
            WHERE atom_id = ?
              AND key IN (?, ?)
            ORDER BY CASE key WHEN 'entity' THEN 0 ELSE 1 END, value
            LIMIT ?
        ),
        facet_degree AS (
            SELECT sf.key, sf.value, COUNT(DISTINCT f.atom_id) AS degree
            FROM seed_facets sf
            JOIN facets f ON f.key = sf.key AND f.value = sf.value
            JOIN atoms a ON a.id = f.atom_id
            WHERE a.status = 'live'
            GROUP BY sf.key, sf.value
            HAVING degree > 1 AND degree <= ?
        )
        SELECT f.atom_id, d.degree
        FROM facet_degree d
        JOIN facets f ON f.key = d.key AND f.value = d.value
        JOIN atoms a ON a.id = f.atom_id
        WHERE a.status = 'live'
          AND f.atom_id != ?
        ORDER BY f.atom_id
        """,
        (
            atomId,
            FACET_ASSOC_KEYS[0],
            FACET_ASSOC_KEYS[1],
            MAX_SEED_FACETS,
            HUB_CAP,
            atomId,
        ),
    ).fetchall()
    scores = defaultdict(float)
    for neighbor, degree in rows:
        scores[neighbor] += 1.0 / (degree ** SPEC_POWER)
    return sorted(scores.items(), key=lambda item: item[0])


def _derivedGraphStats(store):
    rows = store._conn.execute(
        """
        WITH facet_degree AS (
            SELECT f.key, f.value, COUNT(DISTINCT f.atom_id) AS degree
            FROM facets f
            JOIN atoms a ON a.id = f.atom_id
            WHERE f.key IN (?, ?)
              AND a.status = 'live'
            GROUP BY f.key, f.value
            HAVING degree > 1 AND degree <= ?
        )
        SELECT COALESCE(SUM(degree * (degree - 1)), 0)
        FROM facet_degree
        """,
        (FACET_ASSOC_KEYS[0], FACET_ASSOC_KEYS[1], HUB_CAP),
    ).fetchone()
    totalPairs = rows[0] if rows else 0
    rows = store._conn.execute(
        """
        WITH facet_degree AS (
            SELECT f.key, f.value, COUNT(DISTINCT f.atom_id) AS degree
            FROM facets f
            JOIN atoms a ON a.id = f.atom_id
            WHERE f.key IN (?, ?)
              AND a.status = 'live'
            GROUP BY f.key, f.value
            HAVING degree > 1 AND degree <= ?
        )
        SELECT COUNT(DISTINCT f.atom_id)
        FROM facets f
        JOIN atoms a ON a.id = f.atom_id
        JOIN facet_degree d ON d.key = f.key AND d.value = f.value
        WHERE a.status = 'live'
        """,
        (FACET_ASSOC_KEYS[0], FACET_ASSOC_KEYS[1], HUB_CAP),
    ).fetchone()
    seedsWithNeighbors = rows[0] if rows else 0
    return {
        "facet_keys": list(FACET_ASSOC_KEYS),
        "hub_cap": HUB_CAP,
        "max_seed_facets": MAX_SEED_FACETS,
        "seeds_with_neighbors": seedsWithNeighbors,
        "derived_neighbor_pairs": totalPairs,
    }


def _logicalEdgeKey(atomId, neighbor, edgeType):
    left, right = sorted((atomId, neighbor))
    return left, right, edgeType


def _atomStatuses(store, atomIds):
    ids = sorted(set(atomIds))
    if not ids:
        return {}
    placeholders = ",".join("?" for _id in ids)
    rows = store._conn.execute(
        f"SELECT id, status FROM atoms WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _metricsForRankings(queries, rankings):
    hits = [
        gate_mod._rankOfFirstRelevant(ranked, query["relevant"])
        for query, ranked in zip(queries, rankings)
    ]
    return gate_mod._metricsFromHits(hits)


def _delta(new, old):
    return {
        key: new[key] - old[key]
        for key in ("r_at_1", "r_at_5", "r_at_10", "r_at_20", "mrr_at_10")
        if key in new and key in old
    }


def _isTemporalNeighborhoodQuery(query):
    metadata = query.get("metadata") or query.get("meta") or {}
    values = []
    if isinstance(metadata, dict):
        values.extend(str(value) for value in metadata.values())
    text = str(query.get("query", ""))
    values.append(text)
    normalized = " ".join(values).lower().replace("_", "-")
    return (
        "temporal-neighborhood" in normalized
        or "in-flight" in normalized
        or "what else was in flight" in normalized
        or "what-else-was-in-flight" in normalized
    )


def _baselineWithRankings(store, index, embedder, queries,
                          recallK=gate_mod._RECALL_K):
    return _gateWithRecall(
        store, queries,
        lambda query, k: gate_mod.recall(store, index, embedder, query, k=k),
        recallK,
        captureRankings=True,
    )


def _assocWithRankings(store, index, embedder, queries,
                       recallK=gate_mod._RECALL_K):
    return _gateWithRecall(
        store, queries,
        lambda query, k: _recallWithAssoc(store, index, embedder, query, k=k),
        recallK,
        captureRankings=True,
    )


def _assertGateParity(gateBaseline, experimentClone):
    baseline = _parityMetrics(gateBaseline)
    clone = _parityMetrics(experimentClone)
    if baseline != clone:
        raise RuntimeError(
            "arm parity guard failed: gate.py baseline metrics "
            f"{baseline} != experiment clone metrics {clone}; aborting "
            "without emitting an assoc experiment report"
        )


def _assertAssocNotInert(store, baselineRankings, assocRankings):
    queryCount = max(len(baselineRankings), len(assocRankings))
    if queryCount == 0:
        return
    differ = sum(
        1 for baseline, assoc in zip(baselineRankings, assocRankings)
        if baseline != assoc
    )
    differ += abs(len(baselineRankings) - len(assocRankings))
    if differ:
        return
    stats = _derivedGraphStats(store)
    raise RuntimeError(
        "assoc arm inert: "
        f"{differ}/{queryCount} queries differ; refusing to emit a normal "
        "assoc experiment report; "
        f"seeds_with_neighbors={stats['seeds_with_neighbors']}; "
        f"derived_neighbor_pairs={stats['derived_neighbor_pairs']}"
    )


def _parityMetrics(metrics):
    # Latency percentiles are timing observations, not deterministic scoring
    # semantics; the parity guard compares the gate-quality metric contract.
    return {key: metrics.get(key) for key in _PARITY_KEYS if key in metrics}


def _gateWithRecall(store, queries, recallFn, recallK, captureRankings=False):
    refOf = gate_mod._atomToSourceRef(store)
    hits = []
    rankings = []
    latencies = []
    lowConf = 0
    for q in queries:
        t0 = time.perf_counter()
        result = recallFn(q["query"], recallK)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if result["lowConfidence"]:
            lowConf += 1
        ranked = []
        for r in result["results"]:
            ref = refOf.get(r["atomId"])
            if ref is None or ref in q["own"]:
                continue
            ranked.append(ref)
            if len(ranked) >= gate_mod._SCORE_DEPTH:
                break
        rankings.append(ranked)
        hits.append(gate_mod._rankOfFirstRelevant(ranked, q["relevant"]))
    out = gate_mod._metricsFromHits(hits)
    out.update(gate_mod._percentiles(latencies))
    out["low_confidence_rate"] = lowConf / (len(queries) or 1)
    if captureRankings:
        return out, rankings
    return out


def _recallWithAssoc(store, index, embedder, query, project=None, timeScope=None,
                     kinds=None, k=10, tokenBudget=1500):
    now = int(time.time())
    if not query or not query.strip():
        return _emptyResult(store, tokenBudget)

    facetHints = {"query": query}
    if project is not None:
        facetHints["project"] = project
    facet = facetSignal(store, facetHints)
    boostSet = facet["boostSet"]
    filterSet = facet["filterSet"]
    if filterSet is not None and not filterSet:
        return _emptyResult(store, tokenBudget)

    bmHits = bm25(store, query, ASSOC_SIGNAL_K)
    dnHits = dense(index, embedder, query, ASSOC_SIGNAL_K)
    if filterSet is not None:
        bmHits = [pair for pair in bmHits if pair[0] in filterSet]
        dnHits = [pair for pair in dnHits if pair[0] in filterSet]

    seedIds = [atomId for atomId, _score in rrf([bmHits, dnHits])[:ASSOC_SEED_K]]
    assocHits = assocSignal(store, seedIds, ASSOC_SIGNAL_K)
    if filterSet is not None:
        assocHits = [pair for pair in assocHits if pair[0] in filterSet]

    signalHits = {}
    for atomId, _ in bmHits:
        signalHits.setdefault(atomId, set()).add("bm25")
    for atomId, _ in dnHits:
        signalHits.setdefault(atomId, set()).add("dense")
    for atomId, _ in assocHits:
        signalHits.setdefault(atomId, set()).add("assoc")
    for atomId in boostSet:
        signalHits.setdefault(atomId, set()).add("facet")

    fused = rrf([bmHits, dnHits, assocHits])
    if boostSet:
        fused = [
            (atomId, score * FACET_BOOST if atomId in boostSet else score)
            for atomId, score in fused
        ]
    priorHints = {"now": now}
    if timeScope is not None:
        priorHints["timeScope"] = timeScope
    fused = applyPriors(fused, store, priorHints)
    if kinds is not None:
        fused = _filterKinds(store, fused, kinds)
    if not fused:
        return _emptyResult(store, tokenBudget)

    reranked = rerank(query, fused, store)
    assessed = assessTrust(reranked, signalHits, store, now)
    results = assessed[:k]
    payload, tokensUsed, lowConfidence = assemblePayload(store, results, tokenBudget)
    return {
        "results": results,
        "payload": payload,
        "tokensUsed": tokensUsed,
        "lowConfidence": lowConfidence,
    }


def _prepareStore(dbPath, records, embedder, log):
    p = Path(dbPath)
    if p.exists():
        p.unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(p) + suffix)
        if side.exists():
            side.unlink()
    store = openStore(p)
    store._conn.execute("PRAGMA synchronous=NORMAL")
    t0 = time.time()
    stats = backfill(store, records)
    log(f"backfilled {stats['ingested']} atoms in {time.time()-t0:.1f}s")
    t1 = time.time()
    embedded = embedMissing(store, embedder)
    log(f"embedded {embedded} atoms in {time.time()-t1:.1f}s")
    index = FlatIndex().build(store, MODEL_ID)
    return store, index, stats


def _experimentReport(
    corpusDocs,
    stats,
    baseline,
    assoc,
    bm25,
    queries,
    baselineRankings,
    assocRankings,
    derivedGraphStats=None,
):
    comparison = {
        "baseline": baseline,
        "assoc": assoc,
        "delta": _delta(assoc, baseline),
    }
    comparison["temporal_neighborhood"] = compareMetrics(
        queries, baselineRankings, assocRankings
    )["temporal_neighborhood"]
    return {
        "corpus_docs": corpusDocs,
        "backfill": stats,
        "bm25": bm25,
        "v3_baseline": baseline,
        "v3_plus_assoc": assoc,
        "delta": comparison["delta"],
        "assoc_seed_k": ASSOC_SEED_K,
        "assoc_damping_formula": (
            "facet: 1 / degree(f)^SPEC_POWER for shared entity/tag facets; "
            "stored edges: weight(s,t) * weight(t,c) / degree(t)^SPEC_POWER; "
            f"SPEC_POWER={SPEC_POWER}; degree=count(live atoms or unique "
            "incident neighbors)"
        ),
        "assoc_derived_graph": derivedGraphStats or {},
        "temporal_neighborhood": comparison["temporal_neighborhood"],
    }


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    main()
