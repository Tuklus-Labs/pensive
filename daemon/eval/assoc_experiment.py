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


def assocSignal(store, seedAtomIds, k):
    """Two-hop specificity-damped association walk -> ``[(atomId, score)]``.

    Formula, for every seed ``s``, through-node ``t`` adjacent to ``s``, and
    second-hop candidate ``c`` adjacent to ``t``:

        contribution(s,t,c) = weight(s,t) * weight(t,c) / degree(t)^SPEC_POWER

    with ``SPEC_POWER = 1.0`` and ``degree(t) = count(unique incident neighbors)``.
    The linear inverse-degree penalty adapts the v2 specificity precedent:
    a rare shared facet/thread node is discriminative, while a high-degree hub is
    usually popularity. Linear damping is intentionally conservative for this
    build-phase experiment: strong edge weights can still matter, but hub paths
    must pay for every extra neighbor. The walk is deterministic: no sampling,
    score-descending order, atom-id tie-break, and seeds/through nodes excluded
    from the returned candidates.
    """
    if k <= 0:
        return []
    seeds = list(dict.fromkeys(seedAtomIds))
    seedSet = set(seeds)
    if not seeds:
        return []

    scores = defaultdict(float)
    for seed in seeds:
        for through, firstWeight in _neighbors(store, seed):
            if through in seedSet:
                continue
            throughNeighbors = _neighbors(store, through)
            degree = len({neighbor for neighbor, _weight in throughNeighbors})
            if degree == 0:
                continue
            damping = degree ** SPEC_POWER
            for candidate, secondWeight in throughNeighbors:
                if candidate in seedSet or candidate == through:
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
            "assumption": (
                "not reported: gate query records expose no metadata marking "
                "what-else-was-in-flight temporal-neighborhood queries"
            ),
            "n": 0,
        }
    return report


def gateWithAssoc(store, index, embedder, queries, recallK=gate_mod._RECALL_K):
    """Run gate.py's scoring loop with the eval-only association recall."""
    return _gateWithRecall(
        store, queries,
        lambda query, k: _recallWithAssoc(store, index, embedder, query, k=k),
        recallK,
    )


def runChat(exportDir, dbPath, nQueries, embedder, log):
    records, queries, corpusRefs, corpusTexts = gate_mod.buildChatExportBenchmark(
        exportDir, nQueries=nQueries
    )
    log(f"chat corpus: {len(records)} chunk docs, {len(queries)} queries")
    store, index, stats = _prepareStore(dbPath, records, embedder, log)
    try:
        baseline = gate_mod.gate(store, index, embedder, queries)
        assoc = gateWithAssoc(store, index, embedder, queries)
        bm25 = gate_mod.bm25Baseline(corpusRefs, corpusTexts, queries)
        return _experimentReport(len(records), stats, baseline, assoc, bm25, queries)
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
        assoc = gateWithAssoc(store, index, embedder, queries)
        bm25 = gate_mod.bm25Baseline(corpusRefs, corpusTexts, queries)
        report = _experimentReport(len(records), stats, baseline, assoc, bm25, queries)
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
    seen = set()
    out = []
    for edge in edgesFrom(store, atomId):
        key = (edge["id"], edge["dstAtom"])
        if key not in seen:
            seen.add(key)
            out.append((edge["dstAtom"], edge["weight"]))
    for edge in edgesTo(store, atomId):
        key = (edge["id"], edge["srcAtom"])
        if key not in seen:
            seen.add(key)
            out.append((edge["srcAtom"], edge["weight"]))
    return out


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


def _gateWithRecall(store, queries, recallFn, recallK):
    refOf = gate_mod._atomToSourceRef(store)
    hits = []
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
        hits.append(gate_mod._rankOfFirstRelevant(ranked, q["relevant"]))
    out = gate_mod._metricsFromHits(hits)
    out.update(gate_mod._percentiles(latencies))
    out["low_confidence_rate"] = lowConf / (len(queries) or 1)
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


def _experimentReport(corpusDocs, stats, baseline, assoc, bm25, queries):
    comparison = {
        "baseline": baseline,
        "assoc": assoc,
        "delta": _delta(assoc, baseline),
    }
    comparison["temporal_neighborhood"] = compareMetrics(
        queries, [[] for _ in queries], [[] for _ in queries]
    )["temporal_neighborhood"]
    return {
        "corpus_docs": corpusDocs,
        "backfill": stats,
        "bm25": bm25,
        "v3_baseline": baseline,
        "v3_plus_assoc": assoc,
        "delta": comparison["delta"],
        "temporal_neighborhood": comparison["temporal_neighborhood"],
    }


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    main()
