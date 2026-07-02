"""Head-to-head eval gate: v3 recall as a system inside the harness.

This wraps the methodology of ``research/head2head_eval.py`` -- same corpus, same
1,500 real-human-turn queries, same seed-42 sample, same positional ground truth
(the next assistant turn), same query-own-chunk exclusion -- and adds ONE new
system: the full v3 ``recall`` pipeline (signals -> RRF+priors -> cross-encoder
rerank -> trust -> payload). It answers the Phase 2 gate question honestly: does
the v3 core beat the BM25 baseline (R@10 0.583 / MRR@10 0.432) on the same board?

Two corpora, both evaluated by the same ``gate`` core:

  * chat-export  -- the ChatGPT export the harness already knows. Positional
    ground truth (next assistant turn). THIS is the apples-to-apples gate vs BM25.
  * atom corpus  -- the real ~17k Pensive atoms + narratives. No positional
    labels exist for it, so it is scored by a SELF-SUPERVISED PROJECT-sibling
    proxy: an atom's relevant set is the other live atoms sharing its parsed
    project slug (``unclassified`` excluded, that being 88% of rows = "no
    project"). The tighter cluster keys were rejected as unusable on THIS corpus:
    ``session_id``/``episode_id`` are entirely NULL, and ``src:`` tags exist on
    only ~40 rows. This is a topical-coherence read on the real serving corpus
    plus the real end-to-end latency, NOT a labeled accuracy claim, and NOT the
    gate (the gate is the chat-export number); BASELINE_V3.md says so.

The bridge from a recalled atomId back to a ground-truth document id is the
``bulk-import`` provenance ``source_ref`` that ``ingest.backfill`` writes: the
gate builds one atomId -> sourceRef map and scores in sourceRef space, so v3's
ULID atoms line up with the harness's original doc ids.

Run it as a script (NOT pytest -- the gate is a measurement, not a unit test)::

    python3 daemon/eval/gate.py --corpus chat --queries 1500
    python3 daemon/eval/gate.py --corpus atoms --export /path/to/atom_export.jsonl

Dev stores and exports are untracked (daemon/eval/.gitignore); no atom text is
ever written to a tracked file.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make daemon/src importable when run as a standalone script. parents: eval ->
# daemon -> <repo>; daemon/src is the package root the tests' conftest also uses.
_DAEMON_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_DAEMON_SRC) not in sys.path:
    sys.path.insert(0, str(_DAEMON_SRC))

from recall.engine import recall  # noqa: E402
from recall.embedder import Embedder, embedMissing  # noqa: E402
from recall.vector_index import FlatIndex  # noqa: E402
from ingest.backfill import backfill  # noqa: E402
from store.store import openStore  # noqa: E402

MODEL_ID = "BAAI/bge-small-en-v1.5"

# Scoring depth: metrics are computed to @20 (the harness's TOP_K), so the ranked
# list handed to the metric function must reach 20 AFTER the query's own chunks are
# excluded. A query's own turn can carry ~10 chunks, so pull recallK = 40 from
# recall to leave >= 20; rerank caps at the fused top-50, so recallK stays <= 50.
_SCORE_DEPTH = 20
_RECALL_K = 40


# --------------------------------------------------------------------------- #
# metric core                                                                 #
# --------------------------------------------------------------------------- #

def _rankOfFirstRelevant(ranked, relevant):
    """1-based rank of the first sourceRef in ``ranked`` that is ``relevant``."""
    for i, ref in enumerate(ranked, start=1):
        if ref in relevant:
            return i
    return None


def _metricsFromHits(hits):
    """hits: list of (rankOrNone). -> R@1/5/10/20 + MRR@10 over the list."""
    n = len(hits) or 1
    return {
        "r_at_1": sum(1 for r in hits if r and r <= 1) / n,
        "r_at_5": sum(1 for r in hits if r and r <= 5) / n,
        "r_at_10": sum(1 for r in hits if r and r <= 10) / n,
        "r_at_20": sum(1 for r in hits if r and r <= 20) / n,
        "mrr_at_10": sum(1.0 / r for r in hits if r and r <= 10) / n,
        "n": len(hits),
    }


def _percentiles(latenciesMs):
    """p50/p95 of a latency list (ms), nearest-rank; empty -> zeros."""
    if not latenciesMs:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    s = sorted(latenciesMs)
    n = len(s)

    def _p(q):
        # nearest-rank percentile
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return s[idx]

    return {
        "p50_ms": _p(0.50),
        "p95_ms": _p(0.95),
        "mean_ms": sum(s) / n,
    }


def _atomToSourceRef(store):
    """One SELECT -> {atomId: sourceRef} for every backfilled atom."""
    rows = store._conn.execute(
        "SELECT atom_id, source_ref FROM provenance WHERE source = 'bulk-import'"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def gate(store, index, embedder, queries, recallK=_RECALL_K):
    """Run v3 ``recall`` as a system over ``queries`` -> metrics dict.

    The fixed interface core is ``gate(store, index, embedder)`` -- the recall
    stack under test. ``queries`` is the benchmark bound to that stack: a list of
    ``{"query": str, "relevant": set(sourceRef), "own": set(sourceRef)}``. For each
    query the gate runs the WHOLE recall pipeline, maps result atomIds back to
    sourceRefs, drops the query's own docs (the harness's standard query-doc
    removal), keeps the top ``_SCORE_DEPTH`` (20), and records the rank of the first
    relevant doc.

    Returns ``{r_at_1, r_at_5, r_at_10, r_at_20, mrr_at_10, n, p50_ms, p95_ms,
    mean_ms}``. Latency is the wall-clock of each ``recall`` call (informational;
    the formal latency gate is later), so p50/p95 reflect real end-to-end recall
    under whatever GPU contention the run saw.
    """
    refOf = _atomToSourceRef(store)
    hits = []
    latencies = []
    lowConf = 0
    for q in queries:
        t0 = time.perf_counter()
        result = recall(store, index, embedder, q["query"], k=recallK)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if result["lowConfidence"]:
            lowConf += 1
        own = q["own"]
        ranked = []
        for r in result["results"]:
            ref = refOf.get(r["atomId"])
            if ref is None or ref in own:
                continue
            ranked.append(ref)
            if len(ranked) >= _SCORE_DEPTH:
                break
        hits.append(_rankOfFirstRelevant(ranked, q["relevant"]))
    out = _metricsFromHits(hits)
    out.update(_percentiles(latencies))
    out["low_confidence_rate"] = lowConf / (len(queries) or 1)
    return out


# --------------------------------------------------------------------------- #
# BM25 baseline (rank_bm25, same as the harness) on the SAME queries          #
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"\w+")


def bm25Baseline(corpusRefs, corpusTexts, queries):
    """rank_bm25 BM25Okapi over the same corpus + queries -> metrics dict.

    Reproduced in-run so the v3 numbers sit beside a baseline computed on the
    IDENTICAL sample (a faithful wrapper reproduces the published 0.583/0.432 on
    the chat corpus, which is the wrapper's own correctness check). Scored to the
    same ``_SCORE_DEPTH`` as ``gate``."""
    from rank_bm25 import BM25Okapi

    corpusTokens = [_TOKEN.findall(t.lower()) for t in corpusTexts]
    bm25 = BM25Okapi(corpusTokens)
    hits = []
    import numpy as np
    for q in queries:
        scores = bm25.get_scores(_TOKEN.findall(q["query"].lower()))
        order = np.argsort(-scores)[: _SCORE_DEPTH + len(q["own"]) + 8]
        ranked = []
        for i in order:
            ref = corpusRefs[i]
            if ref in q["own"]:
                continue
            ranked.append(ref)
            if len(ranked) >= _SCORE_DEPTH:
                break
        hits.append(_rankOfFirstRelevant(ranked, q["relevant"]))
    return _metricsFromHits(hits)


# --------------------------------------------------------------------------- #
# benchmark builders                                                          #
# --------------------------------------------------------------------------- #

def buildChatExportBenchmark(exportDir, nQueries=1500, seed=42):
    """Reproduce head2head_eval.py's corpus + query sample from the ChatGPT export.

    Returns ``(records, queries, corpusRefs, corpusTexts)``:
      * ``records``  -- export rows for backfill (one per chunk doc).
      * ``queries``  -- the sampled benchmark (query text, relevant refs, own refs).
      * ``corpusRefs``/``corpusTexts`` -- aligned lists for the BM25 baseline.
    """
    import random
    sys.path.insert(0, str(Path.home() / "Projects" / "pensive" / "src"))
    from pensive.ingestion.parsers.chatgpt import ChatGPTParser

    parser = ChatGPTParser(exportDir)
    corpusRefs = []
    corpusTexts = []
    records = []
    # ground-truth index: (conv_id, msg_index) -> {role, ids, chunk0}
    turns = defaultdict(lambda: {"role": None, "ids": [], "chunk0": None})
    for d in parser.parse():
        sa = d.to_sa_dict()
        docId = sa["id"]
        content = sa["content"]
        corpusRefs.append(docId)
        corpusTexts.append(content)
        records.append({
            "sourceId": docId,
            "text": content,
            "kind": "document_chunk",
            "project": None,
            "occurredAt": int(d.timestamp) if getattr(d, "timestamp", None) else None,
            "tags": [],
        })
        conv_id = d.metadata["conv_id"]
        mi = d.metadata["msg_index"]
        role = d.metadata["role"]
        t = turns[(conv_id, mi)]
        t["role"] = role
        t["ids"].append(docId)
        if docId.endswith("-0"):
            t["chunk0"] = content

    candidates = []
    for (conv_id, mi), t in turns.items():
        if t["role"] != "user" or t["chunk0"] is None:
            continue
        nxt = turns.get((conv_id, mi + 1))
        if nxt and nxt["role"] == "assistant" and nxt["ids"]:
            candidates.append((t["chunk0"], frozenset(nxt["ids"]), frozenset(t["ids"])))
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(nQueries, len(candidates)))
    queries = [
        {"query": q, "relevant": set(rel), "own": set(own)}
        for q, rel, own in sample
    ]
    return records, queries, corpusRefs, corpusTexts


def buildSiblingBenchmark(records, clusterKey="tag", nQueries=500, seed=42,
                          excludeKeys=frozenset()):
    """Self-supervised sibling benchmark over export ``records``.

    ``clusterKey='project'`` is what the real atom corpus is scored on: two atoms
    are siblings if they share a parsed project slug (pass ``excludeKeys={'unclassified'}``
    to drop the "no project" bucket). The tighter keys were rejected as unusable on
    that corpus and are kept only for other corpora: ``'session'`` (``session_id``,
    entirely NULL here) and ``'tag'`` (``src:`` tags, only ~40 rows). An atom's
    relevant set is its siblings minus itself; an atom with no sibling is ineligible
    as a query. Returns ``(queries, corpusRefs, corpusTexts, eligibleCount)`` where
    the corpus lists cover EVERY record (the full distractor set), and the queries
    are a seed-sampled subset of the eligible atoms.
    """
    import random
    corpusRefs = [r["sourceId"] for r in records]
    corpusTexts = [r["text"] for r in records]

    # cluster -> set(sourceRef)
    clusters = defaultdict(set)
    keysOf = {}
    for r in records:
        ref = r["sourceId"]
        if clusterKey == "tag":
            keys = list(dict.fromkeys(r.get("tags") or []))
        else:
            # single-value cluster fields: 'session' or 'project'.
            v = r.get(clusterKey)
            keys = [v] if v else []
        # A degenerate bucket (e.g. project='unclassified', 88% of the real
        # corpus = "no project") is not a topical cluster; drop it so it neither
        # forms a giant relevant set nor makes its members eligible queries. Such
        # atoms remain in the corpus as distractors.
        keys = [k for k in keys if k not in excludeKeys]
        keysOf[ref] = keys
        for kkey in keys:
            clusters[kkey].add(ref)

    textOf = {r["sourceId"]: r["text"] for r in records}
    eligible = []
    for r in records:
        ref = r["sourceId"]
        rel = set()
        for kkey in keysOf[ref]:
            rel |= clusters[kkey]
        rel.discard(ref)
        if rel:
            eligible.append({"query": textOf[ref], "relevant": rel, "own": {ref}})

    rng = random.Random(seed)
    sample = rng.sample(eligible, min(nQueries, len(eligible)))
    return sample, corpusRefs, corpusTexts, len(eligible)


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #

def _prepareStore(dbPath, records, embedder, log):
    """Backfill ``records`` into a fresh dev store, embed all, build the index."""
    p = Path(dbPath)
    if p.exists():
        p.unlink()
    for suffix in ("-wal", "-shm"):
        side = Path(str(p) + suffix)
        if side.exists():
            side.unlink()
    store = openStore(p)
    # Dev store only: NORMAL sync keeps a ~100k-row backfill's per-record commits
    # cheap. This is a throwaway file, never production.
    store._conn.execute("PRAGMA synchronous=NORMAL")
    t0 = time.time()
    stats = backfill(store, records)
    log(f"backfilled {stats['ingested']} atoms "
        f"({stats['byKind']}), {stats['tagFacets']} tag + "
        f"{stats['entityFacets']} entity facets in {time.time()-t0:.1f}s")
    t1 = time.time()
    embedded = embedMissing(store, embedder)
    log(f"embedded {embedded} atoms in {time.time()-t1:.1f}s "
        f"({embedded/max(1e-9, time.time()-t1):.0f}/s)")
    index = FlatIndex().build(store, MODEL_ID)
    return store, index, stats


def runChat(exportDir, dbPath, nQueries, embedder, log):
    records, queries, corpusRefs, corpusTexts = buildChatExportBenchmark(
        exportDir, nQueries=nQueries
    )
    log(f"chat corpus: {len(records)} chunk docs, {len(queries)} queries")
    store, index, stats = _prepareStore(dbPath, records, embedder, log)
    t0 = time.time()
    v3 = gate(store, index, embedder, queries)
    log(f"v3 gate done in {time.time()-t0:.1f}s: "
        f"R@10={v3['r_at_10']:.3f} MRR@10={v3['mrr_at_10']:.3f}")
    bm = bm25Baseline(corpusRefs, corpusTexts, queries)
    log(f"bm25 baseline (same sample): R@10={bm['r_at_10']:.3f} "
        f"MRR@10={bm['mrr_at_10']:.3f}")
    store.close()
    return {"corpus_docs": len(records), "backfill": stats, "v3": v3, "bm25": bm}


def runAtoms(exportPath, dbPath, clusterKey, nQueries, embedder, log,
             excludeKeys=frozenset()):
    with open(exportPath, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    log(f"atom corpus: {len(records)} records from {exportPath}")
    queries, corpusRefs, corpusTexts, eligible = buildSiblingBenchmark(
        records, clusterKey=clusterKey, nQueries=nQueries, excludeKeys=excludeKeys
    )
    log(f"sibling benchmark ({clusterKey}): {eligible} eligible, "
        f"{len(queries)} sampled")
    store, index, stats = _prepareStore(dbPath, records, embedder, log)
    t0 = time.time()
    v3 = gate(store, index, embedder, queries)
    log(f"v3 gate done in {time.time()-t0:.1f}s: "
        f"R@10={v3['r_at_10']:.3f} MRR@10={v3['mrr_at_10']:.3f}")
    bm = bm25Baseline(corpusRefs, corpusTexts, queries)
    log(f"bm25 baseline (same sample): R@10={bm['r_at_10']:.3f} "
        f"MRR@10={bm['mrr_at_10']:.3f}")
    store.close()
    return {
        "corpus_docs": len(records), "cluster_key": clusterKey,
        "eligible_queries": eligible, "backfill": stats, "v3": v3, "bm25": bm,
    }


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["chat", "atoms"], required=True)
    ap.add_argument("--export", help="atom export jsonl (for --corpus atoms)")
    ap.add_argument("--export-dir", default=str(Path.home() / "Projects" / "chatgpt-export"))
    ap.add_argument("--db", required=True, help="dev store path (untracked)")
    ap.add_argument("--queries", type=int, default=1500)
    # project is the only usable key on the real atom corpus (session_id NULL,
    # src: tags on ~40 rows); pair it with --exclude-keys unclassified.
    ap.add_argument("--cluster-key", default="project",
                    choices=["project", "session", "tag"])
    ap.add_argument("--exclude-keys", default="",
                    help="comma-separated cluster values to treat as non-clusters "
                         "(e.g. 'unclassified')")
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
        res = runAtoms(args.export, args.db, args.cluster_key, args.queries,
                       embedder, _log, excludeKeys=excludeKeys)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        _log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
