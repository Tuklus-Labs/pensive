#!/usr/bin/env python3
"""Edge campaign proposer: memory-to-chunk candidate pairs, read-only.

For each live memory atom, candidate chunks are scored by shared-entity
specificity: score = sum over shared entities of 1/freq(entity), where freq
counts live atoms carrying the entity. Hub entities (freq > hubCap) are
excluded OUTRIGHT before scoring: a facet shared by hundreds of atoms carries
no relational signal and only manufactures noise pairs (the FILES-style
facets). Same-project pairs get a small additive bonus, tie-broken stably.

Output is self-contained JSONL batches: each line carries both texts
(whitespace-collapsed, truncated) so a verification agent can judge the pair
without ever touching the store. This tool never mutates anything.
"""
import argparse
import json
import re
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_SRC = _TOOLS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from store.store import openStore  # noqa: E402
from recall.embedder import blobToVec  # noqa: E402

__all__ = ["proposeForAtom", "buildFreqMap", "proposeDense", "emitProposals"]

MODEL_ID = "BAAI/bge-small-en-v1.5"

_MEMORY_KINDS = ("atom", "narrative", "snapshot")

# Same-project additive bonus: small on purpose, a tiebreak-plus, never a
# substitute for a shared entity (pairs with NO shared non-hub entity are
# never proposed at all).
_PROJECT_BONUS = 0.05

_WS_RE = re.compile(r"\s+")


def _flat(text, cap):
    return _WS_RE.sub(" ", text or "").strip()[:cap]


def proposeForAtom(store, memId, maxPerAtom=3, hubCap=500, minScore=0.02,
                   freqMap=None):
    """Top candidate chunks for one memory atom -> [{memId, chunkId, score,
    entities, channel}] best-first.

    Entities with live-atom frequency above ``hubCap`` are excluded before
    scoring. Candidates are LIVE document_chunk atoms sharing at least one
    surviving entity. Score = sum(1/freq) over shared surviving entities,
    plus _PROJECT_BONUS when projects match. Results below ``minScore`` are
    dropped; at most ``maxPerAtom`` returned, ordered (score desc, chunkId)
    for determinism.

    ``freqMap``, when given (a ``{value: liveCount}`` dict from
    :func:`buildFreqMap`), skips the per-atom freq CTE entirely: candidates
    come from one indexed IN-list query and scores are computed from the
    map. This is the perf fix for the pilot's 1.12s/atom (the freq CTE
    recomputed the whole live-entity histogram on every call). Behavior with
    ``freqMap=None`` is byte-identical to before this existed.
    """
    if freqMap is None:
        rows = store._conn.execute(
            "WITH mine AS ("
            "  SELECT value FROM facets WHERE atom_id = ? AND key = 'entity'"
            "), freq AS ("
            "  SELECT f.value, COUNT(*) AS n FROM facets f"
            "  JOIN atoms a ON a.id = f.atom_id"
            "  WHERE f.key = 'entity' AND a.status = 'live'"
            "  AND f.value IN (SELECT value FROM mine)"
            "  GROUP BY f.value HAVING COUNT(*) <= ?"
            ") "
            "SELECT f.atom_id, f.value, freq.n, a.project "
            "FROM facets f "
            "JOIN freq ON freq.value = f.value "
            "JOIN atoms a ON a.id = f.atom_id "
            "WHERE f.key = 'entity' AND a.status = 'live' "
            "AND a.kind = 'document_chunk'",
            (memId, hubCap),
        ).fetchall()
    else:
        mine = [r[0] for r in store._conn.execute(
            "SELECT value FROM facets WHERE atom_id = ? AND key = 'entity'",
            (memId,)).fetchall()]
        keep = [v for v in mine if 0 < freqMap.get(v, 0) <= hubCap]
        if not keep:
            return []
        marks = ",".join("?" for _ in keep)
        rows = [
            (chunkId, value, freqMap[value], project)
            for chunkId, value, project in store._conn.execute(
                "SELECT f.atom_id, f.value, a.project FROM facets f "
                "JOIN atoms a ON a.id = f.atom_id "
                f"WHERE f.key = 'entity' AND f.value IN ({marks}) "
                "AND a.status = 'live' AND a.kind = 'document_chunk'",
                keep).fetchall()
        ]
    if not rows:
        return []
    memProject = store._conn.execute(
        "SELECT project FROM atoms WHERE id = ?", (memId,)).fetchone()
    memProject = memProject[0] if memProject else None

    byChunk = {}
    for chunkId, value, n, project in rows:
        entry = byChunk.setdefault(
            chunkId, {"score": 0.0, "entities": [], "project": project})
        entry["score"] += 1.0 / n
        entry["entities"].append(value)
    out = []
    for chunkId, entry in byChunk.items():
        score = entry["score"]
        if memProject is not None and entry["project"] == memProject:
            score += _PROJECT_BONUS
        if score < minScore:
            continue
        out.append({
            "memId": memId,
            "chunkId": chunkId,
            "score": round(score, 6),
            "entities": sorted(entry["entities"]),
            "channel": "entity",
        })
    out.sort(key=lambda p: (-p["score"], p["chunkId"]))
    return out[:maxPerAtom]


def buildFreqMap(store):
    """{entityValue: live-atom count} in one query; the per-run freq cache.

    Computing this once turns proposeForAtom's per-atom freq CTE into a dict
    lookup: the pilot measured 1.12s/atom recomputing it, which is 4.4h over
    the full memory population; this map builds in seconds.
    """
    rows = store._conn.execute(
        "SELECT f.value, COUNT(*) FROM facets f "
        "JOIN atoms a ON a.id = f.atom_id "
        "WHERE f.key = 'entity' AND a.status = 'live' "
        "GROUP BY f.value").fetchall()
    return {r[0]: r[1] for r in rows}


def proposeDense(store, codeIndex, memId, modelId, topK=5, minSim=0.5):
    """Semantic channel: the memory atom's STORED vector vs the code index.

    Never embeds at proposal time: an atom without a stored embedding for
    modelId yields []. Cosine scores below minSim are dropped. Chunks are
    whatever the code-class index holds (live document_chunk by build).
    """
    row = store._conn.execute(
        "SELECT vector FROM embeddings WHERE atom_id = ? AND model_id = ?",
        (memId, modelId)).fetchone()
    if row is None:
        return []
    vec = blobToVec(row[0])
    hits = codeIndex.search(vec, topK)
    return [
        {"memId": memId, "chunkId": chunkId, "score": round(float(sim), 6),
         "entities": [], "channel": "dense"}
        for chunkId, sim in hits if sim >= minSim
    ]


def emitProposals(store, outDir, batchSize=50, textCap=1200,
                  maxPerAtom=3, hubCap=500, minScore=0.02, limit=None,
                  denseK=5, minSim=0.5, modelId=MODEL_ID):
    """Score every live memory atom and write self-contained JSONL batches.

    Two proposal channels feed each atom: the entity channel (shared rare
    facets, scored via a freq map built once up front) and the dense channel
    (stored-vector search against the code-class index, built once up front
    too, and only when ``denseK > 0``). Per atom the two channels are merged
    and deduped by ``chunkId``, keeping whichever entry has the higher score;
    an exact tie keeps the entity entry (the entity list is loaded into the
    merge dict first, so a dense hit only overwrites on a STRICTLY higher
    score, see the merge loop below). The merged list is capped at
    ``maxPerAtom + denseK``.

    Returns ``{"atomsScanned", "proposals", "batches", "skippedNoFacets",
    "entityProposals", "denseProposals", "merged"}``. ``entityProposals`` and
    ``denseProposals`` are pre-dedup totals across all atoms; ``merged``
    (== ``proposals``) is the post-dedup total actually written. ``limit``
    caps the number of memory atoms scanned (pilot waves). Batch files are
    ``proposals-<NNNN>.jsonl`` under ``outDir`` (created).

    Building the code index is lazy: with ``denseK=0`` (or no atoms ever
    reach the dense branch), ``recall.vector_index`` is never imported, so a
    run with the dense channel off never needs usearch installed.
    """
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    kindMarks = ",".join("?" for _ in _MEMORY_KINDS)
    q = (f"SELECT id FROM atoms WHERE status = 'live' "
         f"AND kind IN ({kindMarks}) ORDER BY id")
    params = list(_MEMORY_KINDS)
    if limit is not None:
        q += " LIMIT ?"
        params.append(limit)
    memIds = [r[0] for r in store._conn.execute(q, params).fetchall()]

    report = {"atomsScanned": 0, "proposals": 0, "batches": 0,
              "skippedNoFacets": 0, "entityProposals": 0,
              "denseProposals": 0, "merged": 0}
    batch = []
    batchIdx = 0

    def _flush():
        nonlocal batch, batchIdx
        if not batch:
            return
        path = outDir / f"proposals-{batchIdx:04d}.jsonl"
        with open(path, "w") as fh:
            for line in batch:
                fh.write(json.dumps(line) + "\n")
        report["batches"] += 1
        batchIdx += 1
        batch = []

    textOf = {}

    def _text(atomId):
        if atomId not in textOf:
            row = store._conn.execute(
                "SELECT text, kind FROM atoms WHERE id = ?", (atomId,)
            ).fetchone()
            textOf[atomId] = row
        return textOf[atomId]

    def _ref(atomId):
        row = store._conn.execute(
            "SELECT source_ref FROM provenance WHERE atom_id = ? "
            "AND source_ref IS NOT NULL ORDER BY recorded_at LIMIT 1",
            (atomId,),
        ).fetchone()
        return row[0] if row else None

    freqMap = buildFreqMap(store)
    codeIndex = None
    if denseK > 0:
        from recall.vector_index import buildClassIndexes  # noqa: E402

        codeIndex = buildClassIndexes(store, modelId)["code"]

    for memId in memIds:
        report["atomsScanned"] += 1
        entityProps = proposeForAtom(store, memId, maxPerAtom=maxPerAtom,
                                     hubCap=hubCap, minScore=minScore,
                                     freqMap=freqMap)
        denseProps = (
            proposeDense(store, codeIndex, memId, modelId, topK=denseK,
                        minSim=minSim)
            if denseK > 0 else []
        )
        report["entityProposals"] += len(entityProps)
        report["denseProposals"] += len(denseProps)

        # Dedup by chunkId, higher score wins. Entity entries are seeded
        # first so a dense entry only replaces one on a STRICTLY higher
        # score: an exact tie keeps the entity entry (entity signal is a
        # verified shared fact, dense is a similarity guess, so the tie goes
        # to the more grounded channel).
        merged = {}
        for p in entityProps:
            merged[p["chunkId"]] = p
        for p in denseProps:
            existing = merged.get(p["chunkId"])
            if existing is None or p["score"] > existing["score"]:
                merged[p["chunkId"]] = p
        props = sorted(merged.values(),
                       key=lambda p: (-p["score"], p["chunkId"]))
        props = props[:maxPerAtom + denseK]
        report["merged"] += len(props)

        if not props:
            report["skippedNoFacets"] += 1
            continue
        memText, memKind = _text(memId)
        for p in props:
            chunkText, _ = _text(p["chunkId"])
            batch.append({
                **p,
                "memKind": memKind,
                "memText": _flat(memText, textCap),
                "chunkRef": _ref(p["chunkId"]),
                "chunkText": _flat(chunkText, textCap),
            })
            report["proposals"] += 1
            if len(batch) >= batchSize:
                _flush()
    _flush()
    return report


def main():
    ap = argparse.ArgumentParser(prog="edge-proposer")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-per-atom", type=int, default=3)
    ap.add_argument("--hub-cap", type=int, default=500)
    ap.add_argument("--min-score", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dense-k", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.5)
    args = ap.parse_args()
    store = openStore(Path(args.store))
    try:
        report = emitProposals(
            store, args.out, batchSize=args.batch_size,
            maxPerAtom=args.max_per_atom, hubCap=args.hub_cap,
            minScore=args.min_score, limit=args.limit,
            denseK=args.dense_k, minSim=args.min_sim)
        print(json.dumps(report, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
