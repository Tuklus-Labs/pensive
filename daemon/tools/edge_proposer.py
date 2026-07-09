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

__all__ = ["proposeForAtom", "emitProposals"]

_MEMORY_KINDS = ("atom", "narrative", "snapshot")

# Same-project additive bonus: small on purpose, a tiebreak-plus, never a
# substitute for a shared entity (pairs with NO shared non-hub entity are
# never proposed at all).
_PROJECT_BONUS = 0.05

_WS_RE = re.compile(r"\s+")


def _flat(text, cap):
    return _WS_RE.sub(" ", text or "").strip()[:cap]


def proposeForAtom(store, memId, maxPerAtom=3, hubCap=500, minScore=0.02):
    """Top candidate chunks for one memory atom -> [{memId, chunkId, score,
    entities}] best-first.

    Entities with live-atom frequency above ``hubCap`` are excluded before
    scoring. Candidates are LIVE document_chunk atoms sharing at least one
    surviving entity. Score = sum(1/freq) over shared surviving entities,
    plus _PROJECT_BONUS when projects match. Results below ``minScore`` are
    dropped; at most ``maxPerAtom`` returned, ordered (score desc, chunkId)
    for determinism.
    """
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
        })
    out.sort(key=lambda p: (-p["score"], p["chunkId"]))
    return out[:maxPerAtom]


def emitProposals(store, outDir, batchSize=50, textCap=1200,
                  maxPerAtom=3, hubCap=500, minScore=0.02, limit=None):
    """Score every live memory atom and write self-contained JSONL batches.

    Returns ``{"atomsScanned", "proposals", "batches", "skippedNoFacets"}``.
    ``limit`` caps the number of memory atoms scanned (pilot waves). Batch
    files are ``proposals-<NNNN>.jsonl`` under ``outDir`` (created).
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
              "skippedNoFacets": 0}
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

    for memId in memIds:
        report["atomsScanned"] += 1
        props = proposeForAtom(store, memId, maxPerAtom=maxPerAtom,
                               hubCap=hubCap, minScore=minScore)
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
    args = ap.parse_args()
    store = openStore(Path(args.store))
    try:
        report = emitProposals(
            store, args.out, batchSize=args.batch_size,
            maxPerAtom=args.max_per_atom, hubCap=args.hub_cap,
            minScore=args.min_score, limit=args.limit)
        print(json.dumps(report, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
