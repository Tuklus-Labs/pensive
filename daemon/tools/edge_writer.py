#!/usr/bin/env python3
"""Edge campaign writer: verdicts in, provenance-stamped relates edges out.

Direction contract (pinned by tests, load-bearing for the payload): edges are
``src = chunk, dst = memory, type = 'relates'``, so ``edgesFrom(chunk)``
renders the Tier-2 line ``relates -> p3://<memId> <gist>`` and the enrichment
payload upgrade (Task 3) reads real edges instead of live facet joins.

Every written edge carries its own provenance row (``source='edge-campaign'``,
``source_ref=<campaignRef>``), which makes the campaign a first-class,
auditable, rollback-safe unit: ``rollbackCampaign`` deletes exclusively rows
tied to that ref. This is the ONE sanctioned deletion in the whole Layer 2/3
effort; it can only ever touch rows the campaign itself created. Rollback is
also why weight UPDATES on rerun keep the ORIGINAL provenance row: the edge
remains owned by the campaign that created it.

The writer enforces (src, dst, type) uniqueness by lookup (the schema has no
unique constraint): reruns update weight, never duplicate.
"""
import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_SRC = _TOOLS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from store.store import openStore  # noqa: E402
from util.ulid import ulid  # noqa: E402

__all__ = ["writeVerdicts", "rollbackCampaign", "sampleCampaignEdges"]

_WS_RE = re.compile(r"\s+")


def _now():
    return int(time.time())


def writeVerdicts(store, verdictLines, campaignRef, agent=None):
    """Apply verdict dicts -> relates edges. One transaction, all-or-nothing.

    Keeps write ``src=chunkId, dst=memId, type='relates', weight=confidence``
    with a fresh campaign provenance row. An existing (src,dst,'relates')
    edge gets its weight updated instead (idempotent rerun; provenance row
    kept from the original write). keep=False counts as rejected. A verdict
    naming a missing atom is counted and skipped (the fleet may race a
    supersession; never crash the batch). Malformed lines are counted.
    """
    report = {"written": 0, "updated": 0, "rejected": 0,
              "skippedMissingAtom": 0, "malformed": 0}
    conn = store._conn
    try:
        for line in verdictLines:
            try:
                memId = line["memId"]
                chunkId = line["chunkId"]
                keep = bool(line["keep"])
                confidence = float(line.get("confidence", 1.0))
            except (KeyError, TypeError, ValueError):
                report["malformed"] += 1
                continue
            if not keep:
                report["rejected"] += 1
                continue
            exists = conn.execute(
                "SELECT COUNT(*) FROM atoms WHERE id IN (?, ?)",
                (memId, chunkId)).fetchone()[0]
            if exists != 2:
                report["skippedMissingAtom"] += 1
                continue
            existing = conn.execute(
                "SELECT id FROM edges WHERE src_atom = ? AND dst_atom = ? "
                "AND type = 'relates'", (chunkId, memId)).fetchone()
            if existing:
                conn.execute("UPDATE edges SET weight = ? WHERE id = ?",
                             (confidence, existing[0]))
                report["updated"] += 1
                continue
            provId = ulid()
            conn.execute(
                "INSERT INTO provenance(id, atom_id, source, session_id, "
                "agent, source_ref, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (provId, chunkId, "edge-campaign", None, agent,
                 campaignRef, _now()))
            conn.execute(
                "INSERT INTO edges(id, src_atom, dst_atom, type, weight, "
                "created_at, provenance_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ulid(), chunkId, memId, "relates", confidence, _now(),
                 provId))
            report["written"] += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return report


def rollbackCampaign(store, campaignRef):
    """Delete edges (and their provenance rows) owned by ``campaignRef``.

    The ONLY deletion in the campaign tooling, structurally scoped: both
    DELETEs key on provenance rows whose ``source='edge-campaign' AND
    source_ref=<campaignRef>``; a row without that exact ownership cannot
    match. One transaction. Idempotent (second run deletes nothing).
    """
    conn = store._conn
    try:
        provIds = [r[0] for r in conn.execute(
            "SELECT id FROM provenance WHERE source = 'edge-campaign' "
            "AND source_ref = ?", (campaignRef,)).fetchall()]
        edgesDeleted = 0
        if provIds:
            marks = ",".join("?" for _ in provIds)
            cur = conn.execute(
                f"DELETE FROM edges WHERE provenance_id IN ({marks})",
                provIds)
            edgesDeleted = cur.rowcount
            conn.execute(
                f"DELETE FROM provenance WHERE id IN ({marks})", provIds)
        conn.commit()
        return {"edgesDeleted": edgesDeleted,
                "provenanceDeleted": len(provIds)}
    except Exception:
        conn.rollback()
        raise


def sampleCampaignEdges(store, campaignRef, n, textCap=1200):
    """Random n campaign edges with both texts, for audit re-verification."""
    rows = store._conn.execute(
        "SELECT e.src_atom, e.dst_atom, e.weight FROM edges e "
        "JOIN provenance p ON p.id = e.provenance_id "
        "WHERE p.source = 'edge-campaign' AND p.source_ref = ?",
        (campaignRef,)).fetchall()
    picked = random.sample(rows, min(n, len(rows)))
    out = []
    for chunkId, memId, weight in picked:
        texts = {}
        for key, atomId in (("chunkText", chunkId), ("memText", memId)):
            row = store._conn.execute(
                "SELECT text FROM atoms WHERE id = ?", (atomId,)).fetchone()
            texts[key] = _WS_RE.sub(" ", row[0]).strip()[:textCap] if row else ""
        out.append({"memId": memId, "chunkId": chunkId, "weight": weight,
                    **texts})
    return out


def _backup(storePath):
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(str(storePath) + f".bak-pre-edges-{ts}")
    src = sqlite3.connect(storePath)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return dest


def main():
    ap = argparse.ArgumentParser(prog="edge-writer")
    ap.add_argument("--store", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--agent", default=None)
    args = ap.parse_args()

    store = openStore(Path(args.store))
    try:
        if args.rollback:
            out = rollbackCampaign(store, args.campaign)
        elif args.sample is not None:
            out = sampleCampaignEdges(store, args.campaign, args.sample)
        elif args.verdicts is not None:
            lines = []
            malformedRaw = 0
            with open(args.verdicts) as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        lines.append(json.loads(raw))
                    except json.JSONDecodeError:
                        malformedRaw += 1
            if args.apply:
                backupPath = _backup(args.store)
                out = writeVerdicts(store, lines, args.campaign,
                                    agent=args.agent)
                out["backup"] = str(backupPath)
            else:
                keeps = sum(1 for l in lines
                            if isinstance(l, dict) and l.get("keep"))
                out = {"applied": False, "verdictLines": len(lines),
                       "keeps": keeps}
            out["malformedRawLines"] = malformedRaw
        else:
            print("one of --verdicts/--rollback/--sample required",
                  file=sys.stderr)
            return 2
        print(json.dumps(out, indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
