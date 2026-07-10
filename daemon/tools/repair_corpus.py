#!/usr/bin/env python3
"""Phase A corpus repair CLI: dry-run by default, backup-gated --apply.

Dry-run opens the store read-only-in-spirit (no pass runs) and prints what
WOULD run plus current problem counts. --apply performs a SQLite backup-API
copy (safe on a live WAL database) to <store>.bak-pre-repair-<ts>, then runs
the three passes in order (kv refs, reflib dedup, project backfill) and the
verification invariants. Exit code 1 if verification fails.

Operational note: stop the pensive-v3 daemon (or accept that its resident
index is stale until its next reindex) before an --apply on the live store.

Dry-run purity depends on the store already being at the current schema
version, since an out-of-date store would run migration DDL on open; the
live store is at the current version.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_TOOLS.parent / "src"))

from repair_passes import (  # noqa: E402
    repairKvCacheRefs, dedupReferenceLibrary, backfillProjects,
    totals, verifyRepair,
)
from store.store import openStore  # noqa: E402

_DEFAULT_OLD_DB = Path.home() / "Projects/Aegis/AEGIS/Pensive/kv_cache/vector_meta.db"


def _backup(storePath):
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(str(storePath) + f".bak-pre-repair-{ts}")
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


def _problemCounts(store):
    conn = store._conn
    kv = conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE source_ref LIKE 'kv_cache%'"
    ).fetchone()[0]
    dupLive = conn.execute(
        "SELECT COUNT(*) FROM atoms a JOIN provenance p ON p.atom_id=a.id "
        "WHERE a.kind='document_chunk' AND a.status='live' "
        "AND p.source_ref LIKE 'reference-library/%'").fetchone()[0]
    nullProj = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE kind='document_chunk' "
        "AND project IS NULL").fetchone()[0]
    return {"kvRefs": kv, "reflibDupCandidates": dupLive,
            "nullProjects": nullProj}


def main():
    ap = argparse.ArgumentParser(prog="repair-corpus")
    ap.add_argument("--store", required=True)
    ap.add_argument("--old-db", default=str(_DEFAULT_OLD_DB))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args()

    store = openStore(Path(args.store))
    try:
        out = {"applied": False, "problems": _problemCounts(store)}
        if args.apply:
            backupPath = _backup(args.store)
            out["backup"] = str(backupPath)
            pre = totals(store)
            out["repairKvCacheRefs"] = repairKvCacheRefs(store, args.old_db)
            out["dedupReferenceLibrary"] = dedupReferenceLibrary(
                store, sessionId=args.session_id)
            out["backfillProjects"] = backfillProjects(store)
            out["verify"] = verifyRepair(store, pre)
            out["applied"] = True
        print(json.dumps(out, indent=2))
        if args.apply and not out["verify"]["ok"]:
            return 1
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
