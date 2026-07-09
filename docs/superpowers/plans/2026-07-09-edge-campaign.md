# Edge Materialization Tooling (Phase C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the propose / write / audit tooling for the chunk-to-memory edge campaign, so a verification fleet can turn facet co-occurrence into a real, provenance-stamped, rollback-safe `relates` edge graph that the enrichment payload then prefers over live facet joins.

**Architecture:** Three pieces. `edge_proposer.py` (read-only: scores memory-chunk pairs by shared-entity specificity with hub damping, emits self-contained JSONL batches carrying both texts so verify agents never touch the store). `edge_writer.py` (mutating, backup-gated like the repair CLI: consumes verdict JSONL, writes `relates` edges src=chunk dst=memory with campaign provenance, idempotent, with rollback-by-campaign and random sample-audit export). A small `Enricher` upgrade (prefer real `relates` edges from the chunk, fall back to the facet join). The fleet itself is a Workflow the controller authors separately; it is NOT part of this plan.

**Tech Stack:** Python stdlib + the daemon's store API. Tests with pytest over tmp_path stores.

## Global Constraints

- Work in `~/Projects/pensive/daemon`. Branch `feat/corpus-enrichment`. No new branches/worktrees.
- No em dashes anywhere in code, comments, docstrings, or commit messages.
- Zero deletions of ATOMS ever. The ONLY permitted deletion anywhere in this plan is `edge_writer.py --rollback`, which deletes exclusively edges (and their provenance rows) whose provenance `source_ref` equals the given campaign ref: the campaign owns its own rows. The rollback SQL must be structurally incapable of touching a row without that campaign ref.
- Edge direction: `src = chunk atom, dst = memory atom, type = 'relates'`. This matches the Tier-2 payload grammar (`edgesFrom(chunk)` renders `relates -> p3://<memId> <gist>`); getting it backwards would render nothing. Every task's tests pin the direction.
- Idempotency: re-running the writer with the same verdicts updates weights, never duplicates an edge (the schema has NO unique constraint on (src,dst,type); the writer enforces it by lookup).
- Never touch the live store or run against it from any subagent; tmp stores only. Live runs are controller-only.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
  ```

**Verified store facts:** `facets(atom_id, key, value)`, `key='entity'`; memory kinds `atom|narrative|snapshot`; ~14.2k live memory atoms, ~283k live chunks (post-dedup); `addEdge(store, {src, dst, type, weight?, provenanceId?})`; provenance insert shape as in `store.putAtom`; `supersede` writes provenance with `source_ref`; hub entities exist (e.g. FILES-type) with very high frequency.

---

### Task 1: Edge proposer (`tools/edge_proposer.py`)

**Files:**
- Create: `tools/edge_proposer.py`
- Test: `test/tools/test_edge_proposer.py`

**Interfaces:**
- Produces:
  - `proposeForAtom(store, memId, maxPerAtom=3, hubCap=500, minScore=0.02) -> list[dict]` where each dict is `{"memId", "chunkId", "score", "entities": [..]}`
  - `emitProposals(store, outDir, batchSize=50, textCap=1200, **kw) -> dict` report `{"atomsScanned", "proposals", "batches", "skippedNoFacets"}`; writes `proposals-<NNNN>.jsonl` files where each line carries `memId, chunkId, score, entities, memKind, memText, chunkRef, chunkText` (texts truncated to `textCap` chars, whitespace-collapsed)
  - CLI: `python3 tools/edge_proposer.py --store PATH --out DIR [--max-per-atom N] [--hub-cap N] [--min-score F] [--limit N]` (read-only; `--limit` caps scanned memory atoms for pilot waves)

- [ ] **Step 1: Write the failing tests**

Create `test/tools/test_edge_proposer.py`:

```python
"""Edge proposer: specificity-scored memory-to-chunk candidate pairs."""
import json
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from edge_proposer import proposeForAtom, emitProposals
from store.store import openStore, putAtom, addFacet


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind="atom", project="aegis", sourceRef=None):
    prov = {"source": "claude-code"}
    if sourceRef is not None:
        prov["sourceRef"] = sourceRef
    return putAtom(store, {
        "text": text, "kind": kind, "project": project,
        "importance": 0.0, "provenance": prov,
    })


def test_proposes_rare_shared_entity_chunk_first(store):
    mem = _put(store, "decision about the csr publish lock")
    hot = _put(store, "def compile(): csr publish lock impl",
               kind="document_chunk", sourceRef="projects/p/x.py#c0")
    cold = _put(store, "def unrelated(): pass", kind="document_chunk")
    other = _put(store, "another chunk sharing a commoner entity",
                 kind="document_chunk")
    addFacet(store, mem, "entity", "csr_lock")
    addFacet(store, mem, "entity", "python")
    addFacet(store, hot, "entity", "csr_lock")
    addFacet(store, other, "entity", "python")
    # freq: csr_lock=2 (mem+hot), python=3 (mem+other+below)
    extra = _put(store, "yet another python mention", kind="document_chunk")
    addFacet(store, extra, "entity", "python")
    got = proposeForAtom(store, mem)
    ids = [p["chunkId"] for p in got]
    assert ids[0] == hot                       # rare entity wins
    assert cold not in ids                     # no shared entity
    assert all(p["memId"] == mem for p in got)
    assert got[0]["score"] > 0


def test_hub_entities_are_damped_out(store):
    mem = _put(store, "a memory atom")
    chunk = _put(store, "a chunk", kind="document_chunk")
    addFacet(store, mem, "entity", "hub_value")
    addFacet(store, chunk, "entity", "hub_value")
    # Push hub_value's frequency past the hubCap.
    for i in range(6):
        extra = _put(store, f"filler {i}", kind="document_chunk")
        addFacet(store, extra, "entity", "hub_value")
    got = proposeForAtom(store, mem, hubCap=5)
    assert got == []                           # only shared entity was a hub


def test_cap_and_min_score(store):
    mem = _put(store, "memory with one rare entity")
    addFacet(store, mem, "entity", "rare_e")
    chunks = []
    for i in range(5):
        c = _put(store, f"chunk {i}", kind="document_chunk")
        addFacet(store, c, "entity", "rare_e")
        chunks.append(c)
    got = proposeForAtom(store, mem, maxPerAtom=2)
    assert len(got) == 2
    # A ridiculous min score filters everything.
    assert proposeForAtom(store, mem, minScore=99.0) == []


def test_only_live_memory_and_live_chunks(store):
    mem = _put(store, "memory")
    chunk = _put(store, "chunk", kind="document_chunk")
    addFacet(store, mem, "entity", "shared_e")
    addFacet(store, chunk, "entity", "shared_e")
    store._conn.execute(
        "UPDATE atoms SET status='superseded' WHERE id=?", (chunk,))
    store._conn.commit()
    assert proposeForAtom(store, mem) == []


def test_emitProposals_batches_and_self_contained_lines(store, tmp_path):
    mem = _put(store, "the decision text " + "x" * 3000)
    for i in range(3):
        c = _put(store, f"chunk body {i} " + "y" * 3000,
                 kind="document_chunk", sourceRef=f"projects/p/f{i}.py#c0")
        addFacet(store, c, "entity", "rare_e")
    addFacet(store, mem, "entity", "rare_e")
    out = tmp_path / "props"
    report = emitProposals(store, out, batchSize=2, textCap=100)
    assert report["proposals"] == 3
    assert report["batches"] == 2              # 2 + 1
    files = sorted(out.glob("proposals-*.jsonl"))
    assert len(files) == 2
    lines = [json.loads(l) for f in files for l in f.read_text().splitlines()]
    assert len(lines) == 3
    for line in lines:
        # Self-contained: verify agents never need the store.
        assert set(line) >= {"memId", "chunkId", "score", "entities",
                             "memKind", "memText", "chunkRef", "chunkText"}
        assert len(line["memText"]) <= 100
        assert len(line["chunkText"]) <= 100
        assert "\n" not in line["memText"]     # whitespace-collapsed
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/tools/test_edge_proposer.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'edge_proposer'`.

- [ ] **Step 3: Implement**

Create `tools/edge_proposer.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest test/tools/test_edge_proposer.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/edge_proposer.py test/tools/test_edge_proposer.py
git commit -m "$(cat <<'EOF'
feat(tools): edge campaign proposer

Specificity-scored memory-to-chunk pairs with outright hub-entity
exclusion, per-atom caps, and self-contained JSONL batches so verify
agents never touch the store. Read-only by construction.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 2: Edge writer with rollback and sample audit (`tools/edge_writer.py`)

**Files:**
- Create: `tools/edge_writer.py`
- Test: `test/tools/test_edge_writer.py`

**Interfaces:**
- Produces:
  - `writeVerdicts(store, verdictLines, campaignRef, agent=None) -> dict` report `{"written", "updated", "rejected", "skippedMissingAtom", "malformed"}`. A verdict line is `{"memId", "chunkId", "keep": bool, "confidence": float}`; edges written `src=chunkId, dst=memId, type='relates', weight=confidence`, each with its own provenance row `source='edge-campaign', source_ref=campaignRef`.
  - `rollbackCampaign(store, campaignRef) -> dict` `{"edgesDeleted", "provenanceDeleted"}` deleting ONLY rows tied to that campaignRef.
  - `sampleCampaignEdges(store, campaignRef, n, textCap=1200) -> list[dict]` random n written edges with both texts (for audit re-verification).
  - CLI: `python3 tools/edge_writer.py --store PATH --campaign REF (--verdicts FILE.jsonl [--apply] | --rollback | --sample N)`. Verdict writing is dry-run by default (`--apply` mutates after a `_backup` identical in shape to repair_corpus.py's).

- [ ] **Step 1: Write the failing tests**

Create `test/tools/test_edge_writer.py`:

```python
"""Edge writer: idempotent relates edges with campaign provenance + rollback."""
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from edge_writer import writeVerdicts, rollbackCampaign, sampleCampaignEdges
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _pair(store):
    mem = putAtom(store, {
        "text": "memory", "kind": "atom", "project": "p",
        "importance": 0.0, "provenance": {"source": "claude-code"}})
    chunk = putAtom(store, {
        "text": "chunk", "kind": "document_chunk", "project": "p",
        "importance": 0.0, "provenance": {"source": "bulk-import"}})
    return mem, chunk


def _verdict(mem, chunk, keep=True, confidence=0.9):
    return {"memId": mem, "chunkId": chunk, "keep": keep,
            "confidence": confidence}


def test_writes_edge_chunk_to_memory_with_campaign_provenance(store):
    mem, chunk = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, chunk)], "camp-1")
    assert report["written"] == 1
    edge = store._conn.execute(
        "SELECT src_atom, dst_atom, type, weight, provenance_id "
        "FROM edges WHERE type='relates'").fetchone()
    assert edge[0] == chunk and edge[1] == mem      # DIRECTION: src=chunk
    assert edge[3] == pytest.approx(0.9)
    prov = store._conn.execute(
        "SELECT source, source_ref FROM provenance WHERE id=?",
        (edge[4],)).fetchone()
    assert prov == ("edge-campaign", "camp-1")


def test_rejected_verdicts_write_nothing(store):
    mem, chunk = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, chunk, keep=False)], "camp-1")
    assert report["rejected"] == 1
    assert store._conn.execute(
        "SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_idempotent_rerun_updates_weight_not_duplicate(store):
    mem, chunk = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk, confidence=0.9)], "camp-1")
    report = writeVerdicts(store, [_verdict(mem, chunk, confidence=0.4)],
                           "camp-1")
    assert report["written"] == 0
    assert report["updated"] == 1
    rows = store._conn.execute(
        "SELECT weight FROM edges WHERE type='relates'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.4)


def test_missing_atom_skipped_not_crash(store):
    mem, _ = _pair(store)
    report = writeVerdicts(store, [_verdict(mem, "01FAKEID")], "camp-1")
    assert report["skippedMissingAtom"] == 1
    assert store._conn.execute(
        "SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_malformed_line_counted(store):
    report = writeVerdicts(store, [{"nonsense": True}], "camp-1")
    assert report["malformed"] == 1


def test_rollback_deletes_only_campaign_rows(store):
    mem, chunk = _pair(store)
    mem2, chunk2 = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk)], "camp-A")
    writeVerdicts(store, [_verdict(mem2, chunk2)], "camp-B")
    report = rollbackCampaign(store, "camp-A")
    assert report["edgesDeleted"] == 1
    assert report["provenanceDeleted"] == 1
    left = store._conn.execute(
        "SELECT src_atom FROM edges WHERE type='relates'").fetchall()
    assert left == [(chunk2,)]                 # camp-B untouched
    # Idempotent: rolling back again deletes nothing.
    assert rollbackCampaign(store, "camp-A") == {
        "edgesDeleted": 0, "provenanceDeleted": 0}


def test_sample_returns_written_edges_with_texts(store):
    mem, chunk = _pair(store)
    writeVerdicts(store, [_verdict(mem, chunk)], "camp-1")
    got = sampleCampaignEdges(store, "camp-1", 5)
    assert len(got) == 1
    assert got[0]["memId"] == mem and got[0]["chunkId"] == chunk
    assert got[0]["memText"] == "memory" and got[0]["chunkText"] == "chunk"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/tools/test_edge_writer.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'edge_writer'`.

- [ ] **Step 3: Implement**

Create `tools/edge_writer.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest test/tools/test_edge_writer.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/edge_writer.py test/tools/test_edge_writer.py
git commit -m "$(cat <<'EOF'
feat(tools): edge campaign writer with rollback and sample audit

Verdicts become relates edges (src=chunk, dst=memory) with per-edge
campaign provenance; reruns update weight, never duplicate; rollback
deletes exclusively campaign-owned rows; sample export feeds the audit
wave. Verdict CLI is dry-run by default and backup-gated on --apply.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 3: Enricher prefers real edges

**Files:**
- Modify: `src/recall/enrich.py` (`Enricher.lines` relation source)
- Test: `test/recall/test_enrich.py` (add tests)

**Interfaces:**
- Changed behavior: for a chunk with LIVE outgoing `relates` edges to live memory atoms, `Enricher.lines` renders those (weight desc, capped at `relatedLimit`) INSTEAD of the facet-join `relatedMemory`. Chunks without such edges keep today's facet-join fallback, so the payload works before, during, and after the campaign.

- [ ] **Step 1: Write the failing tests**

Add to `test/recall/test_enrich.py`:

```python
from store.store import addEdge


def test_enricher_prefers_real_relates_edges(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    viaEdge = _put(store, "memory linked by verified edge")
    viaFacet = _put(store, "memory linked only by facet")
    addFacet(store, chunk, "entity", "shared_e")
    addFacet(store, viaFacet, "entity", "shared_e")
    addEdge(store, {"src": chunk, "dst": viaEdge, "type": "relates",
                    "weight": 0.9})
    lines = Enricher(store).lines({"atomId": chunk})
    joined = "\n".join(lines)
    assert f"relates -> p3://{viaEdge}" in joined
    assert viaFacet not in joined              # edges replace the facet join


def test_edge_to_dead_memory_falls_back_to_facets(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    dead = _put(store, "superseded memory")
    live = _put(store, "facet-linked memory")
    addEdge(store, {"src": chunk, "dst": dead, "type": "relates",
                    "weight": 0.9})
    store._conn.execute(
        "UPDATE atoms SET status='superseded' WHERE id=?", (dead,))
    store._conn.commit()
    addFacet(store, chunk, "entity", "shared_e")
    addFacet(store, live, "entity", "shared_e")
    lines = Enricher(store).lines({"atomId": chunk})
    joined = "\n".join(lines)
    assert dead not in joined                  # dead edge target never renders
    assert f"relates -> p3://{live}" in joined # fallback still works


def test_edges_ordered_by_weight_and_capped(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    mems = [_put(store, f"memory {i}") for i in range(4)]
    weights = [0.2, 0.9, 0.5, 0.7]
    for m, w in zip(mems, weights):
        addEdge(store, {"src": chunk, "dst": m, "type": "relates",
                        "weight": w})
    lines = Enricher(store, relatedLimit=2).lines({"atomId": chunk})
    rel = [l for l in lines if l.startswith("relates ->")]
    assert len(rel) == 2
    assert f"p3://{mems[1]}" in rel[0]         # 0.9 first
    assert f"p3://{mems[3]}" in rel[1]         # 0.7 second
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/recall/test_enrich.py -v -k edge`
Expected: FAIL (the facet-join result renders `viaFacet`; edges are ignored today).

- [ ] **Step 3: Implement**

In `src/recall/enrich.py`, add a module-level helper and change `Enricher.lines`'s relation block. New helper (place after `relatedMemory`):

```python
def _edgeRelations(store, chunkId, limit):
    """LIVE relates edges from a chunk -> [(memId, gist)] by weight desc.

    Only edges whose destination is a live atom render (parity with the
    Tier-2 rule: never dangle a pointer at a non-recallable atom). Returns
    [] when the chunk has no live relates edges, which is the signal to fall
    back to the facet join.
    """
    rows = store._conn.execute(
        "SELECT e.dst_atom FROM edges e "
        "JOIN atoms a ON a.id = e.dst_atom "
        "WHERE e.src_atom = ? AND e.type = 'relates' "
        "AND a.status = 'live' "
        "ORDER BY e.weight DESC, e.dst_atom LIMIT ?",
        (chunkId, limit),
    ).fetchall()
    out = []
    for (memId,) in rows:
        atom = getAtom(store, memId)
        if atom is None:
            continue
        out.append((memId, _gist(atom["text"])))
    return out
```

In `Enricher.lines`, replace the relation loop:

```python
        relations = _edgeRelations(
            self._store, result["atomId"], self._relatedLimit)
        if not relations:
            relations = relatedMemory(
                self._store, result["atomId"], self._relatedLimit)
        for memId, gist in relations:
            out.append(f"relates -> p3://{memId} {gist}")
```

Update the module docstring's relation-lines paragraph: real verified edges render when present; the facet join is the fallback for chunks the campaign has not covered.

- [ ] **Step 4: Run the enrichment + payload + engine suites**

Run: `python3 -m pytest test/recall/test_enrich.py test/recall/test_payload_enrich.py test/recall/test_engine.py -v`
Expected: PASS including all pre-existing tests (chunks without edges keep facet-join behavior, so nothing regresses).

- [ ] **Step 5: Run the full daemon suite**

Run: `python3 -m pytest test/ -q`
Expected: 0 failed.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/enrich.py test/recall/test_enrich.py
git commit -m "$(cat <<'EOF'
feat(recall): enrichment prefers verified relates edges

A chunk with live relates edges renders those (weight desc, capped)
instead of the facet join; edge-less chunks keep the fallback, so the
payload works before, during, and after the campaign.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

## After this plan: the fleet (controller-only, NOT plan tasks)

1. Pilot: proposer `--limit 200` -> Workflow verify wave -> writer dry-run -> inspect kill rate and verdict quality -> tune hubCap/minScore if needed.
2. Full campaign under the fleet-director doctrine: propose all ~14.2k atoms, verification Workflow (refute-biased, batch files as agent inputs, structured verdict outputs), writer `--apply` with campaign ref, sample audit wave, rollback gate on audit failure.
3. Daemon restart + live payload check (a campaign-covered chunk renders edge-sourced relates lines).

## Self-Review

- **Spec coverage:** Propose (specificity + hub damping + caps) = Task 1; verify support (self-contained batches, no store access for agents) = Task 1; write (weight=confidence, campaign provenance, idempotent, UNIQUE-by-lookup) = Task 2; gate (sample audit + rollback-by-campaign, deletion scoped to campaign-owned rows only) = Task 2; "feeds Phase B" (edges preferred over facet joins) = Task 3. The fleet itself is deliberately out of plan scope.
- **Placeholder scan:** none; all steps carry complete code and expected outputs.
- **Type consistency:** verdict dict keys match between Task 2 tests and `writeVerdicts`; `Enricher(store, home=None, relatedLimit=3)` signature untouched (Task 3 only changes `.lines` internals); edge direction (src=chunk) is identical in Task 2's writer, Task 2's tests, Task 3's `_edgeRelations` query, and the payload grammar rationale.
- **Self-referential predicate check (the Task-3-of-repair lesson):** the writer's own writes are edges+provenance, which its verdict-driven predicate never scans (it reads verdict FILES, not the store); rollback keys on campaign ownership. The proposer reads facets only, never edges. No pass here can re-trigger on its own output.

---

### Task 4 (amendment, post-pilot): dense proposal channel + freq precompute

**Pilot finding driving this task:** 200-atom pilot yielded 30 proposals with 189 zero-yield atoms. Facet coverage is 98.9%, but the entity vocabulary connects memory to chunks almost entirely through hub values (aegis 26k, claude 16k) and dates; only ~700 rare shared values carry signal. Also 1.12s/atom (the freq CTE recomputes per atom) makes the full run 4.4h. The campaign needs a second, semantic proposal channel and a perf fix. Verification stays the precision gate: propose loose, verify hard.

**Files:**
- Modify: `tools/edge_proposer.py`
- Test: `test/tools/test_edge_proposer.py` (add tests)

**Interfaces:**
- Changed: `proposeForAtom(store, memId, maxPerAtom=3, hubCap=500, minScore=0.02, freqMap=None) -> list[dict]`; when `freqMap` (a `{value: liveCount}` dict) is given, NO freq SQL runs for scoring: shared-entity candidates come from one indexed query and scores use the map. Behavior with `freqMap=None` is unchanged (backward compatible, existing tests untouched).
- New: `buildFreqMap(store) -> dict` (one GROUP BY over live entity facets).
- New: `proposeDense(store, codeIndex, memId, modelId, topK=5, minSim=0.5) -> list[dict]` each `{"memId", "chunkId", "score": cosine, "entities": [], "channel": "dense"}`; fetches the memory atom's STORED vector (`SELECT vector FROM embeddings WHERE atom_id=? AND model_id=?`, via `recall.embedder.blobToVec`), searches the code-class index, filters `minSim`. Missing embedding -> [] (never embeds at proposal time).
- Changed: `emitProposals(store, outDir, ..., denseK=5, minSim=0.5, modelId=MODEL_ID)` builds the freq map once and the code index once (`buildClassIndexes(store, modelId)["code"]`), merges both channels per atom (dedup by chunkId keeping the higher score, entity channel tagged `"channel": "entity"`), caps the merged list at `maxPerAtom + denseK`. Report gains `{"entityProposals", "denseProposals", "merged"}`.

- [ ] **Step 1: Write the failing tests**

Add to `test/tools/test_edge_proposer.py`:

```python
from edge_proposer import buildFreqMap, proposeDense

MODEL_ID = "BAAI/bge-small-en-v1.5"


def test_freqMap_matches_per_atom_scoring(store):
    mem = _put(store, "memory text")
    hot = _put(store, "chunk text", kind="document_chunk")
    addFacet(store, mem, "entity", "rare_e")
    addFacet(store, hot, "entity", "rare_e")
    fm = buildFreqMap(store)
    assert fm["rare_e"] == 2
    withMap = proposeForAtom(store, mem, freqMap=fm)
    without = proposeForAtom(store, mem)
    assert withMap == without           # identical scoring, no freq SQL


def test_proposeDense_finds_semantic_neighbor(store):
    import pytest
    emb = pytest.importorskip("recall.embedder")
    from recall.embedder import Embedder, embedMissing
    from recall.vector_index import buildClassIndexes
    embedder = Embedder(MODEL_ID)
    mem = _put(store, "we fixed the csr matrix publish race with a lock")
    close = _put(store, "def compile(): publish csr matrix under build lock",
                 kind="document_chunk")
    far = _put(store, "banana bread recipe with walnuts",
               kind="document_chunk")
    embedMissing(store, embedder)
    codeIndex = buildClassIndexes(store, MODEL_ID)["code"]
    got = proposeDense(store, codeIndex, mem, MODEL_ID, topK=2, minSim=0.3)
    ids = [p["chunkId"] for p in got]
    assert close in ids
    assert all(p["channel"] == "dense" for p in got)
    assert all(p["score"] >= 0.3 for p in got)
    tight = proposeDense(store, codeIndex, mem, MODEL_ID, topK=2, minSim=0.99)
    assert tight == []                  # threshold filters


def test_proposeDense_missing_embedding_returns_empty(store):
    mem = _put(store, "never embedded")
    class _BombIndex:
        def search(self, vec, k):
            raise AssertionError("must not search without a vector")
    assert proposeDense(store, _BombIndex(), mem, MODEL_ID) == []


def test_emitProposals_merges_channels_dedup_keeps_higher(store, tmp_path):
    import json
    import pytest
    pytest.importorskip("recall.embedder")
    from recall.embedder import Embedder, embedMissing
    embedder = Embedder(MODEL_ID)
    mem = _put(store, "we fixed the csr matrix publish race with a lock")
    both = _put(store, "def compile(): publish csr matrix under build lock",
                kind="document_chunk")
    addFacet(store, mem, "entity", "rare_e")
    addFacet(store, both, "entity", "rare_e")   # entity AND dense candidate
    embedMissing(store, embedder)
    out = tmp_path / "props"
    report = emitProposals(store, out, minSim=0.3)
    lines = [json.loads(l) for f in sorted(out.glob("*.jsonl"))
             for l in f.read_text().splitlines()]
    ours = [l for l in lines if l["chunkId"] == both]
    assert len(ours) == 1               # deduped across channels
    assert report["merged"] == report["proposals"]
    assert report["entityProposals"] >= 1
    assert report["denseProposals"] >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/tools/test_edge_proposer.py -v -k "freqMap or Dense or merges"`
Expected: FAIL, ImportError on buildFreqMap/proposeDense.

- [ ] **Step 3: Implement**

In `tools/edge_proposer.py`: add near the imports

```python
_SRC_IMPORT_NOTE = None  # recall.* resolves via the _SRC path insert above
from recall.embedder import blobToVec  # noqa: E402

MODEL_ID = "BAAI/bge-small-en-v1.5"
```

Add:

```python
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
```

Rework `proposeForAtom` to accept `freqMap=None`: when None, keep the existing CTE query exactly as-is; when given, run the simpler candidate query (no freq CTE) and score from the map:

```python
def proposeForAtom(store, memId, maxPerAtom=3, hubCap=500, minScore=0.02,
                   freqMap=None):
    if freqMap is None:
        rows = store._conn.execute(
            ... the existing CTE query unchanged ...
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
    ... rest of the function unchanged (byChunk accumulation onward),
        with each entity proposal dict gaining "channel": "entity" ...
```

Add:

```python
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
```

In `emitProposals`: build `freqMap = buildFreqMap(store)` and (lazily, only if denseK > 0) `codeIndex = buildClassIndexes(store, modelId)["code"]` once before the loop; per atom collect `props = proposeForAtom(..., freqMap=freqMap)` plus `proposeDense(store, codeIndex, memId, modelId, topK=denseK, minSim=minSim)`; merge with dedup by chunkId keeping the higher score entry; cap merged at `maxPerAtom + denseK`; count `entityProposals`, `denseProposals` (pre-dedup) and `merged` (post-dedup, == proposals). Lazy import of `buildClassIndexes` inside `emitProposals` so the no-dense path (denseK=0) never needs usearch. CLI gains `--dense-k` (default 5) and `--min-sim` (default 0.5).

- [ ] **Step 4: Run the whole proposer suite**

Run: `python3 -m pytest test/tools/test_edge_proposer.py -v`
Expected: PASS, 9 tests (5 original untouched + 4 new).

- [ ] **Step 5: Full tools suite + commit**

Run: `python3 -m pytest test/tools/ -q` (0 failed), then commit:

```bash
cd ~/Projects/pensive/daemon
git add tools/edge_proposer.py test/tools/test_edge_proposer.py
git commit -m "$(cat <<'EOF'
feat(tools): dense proposal channel + freq precompute for edge campaign

Pilot showed the entity vocabulary connects mostly via hubs and dates
(30 proposals from 200 atoms) at 1.12s/atom. buildFreqMap turns per-atom
freq CTEs into dict lookups; proposeDense searches the code-class index
with each memory atom's stored vector; emitProposals merges both channels
with cross-channel dedup. Propose loose, verify hard.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```
