# Layer 5: what the store actually says, before anyone designs anything

> Historical record, written 2026-08-13. It describes the project as it stood then and is kept
> for provenance. Current behavior is documented in [README.md](README.md) and
> [daemon/README.md](daemon/README.md).

Measured 2026-08-13 against the live store (read-only). This exists because I
filed Layer 5 as a finding and got two things wrong in the filing. Ground truth
first, design second.

## Correction 1: importance is not zero

I filed "importance is 0.0 across 296,305 atoms." That is false.

| | |
|---|---|
| atoms total | 320,839 |
| atoms live | 299,728 |
| **importance > 0** | **3,382** |
| importance 0 or NULL | 317,457 |

3,382 atoms carry real importance, and 363 of them sit pinned at exactly the
1.0 cap. Something has been accruing this whole time.

## Correction 2: there are two mechanisms, and they are not duplicates

`accrueImportance` exists twice. My first read was that this was root-5
regeneration (one helper copied into two places, one orphaned) and that a
refactor should collapse them. Wrong, and worth writing down because the names
collide hard enough to invite exactly that mistake:

- **`ambient/distiller.py:_accrueImportance(store, atomId)`** is *write-time
  reinforcement*. A near-dup or re-tail arrives, and the existing atom gets more
  important instead of a twin being written. Per-atom, at emit. **This is wired
  and working**, and it is what produced the 3,382.
- **`lifecycle/importance.py:accrueImportance(store)`** is *read-time usage
  accrual*. It drains `recall_log` and converts retrieval into importance.
  Corpus-wide batch. **This has never run in production.**

Collapsing them would have destroyed the working one to fix the broken one.

## The real gap

`recall_log` holds **345,897 rows, every single one unprocessed**
(`processed_at IS NULL`), covering 10,492 distinct atoms. Retrieval has never
once fed back into importance. That is the actual Layer 5 hole, and it is
narrower and more fixable than what I filed.

## The trap, confirmed with numbers

Draining that log naively promotes the wrong things. The eight most-retrieved
atoms in the backlog, at roughly 5,800 retrievals each, are Charon's own source
code:

| retrievals | what it is |
|---|---|
| 5,851 | a Python `__init__` doing `os.makedirs(self.PENDING_DIR` |
| 5,850 | Charon's `doc_type` → kind mapping notes |
| 5,848 | Go: `narrEvent.Context["doc_type"] = "narrative_fragment"` |
| 5,799 | `class NarrativeStitcher:` (the component silenced earlier tonight) |
| 5,791 | a Go **test fixture** asserting `meta.EmissionID != "test-uuid-123"` |

Naive accrual makes `test-uuid-123` one of the most important memories in a
store whose stated purpose is retaining a person's voice. That is the failure
mode in its clearest possible form.

## Attribution is possible, but not where I first looked

`source_ref` does **not** separate automation from usage. It names the endpoint,
not the caller:

| source_ref | rows | atoms |
|---|---|---|
| mcp.recall | 302,350 | 4,248 |
| mcp.pensive_recall | 29,725 | 6,705 |
| mcp.recall.L2 | 5,021 | 989 |
| mcp.recall.L3 | 4,710 | 1,253 |
| lean.recall.L2 | 2,160 | 58 |
| lean.recall.L3 | 1,560 | 58 |
| mcp.recall_records | 371 | 312 |

I was about to conclude the schema could not attribute usage and that Layer 5
needed a migration first. The `query` column falsifies that. Charon's stitcher
issues a structurally distinctive query, `narrative_fragment session <uuid>`,
which no human or agent would ever type. Splitting on that shape:

| class | rows | atoms | share | retrievals/atom |
|---|---|---|---|---|
| stitcher plumbing | 292,510 | 724 | **84.6%** | 404 |
| genuine recall | 53,387 | 9,870 | 15.4% | 5.4 |

The two populations do not merely differ in volume, they differ in *shape*. 404
retrievals per atom is a machine in a loop. 5.4 across 9,870 atoms is a corpus
being used. The concentration ratio is the tell, and it is 75x.

Spot-checking the genuine side, the top atoms are plausible memory: codex
orchestration campaign records, a lesson about contract holes at worker
interfaces, design-doc content. Not clean, though. One test assertion
(`self.assertIn("## Pensive Recall: AEGIS", out)`) still shows at 405
retrievals, so a source filter is necessary but not sufficient.

## What this means for the design

1. **Exclude stitcher queries from accrual.** One predicate, `query NOT LIKE
   'narrative_fragment session %'`, removes 84.6% of the backlog and the entire
   top-8 contamination. This is a filter on *plumbing*, not on an agent: Charon
   re-firing k=50 recall on an idle timer is not a memory being used.
2. **Damp by concentration, do not just sum.** Summing retrievals lets any future
   loop rediscover the same failure with a different query shape. Importance
   should be sublinear in retrieval count (log, or per-source saturation), so that
   404 hits from one caller cannot outrank a handful of deliberate ones.
3. **The 1.0 cap is already saturating.** 363 atoms sit at exactly 1.0, where
   importance has stopped discriminating among them. Draining 53k genuine rows
   into the same capped scale will widen that plateau. The cap needs to become a
   ranking that still separates at the top, or accrual will produce a tie.
4. **Filtering is not attribution.** The query-shape split works today because
   one automated caller happens to be identifiable. It is a heuristic standing in
   for a missing column. The durable fix is stamping the calling agent on
   `recall_log` at write time, the same way `provenance.agent` is already stamped
   on atoms. Until that exists, any accrual policy is one new automated caller
   away from being wrong again.

## Status

Nothing wired. This is the evidence a design has to survive, not the design.
Every number here is from the live store, read-only, and reproducible with the
queries in this session.

---

## Design returned 2026-08-13, and it found something worse than the filing

Two claims verified against source rather than taken on report.

### The job cannot run, not merely "has not run"

`lifecycle/importance.py:62` stamps the drained rows with:

```python
f"UPDATE recall_log SET processed_at = ? WHERE id IN ({placeholders})"
```

One placeholder per row. Measured against the live store:

```
  SQLITE_LIMIT_VARIABLE_NUMBER  250,000
  unprocessed backlog           345,996
  -> the job CANNOT RUN on production data
```

It raises `too many SQL variables` after roughly 2 seconds, having already
executed ~10,494 atom UPDATEs inside the open transaction which then roll back.
Wire it to a timer without touching it and you get a job that burns CPU forever,
drains nothing, and does so silently if the runner swallows the exception, which
is the normal shape for a background job.

This sharpens the finding at the top of this document. I wrote that read-time
accrual "has never run in production". The stronger and more useful statement is
that it has only ever met fixture-sized inputs, and would have failed the first
time it met real ones. A function whose tests pass and whose production input
exceeds an engine limit is not untested, it is tested against the wrong universe.

The fix is not a bigger chunk size. The stamp becomes a range predicate over a
frontier rowid snapshotted at job start: two bound values regardless of backlog
size, measured at 345,964 rows stamped in 676ms in one statement.

### Accrual must not write to `atoms.importance`, and the reason is a trigger

`store/schema.sql:104`:

```sql
CREATE TRIGGER IF NOT EXISTS atoms_au AFTER UPDATE ON atoms BEGIN
  INSERT INTO fts(fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO fts(rowid, text) VALUES (new.rowid, new.text);
END;
```

No column guard. Any update to any column re-indexes the row's full text.
Interleaved A/B, same connection, 10,000 accruals, three rounds:

| approach | wall | WAL growth |
|---|---:|---:|
| `UPDATE atoms SET importance=...` (fires the trigger) | 424 / 471 / 319 ms | 9.3 / 17.7 / 10.4 MB |
| upsert into a side table | 11.1 / 9.0 / 5.9 ms | 0.9 / 0.5 / 0.5 MB |

36 to 53x wall and 10 to 20x WAL for identical accounting.

So read-time usage belongs in its own table, which also honours the correction
at the top of this document: write-time reinforcement and read-time accrual are
different mechanisms on different scales and must not share a column. It buys
reversibility too, since accrual never mutates a canonical row and there is no
record of pre-accrual importance values to restore.

**A live consequence, filed here because it is not hypothetical.** The WIRED
distiller reinforcement pays this FTS re-index tax today, on every near-duplicate
emit. That is a current cost on the write path, not a future cost of this layer.
