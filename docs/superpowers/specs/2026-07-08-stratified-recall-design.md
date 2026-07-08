# Stratified, intent-fair recall (Layer 1)

Date: 2026-07-08
Status: approved, pre-implementation
Scope: `daemon/` recall path only. No stored-data mutation.

## Problem

The v3 store now holds two populations that were never meant to compete on equal
footing in a single ranking:

| kind | count | role |
|------|-------|------|
| `document_chunk` | 299,347 | bulk-imported source-code corpus |
| `atom` | 14,207 | reasoning / feedback / discovery memory |
| `narrative` | 352 | session narratives |
| `snapshot` | 4 | |

Recall generates candidates as a single kind-blind top-200 per signal
(`bm25` and `dense`, signals.py). With a 21:1 chunk:memory ratio, a generic
query fills those 200 slots with code chunks, so memory atoms frequently never
enter the pipeline at all. The `kinds` filter (`_filterKinds`, engine.py) runs
*after* fusion, so it can only whittle down what already made the pool; it cannot
pull back memory that was crowded out at the door.

Observed directly: an unscoped `"active projects, ongoing work"` recall returned
only code chunks (relevance 0.55 to 0.78); the actual planning/decision memory
was absent. A `kinds:[atom,narrative]` recall returned two results, because two
is all the memory that survived candidate generation, not all the memory that
exists.

This defeats both intended uses of recall at once:
- "What am I working on / what did we decide" (wants memory) returns code.
- "Have we solved X before" (wants the exact code chunk) is not well served
  either, because the chunk that ranks is whatever won the lexical flood, not
  what the query is actually about.

## Root cause

The cross-encoder reranker (rerank.py) is already capable of sorting a
memory-shaped query toward memory and a code-shaped query toward code: its
scores are query-relative and comparable across kinds. It never gets the chance,
because memory is starved at two choke points:

1. **Candidate generation.** One pooled top-200 per signal, dominated by the
   majority population.
2. **Rerank head.** Only the top `RERANK_HEAD` (64) fused candidates are scored
   by the cross-encoder (engine.py). Even if a memory atom sneaks into the pool
   at fused rank 90, it is never reranked, so it cannot win on relevance.

An intent classifier is not needed. The reranker is the intent mechanism. The
fix is to stop starving one population before the reranker sees it.

## Design

Two kind-classes, defined once and structured so more can be added:

```
MEMORY = {atom, narrative, snapshot}
CODE   = {document_chunk}
```

Changes, in pipeline order. Everything not listed is unchanged.

### 1. Per-class candidate generation

- **bm25** (signals.py): run the sanitized FTS `MATCH` once per class, each with
  `AND a.kind IN (...)`, top-`_SIGNAL_K` (200) each. Merge the per-class lists
  into one ranked list by their real `bm25()` score (scores are comparable across
  classes because it is the same scoring function over the same FTS index). The
  never-throws guarantee and token cap are untouched.

- **dense** (vector_index.py, mcp.py `ServeContext`): partition the vector index
  by class. `selectIndex` gains a kind-class argument; `FlatIndex.build` /
  `HnswIndex.build` add `AND a.kind IN (...)` to their embedding-load query; the
  `HNSW_THRESHOLD` count is taken per class. Consequence, for free: the MEMORY
  class (~14.5k live embedded vectors) sits below the 200k threshold and gets an
  exact `FlatIndex` (an upgrade from today's approximate HNSW for those atoms),
  while CODE (~299k) keeps `HnswIndex`. Search each class index for top-`_SIGNAL_K`.

`ServeContext` holds a mapping `{className: VectorIndex}` instead of a single
`self.index`, rebuilt by `reindex()` exactly as today (embed-missing then rebuild,
once per mutation). `recall()` takes the mapping in place of the single `index`
argument; the three call sites (mcp.py:316, mcp.py:385, shadow.py:160) pass it.

Net effect: both classes are guaranteed present in the candidate pool, each
correctly ranked within its own class.

### 2. Fusion, facet boost, priors

Unchanged. RRF over the now-balanced `[bmHits, dnHits]`, the multiplicative facet
boost, and the single-application priors/timeScope stage all operate exactly as
today, now on a pool that contains both classes.

### 3. Stratified rerank head (decisive change)

Build the `RERANK_HEAD`-slot rerank input by **round-robin across the per-class
fused rankings**: best MEMORY, best CODE, second MEMORY, second CODE, and so on,
backfilling from the other class when one runs dry, until the head is full. This
guarantees the cross-encoder scores a fair mix of both classes. Because
cross-encoder scores are query-relative and comparable, the reranked order is
then pure relevance, and the correct population wins by merit:

- memory-shaped query, memory reranks to the top;
- code-shaped query, the relevant chunk reranks to the top.

The untouched fused tail below the head keeps its RRF order, as today.

### 4. Trust, trim, payload

Unchanged. The trust layer, `k`-trim, and tiered payload assembler see the
reranked list exactly as before.

### 5. Explicit `kinds` still honored

If a caller passes `kinds`, recall scopes to those kinds as today. Stratification
runs across only the requested classes: a single-class request degenerates to no
round-robin; a subset stratifies across that subset. The existing
`_filterKinds` short-circuit to the low-confidence sentinel is preserved.

## Tunable constants

In the spirit of the existing `FACET_BOOST` and `RERANK_HEAD` (named, documented,
harness-owned):

- the class definition map;
- per-class `_SIGNAL_K`;
- the rerank-head interleave ratio (default 1:1 round-robin).

These are the knobs the eval tunes. They are not to be guessed.

## Validation

- Run `eval/gate.py` against `BASELINE_V3` before and after; the change must not
  regress the existing gate.
- Add two probe queries to the eval set:
  - a memory-intent query (for example "what am I working on"), asserting a
    memory-kind atom ranks first;
  - a code-intent query naming a specific known solution, asserting the correct
    `document_chunk` ranks first.
- Tune the interleave ratio against these, not by intuition.

## Cost

- One extra small in-memory index (MEMORY class, ~14.5k vectors): negligible
  build time and memory.
- One extra bm25 query per recall (two class queries instead of one pooled).
- Rerank cost unchanged: still `RERANK_HEAD` (64) pairs scored per query.

## Explicitly out of scope (Layer 2 / Layer 3)

Deferred by decision until Layer 1 proves the retrieval shape in real usage:

- **Provenance repair (Layer 2).** Chunk `source_ref` is currently a re-import
  artifact (`kv_cache/vector_meta.db#rowid=N`), not a real path, and ~20k chunks
  have a null project. Making a code hit actionable ("this is
  `pensive/src/spreading.py::_compile`") needs a re-ingest that recovers
  file/symbol provenance.
- **Association enrichment (Layer 3).** The edges table holds one edge total, so
  a code chunk cannot yet be joined to the memory atoms explaining it. Attaching
  "why, and the lessons learned" to a code hit (via shared project + entity
  facets, i.e. spreading activation) is the payload-layer follow-on that depends
  on Layer 2.

Layer 1 mutates no stored data. It is entirely on the recall path.
