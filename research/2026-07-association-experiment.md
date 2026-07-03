# Association signal experiment (Pensive v3, Task 21)

Date: 2026-07-03
Decision: **ADOPT, conditional on a serving-path facet-degree cache (Task 21b).**
Branch: feat/pensive-v3. Harness: `daemon/eval/assoc_experiment.py` (eval only, no serving-path change).

## Question

Does a 2-hop, specificity-damped association walk, seeded by the fused
top-k, add retrieval quality as a fourth fusion signal? The plan's stated
target was the "what else was in flight around X" query class and v3's
known weak spot: rank-1 precision, where BM25 still beat v3 on the
chat-export gate (R@1 0.358 vs 0.329).

## Method

Both gate corpora, each scored with and without the assoc signal fused,
same corpus / queries / seeds / other-signal weights / recall depth per
arm. Metrics from the same `gate.py` code path for both arms.

- Chat-export gate: 97,779 chunk docs, 1,500 real human-turn queries,
  seed 42. This is the decision board (the same sample BM25's published
  0.583/0.432 was measured on).
- Atom-corpus proxy: 17,528 docs, 500 project-sibling queries. Topical
  proxy only (no labeled relevance), reported for domain color.
- Walk: seeds are the top-20 baseline-fused atom ids. Neighbors derive
  from shared `entity`/`tag` facets (atom, facet, atom), the facet value
  is the through-node, contribution damped by `1 / degree^SPEC_POWER`
  with `SPEC_POWER=1.0` and `degree` = count of live atoms carrying that
  facet. Facets with degree > `HUB_CAP=128` are skipped (a shared
  "python" binds nothing after damping; a shared rare entity binds
  strongly). Stored edge rows (supersedes) also participate as direct
  hops. Deterministic, live-only emission.

## Results

Chat-export gate (the decision):

| metric | v3 baseline | v3 + assoc | delta |
|--------|-------------|------------|-------|
| R@1    | 0.3293      | 0.3453     | +0.0160 |
| R@5    | 0.5947      | 0.6100     | +0.0153 |
| R@10   | 0.6800      | 0.6900     | +0.0100 |
| MRR@10 | 0.4426      | 0.4578     | +0.0152 |
| p50 latency | 344 ms | 372 ms     | +28 ms |

Atom-corpus proxy:

| metric | v3 baseline | v3 + assoc | delta |
|--------|-------------|------------|-------|
| R@1    | 0.6680      | 0.6740     | +0.0060 |
| R@5    | 0.8040      | 0.8060     | +0.0020 |
| R@10   | 0.8420      | 0.8340     | -0.0080 |
| MRR@10 | 0.7257      | 0.7316     | +0.0059 |

Derived graph: chat 44,738 seeds with neighbors / 2.10M facet-co-occurrence
pairs; atoms 5,998 / 150k. Seed-k = 20, HUB_CAP = 128.

The temporal-neighborhood subset (the plan's headline target) is
UNAVAILABLE: the gate query builders emit no metadata identifying
"what-else-was-in-flight" queries, so that subset cannot be sliced from
the current corpora. The runner reports it as unavailable rather than
inventing a number. Measuring it needs an annotated query set, which is a
follow-up.

## Reading

The signal helps the chat gate where v3 was weak. R@1 rises 0.016, closing
most of the gap to BM25's 0.358, and MRR@10 rises to 0.4578, which clears
the 0.45 cutover bar the baseline missed by 0.0074. The atom corpus is
flat (v3 already dominates its home turf; one metric drifts negative
inside noise). So the win is concentrated on the exact corpus and metric
the experiment set out to improve.

## Two invalid results were found and discarded before this one

Honesty about the path here, because the first two gate runs looked
clean and were wrong:

1. First run reported exact zero delta on every metric. Diagnosis: the
   walk consumed edge rows, but the v3 store materializes facet
   relationships as facets, not edges (only supersede writes edge rows),
   so both eval stores had zero edges and the walk was inert. The unit
   tests passed because their fixtures inserted edge rows by hand. Fix:
   derive adjacency from facet co-occurrence, and add an inertness guard
   that refuses to emit a report when both arms are byte-identical.
2. Second run measured valid lift but p50 blew to 2665 ms. Diagnosis:
   HUB_CAP skipped broad facets only after a per-query COUNT(DISTINCT)
   over the whole corpus. Fix: precompute facet degrees once, bound the
   per-seed lookup. Retrieval output pinned identical (parity test);
   latency only.

The numbers in the Results table are from the third run, after both fixes.

## Decision and the condition

Adopt the signal. It earns its place on measured lift where v3 needed it.

The serving wire-in is a SEPARATE gated task (Task 21b), not part of
Task 21, because the harness's +28 ms overhead does not translate to
serving as-is. The harness precomputes the facet-degree map once against a
static store. The live store mutates continuously (the distiller adds
atoms), so a once-per-run precompute is the wrong pattern in production:
without a maintained degree source, per-query recall would reintroduce the
O(corpus) scan and blow the 150 ms p95 budget. Task 21b owns:

- a facet-degree cache maintained against the live store (a lifecycle job
  or an on-facet-write counter, not a per-query scan),
- bounded neighbor sets (the HUB_CAP LIMIT already proven here),
- a tuned fusion weight for the fourth signal,
- p95 re-verified under the 150 ms budget with the signal live.

Nothing is lost by deferring the wire-in. The facet back-edges already
persist and back Tier-2 history; the harness and its guards remain in the
tree to re-measure once the cache exists.

## Constants (for Task 21b)

`ASSOC_SEED_K=20`, `SPEC_POWER=1.0`, `HUB_CAP=128`, facet keys
`("entity","tag")`, max-weight logical-edge dedupe. These are eval-tuned
starting points; the serving harness owns final tuning against the p95
budget.
