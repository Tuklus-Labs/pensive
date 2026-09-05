# Risk Model: candidate-scoped facet boosts

## Axis: Invariants

- I1: The scoped boost set equals the old global boost set intersected with the
  fused candidate IDs.
- I2: Moving facet lookup after fusion does not change fused order, final order,
  trust annotations, payload text, token count, or low-confidence state at a
  fixed clock.
- I3: Facets remain a boost only. They never create candidates.
- I4: Only live atoms with canonical lowercase entity values can be boosted.

## Axis: State transitions

- S1: A candidate retired after indexing remains excluded by the canonical atom
  status read.
- S2: An empty fused pool returns the existing empty result without extracting
  entities or querying facets.

## Axis: Boundaries

- B1: Empty candidate IDs produce an empty boost set, distinct from an omitted
  candidate scope, which preserves the legacy global `facetSignal` contract.
- B2: Duplicate and unknown candidate IDs do not duplicate results or raise.
- B3: The engine's maximum pool, two 200-result signals per class plus optional
  auxiliary candidates, remains within SQLite's bound-parameter capacity.

## Axis: Malformed inputs

- M1: Candidate IDs absent from the atom or facet tables are ignored.
- M2: Invalid public recall argument types remain the MCP boundary's concern;
  this internal hint does not broaden the public API.

## Axis: Concurrency

- N/A: the lookup uses call-local immutable IDs and one SQLite read on the
  daemon's existing single serving thread. It adds no cache or shared state.

## Axis: Persistence

- P1: No schema, rows, index contents, export format, or migration changes are
  permitted.
- P2: The query must read canonical atom liveness rather than trust possibly
  stale dense-index membership.

## Axis: Integration contracts

- C1: `facetSignal(store, hints)` keeps project/time filtering and the legacy
  unscoped query behavior for callers that omit `candidateIds`.
- C2: `recall` calls the facet boost only after defensive scope filtering and
  RRF have established the complete candidate pool.
- C3: The candidate query must drive from the facets primary key's `atom_id`
  prefix. Driving from `(key,value)` recreates the 26k-row materialization this
  change exists to remove.
- C4: BM25, dense, HNSW, rerank policy, priors, trust, and payload APIs remain
  unchanged.

## Axis: Regression traps

- [x] boundary: `boundary: empty collection treated as missing collection`.
  `candidateIds=[]` must not mean unscoped.
- [x] concurrency: N/A; no cross-call state is added.
- [x] contract: `contract: optional scope silently ignored`. The signal must
  intersect matches with the supplied pool.
- [x] encoding: N/A; atom IDs and labels remain bound SQLite text values.
- [x] framework: `framework: query planner chooses a low-selectivity index`.
  The SQL must force the atom-first primary-key plan.
- [x] io: N/A; no filesystem, network, shell, or device IO is added.
- [x] persistence: `persistence: stale derived candidate remains live`. The
  canonical atoms table decides status.
- [x] resource: `resource: unbounded match materialization`. A common label may
  have tens of thousands of facets but only the bounded fused pool may be read.
- [x] state: `state: stage moved across its dependency`. Facet membership must
  run after fusion while its boost remains before priors and trust.

## Coverage Matrix

| Risk row | Test name(s) |
|---|---|
| I1, I3, I4, S1, B2, M1, P2 | `test_candidate_scope_equals_global_intersection_for_live_candidates` |
| I2, C2, C4, state trap | `test_recall_candidate_scoped_facets_match_frozen_clock_reference` |
| S2, B1, resource trap | `test_empty_fused_pool_skips_facet_work` |
| B1, C1 | `test_omitted_candidate_scope_preserves_global_facet_contract`, `test_empty_candidate_scope_skips_entity_extraction` |
| C3, framework trap | `test_candidate_scope_forces_atom_first_facet_index` |
| B3 | Existing signal breadth constants cap the pool; verified by the engine spy in the frozen-clock test and documented inspection. |
| P1 | Repository diff and schema/export suites; this change contains no store file. |
