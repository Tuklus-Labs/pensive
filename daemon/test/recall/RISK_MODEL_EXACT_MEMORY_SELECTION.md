# Risk Model: bounded exact memory selection

## Axis: Invariants

- I1: The complete memory kind class uses `FlatIndex` only while its embedded
  live population is at or below 32,000 vectors.
- I2: A memory class above the ceiling uses `HnswIndex`.
- I3: Unscoped, mixed-kind, narrow memory-subset, and code-class calls retain the
  existing count-based selector behavior.
- I4: The chosen index loads the same live, model-keyed, kind-scoped candidate
  universe as before.
- I5: The transition limit is exposed to incremental maintenance, and Flat
  length counts physical rows even after retirement.

## Axis: State transitions

- S1: Crossing from 32,000 to 32,001 embedded live memory rows changes the
  selected implementation from Flat to HNSW.
- S2: A Flat memory index remains valid after incremental add and remove.
- S3: Retiring a Flat row does not reduce its physical footprint; a rebuild is
  what compacts retired rows before continued growth.

## Axis: Boundaries

- B1: Exactly 32,000 memory vectors select Flat.
- B2: Exactly 32,001 memory vectors select HNSW.
- B3: Populations below the general HNSW threshold continue to select Flat for
  every scope.
- B4: Empty memory classes continue to return a usable empty Flat index.

## Axis: Malformed inputs

- N/A: `kinds` remains an internal iterable of canonical kind strings; this
  change introduces no public input or parser.

## Axis: Concurrency

- N/A: Selection performs the existing SQLite count and constructs one
  request-local index during serialized daemon startup.

## Axis: Persistence

- P1: Selection depends only on canonical embedded-live row count and kind scope,
  so an unchanged store reconstructs the same implementation after restart.
- P2: No schema, database row, export format, or serialized sidecar changes.

## Axis: Integration contracts

- X1: `buildClassIndexes` selects exact Flat memory and deterministic HNSW code
  on the measured 21k/283k store shape.
- X2: `selectIndex(store, modelId, kinds=None)` preserves its unscoped behavior.
- X3: The Flat memory result supports allowed-ID search and incremental add/remove
  through the unchanged `VectorIndex` interface.
- X4: The 32k ceiling is named and documented so future growth cannot silently
  turn an unbounded exact scan into the serving path.
- X5: The optional HNSW cache directory passes through unscoped and class factory
  calls while exact Flat memory ignores it.
- X6: `exactSearchLimit(kinds)` returns the maximum inclusive Flat population:
  32,000 for complete memory and 4,999 for every other scope.

## Axis: Regression traps

- [x] boundary: `boundary: wrong comparison at exact limit`. The policy is Flat
  through 32,000 and HNSW starting at 32,001.
- [x] concurrency: N/A; no new shared mutable state or parallel work is added.
- [x] contract: `contract: special case broadens to unintended callers`. Exact
  selection applies only to the complete memory kind set.
- [x] encoding: N/A; vector dtype and atom-ID/key encoding do not change.
- [x] framework: N/A; no framework dispatch or dependency default is involved.
- [x] io: N/A; the selector uses the existing SQLite reads and adds no IO path.
- [x] persistence: `persistence: derived implementation changes after restart`.
  The decision must derive from stable canonical count and scope.
- [x] resource: `resource: exact scan grows without bound`. The named 32k ceiling
  forces HNSW after the measured operating range.
- [x] state: `state: wrong implementation after threshold crossing`. Both sides
  of the transition require direct tests.

## Coverage Matrix

| Risk row | Test or evidence |
|---|---|
| I1, I2, S1, B1, B2, X4, boundary/resource/state traps | `test_complete_memory_class_switches_to_hnsw_above_exact_ceiling` |
| I3, X2, contract trap | `test_exact_memory_policy_does_not_broaden_to_other_scopes` |
| I3, X1 | `test_build_class_indexes_routes_memory_to_flat_and_code_to_hnsw` |
| I4, S2, X3 | `test_selected_exact_memory_index_preserves_scope_add_and_remove` |
| X5 | `test_hnsw_factory_routes_optional_cache_directory` |
| I5, S3, X6 | `test_exact_search_limit_matches_selector_boundaries`, `test_flat_length_counts_retired_physical_rows` |
| B3, B4 | Existing selector tests in `test_hnsw_index.py` and `test_vector_index.py` |
| P1 | Boundary and class-routing tests use only count and kind scope. |
| P2 | Repository diff inspection; no store source is changed. |
