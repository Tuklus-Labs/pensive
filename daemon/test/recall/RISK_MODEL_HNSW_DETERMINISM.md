# Risk Model: deterministic HNSW construction

## Axis: Invariants

- I1: Every HNSW graph uses explicit connectivity 16, insertion expansion 128,
  and search expansion 1024.
- I2: Bulk and incremental insertion use one USearch worker so insertion order
  cannot vary with the host scheduler.
- I3: Identical ordered keys and vectors build byte-identical graphs and return
  identical results.
- I4: Search keeps cosine similarity, descending order, and the existing atom-ID
  mapping.
- I5: A cached graph is accepted only when its fingerprint and native count,
  dimension, and exact key range match the canonical rows.

## Axis: State transitions

- S1: An empty `HnswIndex` creates its first graph with the same construction
  parameters used by a full rebuild.
- S2: Add, remove, and re-add keep the existing key-to-atom-ID mapping and
  retirement semantics.
- S3: A cache miss builds and publishes a snapshot. A cache hit loads without
  rebuilding. Cache failures fall back to canonical construction.

## Axis: Boundaries

- B1: Empty indexes and non-positive `k` continue to return no results.
- B2: A one-vector incremental index is searchable even though serialized
  construction normally starts from a matrix.
- B3: Scoped search remains exact over the allowed IDs and is unaffected by the
  unscoped graph-search expansion.
- B4: Empty stores remain uncached empty indexes.

## Axis: Malformed inputs

- M1: Missing, unsafe, corrupt, or wrong-native-shape cache entries are misses;
  they cannot replace canonical construction or break startup.

## Axis: Concurrency

- C1: Bulk construction must pass `threads=1`; omitting it restores USearch's
  scheduler-dependent `threads=0` default.
- C2: Incremental insertion must also pass `threads=1`, so a future batched
  binding behavior cannot silently restore parallel insertion.

## Axis: Persistence

- P1: A restart rebuild from unchanged canonical rows must reproduce the same
  derived graph and query results.
- P2: An explicitly configured cache may persist a derived native snapshot.
  SQLite rows remain canonical, and any mismatch rebuilds from them.

## Axis: Integration contracts

- X1: The USearch 2.25 API accepts explicit `connectivity`, `expansion_add`,
  `expansion_search`, and per-add `threads` parameters.
- X2: Existing remove, scoped search, score, and candidate-universe tests remain
  green under the new graph parameters.
- X3: The wider search must be assessed against exact retrieval and latency;
  passing quality floors alone is insufficient evidence of preservation.
- X4: `HnswIndex.build`, `selectIndex`, and `buildClassIndexes` accept an optional
  cache directory without changing existing callers.

## Axis: Regression traps

- [x] boundary: `boundary: empty collection treated as missing collection`.
  Empty and first-add paths must not instantiate different graph policies.
- [x] concurrency: `concurrency: ordering assumption without enforced
  serialization`. Ordered SQL rows do not imply ordered graph insertion when
  USearch uses all cores.
- [x] contract: `contract: dependency default silently changes behavior`.
  Every quality-affecting USearch construction setting must be explicit.
- [x] encoding: N/A; vectors remain float32 and atom IDs retain their integer-key
  mapping.
- [x] framework: `framework: thread-count zero means automatic rather than
  disabled`. USearch documents zero as all available cores.
- [x] io: `io: cached native file is corrupt or incomplete`. File-content and
  native-structure checks must run before a graph is accepted.
- [x] persistence: `persistence: restart reconstructs different derived state`.
  The in-memory graph must be reproducible from canonical rows.
- [x] resource: `resource: correctness fix creates unbounded startup cost`.
  Serial construction must be measured on the 283k-vector code class.
- [x] state: `state: alternate initialization path uses stale defaults`. The
  first incremental add must share the full-build constructor.

## Coverage Matrix

| Risk row | Test or evidence |
|---|---|
| I1, I2, C1, X1, contract/framework traps | `test_full_build_pins_quality_and_serialization_parameters` |
| I1, I2, S1, B2, C2, state trap | `test_first_incremental_add_uses_the_full_build_policy` |
| I3, P1, concurrency/persistence traps | `test_repeated_builds_are_byte_and_query_identical` |
| I4, S2, B1, X2 | Existing `test_hnsw_index.py` and `test_incremental_index.py` focused suites |
| B3, X2 | Existing `test_scoped_retrieval.py` HNSW cases |
| I5, S3, B4, M1, P2, X4, IO trap | `test_snapshot_cache_hit_bypasses_graph_rebuild`, `test_invalid_snapshot_structure_rebuilds_from_canonical_rows`, `test_cache_failure_does_not_break_canonical_build`, `test_empty_build_does_not_create_snapshot` |
| X3, resource trap | Isolated real-store startup, exact-reference, latency, and 65-probe evidence |
