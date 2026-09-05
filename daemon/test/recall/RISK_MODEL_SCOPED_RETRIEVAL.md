# Risk Model: scoped recall candidate generation

Scope must define the candidate universe before each signal chooses its top-k.
The unscoped path must retain its existing ANN and SQL behavior.

## Axis: Invariants

- `I1 scope-before-limit`: project, agent, kinds, and inclusive effective time
  constrain BM25, base dense, and auxiliary dense before top-k selection.
- `I2 intersection`: combined constraints use AND semantics.
- `I3 index parity`: Flat and HNSW scoped searches rank the same allowed vectors
  by cosine score; retired rows never return.
- `I4 embed count`: one recall embeds the query once for the base model and once
  for the optional auxiliary model, independent of active class count.
- `I5 unscoped path`: `allowedIds=None` uses the existing Flat or HNSW path and
  does not enumerate a scoped candidate set.

## Axis: State transitions

- `S1 retirement`: an allowed live key can become retired; scoped search must
  exclude the retired key without renumbering later HNSW keys.
- `S2 empty intersection`: a nonempty individual scope can become empty after
  intersection and must short-circuit before embedding.

## Axis: Boundaries

- `B1 cutoff`: a scoped target just beyond the global 200-candidate cutoff must
  still be found.
- `B2 inclusive time`: effective timestamps equal to either time boundary remain
  eligible; timestamps immediately outside do not.
- `B3 empty allowed set`: an empty set returns no dense hits without reading
  vectors.
- Existing `k <= 0`, empty-index, and zero-vector behavior remains covered by
  `test_vector_index.py` and `test_hnsw_index.py`.

## Axis: Malformed inputs

- `M1 unknown IDs`: allowed IDs absent from an index are ignored rather than
  causing key lookup errors.
- Invalid public recall argument types remain the serving boundary's concern;
  this change does not broaden the engine's validation contract.

## Axis: Concurrency

- N/A: search reads an already-built index. This patch adds no shared mutable
  state or cache; existing index rebuild and mutation synchronization is outside
  the scoped search unit.

## Axis: Persistence

- `P1 key mapping`: scoped HNSW lookup must use the stable integer keys already
  persisted in the in-memory graph and must not compact or renumber them.
- N/A for schema migration: no stored format or schema changes.

## Axis: Integration contracts

- `C1 backward call shape`: existing two-argument `search(vec, k)` callers and
  duck-typed indexes remain valid on unscoped recall.
- `C2 lexical SQL`: FTS score ordering and sanitization remain unchanged; scope
  predicates are bound parameters placed before `LIMIT`.
- `C3 kind classes`: explicit kinds narrower than a class, such as `narrative`,
  constrain lexical and dense selection before the class top-k.
- `C4 aux classes`: auxiliary indexes receive the same allowed universe and the
  auxiliary query is embedded once.
- `C5 native class scope`: a whole kind class keeps the native unscoped search
  call for that class, while indexes for classes outside the requested kind
  universe are not searched at all.

## Axis: Regression traps

- [x] `boundary: off-by-one in inclusive vs exclusive range`: effective-time
  endpoints must use inclusive `BETWEEN` semantics.
- [x] `concurrency`: N/A; no new state is shared between calls.
- [x] `contract: duck-typed signature drift`: optional scope must not be passed
  to old index implementations on the unscoped path.
- [x] `encoding`: N/A; IDs and SQL values remain bound strings and integers.
- [x] `framework: library top-k before application filter`: usearch lacks a
  metadata filter, so fixed ANN overfetch cannot establish scoped correctness.
- [x] `io`: N/A; the unit performs no filesystem, network, IPC, or device IO.
- [x] `persistence: stable external key to row mapping`: retired HNSW keys remain
  holes and later keys retain their atom mapping.
- [x] `resource: unbounded full-corpus materialization`: scoped HNSW vector reads
  must use a fixed batch size and a bounded top-k heap.
- [x] `state: stale retired entry leaks through alternate path`: scoped exact
  search must consult live usearch keys, not only the atom-id list.

## Coverage Matrix

| Risk row | Test name(s) covering it |
|----------|--------------------------|
| I1, B1, C2 | `test_project_scope_precedes_bm25_cutoff` |
| I1, B1 | `test_agent_scope_precedes_dense_cutoff` |
| I1, B2 | `test_effective_time_scope_precedes_dense_cutoff_and_is_inclusive` |
| I1, B1, C2, C3 | `test_explicit_narrative_kind_precedes_dense_cutoff`, `test_explicit_narrative_kind_precedes_bm25_cutoff` |
| I2 | `test_combined_scope_intersects_all_constraints`, `test_combined_scope_precedes_bm25_cutoff` |
| I1, C2 combined lexical predicates | `test_combined_scope_precedes_bm25_cutoff` |
| S2, B3 | `test_empty_combined_scope_skips_both_embedders` |
| I3, S1, M1, P1 | `test_scoped_flat_and_hnsw_agree_and_exclude_retired_rows` |
| I4, C1 | `test_l3_embeds_base_query_once_across_two_classes` |
| I1, I4, C4 | `test_aux_dense_scopes_before_top_k_and_embeds_once` |
| I5, C1 | `test_l2_full_class_keeps_old_duck_index_call_shape` |
| I1, I5, C5 | `test_l2_full_class_skips_out_of_scope_aux_indexes` |
