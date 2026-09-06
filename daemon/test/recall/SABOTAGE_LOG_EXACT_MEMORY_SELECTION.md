# Sabotage Log: bounded exact memory selection

All mutations ran in the isolated campaign checkout with the CPU-only runtime,
GPU visibility disabled, and two OpenMP/BLAS workers. Each production mutation
was restored after its test. Replacing the selected test body with a no-op made
the same mutant pass, confirming that the shipped assertions caused the failure.

| Test | Production mutation | Observation |
|---|---|---|
| `test_complete_memory_class_switches_to_hnsw_above_exact_ceiling` | Changed the ceiling comparison from `<=` to `<`. | Failed because 32,000 selected HNSW; weakened test passed. |
| `test_exact_memory_policy_does_not_broaden_to_other_scopes` | Removed the complete-memory kind-set equality check. | Failed for atom-only, mixed, and code scopes because they selected Flat; the weakened atom-only case passed. |
| `test_build_class_indexes_routes_memory_to_flat_and_code_to_hnsw` | Set `EXACT_MEMORY_FLAT_MAX` to zero. | Failed because measured-size memory selected HNSW; weakened test passed. |
| `test_selected_exact_memory_index_preserves_scope_add_and_remove` | Set `EXACT_MEMORY_FLAT_MAX` to zero. | Failed before its scope and mutation checks because the selector returned HNSW; weakened test passed. |
| `test_hnsw_factory_routes_optional_cache_directory` | Dropped `cacheDir` from the selector's HNSW build call. | Failed because the build spy received `None`; weakened test passed. |
| `test_exact_search_limit_matches_selector_boundaries` | Returned the general HNSW limit for every scope. | Failed because complete memory exposed 5,000 instead of 32,000; weakened test passed. |
| `test_flat_length_counts_retired_physical_rows` | Subtracted retired positions from `FlatIndex.__len__`. | Failed because retirement incorrectly reduced physical length from three to two; weakened test passed. |

The initial red run against the old selector also failed the boundary, class
routing, and selected-index contract tests while the non-broadening cases passed.
