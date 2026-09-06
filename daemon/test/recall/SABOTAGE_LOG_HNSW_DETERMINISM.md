# Sabotage Log: deterministic HNSW construction

All runs used the CPU-only test runtime with GPU visibility disabled and two
OpenMP/BLAS workers. Production mutations were applied to the isolated campaign
checkout, tested, and restored. The live service and repository were untouched.

The initial red run against the previous production code failed all three new
tests. The policy tests saw omitted HNSW constructor settings, and the real
2,048-vector test found different serialized bytes from two builds.

| Test | Production mutation | Observed result | Weakened-test result |
|---|---|---|---|
| `test_full_build_pins_quality_and_serialization_parameters` | Set `_HNSW_ADD_THREADS = 0`, restoring USearch's all-core bulk construction. | Failed on `kwargs={'threads': 0}` instead of the required single worker. | Replacing the selected test body with a no-op passed against the same mutant. |
| `test_first_incremental_add_uses_the_full_build_policy` | Set `_HNSW_EXPANSION_SEARCH = 64`, restoring the dependency's narrow-search default. | Failed because first-add construction received search expansion 64 instead of 1024. | Replacing the selected test body with a no-op passed against the same mutant. |
| `test_repeated_builds_are_byte_and_query_identical` | Set `_HNSW_ADD_THREADS = 0`, restoring parallel construction. | Failed because the two real USearch graphs differed in serialized bytes. | Replacing the selected test body with a no-op passed against the same mutant. |
| `test_snapshot_cache_hit_bypasses_graph_rebuild` | Disabled snapshot publication after a cache miss. | Failed loudly because the configured cache directory was absent; weakened test passed. |
| `test_invalid_snapshot_structure_rebuilds_from_canonical_rows` | Returned a loaded native graph before count, dimension, and key validation. | Failed because the two-row graph was accepted for 256 canonical rows; weakened test passed. |
| `test_cache_failure_does_not_break_canonical_build` | Re-raised a fingerprint exception instead of treating it as a cache miss. | Failed on the synthetic cache exception; weakened test passed. |
| `test_empty_build_does_not_create_snapshot` | Removed the empty-row short circuit. | Failed on an attempted first-row read from the empty population; weakened test passed. |

The production mutations are discriminating: configuration assertions catch
silent default drift, the graph assertion catches scheduler-dependent bytes,
and cache tests distinguish verified reuse from canonical fallback.
