# Loudness Audit: deterministic HNSW construction

AST inspection found 22 assertions and zero assertions without explicit
messages in `test_hnsw_determinism.py`.

Every message names the full-build, first-add, serialized-build, key-mapping,
restart determinism, query-order, snapshot, native-shape, fallback, or empty-cache
invariant. Messages include constructor arguments, call counts, graph sizes,
atom IDs, paths, or competing result lists needed to diagnose the failure.

Exemptions: none.
