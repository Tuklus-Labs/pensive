# Index runtime integration risks

| Axis | Risk and proof |
| --- | --- |
| Invariants | Cache settings reach every full/class rebuild. Exact indexes cannot grow past their inclusive limit. |
| Transitions | Adding the first row beyond the exact limit promotes/compacts that class, preserving all live IDs. |
| Boundaries | BLAS default, override and zero opt-out; absent/empty cache setting. |
| Malformed input | Invalid BLAS setting falls back to the measured default. Cache errors remain derived-cache misses. |
| Concurrency | BLAS controls affect this daemon process only; no system/user environment edits. Native cache atomicity is covered in the cache helper suite. |
| Persistence | Default snapshots stay beside this store; explicit opt-out keeps the old uncached factory contract. |
| Integration | Real Store/Flat/HNSW hot-growth test; injected factories inspect full and class-only rebuild calls. |
| Regression traps | boundary: inclusive limit; contract: optional kwargs; state: first over-limit write; resource: thread budget and physical retired rows; io/persistence: cache helper tests; encoding/framework/concurrency: no new wire/async behavior. |

Coverage: test_blas_budget, test_bootstrap_cache_settings,
test_reindex_forwards_cache, test_hot_growth_promotes_exact_index.

Optional writer failure: `test_optional_snapshot_failure_keeps_built_index` first
raised the injected RuntimeError, then passed after adding the final optional
cache boundary. The built canonical index remained searchable.

Unused-model startup: `test_startup_does_not_preload_unused_reranker` observed
an unwanted load before the fix and proves default lazy/explicit preload paths.

Actual isolated source mutations removed the BLAS cap and cache forwarding,
shifted the growth boundary by one, forced unused preload, bypassed checksum
validation, removed a valid retention slot, and ignored publication priority.
All seven failed their targeted tests. Evidence: index-runtime-mutations.json.
All new direct assertions have rule-naming messages.
