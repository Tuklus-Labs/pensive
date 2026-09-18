# pensive Refactoring Targets

> Historical record, written 2026-06-12. It describes the project as it stood then and is kept
> for provenance. Current behavior is documented in [README.md](README.md) and
> [daemon/README.md](daemon/README.md).

Real, citable targets found by reading the source. Pensive is a hardened, 9-audit-pass engine, so most of these are duplication and ergonomics, not correctness bugs. Ordered by payoff. None of them should touch the `_compile()` concurrency contract without the care STYLE.md demands.

**Status (2026-06-12): all 7 landed** on `refactor/penpy-backlog`, full suite 245 passed in 96s.

1. `0cdf619` -- `_ingest_extracted` extracted from the three build loops
2. `bd161e1` -- four spread-and-collect tails unified
3. `6513088` -- `HybridRow` dataclass replaces internal `Dict[str, Any]` rows
4. `d9fdef0` -- degradation-boundary warnings carry `exc_info` under DEBUG
5. `429be8d` -- `compact()` shipped with 10 tests; README/STYLE.md updated
6. `5760e3f` -- CLI report layer split into `boundary_bench_cli.py`
7. `3e49d24` -- `l1_cache` annotated with the `L1CacheLike` Protocol

Line references below are as of 2026-06-05 and have drifted with the fixes; the sections are kept for the rationale record.

### 1. Collapse the three near-identical build loops

- **Location:** `src/pensive/spreading.py` -- `_build_locked` (~701-762), `add_documents` (~768-864), `_build_parallel_locked` (~901-968).
- **Problem:** The per-document graph-construction body (answer node `v:<id>`, the `seen_in_doc` dedup, the `freq = max(..., 1)` clamp, `specificity = 1.0 / (freq ** spec_power)`, `_get_or_add_node`, `_index_entity_node` on new, edge add) is copy-pasted three times with only the edge-add helper differing (`_add_edge_fast` in build, `_add_edge` in add_documents). Three copies means a guard fix (the defense-in-depth freq clamp comment is duplicated verbatim) has to land in three places, and they can silently drift.
- **Proposal:** Extract a private `_ingest_extracted(self, extracted, *, fast_edges: bool)` that runs the shared body and takes the edge-add helper via the flag. The three callers keep their distinct setup (reset, lock, parallel fan-out, generation bump) and delegate the inner loop. No behavior change; one place to fix the clamp and the specificity formula.
- **Est:** ~25k tokens.
- **Priority:** high.

### 2. Unify the four spread-and-collect tail implementations

- **Location:** `src/pensive/spreading.py` -- `_spread_and_collect_bipartite` (~1206), `_spread_and_collect_bipartite_with_ids` (~1261), `_collect_from_array` (~1315), `_collect_from_array_with_ids` (~1351).
- **Problem:** All four share the same tail: `top_k <= 0` guard (with the identical `PENPY-IMP-6` argpartition-degeneracy comment), `flatnonzero(result >= threshold)`, value-node filter, `argpartition` vs `argsort` top-k selection, descending sort. The only real difference is whether the output row is `(label, score)` or `(doc_id, label, score)`. Four copies of a subtle numpy selection (where the `np.argpartition(scores, -0)` bug already bit once) is four places to get it wrong again.
- **Proposal:** Extract `_topk_value_indices(self, result, top_k) -> (nz_global, ordered_scores)` holding the guard + threshold + filter + selection logic once, returning the ordered global indices and scores. The four public-ish collectors become thin row formatters over that. Keep the numba combined-kernel path where it already exists; only the post-kernel selection tail unifies.
- **Est:** ~30k tokens.
- **Priority:** high.

### 3. Type the L2 / hybrid result rows instead of passing `Dict[str, Any]`

- **Location:** `src/pensive/parallel_hybrid.py` (`:347,379,399,441,455,456`), `src/pensive/l2.py` (`:100,186`).
- **Problem:** Internal results flow as `List[Dict[str, Any]]` with string keys (`doc_id`, `score`, `rank`, `source`, `summary`). The keys are an implicit, unchecked schema; a typo in a key name fails silently at runtime, and `_normalize_l2_results` has to defensively reshape arbitrary external rows. STYLE.md tolerates `Dict[str, Any]` only at the external-library mirror seam, not for data this package owns end to end.
- **Proposal:** Define an `@dataclass HybridRow` (mirror the existing `SearchResult`/`L2`-result dataclasses) for the rows that live entirely inside pensive between `_query_l2`, the candidate filter, the agreement boost, and the final formatter. Keep `Dict[str, Any]` only at the boundary where raw faiss/sentence-transformers output enters, normalize once into the dataclass there.
- **Est:** ~35k tokens.
- **Priority:** med.

### 4. Make broad-catch fallbacks distinguishable from real bugs

- **Location:** `src/pensive/parallel_hybrid.py:368` (`SA query failed`), `:393`, `:436` (`Candidate L2 query failed`), `:471`.
- **Problem:** Each catch logs at `warning` and returns an empty/degraded result, which is the right shape for a daily-ops engine. But all four log a one-line message and discard the traceback, so an unexpected `KeyError`/`AttributeError` from a refactor is indistinguishable in the logs from an expected backend hiccup. Debugging a real regression here means it looks like a flaky L2 rather than a code bug.
- **Proposal:** Switch these to `logger.warning("...: %s", e, exc_info=logger.isEnabledFor(logging.DEBUG))` (or unconditional `exc_info=True` at warning) so the traceback is recoverable when debugging, and consider narrowing the caught type where the failure surface is known (e.g. the cross-encoder load at `:154` is genuinely "model unavailable", but the SA query path at `:368` should not be swallowing arbitrary internal errors).
- **Est:** ~12k tokens.
- **Priority:** med.

### 5. Implement the promised `compact()` path

- **Location:** roadmap referenced in `README.md` ("Long-lived processes" section) and `CLAUDE.md`; the append-only buffers live in `src/pensive/spreading.py`.
- **Problem:** The README documents the structural-RSS-growth behavior and tells long-lived ingesting daemons to "periodically serialize with `get_save_data()`, discard the instance, and rebuild" as the workaround, and says "a dedicated `compact()` method is on the roadmap". The roadmap item is unbuilt, so every long-lived consumer hand-rolls the serialize-discard-rebuild dance.
- **Proposal:** Add `SpreadingActivation.compact()` that does the documented round-trip internally: `get_save_data()` -> `from_save_data()` -> swap the new instance's buffers into `self` (or return a fresh compacted instance), defragmenting the Python-side `_idx_to_node`/`_entity_index`/`_token_index` structures. Cover it with a test that asserts query results are identical pre/post compact and that node/edge counts are preserved. Update the README to point at the method instead of the manual recipe.
- **Est:** ~40k tokens.
- **Priority:** med.

### 6. Split the boundary benchmark CLI out of the library module

- **Location:** `src/pensive/boundary_bench.py` (580 lines; `print_boundary_benchmark` at `:465`, `print_boundary_candidate_mining` at `:506`, `print(...)` reporting from `:472`).
- **Problem:** `boundary_bench.py` mixes two concerns: the reusable benchmark/scoring logic (`run_boundary_benchmark`, `summarize_reports`, the case builders) and a `print`-based CLI report. STYLE.md says library code stays silent except through `logging`; the `print` block is the one library-shipped exception and it lives next to importable scoring functions, so importing the scorer pulls in CLI presentation.
- **Proposal:** Keep the pure scoring/summarization functions in `boundary_bench.py`, move the `print_*` reporters and any `__main__` entry into a thin `tools/` or `boundary_bench_cli.py` that imports them. The research benchmark stays runnable; the importable surface stops carrying presentation code.
- **Est:** ~18k tokens.
- **Priority:** low.

### 7. Give `context_bridge` consumers a typed seam (minor nit)

- **Location:** `src/pensive/context_bridge.py:79` -- `__init__(self, l1_cache, ...)`.
- **Problem:** `l1_cache` is an untyped parameter documented as "anything matching `L1CacheLike`". The module already defines a `runtime_checkable` Protocol, so the type information exists but is not applied to the param, and a wrong object only fails when a method is missing at call time.
- **Proposal:** Annotate the param with the existing `L1CacheLike` Protocol so type-checkers and readers see the contract at the constructor. Pure annotation, no runtime change.
- **Est:** ~5k tokens.
- **Priority:** low.
