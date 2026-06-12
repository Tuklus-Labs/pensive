# pensive Engineering Style

Pensive (`pypensive` on PyPI) is the spreading-activation retrieval core: a sparse bipartite entity-graph that answers entity-exact queries in sub-millisecond time at 50M+ documents, and the engine the wider AEGIS Engram/memory layer leans on for recall. The library is published; downstream callers `pip install pypensive` and trust the contract. The graph is also the substrate cognitive subsystems query when they reconstruct what the system knows. That makes correctness load-bearing in ways a one-off script never is.

The failure modes that actually hurt here, grounded in this code:

- **Silent wrong results from a torn graph.** `_compile()` in `spreading.py` converts COO edge buffers to a CSR matrix under concurrent readers and writers. A naive copy or an unchecked publish (the four hazards documented at `spreading.py:338`) leaves `_adj.shape[0] < len(_idx_to_node)`: entities exist but are unreachable through the CSR, queries return plausible-but-wrong answers, and nothing raises. Recall silently rots.
- **Arbitrary code execution via pickle.** `IngestPipeline.load_graph()` unpickles graph files. The HMAC-signature gate (`ingestion/pipeline.py`) is the only thing between a hostile `.pkl` and RCE. Weaken the gate, flip the key precedence, or default `trusted=True` and you have handed every loader a code-exec primitive.
- **Ranking collapse from a bad config.** `decay >= 1.0` amplifies activation without bound across hops and the threshold filter never fires; `spec_power < 0` rewards common entities and buries the discriminative ones. `SpreadingConfig.__post_init__` rejects these for a reason.
- **Unbounded RSS in a long-lived daemon.** The graph is append-only (`add_documents()` never compacts). A consumer that ingests forever grows linearly. Documented and accepted, but a "small" change that adds a per-query cache without an eviction bound turns a known cost into an OOM.
- **A query contract people misread.** `query()` is entity-exact, not natural language. Code or docs that imply otherwise generate confident garbage and erode trust in the whole engine.

This document is the engineering standard a pensive change must meet to land. Heph reads this on SessionStart when working under `/home/aegis/Projects/pensive` and refuses to help violate hard rules without explicit override.

## Core Principles

1. **The concurrency model in `_compile()` is a spec, not an accident.** The optimistic-retry-with-generation-check publish (`spreading.py:338` onward) defends four named races and the synchronous-fallback guarantees forward progress. Treat its docstrings as the contract. Any change to build/`add_documents`/`_compile`/`_reset_graph_state` must preserve: snapshot under `_build_lock`, build CSR outside the lock, publish only if `_graph_generation` is unchanged. If you cannot explain which hazard a line defends, do not touch it.

2. **Pickle loading is a trust boundary, defended in depth.** The disk-key-first precedence in `_load_or_create_key` exists so a hostile `PENSIVE_PICKLE_KEY` cannot forge a signature. `load_graph()` verifies HMAC before `pickle.loads`. Unsigned loads require an explicit, loud `trusted=True`. No change relaxes this without a matching test in `tests/test_pickle_security.py`.

3. **Entity-exact is the contract; say so everywhere.** `query()`, `query_with_doc_ids()`, and `query_analyzed()` match extracted entity surface forms, not questions. Every public docstring and README example that touches query already spells this out. Keep it spelled out. Never add an example that feeds a natural-language sentence to `query()` and pretends it works.

4. **Config guards stay strict.** `SpreadingConfig.__post_init__` (`spreading.py:153`) rejects values with documented failure modes. New config fields that affect ranking get the same treatment: validate at construction, document the failure mode in the raise message, do not silently clamp.

5. **Memory growth is structural, so make it visible, not surprising.** The append-only graph and the bounded substring cache (`_SUBSTR_CACHE_MAX`, `spreading.py:639`) are deliberate. Any new cache or buffer that grows with corpus size needs an explicit bound or an explicit "this grows forever and here is why" note matching the README's longevity section. No silent unbounded growth.

6. **The numba fast path and the numpy fallback must agree.** `_spread_bipartite` ships a JIT kernel and a numpy scatter-max fallback that must produce the same ranking. When numba is absent the library still has to be correct, not merely runnable. Changes to one path require the other to match and a test that exercises both.

7. **Degrade loudly enough to debug.** The hybrid layer (`parallel_hybrid.py`) catches broadly and falls back so a flaky L2 backend does not kill retrieval. That is the right call for a daily-ops engine, but every catch logs with context. A catch that swallows and returns empty without a log is a latent silent failure (see Anti-patterns).

8. **Audit-pass tags are load-bearing breadcrumbs.** `PENPY-*` tags in source map to `tests/test_passN_*.py` (later passes use numbered `PENPY-P5-*`..`PENPY-P9-*`, earlier passes use unnumbered `PENPY-CRIT/IMP/MIN/DEP-*`). When you fix a concurrency, save/load, or ranking bug, find the matching pass test, extend it, and tag the fix. The history of why a guard exists is in those tags; do not erase it.

## Hard Rules

### Error handling
- No bare `except:`. `except Exception` is allowed only at a degradation boundary and only with a logged reason. The hybrid handlers at `parallel_hybrid.py:368`, `:393`, `:436`, `:471` log via `logger.warning` before returning a fallback. Last verified 2026-06-05.
- A broad catch that returns an empty result MUST log first. `parallel_hybrid.py:368` (`logger.warning("SA query failed: %s", e); return [], None`) is the pattern. A silent `return []` after `except` is a review blocker.
- `except BaseException` is reserved for build-rollback only: `spreading.py:696` and `spreading.py:893` catch it to reset graph state so a half-built graph never serves queries, then re-raise. Do not use `BaseException` to swallow; it must re-`raise`. Last verified 2026-06-05.
- `try/except ... pass` is allowed only for genuinely optional cleanup with a comment saying so (e.g. best-effort `delattr` at `spreading.py:558`, `:817`; the stale-tmp `unlink` cleanup at `ingestion/pipeline.py:104`). Never use it to hide a real failure path. Last verified 2026-06-05.
- Wrap-and-translate at boundaries: `from_save_data` / `load_graph` translate low-level unpickle/parse errors into `ValueError` so callers catch a uniform type. Keep that convention.

### Types and structure
- Public methods are fully type-annotated. Return tuples are typed (`List[Tuple[str, float]]`, `List[Tuple[str, str, float]]`). New public surface matches.
- `Dict[str, Any]` is tolerated for the L2/hybrid/bench result dicts that mirror external library shapes (`l2.py:37,100,186`, `parallel_hybrid.py:347,379,399,441,455`, `hybrid_search.py:40,126`, `boundary_bench.py` throughout). It is NOT tolerated for new internal data that you control: define a `@dataclass` (see `SearchResult`, `BoundaryAnalysis`, `SpreadingConfig`). Prefer a typed record over an untyped dict for anything that crosses a function boundary inside the package.
- Untyped object params are allowed only behind a documented duck-typed protocol. `context_bridge.py:79` takes `l1_cache` untyped but documents "anything matching `L1CacheLike`" and a `runtime_checkable` Protocol exists in the module. New cross-component seams get a Protocol, not a bare untyped param. Last verified 2026-06-05.
- Node-type constants (`_ENTITY_TYPE = 0`, `_VALUE_TYPE = 1`) are mirrored in `spreading.py:41` and `boundary.py:19`. If you change the encoding, change both; they are a contract across modules.

### Naming and module size
- `spreading.py` is 1719 lines and is the one acceptable large module: it is a single cohesive engine with a deliberate concurrency story that does not split cleanly. New code does NOT pile on. A genuinely separable concern (a new persistence format, a new spread strategy) gets its own module, as boundary analysis already did.
- Internal helpers stay `_`-prefixed. Public API is exactly what `__init__.py:__all__` exports. Adding to the public surface means adding to `__all__` and documenting it.

### Persistence and security
- Never `pickle.loads` without a verified signature unless the caller passed an explicit `trusted=True` and got a loud `warnings.warn(..., RuntimeWarning)` about it (`ingestion/pipeline.py:265`). Last verified 2026-06-05.
- The HMAC key precedence is disk file, then env var, then generated-and-persisted-at-0600. Do not reorder it. Reordering to env-first reintroduces the forged-signature attack the comment at `ingestion/pipeline.py:53` describes. Last verified 2026-06-05.
- `get_save_data()` emits `format: 'sparse_v1'`. A new format gets a new tag and `from_save_data` keeps loading the old ones. Never break load-compat for graphs already on disk (including legacy networkx pickles).

### Testing
- Run `pytest tests/` before claiming green. 233 tests collect; a clean run is ~80-100s. Verify the tally from output, never from memory.
- Concurrency, save/load, and ranking changes require a test, named for and tagged to the audit pass it belongs to (`test_passN_*`). A fix without a regression test does not land.
- Both spread paths get exercised. A change that only the numba path or only the numpy fallback covers is half-tested.
- L2/hybrid tests skip without `[full]` installed; that is expected, but do not let a core test silently depend on `[full]` deps.

### Dependencies and versions
- Dependency floors in `pyproject.toml` are lower-bound-only and track what the suite has run against (the rationale is in the file comments). Do NOT add upper caps. Do NOT bump or "fix" a floor that looks old: this is an AMD/ROCm workstation with an intentionally current stack (Python 3.14, ROCm 7.x), and a version that looks stale to you is almost certainly correct. Leave it.
- numba is optional and must stay optional. The library imports it behind `try/except ImportError` (`spreading.py:34`) and ships a fallback. Never make numba a hard dependency.
- Core install is numpy + scipy only. New core-path code does not reach for an `[full]`-only dependency.

## Anti-patterns we will not ship

- **Silent-swallow catch.** `except Exception: return []` with no log. Masks real failures as "no results". Name it in review.
- **Optimistic publish without the generation check.** Any rewrite of `_compile()` that publishes a CSR built outside the lock without re-checking `_graph_generation` reintroduces the publish-outside-lock race (`PENPY-P8-CRIT-1`). Hard reject.
- **Env-first pickle key.** Reordering `_load_or_create_key` to consult `PENSIVE_PICKLE_KEY` before the disk file. Reopens the forged-signature path.
- **`trusted=True` as a default or a convenience.** Defaulting unsigned unpickle to trusted, or adding a flag that does it implicitly.
- **NL-query example.** Documentation or a test that feeds `query()` a natural-language sentence and asserts a non-empty result as if NL parsing existed.
- **Clamping a bad config instead of rejecting it.** Silently coercing `decay >= 1.0` or negative `spec_power` to a "safe" value instead of raising. Hides operator error.
- **Unbounded per-query cache.** A memoization layer keyed on query text or corpus size with no eviction bound. The substring cache shows the correct shape (`_SUBSTR_CACHE_MAX`).
- **Untyped internal dict as a return type** for data this package owns, where a dataclass would do.

## Enforcement

- **Heph on SessionStart.** Heph loads this file when working under `/home/aegis/Projects/pensive` and will refuse to help weaken a Hard Rule (pickle gate, concurrency publish, config guards) without an explicit, logged override.
- **`pytest tests/`** is the gate. Green-by-memory is not green; read the tally.
- **`tools/check_wheel_version.py`** guards the stale-wheel release trap (RELEASE.md); run it before any build claims to ship 0.2.0.
- **Audit-pass discipline.** The `PENPY-*` tag (numbered `PENPY-PN-*` from pass 5 on) plus matching `test_passN_*` pairing is the project's own CI-of-record for hardening. New guards join it.
- **`aegis-audit src/pensive`** and **`aegis-async-audit`** for a broader code-intelligence pass when touching the hybrid or concurrency paths.

## Operational SLAs

This is a published library and a recall-critical engine, so the budgets sit above a daily-ops tool but below a court-grade artifact:

- **Query latency:** sub-millisecond on the bipartite fast path at the scales in the README table (~1ms at 1M docs, ~0.45ms at 50M). A change that regresses single-query latency on the bipartite path is a regression to justify, not absorb.
- **Build:** parallel build stays within the README envelope (~12s at 1M docs, ~28min at 50M). `build_parallel` falls back to serial below 2000 docs; keep that threshold honest.
- **Memory:** the scipy.sparse CSR representation is the ~90%-over-networkx win and is non-negotiable. Append-only growth is ~50 bytes/node plus edge buffers; any new structure that grows with the corpus declares its bound.
- **Longevity:** a pure-query workload holds flat RSS indefinitely. A long-lived ingesting daemon calls `compact()` periodically (in-place round-trip under the build lock; results identical pre/post, covered by `tests/test_compact.py`). The manual re-serialize-and-rebuild pattern still works but is no longer required.
- **Install:** `pip install pypensive` pulls numpy + scipy only and must import and serve queries with neither numba nor `[full]` present. `[full]` is additive, never required for the core engine.
- **Observability:** the library logs via the stdlib `logging` module under module-named loggers; it does not `print` outside the `boundary_bench.py` CLI report path (`:472` onward) and the `pensive` CLI. Keep library code silent except through `logger`.

## On scope and humility

The bar moves up, never down. Every audit pass (P1 through P9) tightened this engine and left a tagged test behind; that is the direction. When a change makes the concurrency story simpler-looking but you cannot map each removed line to the hazard it defended, you are not simplifying, you are removing a guard whose cost someone already paid. When in doubt about the pickle gate or the publish race: do the safe thing, write the test, and leave the next person a tag explaining why. Being slower and correct beats being clever and silently wrong, because here "silently wrong" means corrupted recall in a memory system that is supposed to remember.

Last revised: 2026-06-05. Owner: Gary + Heph.
