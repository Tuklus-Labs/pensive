# Contributing to pypensive

## Dev environment

```
git clone https://github.com/Tuklus-Labs/pensive
cd pensive
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,full]"
```

The `dev` extra installs pytest. The `full` extra installs
sentence-transformers + faiss-cpu + rank-bm25 which are needed for
the L2 / ParallelHybrid tests. Without `full` the core spreading-
activation tests still pass but the hybrid retrieval tests skip.

## Running tests

Two suites live in this repository. The library suite:

```
pytest tests/
```

The daemon under `daemon/` has its own suite, run from the repository root
with `python3 -m pytest daemon/test/ -q`; it needs the daemon's dependencies
(`daemon/requirements.txt`) and, for the model-backed tests, an importable
`sentence_transformers`. Contributions to the daemon are gated on that suite,
not this one.

A clean run takes ~80-100s and covers the spreading-activation engine,
the pickle-signing layer, the boundary-analysis diagnostics, the
parser plugins, and the concurrency-safety regressions for `_compile`.

### Concurrency stress test

`tests/test_pass8_fix6_compile_race.py::test_p8_crit1_race_does_not_starve_readers`
runs concurrent writers + readers for a wall-clock duration. The
default is 5 seconds; override with:

```
PENSIVE_CONCURRENCY_DURATION=30 pytest tests/test_pass8_fix6_compile_race.py
```

Use a longer duration when you change anything inside `_compile()` or
its callers. Reader starvation only manifests under sustained
contention, so a 5-second run can miss regressions that a 30-second
run catches.

### Pickle-loading tests need write access to `~/.config/pensive/`

The signing layer reads/writes a key at `~/.config/pensive/pickle.key`.
Tests that exercise save/load must be able to create that directory.
On systems where `$HOME` is read-only, set `XDG_CONFIG_HOME` to a
writable path before invoking pytest.

## Testing methodology

The campaign that produced the current test suite (pass-1 through
pass-9) was built around two non-negotiable rules:

### 1. Every fix has a regression test
Naming convention: `tests/test_passN_<topic>.py`. Per-test docstring
names the finding ID (`PENPY-PN-CRIT/IMP/MIN-K`) and the failure
shape. The body explicitly reconstructs the buggy interleaving
before checking the fix-in-place behavior.

### 2. Every regression test is sabotage-gated
After writing the test, deliberately revert the fix (or break the
relevant invariant) and re-run the test. The test MUST fail with a
diagnostic message that names what was broken -- not just
`AssertionError: 0 != 1`. If the test silently passes under sabotage,
the test is dead weight.

This rule is the lesson from pass-9: P8 fix-6 shipped a generation-
check guard that wasn't actually testable by the existing test
because of a redundant `_dirty=False` short-circuit. Sabotaging only
the gen-check passed all 218 tests. Pass-9 added an isolation test
that reproduces the gen-bump-without-publish interleaving and FAILS
when the gen-check is removed.

Loud-assertion examples:

```python
assert boosted[value] >= bare_score - 1e-9, (
    f"PENPY-P9-IMP-2 violated: context PENALIZED candidate "
    f"{value!r}: bare={bare_score}, with_context={boosted[value]}. "
    f"Context should only boost, never penalize."
)
```

Each new test's commit message documents the sabotage gate -- what
was changed in the code, which test failed, and what the diagnostic
looked like. Search the git log for "Sabotage gate verified:" to see
the pattern.

## Code style

- Python >= 3.10.
- No `from __future__ import annotations` -- the optional numba JIT
  inspects type annotations at runtime.
- Critical sections in `_compile()` are documented at PEP-257 length;
  do not condense them. The concurrency model is explicit because
  re-deriving it from the code alone is hard.
- New patterns added to `patterns.REAL_DATA_PATTERNS` get a unit test
  in `tests/test_mega_extract_spans.py`.

## Submitting changes

- Branch from `main`.
- One logical change per commit. If a single bugfix touches code and
  tests, group them; if it touches three unrelated bugs, split.
- Commit messages: imperative mood, < 72 char subject, body explains
  why the bug exists and how the fix avoids it.
- For regression-fix commits, include the sabotage-gate verification
  in the body.

## Security-sensitive changes

Anything that touches `pensive.ingestion.pipeline` (pickle signing,
key loading, magic header) gets extra scrutiny:

- Update `docs/SECURITY.md` if behavior or threat model changes.
- Add a regression test in `tests/test_pickle_security.py`.
- The sabotage-gate rule applies double here -- a silent-sabotage
  regression in the verification path is an RCE vector.

See also: `docs/SECURITY.md` for the current threat model.
