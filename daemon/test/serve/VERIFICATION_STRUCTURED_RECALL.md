# Structured Recall Verification

Pre-commit evidence, 2026-07-15:

| Gate | Result |
|------|--------|
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve -q` | 105 passed in 14.97s |
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/recall daemon/test/store -q` | 199 passed, 1 skipped in 9.87s |
| `python3 -m compileall -q daemon/src/serve daemon/src/recall daemon/src/store` | exit 0 |
| `git diff --check` | exit 0 |

The skipped test and CPython `swigvarlink` deprecation warning predate this change.

Aggregate wire-cap follow-up evidence, 2026-07-15:

| Gate | Result |
|------|--------|
| Focused cap/wrapper selection | 6 passed, 41 deselected in 0.72s |
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve -q` | 111 passed in 16.87s |
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/recall daemon/test/store -q` | 199 passed, 1 skipped in 10.89s |
| `python3 -m compileall -q daemon/src/serve daemon/src/recall daemon/src/store` | exit 0 |
| `git diff --check` | exit 0 |
| Unslop locator over every changed file | no new candidates; pre-existing MCP section dividers kept |

The cap, UTF-8 accounting, atomic tail-stop, and shared-wrapper production mutations each failed their named regression and were restored. Inverting one assertion in each new test produced 6 failures; restoration produced 6 passes.
