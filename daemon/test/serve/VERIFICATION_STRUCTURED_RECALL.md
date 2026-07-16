# Structured Recall Verification

Pre-commit evidence, 2026-07-15:

| Gate | Result |
|------|--------|
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve -q` | 105 passed in 14.97s |
| `PYTHONPATH=daemon/src python3 -m pytest daemon/test/recall daemon/test/store -q` | 199 passed, 1 skipped in 9.87s |
| `python3 -m compileall -q daemon/src/serve daemon/src/recall daemon/src/store` | exit 0 |
| `git diff --check` | exit 0 |

The skipped test and CPython `swigvarlink` deprecation warning predate this change.
