# Release Checklist

This file documents the steps required to cut a PyPI release of
`pypensive`. The intent is to prevent two release-readiness traps that
have already happened:

* **Stale wheel** -- the previous `dist/` carried a 0.2.0-filenamed
  wheel whose `__init__.py` reported `__version__ = '0.1.1'` and which
  was missing modules added since 0.1.1
  (`boundary.py`, `boundary_bench.py`). Source: pass-5 audit
  (PENPY-P5-IMP-1).
* **Bundled runtime artifact** -- the sdist embedded
  `vector_meta.db` (0 bytes). Source: pass-5 audit (PENPY-P5-MIN-1).

## Pre-release

1. Confirm `pyproject.toml` `version` matches `src/pensive/__init__.py`
   `__version__`. Both move together; a mismatch is the same class of
   bug as the PENPY-P5-IMP-1 stale-wheel issue.
2. From a clean working tree (`git status` reports nothing modified),
   purge stale build artifacts:
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```
   Stale wheels are how 0.1.1 contents shipped under a 0.2.0 filename
   in the first place. **Do not** rely on hatch overwriting an
   existing wheel.
3. Run the full test suite:
   ```bash
   pytest tests/ --ignore=tests/test_performance.py
   ```
   Must be green. Concurrency tests (`test_pass5_p5_concurrency.py`)
   run for >=10s by default; set `PENSIVE_CONCURRENCY_DURATION=30` for
   a longer release-gate sweep.

## Build

```bash
python -m build
```

This produces a fresh sdist + wheel in `dist/`. The
`[tool.hatch.build.targets.sdist]` `exclude` list in `pyproject.toml`
drops `*.db`, `*.faiss`, `*.pkl` so runtime artifacts (e.g.
`vector_meta.db` left over from local development) do not end up in the
tarball.

## Verify

After `python -m build`, run the wheel-version guard:

```bash
python tools/check_wheel_version.py
```

This script asserts that every wheel in `dist/` reports the same
version in its filename, in its embedded `pensive/__init__.py`
`__version__`, and in `pyproject.toml`. A mismatch means the wheel was
built from stale source -- do not upload it.

Sanity-check the sdist contents:

```bash
tar -tzf dist/pypensive-*.tar.gz | grep -E '\.(db|faiss|pkl)$' && {
  echo "ERROR: sdist contains runtime artifacts" >&2
  exit 1
}
```

If the grep matches anything, the `pyproject.toml` sdist `exclude`
rules are not catching all classes of artifact for the current
working-tree state -- audit and update before publishing.

## Publish

Only after the verify step passes:

```bash
twine upload dist/pypensive-<version>*
```

Then tag the release in git and push the tag.

## Rule of thumb

Always rebuild `dist/` from clean HEAD before any release. Don't ship
stale wheels. If the wheel-version guard or the sdist sanity check
complains, fix the root cause before publishing -- not the check.
