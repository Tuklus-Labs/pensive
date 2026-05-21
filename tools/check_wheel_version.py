#!/usr/bin/env python3
"""Guard against publishing a wheel whose embedded __version__ does not
match pyproject.toml.

Pass-5 (PENPY-P5-IMP-1) found dist/pypensive-0.2.0-py3-none-any.whl
shipping with __version__ = '0.1.1' under a 0.2.0 filename, and also
missing the boundary.py / boundary_bench.py modules added since 0.1.1.
A fresh consumer would pip install pypensive==0.2.0 and get a module
without the names listed in __init__'s __all__.

This script:

  1. Reads the project version from pyproject.toml.
  2. For every wheel found in dist/, extracts the version from the
     wheel filename and from the wheel's contained
     src/pensive/__init__.py or pensive/__init__.py.
  3. Fails loudly (non-zero exit, message on stderr) if any of the
     three values disagree.

Intended as a release-readiness check: invoke after `python -m build`
and before any `twine upload` or git tag.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


def _project_version(pyproject: Path) -> str:
    text = pyproject.read_text(encoding='utf-8')
    # Match e.g. version = "0.2.0" under [project].
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find version = ... in {pyproject}")
    return match.group(1)


def _wheel_filename_version(wheel: Path) -> str:
    # e.g. pypensive-0.2.0-py3-none-any.whl -> 0.2.0
    name = wheel.name
    parts = name.split('-')
    if len(parts) < 2:
        raise SystemExit(f"unexpected wheel filename layout: {name}")
    return parts[1]


def _wheel_init_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as zf:
        candidates = [
            n for n in zf.namelist()
            if n.endswith('__init__.py') and 'pensive/' in n.replace('\\', '/')
        ]
        # Prefer the top-level package __init__.py, not a subpackage one.
        top = [n for n in candidates
               if n.replace('\\', '/').rstrip('/').endswith('pensive/__init__.py')]
        if not top:
            raise SystemExit(
                f"could not find pensive/__init__.py in wheel {wheel.name}"
            )
        with zf.open(top[0]) as fh:
            text = fh.read().decode('utf-8')
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise SystemExit(
            f"no __version__ assignment found in {wheel.name}/{top[0]}"
        )
    return match.group(1)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    pyproject = repo / 'pyproject.toml'
    dist = repo / 'dist'

    project_ver = _project_version(pyproject)
    wheels = sorted(dist.glob('*.whl'))
    if not wheels:
        print(f"no wheels in {dist}; nothing to check.")
        return 0

    failed = False
    for wheel in wheels:
        fname_ver = _wheel_filename_version(wheel)
        init_ver = _wheel_init_version(wheel)
        ok = (project_ver == fname_ver == init_ver)
        status = 'OK' if ok else 'MISMATCH'
        print(
            f"[{status}] {wheel.name}: "
            f"pyproject={project_ver} filename={fname_ver} "
            f"__init__={init_ver}"
        )
        if not ok:
            failed = True

    if failed:
        print(
            "\nERROR: wheel version mismatch -- rebuild from clean HEAD "
            "before publishing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
