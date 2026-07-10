# Corpus Repair Batch (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the v3 store's chunk provenance: recover 3,909 kv_cache rowid refs from the retired vector_meta.db, supersede ~16k duplicate reference-library chunks, and backfill derivable null projects: deterministically, idempotently, and with zero deletions.

**Architecture:** A pure helper library (`repair_lib.py`) + a pass-per-problem repair tool (`repair_corpus.py`) in `daemon/tools/`. Every pass takes a store and returns a report dict; the CLI runs dry-run by default and requires `--apply` (which performs a SQLite-API backup first) to mutate. Supersession uses the store's existing atomic `supersede()`; nothing is ever deleted.

**Tech Stack:** Python 3.10+ stdlib + the daemon's own `store.store` API. Tests with pytest over tmp_path stores.

## Global Constraints

- Work in `~/Projects/pensive/daemon`. Run tests from there: `python3 -m pytest test/tools/ -v` (conftest puts `daemon/src` on sys.path; `daemon/tools` needs `sys.path` handling shown in Task 1).
- Branch is `feat/corpus-enrichment` (checked out). Do not create branches/worktrees.
- No em dashes anywhere in code, comments, docstrings, or commit messages.
- ZERO deletions of atoms, ever. Duplicate cleanup is `supersede()` only. The tool must not contain the string `DELETE FROM atoms` in any form.
- Never guess a path: a summary/ref that does not parse is counted in the report and left untouched.
- All passes idempotent: a second run over an already-repaired store reports zero changes.
- Do NOT touch the live store at `~/.local/share/pensive-v3/pensive.db` in any test. Tests use tmp_path stores exclusively. Running the tool against the live store is an operational step the CONTROLLER performs, not any subagent.
- Every git commit message ends with these two trailer lines verbatim:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
  ```

**Store facts the tasks rely on** (verified 2026-07-09):
- Ref roots in the store: `projects/<name>/...` (root `~/Projects/`), `reference-library/...` (root `~/Projects/Aegis/AEGIS/docs/reference-library/`), `kv_cache/vector_meta.db#rowid=N` (broken), `claude-home/...` (`~/.claude/`), `codex-home/...` (`~/.codex/`).
- Retired db: `~/Projects/Aegis/AEGIS/Pensive/kv_cache/vector_meta.db`, table `meta(rowid, summary, ...)`; 3,894 of its rows have summaries like `[files] file /home/aegis/Projects/mission-control/dashboard_metrics.go Created dashboard_metrics.go`; the other rows are `[claude] ...` reasoning text (no path; report, don't guess).
- Duplicate reflib: 16,362 chunks under `reference-library/<f>#cN` (project NULL) vs 16,393 under `projects/Aegis/AEGIS/docs/reference-library/<f>#cN` (project `Aegis`).
- `supersede(store, oldId, newId, provInput)` marks old superseded + writes edge and provenance atomically. `provInput = {"source": ..., "agent"?: ..., "sessionId"?: ..., "sourceRef"?: ...}`.

---

### Task 1: Pure repair helpers (`repair_lib.py`)

**Files:**
- Create: `tools/repair_lib.py` (under `daemon/`)
- Test: `test/tools/test_repair_lib.py` (create dir with empty `__init__.py` if the other test dirs have one; mirror whatever `test/recall/` does)

**Interfaces:**
- Produces:
  - `parseFilesSummary(summary) -> str | None` (absolute path out of a `[files] file <path> ...` summary)
  - `abspathToRef(abspath) -> tuple[str, str | None] | None` (`(ref, project)`; None when the path maps to no known root)
  - `refToProject(ref) -> str | None` (project derivable from a ref, for backfill)

- [ ] **Step 1: Write the failing tests**

Create `test/tools/test_repair_lib.py`:

```python
"""Pure helpers for the Phase A corpus repair (parsing, ref mapping)."""
import sys
from pathlib import Path

# daemon/tools is not a package on sys.path by default; add it the same way
# the tools themselves add daemon/src (parents: tools -> test -> daemon).
_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from repair_lib import parseFilesSummary, abspathToRef, refToProject


def test_parseFilesSummary_extracts_path():
    s = ("[files] file /home/aegis/Projects/mission-control/dashboard_metrics.go "
         "Created dashboard_metrics.go")
    assert parseFilesSummary(s) == \
        "/home/aegis/Projects/mission-control/dashboard_metrics.go"


def test_parseFilesSummary_rejects_non_files_summaries():
    assert parseFilesSummary("[claude] on GPU-context-creation something") is None
    assert parseFilesSummary("[claude] [Narrative: pensive] blah") is None
    assert parseFilesSummary("") is None
    assert parseFilesSummary(None) is None


def test_parseFilesSummary_rejects_relative_path():
    assert parseFilesSummary("[files] file not/an/abs/path Created x") is None


def test_abspathToRef_projects_root():
    ref, project = abspathToRef(
        "/home/aegis/Projects/mission-control/dashboard_metrics.go")
    assert ref == "projects/mission-control/dashboard_metrics.go"
    assert project == "mission-control"


def test_abspathToRef_claude_and_codex_home():
    assert abspathToRef("/home/aegis/.claude/hooks/emit.py") == \
        ("claude-home/hooks/emit.py", None)
    assert abspathToRef("/home/aegis/.codex/config.toml") == \
        ("codex-home/config.toml", None)


def test_abspathToRef_unknown_root_returns_none():
    assert abspathToRef("/etc/passwd") is None
    assert abspathToRef("/home/aegis/Downloads/x.bin") is None


def test_refToProject_variants():
    assert refToProject("projects/obol/internal/api/x.go#c11") == "obol"
    assert refToProject("reference-library/53-hardware.md#c2") == "Aegis"
    assert refToProject("claude-home/hooks/x.py") is None
    assert refToProject("kv_cache/vector_meta.db#rowid=7") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/tools/test_repair_lib.py -v`
Expected: FAIL / collection error, `ModuleNotFoundError: No module named 'repair_lib'`.

- [ ] **Step 3: Write the implementation**

Create `tools/repair_lib.py` (under `daemon/`):

```python
"""Pure helpers for the Phase A corpus repair: summary parsing and ref mapping.

The retired kv_cache store's ``meta.summary`` embeds the original absolute path
for file-derived rows (``[files] file /abs/path <action> ...``); everything else
in that table is reasoning text with no path. These helpers turn that summary
into the store's ref convention and derive project attribution from paths and
refs. Anything that does not parse returns None: the repair tool reports those
rows, it never guesses.

Ref convention (matches the existing import waves):
  /home/aegis/Projects/<name>/<rest>  ->  projects/<name>/<rest>, project <name>
  /home/aegis/.claude/<rest>          ->  claude-home/<rest>, no project
  /home/aegis/.codex/<rest>           ->  codex-home/<rest>, no project
Reference-library refs map to project Aegis (the library lives inside the Aegis
repo at AEGIS/docs/reference-library/).
"""
import re

__all__ = ["parseFilesSummary", "abspathToRef", "refToProject"]

# "[files] file <abspath> <rest>" -- the path is the token after "file ".
# Paths with embedded spaces do not occur in this corpus; a path token that
# does not start with "/" is rejected rather than guessed at.
_FILES_RE = re.compile(r"^\[files\] file (\S+)")

# (prefix, refRoot, projectSegment) -- projectSegment True means the first
# path segment under the prefix is the project name.
_ROOTS = (
    ("/home/aegis/Projects/", "projects/", True),
    ("/home/aegis/.claude/", "claude-home/", False),
    ("/home/aegis/.codex/", "codex-home/", False),
)


def parseFilesSummary(summary):
    """Absolute path out of a ``[files] file <path> ...`` summary, or None."""
    if not summary:
        return None
    m = _FILES_RE.match(summary)
    if m is None:
        return None
    path = m.group(1)
    if not path.startswith("/"):
        return None
    return path


def abspathToRef(abspath):
    """Map an absolute path to ``(ref, project)`` or None for unknown roots."""
    for prefix, refRoot, hasProject in _ROOTS:
        if abspath.startswith(prefix):
            rest = abspath[len(prefix):]
            project = rest.split("/", 1)[0] if hasProject and "/" in rest else (
                rest if hasProject else None)
            # A bare filename directly under Projects/ has no project dir.
            if hasProject and "/" not in rest:
                project = None
            return refRoot + rest, project
    return None


def refToProject(ref):
    """Project derivable from a ref, for null-project backfill; else None.

    ``projects/<name>/...`` yields the name; ``reference-library/...`` yields
    ``Aegis`` (the library lives inside the Aegis repo). Dotfile roots and the
    broken kv_cache refs yield None: dotfiles are not projects, and a kv_cache
    ref carries no path information.
    """
    if ref.startswith("projects/"):
        rest = ref[len("projects/"):]
        if "/" in rest:
            return rest.split("/", 1)[0]
        return None
    if ref.startswith("reference-library/"):
        return "Aegis"
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/tools/test_repair_lib.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/repair_lib.py test/tools/
git commit -m "$(cat <<'EOF'
feat(tools): pure helpers for Phase A corpus repair

parseFilesSummary / abspathToRef / refToProject: retired-store summary
parsing and the ref-root mapping, None over guessing throughout.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 2: kv_cache ref recovery pass

**Files:**
- Create: `tools/repair_passes.py` (under `daemon/`)
- Test: `test/tools/test_repair_passes.py`

**Interfaces:**
- Consumes: Task 1's helpers; `store.store.openStore/putAtom`.
- Produces: `repairKvCacheRefs(store, oldDbPath) -> dict` with keys
  `{"rewritten": int, "projectBackfilled": int, "noPathInSummary": int, "rowidMissing": int}`.

- [ ] **Step 1: Write the failing tests**

Create `test/tools/test_repair_passes.py`:

```python
"""Store-mutation passes for the Phase A corpus repair."""
import sqlite3
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from repair_passes import repairKvCacheRefs
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _putChunk(store, text, sourceRef, project=None):
    return putAtom(store, {
        "text": text, "kind": "document_chunk", "project": project,
        "importance": 0.0,
        "provenance": {"source": "bulk-import", "sourceRef": sourceRef},
    })


def _makeOldDb(tmp_path, rows):
    """rows: list of (rowid, summary). Returns the db path."""
    p = tmp_path / "vector_meta.db"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE meta (rowid INTEGER PRIMARY KEY, summary TEXT)")
    conn.executemany("INSERT INTO meta(rowid, summary) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return p


def test_repairs_ref_and_backfills_project(store, tmp_path):
    aid = _putChunk(store, "chunk body", "kv_cache/vector_meta.db#rowid=7")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created rate.go"),
    ])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 1
    assert report["projectBackfilled"] == 1
    ref = store._conn.execute(
        "SELECT source_ref FROM provenance WHERE atom_id=?", (aid,)
    ).fetchone()[0]
    assert ref == "projects/obol/api/rate.go"
    project = store._conn.execute(
        "SELECT project FROM atoms WHERE id=?", (aid,)).fetchone()[0]
    assert project == "obol"


def test_non_files_summary_reported_not_guessed(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=3")
    old = _makeOldDb(tmp_path, [(3, "[claude] on GPU things, no path here")])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 0
    assert report["noPathInSummary"] == 1
    ref = store._conn.execute(
        "SELECT source_ref FROM provenance LIMIT 1").fetchone()[0]
    assert ref == "kv_cache/vector_meta.db#rowid=3"  # untouched


def test_missing_rowid_reported(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=999")
    old = _makeOldDb(tmp_path, [(1, "[files] file /home/aegis/Projects/x/y.go z")])
    report = repairKvCacheRefs(store, old)
    assert report["rowidMissing"] == 1
    assert report["rewritten"] == 0


def test_existing_project_not_overwritten(store, tmp_path):
    aid = _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=7",
                    project="keep-me")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created"),
    ])
    report = repairKvCacheRefs(store, old)
    assert report["rewritten"] == 1
    assert report["projectBackfilled"] == 0
    assert store._conn.execute(
        "SELECT project FROM atoms WHERE id=?", (aid,)).fetchone()[0] == "keep-me"


def test_idempotent_second_run_is_noop(store, tmp_path):
    _putChunk(store, "body", "kv_cache/vector_meta.db#rowid=7")
    old = _makeOldDb(tmp_path, [
        (7, "[files] file /home/aegis/Projects/obol/api/rate.go Created"),
    ])
    repairKvCacheRefs(store, old)
    second = repairKvCacheRefs(store, old)
    assert second == {"rewritten": 0, "projectBackfilled": 0,
                      "noPathInSummary": 0, "rowidMissing": 0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/tools/test_repair_passes.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'repair_passes'`.

- [ ] **Step 3: Write the implementation**

Create `tools/repair_passes.py`:

```python
"""Store-mutation passes for the Phase A corpus repair.

Each pass takes an open store (and whatever inputs it needs), mutates in one
commit, and returns a flat report dict of counts. Passes are idempotent: the
row predicates that select work exclude already-repaired rows, so a second run
reports zeros. Nothing here deletes an atom; duplicate cleanup (Task 3) uses
the store's supersede().

Daemon-internal reach-through convention: like the recall modules, passes read
and write through store._conn for batched work the public API does not expose.
"""
import re
import sqlite3
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_SRC = _TOOLS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from repair_lib import parseFilesSummary, abspathToRef  # noqa: E402

__all__ = ["repairKvCacheRefs"]

_ROWID_RE = re.compile(r"^kv_cache/vector_meta\.db#rowid=(\d+)$")


def repairKvCacheRefs(store, oldDbPath):
    """Rewrite kv_cache rowid refs from the retired store's summaries.

    For every provenance row on a live-or-superseded document_chunk whose
    source_ref matches ``kv_cache/vector_meta.db#rowid=N``: look up rowid N in
    the retired db, parse the absolute path out of its summary, and rewrite the
    ref to the standard root convention. Backfill atoms.project from the path
    only when project is NULL. Unparseable summaries and missing rowids are
    counted and left untouched (never guess). Idempotent: rewritten refs no
    longer match the predicate.
    """
    report = {"rewritten": 0, "projectBackfilled": 0,
              "noPathInSummary": 0, "rowidMissing": 0}

    rows = store._conn.execute(
        "SELECT p.id, p.atom_id, p.source_ref FROM provenance p "
        "JOIN atoms a ON a.id = p.atom_id "
        "WHERE a.kind = 'document_chunk' AND p.source_ref LIKE 'kv_cache%'"
    ).fetchall()
    if not rows:
        return report

    old = sqlite3.connect(f"file:{oldDbPath}?mode=ro", uri=True)
    try:
        summaryOf = {}
        for provId, atomId, ref in rows:
            m = _ROWID_RE.match(ref)
            if m is None:
                report["noPathInSummary"] += 1
                continue
            rowid = int(m.group(1))
            if rowid not in summaryOf:
                got = old.execute(
                    "SELECT summary FROM meta WHERE rowid = ?", (rowid,)
                ).fetchone()
                summaryOf[rowid] = got[0] if got else None
                if got is None:
                    summaryOf[rowid] = False  # sentinel: rowid absent
            summary = summaryOf[rowid]
            if summary is False:
                report["rowidMissing"] += 1
                continue
            path = parseFilesSummary(summary)
            if path is None:
                report["noPathInSummary"] += 1
                continue
            mapped = abspathToRef(path)
            if mapped is None:
                report["noPathInSummary"] += 1
                continue
            newRef, project = mapped
            store._conn.execute(
                "UPDATE provenance SET source_ref = ? WHERE id = ?",
                (newRef, provId))
            report["rewritten"] += 1
            if project is not None:
                cur = store._conn.execute(
                    "UPDATE atoms SET project = ? "
                    "WHERE id = ? AND project IS NULL",
                    (project, atomId))
                report["projectBackfilled"] += cur.rowcount
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise
    finally:
        old.close()
    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/tools/test_repair_passes.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/repair_passes.py test/tools/test_repair_passes.py
git commit -m "$(cat <<'EOF'
feat(tools): kv_cache ref recovery pass

Rewrites rowid refs from the retired vector_meta.db summaries via the
Task 1 helpers; backfills project only when NULL; unparseable rows are
reported and untouched. Idempotent by predicate.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 3: Reference-library dedup pass

**Files:**
- Modify: `tools/repair_passes.py` (add `dedupReferenceLibrary`)
- Test: `test/tools/test_repair_passes.py` (add tests)

**Interfaces:**
- Consumes: `store.store.supersede`.
- Produces: `dedupReferenceLibrary(store, sessionId=None) -> dict` with keys
  `{"superseded": int, "noTwin": int, "textMismatch": int, "multipleTwins": int}`.

- [ ] **Step 1: Write the failing tests**

Add to `test/tools/test_repair_passes.py`:

```python
from repair_passes import dedupReferenceLibrary


_CANON_ROOT = "projects/Aegis/AEGIS/docs/reference-library/"


def _putPair(store, fname, cN, text, driftedText=None):
    """A reflib duplicate pair: null-project copy + canonical Aegis copy."""
    dup = _putChunk(store, text, f"reference-library/{fname}#c{cN}")
    canon = _putChunk(store, driftedText if driftedText is not None else text,
                      f"{_CANON_ROOT}{fname}#c{cN}", project="Aegis")
    return dup, canon


def test_dedup_supersedes_null_copy_keeps_canonical(store):
    dup, canon = _putPair(store, "53-hw.md", 2, "identical body text")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 1
    statuses = dict(store._conn.execute(
        "SELECT id, status FROM atoms WHERE kind='document_chunk'").fetchall())
    assert statuses[dup] == "superseded"
    assert statuses[canon] == "live"
    # A supersedes edge canon -> dup exists (new -> old convention).
    edge = store._conn.execute(
        "SELECT src_atom, dst_atom FROM edges WHERE type='supersedes'"
    ).fetchone()
    assert edge == (canon, dup)


def test_text_mismatch_left_live_and_reported(store):
    dup, canon = _putPair(store, "99-drift.md", 0, "old text", "revised text")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["textMismatch"] == 1
    statuses = {r[1] for r in store._conn.execute(
        "SELECT id, status FROM atoms").fetchall()}
    assert statuses == {"live"}


def test_no_twin_reported(store):
    _putChunk(store, "orphan body", "reference-library/only-here.md#c0")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["noTwin"] == 1


def test_multiple_twins_reported_not_guessed(store):
    _putChunk(store, "same", "reference-library/multi.md#c1")
    _putChunk(store, "same", f"{_CANON_ROOT}multi.md#c1", project="Aegis")
    _putChunk(store, "same", f"{_CANON_ROOT}multi.md#c1", project="Aegis")
    report = dedupReferenceLibrary(store)
    assert report["superseded"] == 0
    assert report["multipleTwins"] == 1


def test_dedup_idempotent(store):
    _putPair(store, "53-hw.md", 2, "identical body text")
    dedupReferenceLibrary(store)
    second = dedupReferenceLibrary(store)
    assert second == {"superseded": 0, "noTwin": 0,
                      "textMismatch": 0, "multipleTwins": 0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/tools/test_repair_passes.py -v -k dedup`
Expected: FAIL, `ImportError: cannot import name 'dedupReferenceLibrary'`.

- [ ] **Step 3: Write the implementation**

Add to `tools/repair_passes.py` (import `supersede` from `store.store` at the top with the existing imports):

```python
from store.store import supersede  # noqa: E402

_REFLIB_DUP_ROOT = "reference-library/"
_REFLIB_CANON_ROOT = "projects/Aegis/AEGIS/docs/reference-library/"


def dedupReferenceLibrary(store, sessionId=None):
    """Supersede null-root reflib chunks that duplicate the canonical copies.

    For each LIVE document_chunk whose ref is ``reference-library/<tail>``:
    find LIVE canonical twins at ``projects/Aegis/.../reference-library/<tail>``.
    Exactly one twin with EXACTLY matching text: supersede the null-root copy
    (survivor = the canonical copy, which carries project attribution). Zero
    twins, multiple twins, or text drift: count and leave live; a forced merge
    of drifted content would silently lose the difference. Idempotent: the
    LIVE predicate excludes already-superseded copies.
    """
    report = {"superseded": 0, "noTwin": 0, "textMismatch": 0,
              "multipleTwins": 0}
    dups = store._conn.execute(
        "SELECT a.id, p.source_ref, a.text FROM atoms a "
        "JOIN provenance p ON p.atom_id = a.id "
        "WHERE a.kind = 'document_chunk' AND a.status = 'live' "
        "AND p.source_ref LIKE ?",
        (_REFLIB_DUP_ROOT + "%",)
    ).fetchall()
    for dupId, ref, text in dups:
        tail = ref[len(_REFLIB_DUP_ROOT):]
        twins = store._conn.execute(
            "SELECT a.id, a.text FROM atoms a "
            "JOIN provenance p ON p.atom_id = a.id "
            "WHERE a.kind = 'document_chunk' AND a.status = 'live' "
            "AND p.source_ref = ?",
            (_REFLIB_CANON_ROOT + tail,)
        ).fetchall()
        if not twins:
            report["noTwin"] += 1
            continue
        if len(twins) > 1:
            report["multipleTwins"] += 1
            continue
        twinId, twinText = twins[0]
        if twinText != text:
            report["textMismatch"] += 1
            continue
        prov = {"source": "repair-tool", "sourceRef": ref}
        if sessionId is not None:
            prov["sessionId"] = sessionId
        supersede(store, dupId, twinId, prov)
        report["superseded"] += 1
    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/tools/test_repair_passes.py -v`
Expected: PASS, all 10 tests (5 from Task 2 + 5 new).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/repair_passes.py test/tools/test_repair_passes.py
git commit -m "$(cat <<'EOF'
feat(tools): reference-library dedup pass via supersession

Exactly-one-twin + exact-text match supersedes the null-root copy in
favor of the project-attributed canonical copy; drift, orphans, and
ambiguity are reported and left live. Zero deletions.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 4: Project backfill pass + CLI with backup gate + verification

**Files:**
- Modify: `tools/repair_passes.py` (add `backfillProjects` and `verifyRepair`)
- Create: `tools/repair_corpus.py` (CLI)
- Test: `test/tools/test_repair_passes.py` (add tests), `test/tools/test_repair_cli.py`

**Interfaces:**
- Produces:
  - `backfillProjects(store) -> dict` `{"backfilled": int, "unresolvable": int}`
  - `verifyRepair(store, preTotals) -> dict` `{"ok": bool, "atomTotalDelta": int, "liveDelta": int, "kvRefsRemaining": int}`
  - CLI `python3 tools/repair_corpus.py --store PATH [--old-db PATH] [--apply]` (dry-run default: prints per-pass reports as JSON and exits without mutating; `--apply` backs up via the SQLite backup API to `<store>.bak-pre-repair-<YYYYMMDD-HHMMSS>` then runs all passes and prints reports + verification)

- [ ] **Step 1: Write the failing tests**

Add to `test/tools/test_repair_passes.py`:

```python
from repair_passes import backfillProjects, verifyRepair


def test_backfill_from_projects_ref(store):
    aid = _putChunk(store, "b", "projects/sextant/doa/model.py#c4")
    report = backfillProjects(store)
    assert report["backfilled"] == 1
    assert store._conn.execute(
        "SELECT project FROM atoms WHERE id=?", (aid,)).fetchone()[0] == "sextant"


def test_backfill_reflib_gets_aegis_and_dotfiles_stay_null(store):
    r = _putChunk(store, "b", "reference-library/x.md#c0")
    d = _putChunk(store, "b", "claude-home/hooks/emit.py#c1")
    report = backfillProjects(store)
    assert report["backfilled"] == 1
    assert report["unresolvable"] == 1
    got = dict(store._conn.execute(
        "SELECT id, project FROM atoms").fetchall())
    assert got[r] == "Aegis"
    assert got[d] is None


def test_backfill_idempotent(store):
    _putChunk(store, "b", "projects/sextant/doa/model.py#c4")
    backfillProjects(store)
    assert backfillProjects(store) == {"backfilled": 0, "unresolvable": 0}


def test_verifyRepair_flags_count_drift(store):
    _putChunk(store, "b", "projects/x/y.py#c0")
    pre = {"total": 99, "live": 99}   # wrong on purpose
    v = verifyRepair(store, pre)
    assert v["ok"] is False
    assert v["atomTotalDelta"] != 0


def test_verifyRepair_ok_when_totals_hold(store):
    _putChunk(store, "b", "projects/x/y.py#c0")
    pre = {"total": 1, "live": 1}
    v = verifyRepair(store, pre)
    assert v["ok"] is True
    assert v["kvRefsRemaining"] == 0
```

Create `test/tools/test_repair_cli.py`:

```python
"""CLI behavior: dry-run mutates nothing; --apply backs up first."""
import json
import subprocess
import sys
from pathlib import Path

_DAEMON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DAEMON / "tools"))
sys.path.insert(0, str(_DAEMON / "src"))

from store.store import openStore, putAtom


def _seed(tmp_path):
    p = tmp_path / "mem.db"
    s = openStore(p)
    putAtom(s, {
        "text": "b", "kind": "document_chunk", "project": None,
        "importance": 0.0,
        "provenance": {"source": "bulk-import",
                       "sourceRef": "projects/obol/api/x.go#c0"},
    })
    s.close()
    return p


def _run(args):
    return subprocess.run(
        [sys.executable, str(_DAEMON / "tools" / "repair_corpus.py"), *args],
        capture_output=True, text=True)


def test_dry_run_default_mutates_nothing(tmp_path):
    p = _seed(tmp_path)
    r = _run(["--store", str(p)])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["applied"] is False
    s = openStore(p)
    assert s._conn.execute(
        "SELECT project FROM atoms").fetchone()[0] is None  # untouched
    s.close()
    assert not list(tmp_path.glob("*.bak-pre-repair-*"))


def test_apply_backs_up_then_repairs(tmp_path):
    p = _seed(tmp_path)
    r = _run(["--store", str(p), "--apply"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["applied"] is True
    assert out["backfillProjects"]["backfilled"] == 1
    assert out["verify"]["ok"] is True
    backups = list(tmp_path.glob("*.bak-pre-repair-*"))
    assert len(backups) == 1
    s = openStore(p)
    assert s._conn.execute(
        "SELECT project FROM atoms").fetchone()[0] == "obol"
    s.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/tools/ -v -k "backfill or verify or cli or dry_run or apply"`
Expected: FAIL, `ImportError` on `backfillProjects` and missing `repair_corpus.py`.

- [ ] **Step 3: Write the implementations**

Add to `tools/repair_passes.py`:

```python
from repair_lib import refToProject  # noqa: E402  (with the existing imports)


def backfillProjects(store):
    """Backfill NULL projects derivable from refs; count the rest.

    Only document_chunk atoms; only where project IS NULL; the derivation is
    refToProject (projects/<name>/ and reference-library/ resolve, dotfile and
    kv_cache roots do not). Idempotent: backfilled rows leave the predicate.
    """
    report = {"backfilled": 0, "unresolvable": 0}
    rows = store._conn.execute(
        "SELECT a.id, p.source_ref FROM atoms a "
        "JOIN provenance p ON p.atom_id = a.id "
        "WHERE a.kind = 'document_chunk' AND a.project IS NULL"
    ).fetchall()
    try:
        for atomId, ref in rows:
            project = refToProject(ref) if ref else None
            if project is None:
                report["unresolvable"] += 1
                continue
            store._conn.execute(
                "UPDATE atoms SET project = ? WHERE id = ? AND project IS NULL",
                (project, atomId))
            report["backfilled"] += 1
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise
    return report


def totals(store):
    """The invariant counters verifyRepair checks against."""
    total = store._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    live = store._conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE status='live'").fetchone()[0]
    return {"total": total, "live": live}


def verifyRepair(store, preTotals):
    """Post-run invariants: no atom created or destroyed; kv refs gone or known.

    ``atomTotalDelta`` must be zero (supersession changes status, never count).
    ``liveDelta`` is informational (dedup reduces live count by design).
    ``ok`` is True when the total held.
    """
    post = totals(store)
    kvRemaining = store._conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE source_ref LIKE 'kv_cache%'"
    ).fetchone()[0]
    delta = post["total"] - preTotals["total"]
    return {
        "ok": delta == 0,
        "atomTotalDelta": delta,
        "liveDelta": post["live"] - preTotals["live"],
        "kvRefsRemaining": kvRemaining,
    }
```

Create `tools/repair_corpus.py`:

```python
#!/usr/bin/env python3
"""Phase A corpus repair CLI: dry-run by default, backup-gated --apply.

Dry-run opens the store read-only-in-spirit (no pass runs) and prints what
WOULD run plus current problem counts. --apply performs a SQLite backup-API
copy (safe on a live WAL database) to <store>.bak-pre-repair-<ts>, then runs
the three passes in order (kv refs, reflib dedup, project backfill) and the
verification invariants. Exit code 1 if verification fails.

Operational note: stop the pensive-v3 daemon (or accept that its resident
index is stale until its next reindex) before an --apply on the live store.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_TOOLS.parent / "src"))

from repair_passes import (  # noqa: E402
    repairKvCacheRefs, dedupReferenceLibrary, backfillProjects,
    totals, verifyRepair,
)
from store.store import openStore  # noqa: E402

_DEFAULT_OLD_DB = Path.home() / "Projects/Aegis/AEGIS/Pensive/kv_cache/vector_meta.db"


def _backup(storePath):
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(str(storePath) + f".bak-pre-repair-{ts}")
    src = sqlite3.connect(storePath)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def _problemCounts(store):
    conn = store._conn
    kv = conn.execute(
        "SELECT COUNT(*) FROM provenance WHERE source_ref LIKE 'kv_cache%'"
    ).fetchone()[0]
    dupLive = conn.execute(
        "SELECT COUNT(*) FROM atoms a JOIN provenance p ON p.atom_id=a.id "
        "WHERE a.kind='document_chunk' AND a.status='live' "
        "AND p.source_ref LIKE 'reference-library/%'").fetchone()[0]
    nullProj = conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE kind='document_chunk' "
        "AND project IS NULL").fetchone()[0]
    return {"kvRefs": kv, "reflibDupCandidates": dupLive,
            "nullProjects": nullProj}


def main():
    ap = argparse.ArgumentParser(prog="repair-corpus")
    ap.add_argument("--store", required=True)
    ap.add_argument("--old-db", default=str(_DEFAULT_OLD_DB))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args()

    store = openStore(Path(args.store))
    try:
        out = {"applied": False, "problems": _problemCounts(store)}
        if args.apply:
            backupPath = _backup(args.store)
            out["backup"] = str(backupPath)
            pre = totals(store)
            out["repairKvCacheRefs"] = repairKvCacheRefs(store, args.old_db)
            out["dedupReferenceLibrary"] = dedupReferenceLibrary(
                store, sessionId=args.session_id)
            out["backfillProjects"] = backfillProjects(store)
            out["verify"] = verifyRepair(store, pre)
            out["applied"] = True
        print(json.dumps(out, indent=2))
        if args.apply and not out["verify"]["ok"]:
            return 1
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tool tests**

Run: `python3 -m pytest test/tools/ -v`
Expected: PASS (7 lib + 10 pass + 5 backfill/verify + 2 CLI = 24 tests).

- [ ] **Step 5: Run the full daemon suite (no regressions)**

Run: `python3 -m pytest test/ -q`
Expected: 393 + 24 = 417 passed (or current total + 24), 1 skipped, 0 failed.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/pensive/daemon
git add tools/repair_passes.py tools/repair_corpus.py test/tools/
git commit -m "$(cat <<'EOF'
feat(tools): project backfill, verification invariants, backup-gated CLI

repair_corpus.py: dry-run default with problem counts; --apply performs a
SQLite backup-API copy then runs the three passes plus invariant checks
(atom total constant, kv refs accounted). Exit 1 on invariant failure.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

## Post-implementation (controller-only, NOT a subagent task)

1. Dry-run against the live store; sanity-check problem counts (~3,909 / ~16,362 / ~20,271).
2. Stop `pensive-v3.service`, run `--apply --session-id <this session>`, inspect the report, restart the service.
3. Post-repair recall spot-checks: a reflib query returns no duplicate-text pair; a repaired kv_cache chunk surfaces with its real ref.

## Self-Review

- **Spec coverage:** Phase A items 1/2/3 map to Tasks 2/3/4; backup gate + verification invariants are Task 4; idempotency is tested per pass; "never guess" paths all return report counts. The spec's "every rewritten ref resolves to an existing file OR is flagged" invariant is deliberately implemented as a REPORT field rather than a hard check (files legitimately deleted since import must not fail the run); the dry-run report and post-run spot-checks cover it.
- **Placeholder scan:** none; every step carries full code and expected output.
- **Type consistency:** report dict keys match between implementations and tests; `supersede(store, oldId, newId, provInput)` called with the store API's exact signature; `_putChunk`/`_makeOldDb` helpers defined in the test file that uses them.
