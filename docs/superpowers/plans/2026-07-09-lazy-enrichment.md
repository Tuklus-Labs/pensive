# Lazy Recall Enrichment (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a recall result is a code chunk, attach (at serve time, for surfaced hits only) its resolved file location and 2-3 related memory atoms, so "have we solved X" returns the chunk plus where it lives and what we learned.

**Architecture:** Two new pure-ish modules in `src/recall/` (`refs.py`: ref-to-path resolution; `enrich.py`: chunk location by text search + related-memory by shared entity facets) feed an optional `enricher` parameter on `assemblePayload`. Default `None` preserves byte-identical legacy behavior. Attachments are furniture lines in the existing tier grammar and degrade first under budget: related-memory lines drop, then the location line, then the normal atomic entry drop.

**Tech Stack:** Python stdlib + the daemon's store API. No new dependencies. Tests with pytest; the payload tests need no model, the end-to-end test uses the session embedder/reranker fixtures.

## Global Constraints

- Work in `~/Projects/pensive/daemon`. Branch `feat/corpus-enrichment`. No new branches/worktrees.
- No em dashes anywhere in code, comments, docstrings, or commit messages.
- Payload no-slop contract holds: every furniture line we add is plain text, prefixed, one line; bodies stay verbatim; no markdown furniture, no emoji.
- `assemblePayload(store, results, tokenBudget)` with no enricher must remain BYTE-IDENTICAL to today for every input (all existing payload/engine/serve tests must pass unmodified).
- Enrichment must never raise on missing/unreadable files, refs with unknown roots, or text that no longer matches the file: every failure degrades to "no attachment", silently at serve time, because recall must answer even when the filesystem moved.
- File reads are bounded: refuse files over 5 MB (`_MAX_FILE_BYTES`), read once per surfaced hit.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
  ```
- Tests never touch the live store or real repos; tmp_path only.

**Verified store facts:** ref roots map as `projects/` -> `~/Projects/`, `reference-library/` -> `~/Projects/Aegis/AEGIS/docs/reference-library/`, `claude-home/` -> `~/.claude/`, `codex-home/` -> `~/.codex/`. Refs carry `#cN` fragments. Facets: `facets(atom_id, key, value)` with `key='entity'`, values lowercase. Memory kinds: `atom|narrative|snapshot`. Tier-2 edge grammar: `<type> -> p3://<dst> <gist>`.

---

### Task 1: Ref-to-path resolution (`src/recall/refs.py`)

**Files:**
- Create: `src/recall/refs.py`
- Test: `test/recall/test_refs.py`

**Interfaces:**
- Produces: `refToPath(ref, home=None) -> pathlib.Path | None`. Strips any `#...` fragment, maps the root, returns an absolute Path without touching the filesystem (existence is the caller's concern). `home` overrides `Path.home()` for tests.

- [ ] **Step 1: Write the failing tests**

Create `test/recall/test_refs.py`:

```python
"""Ref-to-filesystem-path resolution for serve-time enrichment."""
from pathlib import Path

from recall.refs import refToPath

_HOME = Path("/fake/home")


def test_projects_root():
    assert refToPath("projects/obol/api/rate.go#c11", home=_HOME) == \
        _HOME / "Projects/obol/api/rate.go"


def test_reference_library_root():
    assert refToPath("reference-library/53-hw.md#c2", home=_HOME) == \
        _HOME / "Projects/Aegis/AEGIS/docs/reference-library/53-hw.md"


def test_dotfile_roots():
    assert refToPath("claude-home/hooks/emit.py", home=_HOME) == \
        _HOME / ".claude/hooks/emit.py"
    assert refToPath("codex-home/config.toml#c0", home=_HOME) == \
        _HOME / ".codex/config.toml"


def test_unknown_root_and_junk_return_none():
    assert refToPath("kv_cache/vector_meta.db#rowid=7", home=_HOME) is None
    assert refToPath("", home=_HOME) is None
    assert refToPath(None, home=_HOME) is None


def test_traversal_rejected():
    # A ref must never escape its root: reject any .. segment outright.
    assert refToPath("projects/../../etc/passwd", home=_HOME) is None
    assert refToPath("projects/x/../../../etc/shadow#c1", home=_HOME) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/recall/test_refs.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'recall.refs'`.

- [ ] **Step 3: Implement**

Create `src/recall/refs.py`:

```python
"""Ref-to-filesystem-path resolution: the serve-time inverse of the import
waves' ref convention.

A source_ref names a file relative to one of the known import roots plus an
optional ``#<fragment>`` (chunk index). This module maps a ref back to the
absolute path WITHOUT touching the filesystem; existence and readability are
the caller's concern (enrich handles missing files as "no attachment").

Unknown roots (including the kv_cache refs Phase A could not recover) and any
ref containing a ``..`` segment resolve to None: a ref is data from the store,
not a trusted path, and must never escape its root.
"""
from pathlib import Path

__all__ = ["refToPath"]

# refRoot -> path relative to $HOME. Ordered; first prefix match wins.
_ROOTS = (
    ("projects/", "Projects"),
    ("reference-library/", "Projects/Aegis/AEGIS/docs/reference-library"),
    ("claude-home/", ".claude"),
    ("codex-home/", ".codex"),
)


def refToPath(ref, home=None):
    """Absolute Path for a store ref, or None for unknown/unsafe refs."""
    if not ref:
        return None
    base = ref.split("#", 1)[0]
    if not base:
        return None
    parts = base.split("/")
    if ".." in parts:
        return None
    root = home if home is not None else Path.home()
    for prefix, rel in _ROOTS:
        if base.startswith(prefix):
            rest = base[len(prefix):]
            if not rest:
                return None
            return root / rel / rest
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest test/recall/test_refs.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/refs.py test/recall/test_refs.py
git commit -m "$(cat <<'EOF'
feat(recall): ref-to-path resolution for serve-time enrichment

Maps store refs back through the four import roots; unknown roots and
any ..-segment ref resolve to None (a ref is data, never a trusted path).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 2: Chunk location + related memory (`src/recall/enrich.py`)

**Files:**
- Create: `src/recall/enrich.py`
- Test: `test/recall/test_enrich.py`

**Interfaces:**
- Consumes: `refToPath` (Task 1), `store._conn` (sanctioned batched reads), `recall.payload._gist` semantics (re-implemented locally as `_gist` import from payload).
- Produces:
  - `locateChunk(chunkText, path) -> tuple[int, int] | None` (1-indexed inclusive line span of the chunk's text in the file, whitespace-normalized; None on miss/missing/oversized file)
  - `relatedMemory(store, atomId, limit=3) -> list[tuple[str, str]]` (`(memoryAtomId, gist)` best-first by shared-entity specificity)
  - `Enricher` class: `Enricher(store)` with `.lines(result) -> list[str]` returning the furniture lines for one recall result (empty for non-chunks and on any failure)

- [ ] **Step 1: Write the failing tests**

Create `test/recall/test_enrich.py`:

```python
"""Serve-time enrichment: chunk location + related-memory attachment."""
import pytest

from recall.enrich import locateChunk, relatedMemory, Enricher
from store.store import openStore, putAtom, addFacet


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind="atom", project="aegis", sourceRef=None):
    prov = {"source": "claude-code"}
    if sourceRef is not None:
        prov["sourceRef"] = sourceRef
    return putAtom(store, {
        "text": text, "kind": kind, "project": project,
        "importance": 0.0, "provenance": prov,
    })


# ---- locateChunk ----------------------------------------------------------

def test_locate_exact_span(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("line one\ndef target():\n    return 42\nline four\n")
    assert locateChunk("def target():\n    return 42", f) == (2, 3)


def test_locate_whitespace_normalized(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a\n\ndef target():\n\n    return   42\nz\n")
    # The stored chunk collapsed blank lines and run-length spaces; the span
    # still resolves to the file's own line numbers.
    assert locateChunk("def target():\n    return 42", f) == (3, 5)


def test_locate_miss_missing_and_oversize(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("nothing relevant\n")
    assert locateChunk("absent text", f) is None
    assert locateChunk("anything", tmp_path / "gone.py") is None
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    assert locateChunk("anything", big) is None


# ---- relatedMemory --------------------------------------------------------

def test_related_ranks_by_shared_entity_specificity(store):
    chunk = _put(store, "def compile(): pass", kind="document_chunk",
                 sourceRef="projects/pensive/src/x.py#c0")
    rare = _put(store, "decision about the csr publish lock")
    common = _put(store, "note mentioning python again")
    other = _put(store, "unrelated memory")
    codeTwin = _put(store, "another chunk", kind="document_chunk")
    addFacet(store, chunk, "entity", "csr_lock")
    addFacet(store, chunk, "entity", "python")
    addFacet(store, rare, "entity", "csr_lock")          # rare: only 2 atoms
    addFacet(store, common, "entity", "python")
    addFacet(store, codeTwin, "entity", "python")        # python: 3 atoms
    addFacet(store, codeTwin, "entity", "csr_lock")      # chunk kind: excluded
    got = relatedMemory(store, chunk, limit=3)
    ids = [g[0] for g in got]
    assert ids[0] == rare                 # rarer shared entity outranks common
    assert common in ids
    assert other not in ids               # no shared entity
    assert codeTwin not in ids            # memory kinds only


def test_related_empty_when_no_facets(store):
    chunk = _put(store, "body", kind="document_chunk")
    assert relatedMemory(store, chunk, limit=3) == []


# ---- Enricher -------------------------------------------------------------

def test_enricher_lines_for_chunk(store, tmp_path):
    f = tmp_path / "Projects/obol/api/rate.go"
    f.parent.mkdir(parents=True)
    f.write_text("package api\nfunc Rate() int { return 1 }\n")
    chunk = _put(store, "func Rate() int { return 1 }",
                 kind="document_chunk",
                 sourceRef="projects/obol/api/rate.go#c0")
    mem = _put(store, "we capped the rate limiter at one")
    addFacet(store, chunk, "entity", "rate_limiter")
    addFacet(store, mem, "entity", "rate_limiter")
    e = Enricher(store, home=tmp_path)
    lines = e.lines({"atomId": chunk})
    assert lines[0] == "at projects/obol/api/rate.go#L2-2"
    assert lines[1].startswith(f"relates -> p3://{mem} ")


def test_enricher_empty_for_memory_kind_and_on_any_failure(store):
    mem = _put(store, "a memory atom")
    assert Enricher(store).lines({"atomId": mem}) == []
    chunk = _put(store, "body", kind="document_chunk",
                 sourceRef="kv_cache/vector_meta.db#rowid=1")
    # Unknown root: no location; no facets: no relations; empty, no raise.
    assert Enricher(store).lines({"atomId": chunk}) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/recall/test_enrich.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'recall.enrich'`.

- [ ] **Step 3: Implement**

Create `src/recall/enrich.py`:

```python
"""Serve-time enrichment for code-chunk recall results.

Two attachments, computed lazily for SURFACED hits only (top-k is small, so
this is a handful of file reads and one facet query per recall):

- **Location line** ``at <ref-path>#L<start>-<end>``: the chunk's stored text
  located in its source file by whitespace-normalized search. The chunker that
  produced the corpus is irrelevant because the text itself is the key.
- **Relation lines** ``relates -> p3://<atomId> <gist>``: memory-kind atoms
  sharing entity facets with the chunk, rarest-shared-entity first (specificity
  = 1/frequency, summed over shared entities). Same grammar as Tier-2 edge
  lines, so when the Phase C campaign materializes real relates edges the
  payload format does not change.

Failure policy: every miss (unknown ref root, missing/oversized/binary file,
text not found, no shared facets) yields NO line, never an exception. Recall
must answer even when the filesystem has moved on. Enrichment failures are
invisible by design; the dry corpus-health numbers live in the repair tool's
report, not in serve-time noise.
"""
import re

from recall.refs import refToPath
from store.store import getAtom

__all__ = ["locateChunk", "relatedMemory", "Enricher"]

# Files over this size are not searched (binary blobs, giant logs).
_MAX_FILE_BYTES = 5 * 1024 * 1024

# Collapse all whitespace runs to single spaces for matching; the stored chunk
# and the on-disk file may disagree about blank lines and indentation width.
_WS_RE = re.compile(r"\s+")

# Gist rendering matches payload._gist (80 chars, whitespace collapsed).
_GIST_CHARS = 80


def _norm(text):
    return _WS_RE.sub(" ", text).strip()


def _gist(text):
    flat = _norm(text)
    return flat[:_GIST_CHARS]


def locateChunk(chunkText, path):
    """1-indexed inclusive (start, end) line span of chunkText in path, or None.

    Match is on whitespace-normalized text. The span is found by scanning
    normalized prefixes: walk the file's lines accumulating a normalized
    window, and slide the window start forward when it can no longer prefix
    the target. O(lines * window) worst case, fine for source files.
    """
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        lines = path.read_text(errors="strict").splitlines()
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    target = _norm(chunkText)
    if not target:
        return None
    for start in range(len(lines)):
        acc = ""
        for end in range(start, len(lines)):
            piece = _norm(lines[end])
            acc = (acc + " " + piece).strip() if piece else acc
            if acc == target:
                return (start + 1, end + 1)
            if len(acc) > len(target) or (acc and not target.startswith(acc)):
                break
    return None


_MEMORY_KINDS = ("atom", "narrative", "snapshot")


def relatedMemory(store, atomId, limit=3):
    """Memory atoms sharing entity facets with atomId, rarest-first.

    Score per candidate = sum over shared entity values of 1/freq(value),
    where freq counts LIVE atoms carrying that entity facet, so rare shared
    entities dominate and hub entities (FILES-style) contribute almost
    nothing. Live memory kinds only; the source atom itself excluded.
    Returns [(memoryAtomId, gist)] best-first, at most limit.
    """
    kindMarks = ",".join("?" for _ in _MEMORY_KINDS)
    rows = store._conn.execute(
        "WITH mine AS ("
        "  SELECT value FROM facets WHERE atom_id = ? AND key = 'entity'"
        "), freq AS ("
        "  SELECT f.value, COUNT(*) AS n FROM facets f"
        "  JOIN atoms a ON a.id = f.atom_id"
        "  WHERE f.key = 'entity' AND a.status = 'live'"
        "  AND f.value IN (SELECT value FROM mine)"
        "  GROUP BY f.value"
        ") "
        "SELECT f.atom_id, SUM(1.0 / freq.n) AS score "
        "FROM facets f "
        "JOIN freq ON freq.value = f.value "
        "JOIN atoms a ON a.id = f.atom_id "
        f"WHERE f.key = 'entity' AND a.status = 'live' AND a.kind IN ({kindMarks}) "
        "AND f.atom_id != ? "
        "GROUP BY f.atom_id ORDER BY score DESC, f.atom_id LIMIT ?",
        (atomId, *_MEMORY_KINDS, atomId, limit),
    ).fetchall()
    out = []
    for memId, _score in rows:
        atom = getAtom(store, memId)
        if atom is None:
            continue
        out.append((memId, _gist(atom["text"])))
    return out


class Enricher:
    """Per-recall enrichment: furniture lines for one result dict.

    ``lines(result)`` returns [] for non-chunk atoms and on every failure
    path. ``home`` overrides the filesystem root for tests.
    """

    def __init__(self, store, home=None, relatedLimit=3):
        self._store = store
        self._home = home
        self._relatedLimit = relatedLimit

    def lines(self, result):
        atom = getAtom(self._store, result["atomId"])
        if atom is None or atom["kind"] != "document_chunk":
            return []
        out = []
        ref = self._sourceRef(result["atomId"])
        if ref:
            path = refToPath(ref, home=self._home)
            if path is not None:
                span = locateChunk(atom["text"], path)
                if span is not None:
                    base = ref.split("#", 1)[0]
                    out.append(f"at {base}#L{span[0]}-{span[1]}")
        for memId, gist in relatedMemory(
                self._store, result["atomId"], self._relatedLimit):
            out.append(f"relates -> p3://{memId} {gist}")
        return out

    def _sourceRef(self, atomId):
        row = self._store._conn.execute(
            "SELECT source_ref FROM provenance WHERE atom_id = ? "
            "AND source_ref IS NOT NULL ORDER BY recorded_at LIMIT 1",
            (atomId,),
        ).fetchone()
        return row[0] if row else None
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest test/recall/test_enrich.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/enrich.py test/recall/test_enrich.py
git commit -m "$(cat <<'EOF'
feat(recall): chunk location + related-memory enrichment

locateChunk finds the stored text's line span by whitespace-normalized
search (chunker-independent); relatedMemory ranks live memory atoms by
rarest-shared-entity specificity; Enricher renders both as tier-grammar
furniture lines, empty on every failure path.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 3: Payload integration with per-entry degrade

**Files:**
- Modify: `src/recall/payload.py` (`assemblePayload` signature + one helper)
- Test: `test/recall/test_engine.py` has payload tests? No: payload tests live inside `test/recall/test_engine.py`'s payload section; ADD new tests to a new `test/recall/test_payload_enrich.py` instead (do not disturb the existing file).

**Interfaces:**
- Produces: `assemblePayload(store, results, tokenBudget, enricher=None)`. With `enricher=None`: byte-identical legacy behavior. With an enricher: each Tier-1 entry gains the enricher's lines appended after the provenance line; if the enriched entry does not fit the remaining budget, degrade per-entry: drop `relates` lines one at a time from the end, then the `at` line, then fit the bare entry; only then the normal atomic drop.

- [ ] **Step 1: Write the failing tests**

Create `test/recall/test_payload_enrich.py`:

```python
"""assemblePayload enrichment: attachment lines + per-entry degrade order."""
import pytest

from recall.payload import assemblePayload, estimateTokens
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind="document_chunk"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "aegis",
        "importance": 0.0, "provenance": {"source": "bulk-import"},
    })


def _result(atomId):
    return {"atomId": atomId, "score": 1.0, "confidence": 0.9,
            "shouldTrust": True, "why": "why"}


class _FakeEnricher:
    def __init__(self, lines):
        self._lines = lines

    def lines(self, result):
        return list(self._lines)


def test_no_enricher_is_legacy_identical(store):
    aid = _put(store, "chunk body")
    legacy = assemblePayload(store, [_result(aid)], 1500)
    explicit = assemblePayload(store, [_result(aid)], 1500, enricher=None)
    assert legacy == explicit


def test_enricher_lines_appended_after_provenance(store):
    aid = _put(store, "chunk body")
    enr = _FakeEnricher(["at projects/x/y.py#L1-2",
                         "relates -> p3://SOME_ID a gist"])
    payload, _, _ = assemblePayload(store, [_result(aid)], 1500, enricher=enr)
    lines = payload.splitlines()
    assert lines[-2] == "at projects/x/y.py#L1-2"
    assert lines[-1] == "relates -> p3://SOME_ID a gist"
    assert "chunk body" in payload


def test_degrade_drops_relates_then_at_before_entry(store):
    aid = _put(store, "chunk body")
    enr = _FakeEnricher(["at projects/x/y.py#L1-2",
                         "relates -> p3://A first gist",
                         "relates -> p3://B second gist"])
    bare, bareTokens, _ = assemblePayload(store, [_result(aid)], 1500)
    # Budget exactly the bare entry: every attachment must drop, entry kept.
    payload, _, _ = assemblePayload(store, [_result(aid)], bareTokens,
                                    enricher=enr)
    assert payload == bare
    # Budget = bare + the at-line only: relates lines drop, at-line stays.
    atLine = "at projects/x/y.py#L1-2"
    midBudget = estimateTokens(bare + "\n" + atLine)
    payload2, _, _ = assemblePayload(store, [_result(aid)], midBudget,
                                     enricher=enr)
    assert atLine in payload2
    assert "relates ->" not in payload2


def test_enricher_exception_never_breaks_payload(store):
    aid = _put(store, "chunk body")

    class _Bomb:
        def lines(self, result):
            raise RuntimeError("enrichment exploded")

    payload, _, _ = assemblePayload(store, [_result(aid)], 1500,
                                    enricher=_Bomb())
    assert "chunk body" in payload  # entry rendered bare, no raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/recall/test_payload_enrich.py -v`
Expected: FAIL, `TypeError: assemblePayload() got an unexpected keyword argument 'enricher'`.

- [ ] **Step 3: Implement**

In `src/recall/payload.py`, add after `tier1Entry`:

```python
def _enrichedEntry(store, result, enricher, remainingBudget):
    """Tier-1 entry plus enricher lines, degraded to fit remainingBudget.

    Degrade order (spec: attachments go first, bodies are never touched):
    drop ``relates`` lines from the end one at a time, then the ``at`` line,
    then the bare entry. Returns the best-fitting string, which may still
    exceed remainingBudget (the caller's atomic-drop rule then applies to the
    WHOLE entry, exactly as for a bare oversized entry). An enricher that
    raises is treated as no enricher for this entry: serve-time enrichment is
    best-effort by contract and must never break recall.
    """
    entry = tier1Entry(store, result)
    try:
        extra = list(enricher.lines(result))
    except Exception:
        extra = []
    while extra:
        candidate = entry + _LINE_SEPARATOR + _LINE_SEPARATOR.join(extra)
        if estimateTokens(candidate) <= remainingBudget:
            return candidate
        extra.pop()  # relates lines shed from the end; the at line goes last
    return entry
```

Change `assemblePayload`'s signature and its entry loop (the only edits; the low-confidence path and everything else stay untouched):

```python
def assemblePayload(store, results, tokenBudget, enricher=None):
```

and inside the trusted-results loop, replace `entry = tier1Entry(store, result)` with:

```python
        if enricher is not None:
            joined = _ENTRY_SEPARATOR.join(entries) if entries else ""
            used = estimateTokens(joined) if entries else 0
            sep = estimateTokens(_ENTRY_SEPARATOR) if entries else 0
            entry = _enrichedEntry(store, result, enricher,
                                   tokenBudget - used - sep)
        else:
            entry = tier1Entry(store, result)
```

Also extend the function docstring with one sentence: "``enricher`` (optional) appends per-result furniture lines that degrade before anything else under budget; None preserves the exact legacy payload."

- [ ] **Step 4: Run new tests AND the full existing payload/engine suites**

Run: `python3 -m pytest test/recall/test_payload_enrich.py test/recall/test_engine.py -v`
Expected: PASS. Every pre-existing test must pass UNMODIFIED (the no-enricher path is byte-identical).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/payload.py test/recall/test_payload_enrich.py
git commit -m "$(cat <<'EOF'
feat(recall): optional enricher on assemblePayload with per-entry degrade

Attachments join the entry after the provenance line and shed first under
budget (relates lines from the end, then the at line, never a body). No
enricher = byte-identical legacy payload; a raising enricher = bare entry.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 4: Engine + serve wiring, end to end

**Files:**
- Modify: `src/recall/engine.py` (`recall(...)` grows `enrich=False`; builds an `Enricher` and passes it to `assemblePayload`)
- Modify: `src/serve/mcp.py` (`handle_recall` passes `enrich=True`; `handle_pensive_recall` stays legacy)
- Test: `test/recall/test_engine.py` (one new behavioral test), `test/serve/test_mcp.py` (extend the existing Task-5 handler test)

**Interfaces:**
- Produces: `recall(store, indexes, embedder, query, project=None, timeScope=None, kinds=None, k=10, tokenBudget=1500, enrich=False)`. When `enrich` is True the final `assemblePayload` call receives `Enricher(store)`; everything upstream is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `test/recall/test_engine.py` (uses the existing `store`/`embedder`/`_rerankerWarm` fixtures and `_put`/`buildClassIndexes` helpers already in the file):

```python
def test_enriched_recall_attaches_location_and_relations(
        store, embedder, _rerankerWarm, tmp_path, monkeypatch):
    import recall.enrich as enrich_mod
    # Point the enricher's filesystem root at tmp_path via the Enricher the
    # engine constructs: patch the class default rather than threading a home
    # argument through recall() (serve never needs a non-default home).
    f = tmp_path / "Projects/obol/api/rate.go"
    f.parent.mkdir(parents=True)
    f.write_text("package api\nfunc Rate() int { return 1 }\n")
    origInit = enrich_mod.Enricher.__init__

    def patchedInit(self, s, home=None, relatedLimit=3):
        origInit(self, s, home=tmp_path, relatedLimit=relatedLimit)

    monkeypatch.setattr(enrich_mod.Enricher, "__init__", patchedInit)

    chunk = putAtom(store, {
        "text": "func Rate() int { return 1 }", "kind": "document_chunk",
        "project": "obol", "importance": 0.0,
        "provenance": {"source": "bulk-import",
                       "sourceRef": "projects/obol/api/rate.go#c0"},
    })
    mem = _put(store, "we capped the rate limiter at one per second")
    addFacet(store, chunk, "entity", "rate_limiter")
    addFacet(store, mem, "entity", "rate_limiter")
    idx = buildClassIndexes(store, MODEL_ID)
    out = recall(store, idx, embedder, "rate limiter cap function",
                 k=3, enrich=True)
    assert "at projects/obol/api/rate.go#L2-2" in out["payload"]
    assert f"relates -> p3://{mem}" in out["payload"]
```

Extend the Task-5 serve test in `test/serve/test_mcp.py` (`test_serve_context_holds_class_indexes_and_recall_prefers_memory`) with two lines at the end, asserting the enriched path does not break the plain-memory case:

```python
        # Enrichment on the native handler must not disturb memory results.
        assert "Authelia" in out
```

(The handler change itself flips `handle_recall` to `enrich=True`; the existing assertion already covers it, the comment documents intent. If the existing test fails after wiring, that IS a finding to fix, not to relax.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest test/recall/test_engine.py::test_enriched_recall_attaches_location_and_relations -v`
Expected: FAIL, `TypeError: recall() got an unexpected keyword argument 'enrich'`.

- [ ] **Step 3: Implement**

In `src/recall/engine.py`: add the import `from recall.enrich import Enricher` with the other recall imports; change the signature to

```python
def recall(store, indexes, embedder, query, project=None, timeScope=None,
           kinds=None, k=10, tokenBudget=1500, enrich=False):
```

document the new arg in the docstring options list:

```
    - ``enrich``: attach serve-time enrichment lines (chunk file location and
      related memory) to document_chunk results. Off by default; the eval gate
      and legacy callers measure the bare pipeline.
```

and change the final assembly line to:

```python
    enricher = Enricher(store) if enrich else None
    payload, tokensUsed, lowConfidence = assemblePayload(
        store, results, tokenBudget, enricher=enricher)
```

(`_emptyResult` stays enricher-free: the sentinel has no entries to enrich.)

In `src/serve/mcp.py` `handle_recall`, add `enrich=True` to the recall call:

```python
    out = recall(
        ctx.store, ctx.indexes, ctx.embedder, query,
        project=project, timeScope=timeScope, kinds=kinds,
        k=k, tokenBudget=tokenBudget, enrich=True,
    )
```

`handle_pensive_recall` (the legacy listing) stays unchanged.

- [ ] **Step 4: Run the affected suites**

Run: `python3 -m pytest test/recall/test_engine.py test/recall/test_payload_enrich.py test/serve/ -v`
Expected: PASS, including the new behavioral test and all pre-existing tests.

- [ ] **Step 5: Run the full daemon suite**

Run: `python3 -m pytest test/ -q`
Expected: 0 failed.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/engine.py src/serve/mcp.py test/recall/test_engine.py test/serve/test_mcp.py
git commit -m "$(cat <<'EOF'
feat(recall,serve): wire lazy enrichment through recall and the MCP handler

recall() grows enrich=False (gate and legacy callers measure the bare
pipeline); the native handle_recall serves enrich=True, so code hits carry
their file location and related memory.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

## Post-implementation (controller-only)

Restart `pensive-v3.service` after Phase A's apply + Phase B's merge, then live-verify: a code-intent recall ("csr compile lock" style) whose top hit is a chunk shows an `at projects/...#L...` line and, where facets overlap, `relates ->` lines.

## Self-Review

- **Spec coverage:** Phase B item 1 (resolved location, no symbol parsing, misses degrade to bare ref) = Tasks 1-2; item 2 (related memory via entity facets, 2-3 gists, specificity-ranked) = Task 2; budget rules (attachments degrade first) = Task 3; "serve time, top hits only" = Task 4 (enrichment runs inside assemblePayload on surfaced entries only). Tree-sitter symbols explicitly out of scope, matching the spec.
- **Placeholder scan:** none; all steps carry complete code and expected outputs.
- **Type consistency:** `Enricher(store, home=None, relatedLimit=3)` matches Task 4's monkeypatch; `assemblePayload(..., enricher=None)` matches Task 3's tests and Task 4's engine call; `refToPath(ref, home=None)` matches both callers; report grammar lines (`at ...#L<s>-<e>`, `relates -> p3://<id> <gist>`) are identical across Tasks 2, 3, and 4.
