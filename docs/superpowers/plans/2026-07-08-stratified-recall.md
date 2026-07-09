# Stratified, Intent-Fair Recall (Layer 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the 299k-chunk code corpus from starving the 14.5k-atom reasoning memory in unscoped recall, by stratifying candidate generation and the rerank head by kind-class and letting the cross-encoder decide relevance.

**Architecture:** Two kind-classes (MEMORY = atom/narrative/snapshot, CODE = document_chunk). Candidates are generated per class (kind-scoped bm25 + one partitioned dense index per class), fused and prior-weighted globally as today, then the rerank head is filled by round-robin across the per-class fused rankings so the cross-encoder scores a fair mix. Everything downstream (trust, payload) is unchanged. No stored data is mutated; the whole change is on the recall path.

**Tech Stack:** Python 3.10+, SQLite/FTS5, numpy, usearch (HNSW), sentence-transformers (bge-small embedder + bge-reranker-base cross-encoder). All work is in `~/Projects/pensive/daemon`.

## Global Constraints

- Work in `~/Projects/pensive/daemon`. Run tests from that directory: `python3 -m pytest <path> -v` (the repo conftest puts `daemon/src` and `daemon/test` on `sys.path`).
- Branch is already `feat/stratified-recall` (checked out). Do not create worktrees.
- No em dashes anywhere in code, comments, docstrings, or commit messages. Use commas, colons, parentheses, or restructure.
- Match the surrounding house style: two-space-indented SQL string continuation, docstrings that state the contract and the failure mode, named constants over magic numbers, loud failures (raise, never silently drop) consistent with the existing "decades rule" comments.
- `MODEL_ID = "BAAI/bge-small-en-v1.5"`.
- Every git commit message ends with these two trailer lines verbatim:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
  ```
- Do not mutate the live store at `~/.local/share/pensive-v3/pensive.db`. Tests use throwaway `tmp_path` stores only.
- `selectIndex` must stay backward compatible: existing single-index callers (`reembed.py`, and any not touched by this plan) call `selectIndex(store, modelId)` with no kinds and must keep working (kinds defaults to None = all live atoms = today's behavior).

---

### Task 1: Kind-class strata module (pure helpers)

The foundation: the class definitions and the four pure functions the engine, signals, and index layers all consume. No model, no DB fixture beyond a plain store for `splitByClass`.

**Files:**
- Create: `src/recall/strata.py`
- Test: `test/recall/test_strata.py`

**Interfaces:**
- Produces:
  - `MEMORY_KINDS = ("atom", "narrative", "snapshot")`, `CODE_KINDS = ("document_chunk",)`
  - `KIND_CLASSES = (("memory", MEMORY_KINDS), ("code", CODE_KINDS))`
  - `classesForKinds(kinds) -> list[tuple[str, tuple[str, ...]]]`
  - `kindInClause(kinds, alias="a") -> tuple[str, tuple[str, ...]]`
  - `interleave(perClassLists, cap) -> list`
  - `splitByClass(fused, store, classNames) -> dict[str, list]`

- [ ] **Step 1: Write the failing tests**

Create `test/recall/test_strata.py`:

```python
"""Pure kind-class strata helpers (Task 1 of stratified recall)."""
import pytest

from recall.strata import (
    KIND_CLASSES,
    classesForKinds,
    kindInClause,
    interleave,
    splitByClass,
)
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "aegis",
        "importance": 0.0, "provenance": {"source": "claude-code"},
    })


def test_classesForKinds_none_returns_all_classes():
    assert classesForKinds(None) == list(KIND_CLASSES)


def test_classesForKinds_subset_narrows_to_overlapping_classes():
    # Only 'atom' requested: the memory class survives, narrowed to ('atom',);
    # the code class drops out entirely.
    assert classesForKinds(["atom"]) == [("memory", ("atom",))]


def test_classesForKinds_code_only():
    assert classesForKinds(["document_chunk"]) == [("code", ("document_chunk",))]


def test_classesForKinds_unknown_kind_yields_no_classes():
    assert classesForKinds(["nonsense"]) == []


def test_kindInClause_empty_is_noop():
    assert kindInClause(None) == ("", ())
    assert kindInClause(()) == ("", ())


def test_kindInClause_builds_placeholders_and_params():
    frag, params = kindInClause(("atom", "narrative"), alias="a")
    assert frag == " AND a.kind IN (?,?)"
    assert params == ("atom", "narrative")


def test_interleave_round_robins_and_backfills():
    mem = [("m1", 9), ("m2", 8), ("m3", 7)]
    code = [("c1", 9), ("c2", 8)]
    # cap 4: m1, c1, m2, c2
    assert interleave([mem, code], 4) == [("m1", 9), ("c1", 9), ("m2", 8), ("c2", 8)]
    # cap larger than total: everything, backfilling from mem once code is dry
    assert interleave([mem, code], 99) == [
        ("m1", 9), ("c1", 9), ("m2", 8), ("c2", 8), ("m3", 7)
    ]


def test_interleave_single_list_is_prefix():
    mem = [("m1", 3), ("m2", 2), ("m3", 1)]
    assert interleave([mem], 2) == [("m1", 3), ("m2", 2)]


def test_splitByClass_buckets_by_kind_preserving_order(store):
    a = _put(store, "a memory atom", "atom")
    n = _put(store, "a narrative", "narrative")
    c = _put(store, "some code chunk", "document_chunk")
    fused = [(c, 9.0), (a, 8.0), (n, 7.0)]
    out = splitByClass(fused, store, ["memory", "code"])
    assert out["memory"] == [(a, 8.0), (n, 7.0)]
    assert out["code"] == [(c, 9.0)]


def test_splitByClass_drops_atoms_whose_class_not_requested(store):
    a = _put(store, "a memory atom", "atom")
    c = _put(store, "some code chunk", "document_chunk")
    fused = [(c, 9.0), (a, 8.0)]
    # Only the memory bucket requested: the chunk is not returned in any bucket.
    out = splitByClass(fused, store, ["memory"])
    assert out == {"memory": [(a, 8.0)]}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/recall/test_strata.py -v`
Expected: FAIL / collection error, `ModuleNotFoundError: No module named 'recall.strata'`.

- [ ] **Step 3: Write the implementation**

Create `src/recall/strata.py`:

```python
"""Kind-class stratification: which atom kinds form a recall population, and the
pure helpers that keep every population fairly represented at recall's two choke
points (candidate generation and the rerank head).

The store now holds two populations that must not compete on raw count: a large
bulk-imported code corpus (``document_chunk``) and a smaller, higher-value
reasoning memory (``atom``/``narrative``/``snapshot``). A single kind-blind
candidate pool lets the majority starve the minority before the cross-encoder
reranker (the real relevance judge) ever sees it. These helpers let the engine
generate candidates per class and fill the rerank head by round-robin, so both
populations reach the reranker and the winner is decided on relevance, not count.

Nothing here touches the model or the vector index; it is deliberately dependency
-free so both ``signals`` and ``vector_index`` can import it without a cycle.
"""

__all__ = [
    "MEMORY_KINDS",
    "CODE_KINDS",
    "KIND_CLASSES",
    "classesForKinds",
    "kindInClause",
    "interleave",
    "splitByClass",
]

# The two populations. MEMORY is the reasoning memory; CODE is the bulk source
# corpus. Ordered tuples so round-robin interleaving is deterministic.
MEMORY_KINDS = ("atom", "narrative", "snapshot")
CODE_KINDS = ("document_chunk",)

# (className, kinds) in the order the rerank head round-robins them. Adding a
# third class later is a one-line edit here; every consumer reads this tuple.
KIND_CLASSES = (
    ("memory", MEMORY_KINDS),
    ("code", CODE_KINDS),
)


def classesForKinds(kinds):
    """The KIND_CLASSES entries to stratify over for a given ``kinds`` request.

    ``kinds=None`` (unscoped recall) returns every class, each with its full kind
    tuple: this is the case the whole feature exists for. A ``kinds`` subset
    returns only the classes that overlap it, each narrowed to the overlapping
    kinds, so an explicit ``kinds=['atom']`` request stratifies over just the
    memory class scoped to atoms. A ``kinds`` naming no known kind returns ``[]``,
    which the engine reads as "nothing eligible" and short-circuits to the
    low-confidence sentinel, matching the existing empty-kinds contract.
    """
    if kinds is None:
        return list(KIND_CLASSES)
    wanted = set(kinds)
    out = []
    for name, classKinds in KIND_CLASSES:
        overlap = tuple(k for k in classKinds if k in wanted)
        if overlap:
            out.append((name, overlap))
    return out


def kindInClause(kinds, alias="a"):
    """Build an ``AND <alias>.kind IN (?,?...)`` SQL fragment and its params.

    Returns ``("", ())`` for an empty/None ``kinds`` so callers can unconditionally
    concatenate the fragment and extend their params. ``alias`` is the atoms-table
    alias in the caller's query (every current caller uses ``a``). Values are bound
    parameters, never interpolated, so arbitrary kind strings are injection-safe.
    """
    if not kinds:
        return "", ()
    placeholders = ",".join("?" for _ in kinds)
    return f" AND {alias}.kind IN ({placeholders})", tuple(kinds)


def interleave(perClassLists, cap):
    """Round-robin merge of per-class ranked lists into one list of length <= cap.

    Takes one item from each list in turn (list order = class order), preserving
    each list's internal order, and backfills from the lists that still have items
    once others run dry. Stops as soon as ``cap`` items are collected. With a
    single non-empty list this is exactly that list's ``[:cap]`` prefix, so a
    single-class recall degrades to today's behavior. Empty lists are skipped.
    """
    result = []
    positions = [0] * len(perClassLists)
    while len(result) < cap:
        advanced = False
        for i, lst in enumerate(perClassLists):
            if positions[i] < len(lst):
                result.append(lst[positions[i]])
                positions[i] += 1
                advanced = True
                if len(result) >= cap:
                    break
        if not advanced:
            break
    return result


def splitByClass(fused, store, classNames):
    """Group ``fused`` ``[(atomId, score)]`` into per-class sublists.

    One batched ``SELECT id, kind`` maps each fused atom to its class via
    KIND_CLASSES. Returns ``{className: [(atomId, score), ...]}`` for every name in
    ``classNames``, each sublist in the input ``fused`` order. Atoms whose kind
    belongs to a class not in ``classNames`` (or to no class) are dropped, so a
    scoped recall never leaks an out-of-class atom into the rerank head. An empty
    ``fused`` returns a bucket-per-name dict of empty lists.
    """
    buckets = {name: [] for name in classNames}
    if not fused:
        return buckets
    kindToClass = {}
    for name, classKinds in KIND_CLASSES:
        for k in classKinds:
            kindToClass[k] = name
    atomIds = [atomId for atomId, _ in fused]
    placeholders = ",".join("?" for _ in atomIds)
    rows = store._conn.execute(
        f"SELECT id, kind FROM atoms WHERE id IN ({placeholders})",
        tuple(atomIds),
    ).fetchall()
    classById = {r[0]: kindToClass.get(r[1]) for r in rows}
    for atomId, score in fused:
        name = classById.get(atomId)
        if name in buckets:
            buckets[name].append((atomId, score))
    return buckets
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/recall/test_strata.py -v`
Expected: PASS, all 9 tests.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/strata.py test/recall/test_strata.py
git commit -m "$(cat <<'EOF'
feat(recall): kind-class strata helpers for stratified recall

Pure, dependency-free foundation: two kind-classes (memory vs code) and
the classesForKinds / kindInClause / interleave / splitByClass helpers the
engine, signals, and index layers consume.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 2: Kind-scoped bm25 candidate generation

Give the lexical signal a `kinds` filter so the engine can pull per-class bm25 candidates.

**Files:**
- Modify: `src/recall/signals.py` (the `bm25` function and its import block)
- Test: `test/recall/test_signals.py` (add tests)

**Interfaces:**
- Consumes: `kindInClause` from Task 1.
- Produces: `bm25(store, query, k=_DEFAULT_K, kinds=None) -> [(atomId, score)]`. When `kinds` is a non-empty iterable, only atoms of those kinds are returned; `kinds=None` is today's behavior (all live atoms).

- [ ] **Step 1: Write the failing test**

Add to `test/recall/test_signals.py` (match the file's existing fixture/import style; it already builds a `store` and `_put`s atoms):

```python
def test_bm25_kinds_filter_scopes_to_requested_kinds(store):
    from recall.signals import bm25
    from store.store import putAtom

    def put(text, kind):
        return putAtom(store, {
            "text": text, "kind": kind, "project": "aegis",
            "importance": 0.0, "provenance": {"source": "claude-code"},
        })

    memId = put("the migration plan for the ingest pipeline", "atom")
    codeId = put("the migration plan for the ingest pipeline", "document_chunk")

    allHits = {a for a, _ in bm25(store, "migration ingest pipeline", 200)}
    assert memId in allHits and codeId in allHits

    memOnly = {a for a, _ in bm25(store, "migration ingest pipeline", 200,
                                  kinds=("atom", "narrative", "snapshot"))}
    assert memId in memOnly
    assert codeId not in memOnly

    codeOnly = {a for a, _ in bm25(store, "migration ingest pipeline", 200,
                                   kinds=("document_chunk",))}
    assert codeId in codeOnly
    assert memId not in codeOnly
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest test/recall/test_signals.py::test_bm25_kinds_filter_scopes_to_requested_kinds -v`
Expected: FAIL with `TypeError: bm25() got an unexpected keyword argument 'kinds'`.

- [ ] **Step 3: Write the implementation**

In `src/recall/signals.py`, add the strata import near the other `recall`-package imports (after the existing `from pensive...` block is fine; it is a sibling module):

```python
from recall.strata import kindInClause  # noqa: E402  (path set above)
```

Replace the `bm25` function with:

```python
def bm25(store, query, k=_DEFAULT_K, kinds=None):
    """Lexical signal: FTS5 BM25 over live atom text -> ``[(atomId, score)]``.

    ``score`` is the negated ``bm25()`` cost, so higher = better and the list is
    already best-first. ``kinds`` (when a non-empty iterable) restricts the result
    to atoms of those kinds via an ``AND a.kind IN (...)`` clause, so the engine
    can pull a separate per-class candidate list; ``kinds=None`` is unrestricted
    (every live atom). Returns [] for ``k <= 0`` or a query with no searchable
    token (parity with the vector-index contract; never raises on raw text).
    """
    if k <= 0:
        return []
    match = _sanitizeFtsQuery(query)
    if match is None:
        return []
    kindClause, kindParams = kindInClause(kinds, alias="a")
    rows = store._conn.execute(
        "SELECT a.id, -bm25(fts) AS score "
        "FROM fts JOIN atoms a ON a.rowid = fts.rowid "
        "WHERE fts MATCH ? AND a.status = 'live'" + kindClause + " "
        "ORDER BY bm25(fts) "
        "LIMIT ?",
        (match, *kindParams, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/recall/test_signals.py -v`
Expected: PASS (the new test plus all existing signal tests, which call `bm25` without `kinds` and must be unaffected).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/signals.py test/recall/test_signals.py
git commit -m "$(cat <<'EOF'
feat(recall): kind filter on bm25 candidate generation

bm25() gains an optional kinds arg (AND a.kind IN ...) so the engine can
pull a separate per-class lexical candidate list. kinds=None is unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 3: Partitioned dense indexes per kind-class

Add a `kinds` filter to the dense index build path and a `buildClassIndexes` helper that returns one built `VectorIndex` per class. This is what lets memory get an exact FlatIndex while code stays on HNSW.

**Files:**
- Modify: `src/recall/vector_index.py` (`_countEmbeddedLive`, `FlatIndex.build`, `selectIndex`, `__all__`, add `buildClassIndexes`)
- Modify: `src/recall/hnsw_index.py` (`HnswIndex.build`)
- Test: `test/recall/test_vector_index.py` (create if absent; if a vector-index test file already exists under a different name, add there instead)

**Interfaces:**
- Consumes: `KIND_CLASSES` from Task 1.
- Produces:
  - `_countEmbeddedLive(store, modelId, kinds=None) -> int`
  - `FlatIndex.build(self, store, modelId, kinds=None) -> self`
  - `HnswIndex.build(self, store, modelId, kinds=None) -> self`
  - `selectIndex(store, modelId, kinds=None) -> VectorIndex` (built)
  - `buildClassIndexes(store, modelId) -> dict[str, VectorIndex]` keyed by class name from `KIND_CLASSES`

- [ ] **Step 1: Write the failing tests**

Create `test/recall/test_vector_index.py`:

```python
"""Partitioned per-class dense indexes (Task 3 of stratified recall)."""
import pytest

from recall.embedder import Embedder, embedMissing
from recall.vector_index import (
    FlatIndex,
    selectIndex,
    buildClassIndexes,
    _countEmbeddedLive,
)
from store.store import openStore, putAtom

MODEL_ID = "BAAI/bge-small-en-v1.5"

pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "aegis",
        "importance": 0.0, "provenance": {"source": "claude-code"},
    })


def test_countEmbeddedLive_respects_kinds(store, embedder):
    _put(store, "a memory atom about planning", "atom")
    _put(store, "def compile(): pass", "document_chunk")
    _put(store, "another chunk of code", "document_chunk")
    embedMissing(store, embedder)
    assert _countEmbeddedLive(store, MODEL_ID) == 3
    assert _countEmbeddedLive(store, MODEL_ID,
                              kinds=("atom", "narrative", "snapshot")) == 1
    assert _countEmbeddedLive(store, MODEL_ID, kinds=("document_chunk",)) == 2


def test_flatindex_build_with_kinds_excludes_other_kinds(store, embedder):
    memId = _put(store, "the ingest pipeline migration decision", "atom")
    codeId = _put(store, "the ingest pipeline migration decision", "document_chunk")
    embedMissing(store, embedder)
    idx = FlatIndex().build(store, MODEL_ID, kinds=("atom", "narrative", "snapshot"))
    hitIds = {a for a, _ in idx.search(
        embedder.embed(["ingest pipeline migration"])[0], 10)}
    assert memId in hitIds
    assert codeId not in hitIds


def test_buildClassIndexes_partitions_by_class(store, embedder):
    memId = _put(store, "a reasoning memory atom", "atom")
    codeId = _put(store, "some source code chunk", "document_chunk")
    embedMissing(store, embedder)
    indexes = buildClassIndexes(store, MODEL_ID)
    assert set(indexes.keys()) == {"memory", "code"}
    memHits = {a for a, _ in indexes["memory"].search(
        embedder.embed(["reasoning memory"])[0], 10)}
    codeHits = {a for a, _ in indexes["code"].search(
        embedder.embed(["source code"])[0], 10)}
    assert memId in memHits and codeId not in memHits
    assert codeId in codeHits and memId not in codeHits


def test_selectIndex_backward_compatible_no_kinds(store, embedder):
    # The existing single-index contract must be untouched: no kinds arg builds an
    # index over every live embedded atom.
    _put(store, "a memory atom", "atom")
    _put(store, "a code chunk", "document_chunk")
    embedMissing(store, embedder)
    idx = selectIndex(store, MODEL_ID)
    allHits = {a for a, _ in idx.search(embedder.embed(["atom chunk"])[0], 10)}
    assert len(allHits) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest test/recall/test_vector_index.py -v`
Expected: FAIL, `ImportError: cannot import name 'buildClassIndexes'` (and `TypeError` on the `kinds=` calls).

- [ ] **Step 3: Write the implementation**

In `src/recall/vector_index.py`:

Add the strata import at the top (after `from recall.embedder import blobToVec`):

```python
from recall.strata import KIND_CLASSES, kindInClause
```

Update `__all__`:

```python
__all__ = ["VectorIndex", "FlatIndex", "selectIndex", "buildClassIndexes",
           "HNSW_THRESHOLD"]
```

Replace `FlatIndex.build` with:

```python
    def build(self, store, modelId, kinds=None):
        kindClause, kindParams = kindInClause(kinds, alias="a")
        rows = store._conn.execute(
            "SELECT e.atom_id, e.vector FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live'" + kindClause + " "
            "ORDER BY e.atom_id",
            (modelId, *kindParams),
        ).fetchall()
        self._atomIds = [r[0] for r in rows]
        if rows:
            # np.stack copies the read-only frombuffer views into one owned,
            # writable (n, dim) array.
            self._matrix = np.stack([blobToVec(r[1]) for r in rows]).astype(
                np.float32, copy=False
            )
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        return self
```

Replace `_countEmbeddedLive` with:

```python
def _countEmbeddedLive(store, modelId, kinds=None):
    """Count embedded LIVE atoms for ``modelId`` -- the size the switch keys on.

    Mirrors the build-time filter EXACTLY (``embeddings`` joined to LIVE ``atoms``
    for this model, same optional ``kinds`` restriction), so the count equals the
    number of vectors the matching index would actually load. A raw ``embeddings``
    row count would over-count superseded atoms that are never in the index and
    could pick HNSW for a store that is small once the dead rows are excluded.
    """
    kindClause, kindParams = kindInClause(kinds, alias="a")
    return store._conn.execute(
        "SELECT COUNT(*) FROM embeddings e "
        "JOIN atoms a ON a.id = e.atom_id "
        "WHERE e.model_id = ? AND a.status = 'live'" + kindClause,
        (modelId, *kindParams),
    ).fetchone()[0]
```

Replace `selectIndex` with:

```python
def selectIndex(store, modelId, kinds=None):
    """Build and return the right ``VectorIndex`` for this population's size.

    Below :data:`HNSW_THRESHOLD` embedded live atoms (of ``kinds``, if given), an
    exact ``FlatIndex``; at or above it, the approximate ``HnswIndex``. Returns the
    index already BUILT. ``kinds`` scopes the index to one kind-class so a caller
    can hold one index per population; ``kinds=None`` preserves the original
    single-index contract (every live embedded atom). The count is taken over the
    SAME ``kinds``, so a small minority class rides the exact flat scan even when
    the whole store is past the threshold. ``HnswIndex`` is imported lazily so that
    importing this module (and using ``FlatIndex`` at shadow scale) never requires
    usearch.
    """
    if _countEmbeddedLive(store, modelId, kinds) >= HNSW_THRESHOLD:
        from recall.hnsw_index import HnswIndex

        return HnswIndex().build(store, modelId, kinds)
    return FlatIndex().build(store, modelId, kinds)


def buildClassIndexes(store, modelId):
    """One built ``VectorIndex`` per kind-class in ``KIND_CLASSES``, keyed by name.

    ``{"memory": <index>, "code": <index>}`` today. Each class picks flat vs HNSW
    independently by its own live-embedded count, so the small memory population
    gets an exact scan while the large code population gets the approximate graph.
    An empty class (no live embedded atoms of its kinds) yields an empty but valid
    index whose ``search`` returns ``[]``.
    """
    return {
        name: selectIndex(store, modelId, kinds)
        for name, kinds in KIND_CLASSES
    }
```

In `src/recall/hnsw_index.py`, replace `HnswIndex.build` with (add the `kinds` param and the same clause; keep the identical-candidate-universe comment honest):

```python
    def build(self, store, modelId, kinds=None):
        # EXACTLY FlatIndex.build's query: same LIVE filter, same model, same
        # optional kinds restriction, same ORDER BY -- the flat and HNSW indexes
        # for a given class must load one identical candidate universe.
        from recall.strata import kindInClause

        kindClause, kindParams = kindInClause(kinds, alias="a")
        rows = store._conn.execute(
            "SELECT e.atom_id, e.vector FROM embeddings e "
            "JOIN atoms a ON a.id = e.atom_id "
            "WHERE e.model_id = ? AND a.status = 'live'" + kindClause + " "
            "ORDER BY e.atom_id",
            (modelId, *kindParams),
        ).fetchall()
        self._atomIds = [r[0] for r in rows]
        if not rows:
            self._index = None
            return self
        # np.stack copies the read-only frombuffer views into one owned, writable
        # (n, dim) float32 array -- what usearch.add wants.
        matrix = np.stack([blobToVec(r[1]) for r in rows]).astype(
            np.float32, copy=False
        )
        index = Index(ndim=matrix.shape[1], metric="cos", dtype="f32")
        # Keys 0..n-1 index straight into self._atomIds; the add order matches the
        # ORDER BY atom_id row order.
        index.add(np.arange(len(self._atomIds), dtype=np.int64), matrix)
        self._index = index
        return self
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest test/recall/test_vector_index.py test/recall/test_hnsw_index.py -v`
Expected: PASS. The existing `test_hnsw_index.py` calls `build(store, modelId)` with no kinds and must still pass (kinds defaults to None).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/vector_index.py src/recall/hnsw_index.py test/recall/test_vector_index.py
git commit -m "$(cat <<'EOF'
feat(recall): partitioned per-class dense indexes

selectIndex/build/_countEmbeddedLive gain an optional kinds filter and a
new buildClassIndexes returns one built index per kind-class. The small
memory class rides an exact FlatIndex; code stays on HNSW. selectIndex with
no kinds is unchanged (backward compatible).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 4: Stratified recall engine

The core. Per-class candidate generation and a round-robin rerank head, with `recall`'s single `index` argument becoming an `indexes` dict. Update the engine tests and fixtures to the new signature.

**Files:**
- Modify: `src/recall/engine.py` (`recall` signature and body; keep `_emptyResult` and `_filterKinds`)
- Modify: `test/recall/test_engine.py` (imports, `_index` fixture, `_stub_recall_dependencies`, and every `recall(...)` call site in the file)
- Test: `test/recall/test_engine.py` (add the two behavioral tests below)

**Interfaces:**
- Consumes: `classesForKinds`, `interleave`, `splitByClass` (Task 1); `bm25(..., kinds=)` (Task 2); `buildClassIndexes` (Task 3); existing `dense`, `rrf`, `applyPriors`, `rerank`, `assessTrust`, `assemblePayload`, `FACET_BOOST`, `RERANK_HEAD`.
- Produces: `recall(store, indexes, embedder, query, project=None, timeScope=None, kinds=None, k=10, tokenBudget=1500) -> {results, payload, tokensUsed, lowConfidence}`. `indexes` is `{className: VectorIndex}` from `buildClassIndexes`.

- [ ] **Step 1: Write the failing behavioral tests**

Add to `test/recall/test_engine.py`. These use the real embedder + reranker (session fixtures `embedder` and `_rerankerWarm` already exist in the file). Add a fixture that builds class indexes:

```python
from recall.vector_index import buildClassIndexes


@pytest.fixture
def indexes(store, embedder):
    embedMissing(store, embedder)
    return buildClassIndexes(store, MODEL_ID)


def test_memory_intent_query_surfaces_memory_over_code(
        store, embedder, indexes, _rerankerWarm):
    # A store flooded with code chunks (the real-world 20:1 shape, scaled down)
    # plus a few reasoning-memory atoms. A memory-shaped query must return a
    # memory-kind atom at the top, not a code chunk, because both classes reached
    # the reranker and the reranker sorted on relevance.
    for i in range(40):
        _put(store, f"def handler_{i}(req): return route(req, table_{i})",
             kind="document_chunk")
    good = _put(store,
                "we decided to migrate the ingest pipeline to stratified recall "
                "next sprint because the code corpus was starving memory",
                kind="atom")
    idx = buildClassIndexes(store, MODEL_ID)  # rebuild after inserts
    out = recall(store, idx, embedder,
                 "what did we decide about the ingest pipeline migration", k=5)
    topId = out["results"][0]["atomId"]
    assert topId == good


def test_code_intent_query_surfaces_the_right_chunk(
        store, embedder, _rerankerWarm):
    for i in range(20):
        _put(store, f"notes on sprint planning meeting number {i}", kind="atom")
    target = _put(store,
                  "def _compile(self): acquire build_lock then publish the csr "
                  "matrix only if the graph generation has not moved",
                  kind="document_chunk")
    idx = buildClassIndexes(store, MODEL_ID)
    out = recall(store, idx, embedder,
                 "csr matrix compile build lock generation publish", k=5)
    topId = out["results"][0]["atomId"]
    assert topId == target


def test_rerank_head_is_interleaved_across_classes(store, monkeypatch):
    # White-box: with equal-length per-class fused lists, the head handed to
    # rerank must alternate memory, code, memory, code ... (choke point 2 fixed).
    mem = [_put(store, f"memory {i}", kind="atom") for i in range(5)]
    code = [_put(store, f"code {i}", kind="document_chunk") for i in range(5)]
    # fused order: all memory first, then all code. Naive fused[:RERANK_HEAD] would
    # put every memory ahead of every code; interleave must alternate them.
    fused = [(a, 100 - i) for i, a in enumerate(mem)] + \
            [(a, 50 - i) for i, a in enumerate(code)]

    seen = {}
    monkeypatch.setattr(engine, "facetSignal",
                        lambda store, hints: {"boostSet": set(), "filterSet": None})
    monkeypatch.setattr(engine, "bm25", lambda store, query, k, kinds=None: [])
    monkeypatch.setattr(engine, "dense", lambda index, embedder, query, k: [])
    monkeypatch.setattr(engine, "rrf", lambda hits: list(fused))
    monkeypatch.setattr(engine, "applyPriors",
                        lambda fusedPairs, store, hints: fusedPairs)

    def rerankSpy(query, candidates, store):
        seen["head"] = [a for a, _ in candidates]
        return list(candidates)

    monkeypatch.setattr(engine, "rerank", rerankSpy)
    monkeypatch.setattr(engine, "assessTrust",
                        lambda pairs, hits, store, now: [
                            _result(a, 0.9, True, score=s) for a, s in pairs])
    monkeypatch.setattr(engine, "assemblePayload",
                        lambda store, results, tokenBudget: ("payload", 1, False))

    recall(store, {"memory": None, "code": None}, _BombEmbedder(),
           "irrelevant", k=10)
    head = seen["head"]
    # First four head slots alternate memory, code, memory, code.
    assert head[0] in mem and head[1] in code
    assert head[2] in mem and head[3] in code
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python3 -m pytest test/recall/test_engine.py::test_rerank_head_is_interleaved_across_classes -v`
Expected: FAIL. Before the engine change, `recall` takes a single `index` and builds `fused[:RERANK_HEAD]`, so either the signature rejects the dict or the head is memory-first (no alternation).

- [ ] **Step 3: Rewrite the engine `recall` body**

In `src/recall/engine.py`, add the strata import near the top imports:

```python
from recall.strata import classesForKinds, interleave, splitByClass
```

Replace the `recall` signature and body (keep `_emptyResult` and `_filterKinds` unchanged below). New signature and the changed sections:

```python
def recall(store, indexes, embedder, query, project=None, timeScope=None,
           kinds=None, k=10, tokenBudget=1500):
    """Run the full recall pipeline and assemble a tiered payload.

    ``store`` is the canonical store, ``indexes`` a ``{className: VectorIndex}``
    map (one built dense index per kind-class, from ``buildClassIndexes``), and
    ``embedder`` a loaded ``Embedder``. Candidates are generated PER CLASS (a
    kind-scoped bm25 list plus that class's own dense index), so the large code
    corpus can no longer starve the smaller reasoning memory out of the pool.
    Fusion, the facet boost, and priors run globally as before; the rerank head is
    then filled by round-robin across the per-class fused rankings, so the
    cross-encoder scores a fair mix and its scores decide the final order. Options:

    - ``project``: restrict recall to that project's live atoms (facet filter).
    - ``timeScope`` ``(startUnix, endUnix)``: restrict to atoms whose effective
      time falls in the inclusive window (applied once, in the priors stage).
    - ``kinds``: restrict to these atom kinds. Stratification runs over only the
      classes overlapping ``kinds`` (a single-class request degrades to today's
      non-stratified behavior).
    - ``k``: number of results to return (default 10).
    - ``tokenBudget``: payload budget by the conservative heuristic (default 1500).

    Returns the ``RecallResult`` dict ``{results, payload, tokensUsed,
    lowConfidence}``.
    """
    now = int(time.time())

    if not query or not query.strip():
        return _emptyResult(store, tokenBudget)

    facetHints = {"query": query}
    if project is not None:
        facetHints["project"] = project
    facet = facetSignal(store, facetHints)
    boostSet = facet["boostSet"]
    filterSet = facet["filterSet"]

    if filterSet is not None and not filterSet:
        return _emptyResult(store, tokenBudget)

    # 2. Per-class candidate generation. classesForKinds(None) is every class;
    #    a kinds subset narrows to the overlapping classes. No class -> sentinel.
    classes = classesForKinds(kinds)
    if not classes:
        return _emptyResult(store, tokenBudget)
    classNames = [name for name, _ in classes]

    bmHits = []
    dnHits = []
    for name, classKinds in classes:
        bmHits.extend(bm25(store, query, _SIGNAL_K, kinds=classKinds))
        classIndex = indexes.get(name)
        if classIndex is not None:
            dnHits.extend(dense(classIndex, embedder, query, _SIGNAL_K))

    signalHits = {}
    for atomId, _ in bmHits:
        signalHits.setdefault(atomId, set()).add("bm25")
    for atomId, _ in dnHits:
        signalHits.setdefault(atomId, set()).add("dense")
    for atomId in boostSet:
        signalHits.setdefault(atomId, set()).add("facet")

    if filterSet is not None:
        bmHits = [pair for pair in bmHits if pair[0] in filterSet]
        dnHits = [pair for pair in dnHits if pair[0] in filterSet]

    fused = rrf([bmHits, dnHits])

    if boostSet:
        fused = [
            (atomId, score * FACET_BOOST if atomId in boostSet else score)
            for atomId, score in fused
        ]

    priorHints = {"now": now}
    if timeScope is not None:
        priorHints["timeScope"] = timeScope
    fused = applyPriors(fused, store, priorHints)

    # kinds filter before the head is built: the per-class dense index for a class
    # returns every kind in that class, so an explicit narrow kinds request (e.g.
    # only 'narrative') still needs this to drop the other in-class kinds.
    if kinds is not None:
        fused = _filterKinds(store, fused, kinds)

    if not fused:
        return _emptyResult(store, tokenBudget)

    # 8. Stratified rerank head: round-robin across the per-class fused rankings so
    #    the cross-encoder scores a fair mix of both classes, then let its scores
    #    decide. The untouched fused tail keeps global RRF order below the head.
    perClass = splitByClass(fused, store, classNames)
    rerankHead = interleave([perClass[name] for name in classNames], RERANK_HEAD)
    rerankedHead = rerank(query, rerankHead, store)
    rerankedIds = {atomId for atomId, _ in rerankedHead}
    headIds = {atomId for atomId, _ in rerankHead}
    reranked = (
        rerankedHead
        + [pair for pair in rerankHead if pair[0] not in rerankedIds]
        + [pair for pair in fused if pair[0] not in headIds]
    )

    assessed = assessTrust(reranked, signalHits, store, now)

    results = assessed[:k]
    payload, tokensUsed, lowConfidence = assemblePayload(store, results, tokenBudget)
    return {
        "results": results,
        "payload": payload,
        "tokensUsed": tokensUsed,
        "lowConfidence": lowConfidence,
    }
```

Update the module docstring's pipeline sketch (top of file) so the `signals -> ... -> tiered payload` line reads `per-class signals -> RRF fusion -> facet boost -> priors -> kinds filter -> round-robin rerank head -> cross-encoder rerank -> trust -> tiered payload`. Keep it one honest sentence per the house style.

- [ ] **Step 4: Update the existing engine tests to the new signature**

In `test/recall/test_engine.py`:

Change the `_index` fixture helper and any full-pipeline test that calls `recall(store, index, ...)`. The `_index(store, embedder)` helper returns a single `FlatIndex`; add a parallel `_indexes(store, embedder)` returning `buildClassIndexes` and switch the full-pipeline (real-model) tests to it:

```python
def _indexes(store, embedder):
    embedMissing(store, embedder)
    return buildClassIndexes(store, MODEL_ID)
```

In `_stub_recall_dependencies`, update the `bm25` and `dense` stub signatures to match the new call shapes (bm25 now receives `kinds`, dense is called per class):

```python
    monkeypatch.setattr(engine, "bm25", lambda store, query, k, kinds=None: [])
    monkeypatch.setattr(engine, "dense", lambda index, embedder, query, k: [])
```

Every stub-based test that calls `recall(store, <index>, embedder, ...)` must pass an `indexes` dict instead. For stub tests (dense is stubbed and ignores its index arg) pass `{"memory": _SENTINEL, "code": _SENTINEL}` where `_SENTINEL = object()`, so `indexes.get(name)` is truthy and the stubbed `dense` is still exercised per class. For real-model full-pipeline tests, pass `_indexes(store, embedder)`.

Grep the file for `recall(` and fix each call site accordingly:

Run: `rg -n 'recall\(store,' test/recall/test_engine.py`

For each hit, replace the second positional arg (the old single index) with the appropriate `indexes` dict per the rule above.

- [ ] **Step 5: Run the full engine test file to verify it passes**

Run: `python3 -m pytest test/recall/test_engine.py -v`
Expected: PASS, including the three new behavioral tests and every migrated existing test.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/recall/engine.py test/recall/test_engine.py
git commit -m "$(cat <<'EOF'
feat(recall): stratified candidate generation and rerank head

recall() now takes a per-class indexes dict, generates candidates per
kind-class (kind-scoped bm25 + that class's dense index), and fills the
rerank head by round-robin across per-class fused rankings so the
cross-encoder scores a fair memory/code mix and sorts on relevance. Fixes
unscoped recall returning only code. Fusion/priors/trust/payload unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 5: Serving wiring (ServeContext + handlers)

Make the daemon hold per-class indexes and pass them through the three serving recall call sites.

**Files:**
- Modify: `src/serve/mcp.py` (`ServeContext.__init__`, `ServeContext.reindex`, `handle_pensive_recall`, `handle_recall`, and the `from recall.vector_index import ...` line)
- Modify: `src/serve/shadow.py` (the `recall(...)` call around line 160)
- Test: `test/serve/test_mcp.py` (add a handler-level test)

**Interfaces:**
- Consumes: `buildClassIndexes` (Task 3); `recall(store, indexes, embedder, ...)` (Task 4).
- Produces: `ServeContext.indexes` (dict, replaces `ServeContext.index`), rebuilt by `reindex()`.

- [ ] **Step 1: Write the failing test**

Add to `test/serve/test_mcp.py` (follow the file's existing ServeContext construction pattern; it builds a store, an `Embedder`, and a `ServeContext`):

```python
def test_serve_context_holds_class_indexes_and_recall_prefers_memory(tmp_path):
    from recall.embedder import Embedder
    from serve.mcp import ServeContext, handle_recall
    from store.store import openStore, putAtom

    store = openStore(tmp_path / "mem.db")
    try:
        for i in range(30):
            putAtom(store, {
                "text": f"def route_{i}(r): return dispatch(r, {i})",
                "kind": "document_chunk", "project": "aegis",
                "importance": 0.0, "provenance": {"source": "bulk-import"},
            })
        putAtom(store, {
            "text": "we chose Authelia forward-auth as the default web perimeter",
            "kind": "atom", "project": "aegis",
            "importance": 0.0, "provenance": {"source": "claude-code"},
        })
        ctx = ServeContext(store, Embedder("BAAI/bge-small-en-v1.5"),
                           "BAAI/bge-small-en-v1.5")
        assert set(ctx.indexes.keys()) == {"memory", "code"}
        out = handle_recall(ctx, {"query": "what did we choose for web auth",
                                  "k": 3})
        assert "Authelia" in out
    finally:
        store.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest test/serve/test_mcp.py::test_serve_context_holds_class_indexes_and_recall_prefers_memory -v`
Expected: FAIL with `AttributeError: 'ServeContext' object has no attribute 'indexes'`.

- [ ] **Step 3: Update the serving code**

In `src/serve/mcp.py`:

Change the import line 46:

```python
from recall.vector_index import buildClassIndexes
```

In `ServeContext.__init__`, replace `self.index = FlatIndex()` with:

```python
        self.indexes = {}
```

In `ServeContext.reindex`, replace `self.index = selectIndex(self.store, self.modelId)` with:

```python
        self.indexes = buildClassIndexes(self.store, self.modelId)
```

Update the `ServeContext` docstring's index sentence to say it holds one dense index per kind-class (from `buildClassIndexes`), rebuilt by `reindex`, so memory and code are searched from separate pools.

In `handle_pensive_recall` (around line 315), change the recall call:

```python
    out = recall(
        ctx.store, ctx.indexes, ctx.embedder, query,
        project=project, k=limit, tokenBudget=ctx.defaultTokenBudget,
    )
```

In `handle_recall` (around line 384), change the recall call:

```python
    out = recall(
        ctx.store, ctx.indexes, ctx.embedder, query,
        project=project, timeScope=timeScope, kinds=kinds,
        k=k, tokenBudget=tokenBudget,
    )
```

In `src/serve/shadow.py` (around line 160), change `ctx.index` to `ctx.indexes` in the recall call:

```python
        out = recall(
            ctx.store, ctx.indexes, ctx.embedder, query,
```

- [ ] **Step 4: Run the serving tests to verify they pass**

Run: `python3 -m pytest test/serve/ -v`
Expected: PASS. Any existing test referencing `ctx.index` must be updated to `ctx.indexes`; grep first: `rg -n 'ctx\.index|\.index\b' test/serve/`.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/pensive/daemon
git add src/serve/mcp.py src/serve/shadow.py test/serve/test_mcp.py
git commit -m "$(cat <<'EOF'
feat(serve): hold per-class dense indexes and pass through recall

ServeContext.index becomes ServeContext.indexes (buildClassIndexes),
rebuilt in reindex(); the pensive_recall, recall, and shadow call sites
pass the dict. Memory and code are now searched from separate pools.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

### Task 6: Eval wiring and the no-regression gate

Update the two eval callers to the new signature and run the gate to prove the pure-code corpus (single class) is not regressed by stratification.

**Files:**
- Modify: `eval/gate.py` (`gate` signature, the `recall(...)` call at line 141, and the `FlatIndex().build` at line 337)
- Modify: `eval/assoc_experiment.py` (the `FlatIndex().build` at line 649 and the recall lambda at line 480)
- Test: `test/recall/test_gate_smoke.py` (create: a fast smoke test that exercises `gate` end to end on a tiny store, no GPU-heavy 1500-query run)

**Interfaces:**
- Consumes: `buildClassIndexes` (Task 3); `recall(store, indexes, ...)` (Task 4).
- Produces: `gate(store, indexes, embedder, queries, recallK=_RECALL_K)` (the second positional arg is now the indexes dict).

- [ ] **Step 1: Write the failing smoke test**

Create `test/recall/test_gate_smoke.py`:

```python
"""Fast smoke test that the eval gate runs against the new recall signature.

The full 1,500-query gate needs the ChatGPT export and heavy GPU; this proves
gate() is wired to the stratified recall (indexes dict) on a tiny store so a
subagent can verify without the big run. The full-gate command is in the plan.
"""
import pytest

from recall.embedder import Embedder, embedMissing
from recall.vector_index import buildClassIndexes
from store.store import openStore, putAtom
import eval.gate as gate_mod

MODEL_ID = "BAAI/bge-small-en-v1.5"

pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(MODEL_ID)


def test_gate_runs_with_indexes_dict(tmp_path, embedder):
    store = openStore(tmp_path / "mem.db")
    try:
        ids = []
        for i in range(6):
            ids.append(putAtom(store, {
                "text": f"document chunk number {i} about routing and dispatch",
                "kind": "document_chunk", "project": "aegis",
                "importance": 0.0, "provenance": {"source": "bulk-import"},
            }))
        embedMissing(store, embedder)
        indexes = buildClassIndexes(store, MODEL_ID)
        queries = [{"query": "routing and dispatch",
                    "relevant": set(), "own": set()}]
        out = gate_mod.gate(store, indexes, embedder, queries)
        # Metrics dict is well formed and the run did not raise.
        assert "r_at_10" in out and "n" in out and out["n"] == 1
    finally:
        store.close()
```

Note: if `import eval.gate` does not resolve under the repo's test path config, import via the path the other eval-touching tests use; check `rg -n "import.*gate" test/` first and mirror it.

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `python3 -m pytest test/recall/test_gate_smoke.py -v`
Expected: FAIL. `gate` still calls `recall(store, index, embedder, ...)` expecting a single index, so passing the dict path is not yet wired (or the signature mismatch surfaces downstream).

- [ ] **Step 3: Update the eval callers**

In `eval/gate.py`:

Add the import near the existing `from recall.vector_index import FlatIndex` line:

```python
from recall.vector_index import FlatIndex, buildClassIndexes  # noqa: E402
```

Change the `gate` signature and its recall call. Rename the second parameter to `indexes` and pass it through:

```python
def gate(store, indexes, embedder, queries, recallK=_RECALL_K):
```

and at the recall call (was line 141):

```python
        result = recall(store, indexes, embedder, q["query"], k=recallK)
```

Update the `gate` docstring's "fixed interface core is `gate(store, index, embedder)`" line to `gate(store, indexes, embedder)`.

At the backfill index build (was line 337), replace:

```python
    index = buildClassIndexes(store, MODEL_ID)
    return store, index, stats
```

Any caller in `gate.py` that unpacked `store, index, stats` and later called `gate(store, index, ...)` now passes the dict unchanged (the variable is still named `index` at those sites or can be renamed; the value is a dict and `gate` treats it as `indexes`). Grep and verify: `rg -n 'gate\(|runChat|runAtoms|index' eval/gate.py`.

In `eval/assoc_experiment.py`:

At the index build (was line 649), replace `FlatIndex().build(store, MODEL_ID)` with:

```python
    index = buildClassIndexes(store, MODEL_ID)
```

The recall lambda (was line 480) `lambda query, k: gate_mod.recall(store, index, embedder, query, k=k)` now passes the dict as the second arg unchanged; it is already correct once `index` is the dict.

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `python3 -m pytest test/recall/test_gate_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full daemon test suite**

Run: `python3 -m pytest test/ -q`
Expected: PASS (no regressions across recall, serve, eval). Record the exact pass/fail tally to a file before claiming green:

```bash
python3 -m pytest test/ -q | tail -5 | tee /tmp/claude-1000/-home-aegis/d0df4bef-0c98-4044-8936-8f41bc18123e/scratchpad/stratified-test-tally.txt
```

- [ ] **Step 6: Run the no-regression gate (documented; needs the ChatGPT export + GPU)**

The gate corpus is 100% `document_chunk` (a single class), so stratification must be a no-op there and R@10 must stay at the BASELINE_V3 value of 0.680. Run it if the export dir is present:

```bash
cd ~/Projects/pensive/daemon
python3 eval/gate.py --corpus chat --queries 1500
```

Expected: `v3 recall (full pipeline)` R@10 within noise of 0.680 (BASELINE_V3.md). If R@10 drops materially, the stratification changed single-class behavior and Task 4's head construction must be re-checked (the single-class interleave must equal `fused[:RERANK_HEAD]`). If the export dir is not available on this machine, note that in the commit body and leave the smoke test as the automated proof.

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/pensive/daemon
git add eval/gate.py eval/assoc_experiment.py test/recall/test_gate_smoke.py
git commit -m "$(cat <<'EOF'
feat(eval): wire gate and assoc_experiment to stratified recall

gate() takes the per-class indexes dict; both eval harnesses build indexes
via buildClassIndexes. Smoke test proves the gate runs against the new
signature; the full 1500-query gate is the single-class no-regression check
(R@10 must stay ~0.680, since the chat corpus is all document_chunk).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016EoCM1vfsbmuVNCqMwRjnK
EOF
)"
```

---

## Post-implementation: restart the live daemon

After all tasks pass, the running Pensive v3 daemon must be restarted to load the new code and rebuild per-class indexes over the live store. Do NOT kill it without a flush per the process-management rule. Identify it (`ps`/systemctl user unit), send SIGINT, wait for graceful exit, then restart via its normal supervisor. Confirm a post-restart unscoped recall (for example "what am I working on") returns memory atoms, not code chunks. This step is operational, not part of the test cycle, and is done last.

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-08-stratified-recall-design.md`):
- Per-class bm25: Task 2. Partitioned dense indexes (memory flat / code HNSW): Task 3. Round-robin stratified rerank head: Task 4. Fusion/priors/trust/payload unchanged: Task 4 (preserved sections). Explicit `kinds` still honored: Task 4 (`_filterKinds` kept + `classesForKinds` narrowing). Tunable constants (`KIND_CLASSES`, `_SIGNAL_K`, `RERANK_HEAD`): Tasks 1 and 4. Validation via `gate.py` + two probe queries: Task 6 (gate) and Task 4 (the memory-intent and code-intent behavioral tests are the two probes). No stored-data mutation: every task uses tmp stores; live restart is operational only.
- Gap check: the spec's "interleave ratio default 1:1" is realized by `interleave` taking one item per class per round (Task 1). No spec requirement is left without a task.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows full code; every test step shows the test; every command shows expected output.

**Type consistency:** `recall(store, indexes, embedder, ...)` (Task 4) matches the dict produced by `buildClassIndexes` (Task 3) and passed by `ServeContext.indexes` (Task 5) and `gate(store, indexes, ...)` (Task 6). `bm25(..., kinds=None)` (Task 2) matches the engine's per-class call (Task 4). `classesForKinds`/`interleave`/`splitByClass` signatures (Task 1) match their engine call sites (Task 4). `selectIndex(store, modelId, kinds=None)` (Task 3) stays backward compatible for untouched callers per the Global Constraints.
