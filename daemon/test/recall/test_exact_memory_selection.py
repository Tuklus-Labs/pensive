"""Routing tests for the bounded exact memory index policy.

Risk rows map to ``RISK_MODEL_EXACT_MEMORY_SELECTION.md``.
"""
import time

import numpy as np
import pytest

import recall.vector_index as vector_index
from recall.embedder import vecToBlob
from recall.hnsw_index import HnswIndex
from recall.strata import CODE_KINDS, MEMORY_KINDS
from recall.vector_index import (
    EXACT_MEMORY_FLAT_MAX,
    HNSW_THRESHOLD,
    FlatIndex,
    buildClassIndexes,
    exactSearchLimit,
    selectIndex,
)
from store.store import openStore, putAtom


MODEL_ID = "BAAI/bge-small-en-v1.5"


class _CountResult:
    def __init__(self, count):
        self._count = count

    def fetchone(self):
        return (self._count,)

    def fetchall(self):
        return []


class _CountConnection:
    def __init__(self, counts):
        self._counts = counts

    def execute(self, sql, params):
        scope = frozenset(params[1:])
        count = self._counts(scope) if callable(self._counts) else self._counts
        return _CountResult(count)


class _CountStore:
    def __init__(self, counts):
        self._conn = _CountConnection(counts)


@pytest.fixture
def store(tmp_path):
    value = openStore(tmp_path / "exact-memory.db")
    try:
        yield value
    finally:
        value.close()


def _unit(seed, dim=16):
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _putEmbedded(store, text, vector):
    atomId = putAtom(store, {
        "text": text,
        "kind": "atom",
        "project": "selector-test",
        "provenance": {"source": "codex-test"},
    })
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        (atomId, MODEL_ID, vecToBlob(vector), int(time.time())),
    )
    store._conn.commit()
    return atomId


def test_complete_memory_class_switches_to_hnsw_above_exact_ceiling():
    """I1/I2/S1/B1/B2: the exact-memory ceiling has both sides pinned."""
    atCeiling = selectIndex(
        _CountStore(32_000), MODEL_ID, kinds=MEMORY_KINDS)
    aboveCeiling = selectIndex(
        _CountStore(32_001), MODEL_ID, kinds=MEMORY_KINDS)

    assert isinstance(atCeiling, FlatIndex), (
        "exact-memory boundary invariant violated: the complete memory class "
        f"at 32000 vectors selected {type(atCeiling).__name__}")
    assert isinstance(aboveCeiling, HnswIndex), (
        "exact-memory boundary invariant violated: the complete memory class "
        f"at 32001 vectors selected {type(aboveCeiling).__name__}")


def test_exact_search_limit_matches_selector_boundaries():
    """I5/X6: incremental maintenance reads the selector's transition limit."""
    assert exactSearchLimit(MEMORY_KINDS) == EXACT_MEMORY_FLAT_MAX == 32_000, (
        "memory transition-limit invariant violated: complete memory must expose "
        f"32000, helper={exactSearchLimit(MEMORY_KINDS)} "
        f"constant={EXACT_MEMORY_FLAT_MAX}")
    for kinds in (None, ("atom",), (*MEMORY_KINDS, "document_chunk"), CODE_KINDS):
        assert exactSearchLimit(kinds) == HNSW_THRESHOLD - 1 == 4_999, (
            "general transition-limit invariant violated: non-memory scope must "
            f"expose inclusive Flat maximum 4999, kinds={kinds!r} "
            f"helper={exactSearchLimit(kinds)}")


def test_flat_length_counts_retired_physical_rows():
    """I5/S3: retirement masks a row but does not reclaim its memory footprint."""
    index = FlatIndex()
    vectors = [_unit(seed) for seed in range(3)]
    for number, vector in enumerate(vectors):
        index.add(f"atom-{number}", vector)
    assert len(index) == 3, (
        "Flat physical-length invariant violated: three added rows must report "
        f"length three, length={len(index)} atomIds={index._atomIds!r}")
    assert index.remove("atom-1") is True, (
        "Flat retirement setup invariant violated: present middle row did not "
        "report successful removal")
    assert len(index) == 3, (
        "Flat physical-length invariant violated: retiring a masked row must not "
        f"shrink storage, length={len(index)} retired={index._retired!r}")


@pytest.mark.parametrize("kinds", [
    None,
    ("atom",),
    (*MEMORY_KINDS, "document_chunk"),
    CODE_KINDS,
])
def test_exact_memory_policy_does_not_broaden_to_other_scopes(kinds):
    """I3/X2: only the complete memory class receives the bounded exception."""
    selected = selectIndex(_CountStore(21_252), MODEL_ID, kinds=kinds)

    assert isinstance(selected, HnswIndex), (
        "exact-memory scope invariant violated: a non-memory-class selection "
        f"used {type(selected).__name__}; kinds={kinds!r}")


def test_build_class_indexes_routes_memory_to_flat_and_code_to_hnsw():
    """I3/X1: the measured class sizes route to exact memory and HNSW code."""
    counts = {
        frozenset(MEMORY_KINDS): 21_252,
        frozenset(CODE_KINDS): 282_985,
    }
    indexes = buildClassIndexes(
        _CountStore(lambda scope: counts[scope]), MODEL_ID)

    assert isinstance(indexes["memory"], FlatIndex), (
        "class-routing invariant violated: 21252-vector memory class selected "
        f"{type(indexes['memory']).__name__}")
    assert isinstance(indexes["code"], HnswIndex), (
        "class-routing invariant violated: 282985-vector code class selected "
        f"{type(indexes['code']).__name__}")


def test_hnsw_factory_routes_optional_cache_directory(tmp_path, monkeypatch):
    """X5: selector factories pass cache configuration only to HNSW builds."""
    calls = []

    def recordBuild(self, store, modelId, kinds=None, cacheDir=None):
        calls.append((modelId, tuple(kinds) if kinds is not None else None, cacheDir))
        return self

    monkeypatch.setattr(HnswIndex, "build", recordBuild)
    cache = tmp_path / "index-cache"
    selected = selectIndex(
        _CountStore(32_001), MODEL_ID, kinds=MEMORY_KINDS, cacheDir=cache)
    assert isinstance(selected, HnswIndex), (
        "cache-routing setup invariant violated: above-ceiling memory did not "
        f"select HNSW, selected={type(selected).__name__}")
    assert calls == [(MODEL_ID, tuple(MEMORY_KINDS), cache)], (
        "selector cache-routing invariant violated: direct HNSW build did not "
        f"receive the cache directory exactly once, calls={calls!r}")

    calls.clear()
    counts = {
        frozenset(MEMORY_KINDS): 21_252,
        frozenset(CODE_KINDS): 282_985,
    }
    indexes = buildClassIndexes(
        _CountStore(lambda scope: counts[scope]), MODEL_ID, cacheDir=cache)
    assert isinstance(indexes["memory"], FlatIndex), (
        "class cache-routing invariant violated: bounded memory must remain Flat, "
        f"selected={type(indexes['memory']).__name__}")
    assert calls == [(MODEL_ID, tuple(CODE_KINDS), cache)], (
        "class cache-routing invariant violated: only code HNSW should receive "
        f"the configured cache directory, calls={calls!r}")


def test_selected_exact_memory_index_preserves_scope_add_and_remove(
        store, monkeypatch):
    """I4/S2/X3: selected Flat memory keeps the full VectorIndex contract."""
    vectors = [_unit(seed) for seed in range(6)]
    atomIds = [
        _putEmbedded(store, f"memory {i}", vector)
        for i, vector in enumerate(vectors)
    ]
    monkeypatch.setattr(vector_index, "HNSW_THRESHOLD", 1)
    index = selectIndex(store, MODEL_ID, kinds=MEMORY_KINDS)

    assert isinstance(index, FlatIndex), (
        "exact-memory selection invariant violated: a full memory class below "
        f"the ceiling selected {type(index).__name__}")
    allowed = {atomIds[1], atomIds[4]}
    scoped = [atomId for atomId, _score in index.search(
        vectors[0], k=10, allowedIds=allowed)]
    assert set(scoped) == allowed, (
        "exact-memory scope invariant violated: allowed-ID search returned "
        f"{scoped!r} for allowed={sorted(allowed)!r}")

    lateVector = _unit(99)
    index.add("late", lateVector)
    assert index.search(lateVector, 1)[0][0] == "late", (
        "exact-memory add invariant violated: an incrementally added atom did "
        f"not rank first; hits={index.search(lateVector, 3)!r}")
    assert index.remove("late") is True, (
        "exact-memory remove invariant violated: removing a present atom must "
        "report success")
    after = [atomId for atomId, _score in index.search(lateVector, 10)]
    assert "late" not in after, (
        "exact-memory retirement invariant violated: removed atom remains "
        f"recallable; hits={after!r}")
