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
