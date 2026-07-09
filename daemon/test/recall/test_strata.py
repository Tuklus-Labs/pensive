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
