"""Gate for the hub-entity ceiling in relatedMemory.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    An entity value shared by a large fraction of the store is EXCLUDED from the
    related-memory query, not merely down-weighted inside it. Rare shared
    entities still decide the ranking, and a chunk whose only entities are hubs
    returns nothing rather than scanning the store to earn a score near zero.

Why. `relatedMemory` scores candidates by sum(1/freq) over shared entity facets,
and its docstring says hub entities "contribute almost nothing". That is true of
the SCORE and false of the COST: a hub value is still joined, grouped and summed
before being down-weighted. Measured on the live store, 2026-08-13:

    getAtom (+ all provenance)      0.01 ms
    _sourceRef                      0.01 ms
    _edgeRelations                  0.08 ms
    relatedMemory (fallback)     1182.23 ms   <- the entire enrichment cost
    Enricher.lines total         1185.49 ms

Only 2,810 `relates` edges exist across ~283k live chunks, so nearly every chunk
takes this fallback, and L3 with enrichment measured 2,323 ms p50 against a
125 ms budget.

This is the same defect as the lexical one fixed earlier the same night: a term
whose score contribution is near zero dominating the work because it is filtered
AFTER the join instead of before it. Both are fixed the same way, and the shape
is worth recognizing on sight.
"""
import pytest

from recall.enrich import relatedMemory, HUB_ENTITY_MAX_FREQ
from store.store import openStore, putAtom, addFacet


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _atom(store, text, kind="atom"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "p", "importance": 0.0,
        "provenance": {"source": "explicit-emit"},
    })


def test_a_rare_shared_entity_still_relates(store):
    chunk = _atom(store, "a chunk body", kind="document_chunk")
    mem = _atom(store, "the memory that shares a rare entity")
    addFacet(store, chunk, "entity", "phosphorescent-marker")
    addFacet(store, mem, "entity", "phosphorescent-marker")

    out = relatedMemory(store, chunk, limit=3)
    assert [m for m, _ in out] == [mem]


def test_a_hub_entity_is_excluded_not_merely_downweighted(store):
    """The hub is shared with far more atoms than the ceiling allows, so it must
    not reach the join at all. If it were only down-weighted, this chunk would
    still relate to every one of them."""
    chunk = _atom(store, "a chunk body", kind="document_chunk")
    hubMembers = [_atom(store, f"memory {i} carrying the hub entity")
                  for i in range(HUB_ENTITY_MAX_FREQ + 5)]
    addFacet(store, chunk, "entity", "hub-value")
    for m in hubMembers:
        addFacet(store, m, "entity", "hub-value")

    out = relatedMemory(store, chunk, limit=3)
    assert out == [], f"a hub entity produced relations: {out!r}"


def test_a_rare_entity_survives_alongside_a_hub(store):
    """The ceiling must drop only the hub, never the whole query: a chunk that
    shares one hub and one rare entity still relates through the rare one."""
    chunk = _atom(store, "a chunk body", kind="document_chunk")
    rareMate = _atom(store, "the memory sharing the rare entity")
    hubMembers = [_atom(store, f"memory {i}") for i in range(HUB_ENTITY_MAX_FREQ + 5)]
    addFacet(store, chunk, "entity", "hub-value")
    addFacet(store, chunk, "entity", "rare-value")
    addFacet(store, rareMate, "entity", "rare-value")
    for m in hubMembers:
        addFacet(store, m, "entity", "hub-value")

    out = relatedMemory(store, chunk, limit=3)
    assert [m for m, _ in out] == [rareMate]


def test_no_entities_returns_empty_without_error(store):
    chunk = _atom(store, "a chunk with no entity facets", kind="document_chunk")
    assert relatedMemory(store, chunk, limit=3) == []


def test_the_ceiling_is_a_named_constant(store):
    """Named so a drift is loud, and so the cost reasoning has somewhere to live."""
    assert isinstance(HUB_ENTITY_MAX_FREQ, int) and HUB_ENTITY_MAX_FREQ > 1
