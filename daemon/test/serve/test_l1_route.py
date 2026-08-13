"""Gate for the L1 identity tier: retrieval by NAME, never by similarity.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    Asking for an atom by its id returns exactly that atom, asking for an id
    that cannot exist returns nothing, and neither answer involves ranking, a
    model, or a score. The two answers are DISTINGUISHABLE, which is the whole
    point: a tier that returns the same response for present and absent cannot
    be used to look anything up.

Why this tier exists at all. `Aegis/CLAUDE.md:113` specifies an L1 hot tier and
the v3 rewrite dropped it, leaving one monolithic recall path where every caller
pays for a cross-encoder. Two live consequences, both measured 2026-08-12:

  * Charon answered a SET-MEMBERSHIP question ("which fragments belong to
    session <uuid>?") with a semantic top-k recall, producing 91.4% of all
    traffic in the store's history and retrieving its own source code, because
    the best lexical match for "narrative_fragment session" in a corpus holding
    Charon's source IS Charon's source.
  * Standing rules cannot be served, because serving a rule by similarity means
    the rule has to win a ranking contest to reach the agent, and a rule that
    might not arrive is not a rule.

Budget is P95 <= 1ms client-observed, which is why this is a plain route and not
an MCP tool: the MCP tools/call envelope alone measured 2.486ms P95 for a
142-byte error touching no store, while the bare HTTP route floor is 0.297ms.
No amount of optimizing the handler would have reached the budget through MCP.
"""
import pytest

from serve.l1 import handleGet, handleLookup, MAX_ID_LEN, MAX_LOOKUP_LIMIT
from store.store import openStore, putAtom, addFacet, supersede

IMPOSSIBLE_ID = "01ZZZZZZZZZZZZZZZZZZZZZZZZ"


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text="an atom", kind="atom", project="aegis"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": project, "importance": 0.0,
        "provenance": {"source": "explicit-emit", "agent": "heph"},
    })


# --------------------------------------------------------------------------- #
# the discriminating pair
# --------------------------------------------------------------------------- #


def test_a_present_atom_comes_back_whole(store):
    atomId = _put(store, "the body that must round-trip")
    status, payload = handleGet(store, atomId)
    assert status == 200
    assert payload["id"] == atomId
    assert payload["text"] == "the body that must round-trip"
    assert payload["kind"] == "atom"


def test_an_impossible_id_is_a_clean_absence(store):
    """404, not an error and not a result. A route that 500s on a bad id is not
    discriminating, and a route that answers 404 for BOTH present and absent ids
    (which is what "no route at all" looks like) proves nothing either."""
    _put(store)
    status, payload = handleGet(store, IMPOSSIBLE_ID)
    assert status == 404
    assert payload.get("found") is False


def test_present_and_absent_are_distinguishable(store):
    """The pair, asserted together. This is the property the gate's canary needs
    and could not have while the route was missing."""
    atomId = _put(store)
    present, _ = handleGet(store, atomId)
    absent, _ = handleGet(store, IMPOSSIBLE_ID)
    assert present == 200 and absent == 404, (
        f"present={present} absent={absent}: the tier cannot tell them apart"
    )


# --------------------------------------------------------------------------- #
# identity retrieval is not ranked retrieval
# --------------------------------------------------------------------------- #


def test_no_score_confidence_or_ranking_appears_in_an_l1_answer(store):
    """L1 returns a row, not a judgement. A score here would invite a caller to
    threshold on it, which re-introduces the ranking contest this tier exists to
    avoid."""
    atomId = _put(store)
    _, payload = handleGet(store, atomId)
    for forbidden in ("score", "confidence", "shouldTrust", "why", "rank"):
        assert forbidden not in payload, f"L1 leaked a ranking field: {forbidden}"


def test_a_superseded_atom_is_returned_with_its_status(store):
    """Identity retrieval answers "what is this id", which is a different
    question from "what should you believe". A superseded atom still EXISTS, so
    L1 returns it and says so rather than hiding it; the trust layer is what
    withholds belief, and duplicating that judgement here would put two
    mechanisms on one question."""
    old = _put(store, "the old body")
    new = _put(store, "the new body")
    supersede(store, old, new, {"source": "explicit-emit"})
    status, payload = handleGet(store, old)
    assert status == 200
    assert payload["status"] == "superseded"


# --------------------------------------------------------------------------- #
# lookup by facet: the operation Charon actually needed
# --------------------------------------------------------------------------- #


def test_lookup_returns_ids_for_an_exact_facet_match(store):
    a = _put(store, "first")
    b = _put(store, "second")
    _put(store, "unrelated")
    addFacet(store, a, "tag", "session-abc")
    addFacet(store, b, "tag", "session-abc")

    status, payload = handleLookup(store, "tag", "session-abc")
    assert status == 200
    assert set(payload["ids"]) == {a, b}


def test_lookup_for_a_value_that_does_not_exist_is_empty_not_nearest(store):
    """The exact defect that made Charon retrieve its own source: a similarity
    search cannot report absence, so an empty set came back as nearest
    neighbours. An exact lookup CAN report absence, and must."""
    a = _put(store)
    addFacet(store, a, "tag", "session-abc")
    status, payload = handleLookup(store, "tag", "session-does-not-exist")
    assert status == 200
    assert payload["ids"] == []


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #


def test_an_overlong_id_is_refused_without_touching_the_store(store):
    status, payload = handleGet(store, "x" * (MAX_ID_LEN + 1))
    assert status == 400


def test_lookup_limit_is_capped(store):
    ids = [_put(store, f"atom {i}") for i in range(5)]
    for i in ids:
        addFacet(store, i, "tag", "many")
    status, payload = handleLookup(store, "tag", "many", limit=10_000)
    assert status == 200
    assert payload["limit"] <= MAX_LOOKUP_LIMIT


def test_a_blank_id_is_refused(store):
    status, _ = handleGet(store, "")
    assert status == 400
