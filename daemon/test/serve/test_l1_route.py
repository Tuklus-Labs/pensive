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
from store.migrate import CURRENT_SCHEMA_VERSION

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


# --------------------------------------------------------------------------- #
# lookup does no sorting, at any store size
# --------------------------------------------------------------------------- #
#
# RISK MODEL. /lookup is unauthenticated, synchronous, and runs INLINE on the
# daemon's event-loop thread -- the one thread that also serves /mcp, /get,
# /brief and every other route. Its query filters on (key, value) and then asks
# for ORDER BY f.atom_id. `idx_facets_kv` is on (key, value) ALONE, so it cannot
# produce that order, and SQLite answers by materializing every matching facet
# row into a temp B-tree and sorting it before the LIMIT is applied. The LIMIT
# bounds the RESPONSE. It does not bound the WORK.
#
# That makes the cost a function of how many rows share the requested value, and
# the caller picks the value. The live facets table holds ~1.25M rows and facet
# values are guessable rather than secret ('project'/'aegis', a session id, a
# tag), so a single request against a hot value stalls the whole daemon for
# everyone. Measured on a 1.25M-row fixture with 200k rows under one value:
# 58ms per lookup with the temp B-tree, 0.08ms with a covering (key, value,
# atom_id) index. This route's budget is P95 <= 1ms.
#
# THE INSTRUMENT, and why it is built this way: the plan is read off the
# statement the handler ACTUALLY executed, captured with `set_trace_callback`
# (which hands back the SQL with its parameters already bound). A test that
# re-typed the query into its own body would keep passing after the handler
# drifted away from it -- it would be measuring the test, not the daemon.
#
# The plan assertion alone is not enough, because there are two ways to make a
# temp B-tree disappear and only one of them is a fix: adding the index, or
# deleting the ORDER BY. `test_lookup_ids_come_back_in_a_stable_order` is what
# makes the second one fail, so the pair holds the real property -- sorted
# output, obtained without sorting.


def _plansOf(store, run):
    """Query plans of every SELECT ``run()`` issues, as flat detail strings.

    Traced rather than re-typed: `set_trace_callback` reports the expanded SQL
    of each statement as it executes, so the plan below belongs to the query the
    handler really ran, with the real parameter values the planner saw.
    """
    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        run()
    finally:
        store._conn.set_trace_callback(None)

    details = []
    for sql in statements:
        if not sql.lstrip().upper().startswith("SELECT"):
            continue
        details.extend(
            row[-1] for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert details, "no SELECT was traced; the instrument saw nothing to plan"
    return details


def test_lookup_does_not_sort_the_facets_it_matches(store):
    """The finding, stated as a plan assertion: no temp B-tree, at any size.

    A sort here is unbounded work bought with one unauthenticated request, on
    the thread that serves everything else.
    """
    ids = [_put(store, f"atom {i}") for i in range(3)]
    for atomId in ids:
        addFacet(store, atomId, "tag", "hot")

    plan = _plansOf(store, lambda: handleLookup(store, "tag", "hot"))

    assert not any("USE TEMP B-TREE" in line for line in plan), (
        "handleLookup sorts its matches in a temp B-tree: every row carrying "
        "the requested facet value is materialized and sorted BEFORE the LIMIT "
        "applies, so a hot value turns one request into unbounded work on the "
        "event-loop thread.\nplan:\n  " + "\n  ".join(plan)
    )


def test_lookup_reads_its_answer_out_of_a_covering_index(store):
    """The positive half: the ordering is SATISFIED by an index, not sorted.

    Asserting only the absence of the temp B-tree would also pass if the
    ordering had simply been dropped, or if the planner had switched to scanning
    the primary key in atom_id order -- which removes the sort but replaces it
    with a full table scan, and is measurably WORSE for the absent-value lookup
    this tier exists to answer.
    """
    ids = [_put(store, f"atom {i}") for i in range(3)]
    for atomId in ids:
        addFacet(store, atomId, "tag", "hot")

    plan = _plansOf(store, lambda: handleLookup(store, "tag", "hot"))

    assert any("COVERING INDEX idx_facets_kv_atom" in line for line in plan), (
        "the facet lookup is not being served by the covering "
        "(key, value, atom_id) index; it is reaching the answer some other way, "
        "and the other ways all scan.\nplan:\n  " + "\n  ".join(plan)
    )


def test_lookup_plan_holds_for_a_value_that_is_absent(store):
    """Absence is the operation this tier exists for, so it gets its own plan.

    The planner picks a plan per parameter VALUE once statistics exist, and the
    absent case is the one that goes wrong differently: a plan that terminates
    early on a hot value can still walk the entire facets table to prove a value
    is not there.
    """
    a = _put(store)
    addFacet(store, a, "tag", "hot")

    plan = _plansOf(
        store, lambda: handleLookup(store, "tag", "session-does-not-exist"))

    assert not any("USE TEMP B-TREE" in line for line in plan), (
        "proving a facet value ABSENT sorts: \n  " + "\n  ".join(plan))
    assert not any(line.startswith("SCAN f") for line in plan), (
        "proving a facet value ABSENT scans the whole facets table:\n  "
        + "\n  ".join(plan))


def test_lookup_ids_come_back_in_a_stable_order(store):
    """The property the ORDER BY is FOR, pinned so it cannot be traded away.

    There are two ways to make the temp B-tree disappear and only one of them
    is a fix: add the index, or delete the ORDER BY. This test exists to fail on
    the second one -- and the OUTPUT assertion alone cannot do it, which was
    caught by sabotage rather than by reading. With the covering index in place
    the rows already arrive in atom_id order, so deleting the ORDER BY leaves
    every observable answer here identical. The sabotage passed all fifteen
    tests in this file.

    So the ordering is asserted where it is actually guaranteed: in the
    statement. An index that happens to emit sorted rows is a plan, and plans
    change -- adding a column, running ANALYZE, or a future SQLite release can
    pick a different one, at which point a truncated lookup silently starts
    returning a different subset of the same set. The ORDER BY is what makes the
    contract survive that; the output check below is what makes the ORDER BY
    mean the right thing.
    """
    ids = [_put(store, f"atom {i}") for i in range(12)]
    for atomId in ids:
        addFacet(store, atomId, "tag", "many")

    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        _, payload = handleLookup(store, "tag", "many", limit=5)
    finally:
        store._conn.set_trace_callback(None)

    lookupSql = [s for s in statements if "FROM facets" in s]
    assert lookupSql, "no facet lookup was traced"
    assert all("ORDER BY" in s.upper() for s in lookupSql), (
        "the facet lookup no longer REQUESTS an order. The rows may still come "
        "back sorted today because the covering index emits them that way, but "
        "that is a query plan, not a contract: the day the planner picks "
        "differently, a truncated lookup starts returning a different subset "
        f"with no other symptom.\nsql: {lookupSql}")

    assert payload["ids"] == sorted(payload["ids"]), (
        f"lookup ids are not ordered: {payload['ids']}")
    assert payload["ids"] == sorted(ids)[:5], (
        "a truncated lookup did not return the first ids by atom_id, so WHICH "
        "ids a caller gets depends on the query plan rather than the contract")
    assert payload["truncated"] is True


def test_the_covering_index_reaches_an_existing_store_through_the_ladder(tmp_path):
    """An existing 1.6GB store is upgraded in place, not rebuilt.

    The index has to arrive through the forward-only migration ladder, because
    the stores that need it are the ones that already exist -- a fresh
    schema.sql only ever helps a database nobody has. Rewinding a real store to
    v2 and reopening exercises the actual ladder step rather than a copy of it.
    """
    dbPath = tmp_path / "mem.db"
    s = openStore(dbPath)
    try:
        assert CURRENT_SCHEMA_VERSION >= 3, (
            "adding an index to an existing store is a schema change and needs "
            "its own version; the ladder was not advanced")
        # Rewind to exactly what a pre-migration store looks like: v2 stamped,
        # index absent.
        s._conn.execute("DROP INDEX IF EXISTS idx_facets_kv_atom")
        s._setVersion(2)
        s._commit()
    finally:
        s.close()

    s2 = openStore(dbPath)                       # the ladder runs on open
    try:
        assert s2.schemaVersion() == CURRENT_SCHEMA_VERSION
        row = s2._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_facets_kv_atom",),
        ).fetchone()
        assert row is not None, (
            "reopening a v2 store did not add the covering facet index, so "
            "every store that already exists keeps the unbounded sort")
        assert "atom_id" in row[0]
    finally:
        s2.close()
