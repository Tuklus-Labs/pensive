"""Trust layer: honest confidence, shouldTrust, one-clause why, and the
supersession invariant.

Two things under test, both pure Python on the scoring side over a real store:

- ``assessTrust(reranked, signalHits, Store, now)`` annotates each reranked
  ``(atomId, score)`` with a ``confidence`` in [0,1] blending signal agreement,
  the top1-top2 rerank gap (disambiguation), and a temporal decay term; a
  ``shouldTrust`` boolean gated on ``TRUST_FLOOR``; and a one-clause plain-text
  ``why`` built from the actual evidence.

- The CRITICAL invariant: a ``superseded`` atom NEVER surfaces alone. If it
  appears at all it carries a ``supersededBy`` key pointing at the LIVE end of
  its supersession chain, with confidence capped below ``TRUST_FLOOR`` so
  ``shouldTrust`` is False. A chain that ends in a tombstone or a missing atom
  drops the entry entirely; a cycle raises.

No model load: reranked lists and signalHits are constructed by hand; the store
is a live SQLite db built by the fixture. signalHits is a dict
``atomId -> set(signal names)`` with names from {"bm25", "dense", "facet"}.
"""
import pytest

from recall.trust import (
    assessTrust,
    TRUST_FLOOR,
    SUPERSEDED_CONF_CAP,
)
from store.store import openStore, putAtom, supersede

# Fixed reference "now" so ages are deterministic regardless of the wall clock.
# Atoms whose age matters set occurred_at explicitly (COALESCE uses it), so the
# test never depends on created_at (stamped to the real clock).
NOW = 2_000_000_000
DAY = 86_400


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text="atom text", importance=0.0, occurredAt=NOW, project="aegis"):
    atomInput = {
        "text": text,
        "kind": "atom",
        "project": project,
        "importance": importance,
        "provenance": {"source": "claude-code"},
        "occurredAt": occurredAt,
    }
    return putAtom(store, atomInput)


def _tombstone(store, atomId):
    # Task 19 owns the lifecycle jobs that tombstone atoms; the trust tests only
    # need the resulting state, so set it directly on the real store.
    store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (atomId,))
    store._conn.commit()


def _by_id(out):
    return {row["atomId"]: row for row in out}


# --------------------------------------------------------------------------- #
# Step 1: agreement + a decisive gap outrank a single signal with a tight gap  #
# --------------------------------------------------------------------------- #

def test_dual_signal_decisive_gap_beats_single_signal_tight_gap(store):
    # Brief Step 1. Two scenarios, each with its own top-1:
    #   both/decisive: hit by bm25 AND dense, a large lead over the runner-up
    #   single/tight:  hit by bm25 only, a razor-thin lead over the runner-up
    # The dual-signal decisive-gap atom must earn strictly higher confidence.
    both = _put(store, "hit by both signals, clear winner")
    otherA = _put(store, "runner up A")
    single = _put(store, "hit by one signal, near tie")
    otherB = _put(store, "runner up B")

    dual = assessTrust(
        [(both, 5.0), (otherA, 1.0)],            # decisive 4.0 gap
        {both: {"bm25", "dense"}, otherA: {"bm25"}},
        store,
        NOW,
    )
    lone = assessTrust(
        [(single, 5.0), (otherB, 4.9)],          # tight 0.1 gap
        {single: {"bm25"}, otherB: {"dense"}},
        store,
        NOW,
    )

    cDual = _by_id(dual)[both]["confidence"]
    cLone = _by_id(lone)[single]["confidence"]
    assert cDual > cLone
    assert _by_id(dual)[both]["shouldTrust"] is True
    # both signals + a decisive gap read into the why verbatim
    assert "both signals agree" in _by_id(dual)[both]["why"]
    assert "decisive gap" in _by_id(dual)[both]["why"]


def test_dual_signal_scores_higher_than_single_at_equal_gap(store):
    # Isolate the agreement term: same gap, same recency; only the signal set
    # differs. Both-signals must out-confidence single-signal.
    both = _put(store, "both signals")
    single = _put(store, "one signal")
    tail = _put(store, "tail")

    outBoth = _by_id(assessTrust(
        [(both, 5.0), (tail, 4.0)], {both: {"bm25", "dense"}, tail: set()}, store, NOW
    ))[both]
    outSingle = _by_id(assessTrust(
        [(single, 5.0), (tail, 4.0)], {single: {"bm25"}, tail: set()}, store, NOW
    ))[single]
    assert outBoth["confidence"] > outSingle["confidence"]


def test_facet_adds_a_smaller_increment_than_a_second_primary(store):
    # A facet hit lifts confidence, but by less than adding the second primary
    # signal would. Pins the "facet increment is smaller" contract.
    base = _put(store, "bm25 only")
    withFacet = _put(store, "bm25 + facet")
    withDense = _put(store, "bm25 + dense")
    tail = _put(store, "tail")

    def conf(atomId, signals):
        return _by_id(assessTrust(
            [(atomId, 5.0), (tail, 4.0)], {atomId: signals, tail: set()}, store, NOW
        ))[atomId]["confidence"]

    cBase = conf(base, {"bm25"})
    cFacet = conf(withFacet, {"bm25", "facet"})
    cDense = conf(withDense, {"bm25", "dense"})
    assert cBase < cFacet < cDense


def test_recent_atom_outconfidences_an_old_one_all_else_equal(store):
    # The temporal term: identical signals and gap, differing only in age. Recent
    # beats old, and the why labels reflect it.
    recent = _put(store, "recorded now", occurredAt=NOW)
    old = _put(store, "recorded long ago", occurredAt=NOW - 800 * DAY)
    tail = _put(store, "tail")

    outR = _by_id(assessTrust(
        [(recent, 5.0), (tail, 4.0)], {recent: {"bm25"}, tail: set()}, store, NOW
    ))[recent]
    outO = _by_id(assessTrust(
        [(old, 5.0), (tail, 4.0)], {old: {"bm25"}, tail: set()}, store, NOW
    ))[old]
    assert outR["confidence"] > outO["confidence"]
    assert "recent" in outR["why"]
    assert "aged" in outO["why"]


def test_should_trust_gates_on_the_named_floor(store):
    # shouldTrust is exactly confidence >= TRUST_FLOOR. A no-signal, tail-position
    # atom lands below the floor; a both-signals recent top atom lands above.
    weak = _put(store, "weak")
    strong = _put(store, "strong")
    tail = _put(store, "tail")

    out = _by_id(assessTrust(
        [(strong, 5.0), (weak, 1.0), (tail, 0.5)],
        {strong: {"bm25", "dense"}, weak: set(), tail: set()},
        store,
        NOW,
    ))
    assert out[strong]["shouldTrust"] is (out[strong]["confidence"] >= TRUST_FLOOR)
    assert out[strong]["shouldTrust"] is True
    assert out[weak]["confidence"] < TRUST_FLOOR
    assert out[weak]["shouldTrust"] is False


# --------------------------------------------------------------------------- #
# Supersession invariant (CRITICAL)                                           #
# --------------------------------------------------------------------------- #

def _assert_no_orphan_superseded(store, out):
    """The invariant, checked against the live store: no output row is a
    superseded atom without a supersededBy key pointing at a LIVE atom."""
    for row in out:
        status = store._conn.execute(
            "SELECT status FROM atoms WHERE id = ?", (row["atomId"],)
        ).fetchone()[0]
        if status == "superseded":
            assert "supersededBy" in row, f"orphan superseded row: {row}"
            succStatus = store._conn.execute(
                "SELECT status FROM atoms WHERE id = ?", (row["supersededBy"],)
            ).fetchone()[0]
            assert succStatus == "live", f"supersededBy is not live: {row}"


def test_superseded_with_live_successor_surfaces_chained_and_untrusted(store):
    old = _put(store, "the old fact")
    new = _put(store, "the new fact")
    supersede(store, old, new, {"source": "claude-code"})

    out = assessTrust([(old, 5.0)], {old: {"bm25", "dense"}}, store, NOW)
    assert len(out) == 1
    row = out[0]
    assert row["atomId"] == old
    assert row["supersededBy"] == new
    assert row["shouldTrust"] is False
    assert row["confidence"] < TRUST_FLOOR
    assert row["confidence"] <= SUPERSEDED_CONF_CAP
    assert row["why"] == "superseded by newer atom"
    _assert_no_orphan_superseded(store, out)


def test_superseded_alone_with_tombstoned_successor_is_dropped(store):
    # The successor exists but has been tombstoned: the superseded atom has no
    # LIVE successor, so it must never surface -- dropped entirely.
    old = _put(store, "the old fact")
    new = _put(store, "the retracted new fact")
    supersede(store, old, new, {"source": "claude-code"})
    _tombstone(store, new)

    out = assessTrust([(old, 5.0)], {old: {"bm25", "dense"}}, store, NOW)
    assert out == []


def test_multi_hop_chain_resolves_to_the_live_end(store):
    # A superseded by B, B superseded by C, C live. The surviving row points at
    # C (the LIVE end), not B (itself superseded).
    a = _put(store, "fact v1")
    b = _put(store, "fact v2")
    c = _put(store, "fact v3")
    supersede(store, a, b, {"source": "claude-code"})
    supersede(store, b, c, {"source": "claude-code"})

    out = assessTrust([(a, 5.0)], {a: {"bm25"}}, store, NOW)
    assert len(out) == 1
    assert out[0]["atomId"] == a
    assert out[0]["supersededBy"] == c        # live end, not the superseded b
    assert out[0]["shouldTrust"] is False
    _assert_no_orphan_superseded(store, out)


def test_multi_hop_chain_ending_in_tombstone_is_dropped(store):
    a = _put(store, "fact v1")
    b = _put(store, "fact v2")
    c = _put(store, "fact v3, later retracted")
    supersede(store, a, b, {"source": "claude-code"})
    supersede(store, b, c, {"source": "claude-code"})
    _tombstone(store, c)

    out = assessTrust([(a, 5.0)], {a: {"bm25"}}, store, NOW)
    assert out == []


def test_supersession_cycle_raises_naming_the_atoms(store):
    # Canonical corruption: A supersedes B and B supersedes A. The walk must
    # detect the cycle and raise loudly, naming the atoms.
    a = _put(store, "atom a")
    b = _put(store, "atom b")
    supersede(store, a, b, {"source": "claude-code"})   # edge b->a, a superseded
    supersede(store, b, a, {"source": "claude-code"})   # edge a->b, b superseded

    with pytest.raises(ValueError) as exc:
        assessTrust([(a, 5.0)], {a: {"bm25"}}, store, NOW)
    msg = str(exc.value)
    assert "cycle" in msg
    assert a in msg and b in msg


def test_self_supersession_cycle_raises(store):
    # Degenerate cycle: an atom that supersedes itself.
    a = _put(store, "self-superseding atom")
    supersede(store, a, a, {"source": "claude-code"})

    with pytest.raises(ValueError) as exc:
        assessTrust([(a, 5.0)], {a: {"bm25"}}, store, NOW)
    assert "cycle" in str(exc.value)


def test_invariant_holds_across_a_mixed_result_set(store):
    # A realistic mix: two live atoms, one superseded-with-live-successor, one
    # superseded-with-tombstoned-successor, one tombstoned outright. The
    # invariant must hold and the drops must actually be dropped.
    live1 = _put(store, "live fact one")
    live2 = _put(store, "live fact two")
    supOld = _put(store, "superseded, has live successor")
    supNew = _put(store, "the successor")
    supersede(store, supOld, supNew, {"source": "claude-code"})
    deadOld = _put(store, "superseded, successor tombstoned")
    deadNew = _put(store, "tombstoned successor")
    supersede(store, deadOld, deadNew, {"source": "claude-code"})
    _tombstone(store, deadNew)
    tomb = _put(store, "tombstoned outright")
    _tombstone(store, tomb)

    reranked = [
        (live1, 9.0), (supOld, 8.0), (deadOld, 7.0),
        (tomb, 6.0), (live2, 5.0), (supNew, 4.0),
    ]
    signalHits = {a: {"bm25", "dense"} for a, _ in reranked}
    out = assessTrust(reranked, signalHits, store, NOW)

    ids = [row["atomId"] for row in out]
    assert deadOld not in ids          # tombstoned successor -> dropped
    assert tomb not in ids             # tombstoned outright -> dropped
    assert live1 in ids and live2 in ids and supOld in ids and supNew in ids
    _assert_no_orphan_superseded(store, out)
    # output preserves reranked order minus drops
    assert ids == [live1, supOld, live2, supNew]


# --------------------------------------------------------------------------- #
# Sabotage + property checks                                                  #
# --------------------------------------------------------------------------- #

def test_empty_reranked_is_empty(store):
    assert assessTrust([], {}, store, NOW) == []


def test_all_superseded_with_no_live_successors_is_empty(store):
    a = _put(store, "old a")
    an = _put(store, "successor a")
    supersede(store, a, an, {"source": "claude-code"})
    _tombstone(store, an)
    b = _put(store, "old b")
    bn = _put(store, "successor b")
    supersede(store, b, bn, {"source": "claude-code"})
    _tombstone(store, bn)

    out = assessTrust([(a, 5.0), (b, 4.0)], {a: {"bm25"}, b: {"dense"}}, store, NOW)
    assert out == []


def test_all_superseded_with_live_successors_all_chained(store):
    a = _put(store, "old a")
    an = _put(store, "successor a")
    supersede(store, a, an, {"source": "claude-code"})
    b = _put(store, "old b")
    bn = _put(store, "successor b")
    supersede(store, b, bn, {"source": "claude-code"})

    out = assessTrust([(a, 5.0), (b, 4.0)], {a: {"bm25"}, b: {"dense"}}, store, NOW)
    assert {row["atomId"] for row in out} == {a, b}
    for row in out:
        assert "supersededBy" in row
        assert row["shouldTrust"] is False
    _assert_no_orphan_superseded(store, out)


def test_unknown_atom_raises_loudly(store):
    real = _put(store, "a real atom")
    with pytest.raises(ValueError) as exc:
        assessTrust(
            [(real, 5.0), ("GHOST_NOT_IN_STORE", 4.0)],
            {real: {"bm25"}, "GHOST_NOT_IN_STORE": {"bm25"}},
            store,
            NOW,
        )
    assert "GHOST_NOT_IN_STORE" in str(exc.value)


def test_missing_signal_hits_entry_is_lenient_not_loud(store):
    # An atom absent from signalHits is treated as empty (minimal agreement),
    # not a crash: signalHits is explanation data, unlike store consistency.
    a = _put(store, "atom with no signal record")
    tail = _put(store, "tail")
    out = _by_id(assessTrust([(a, 5.0), (tail, 4.0)], {}, store, NOW))
    assert 0.0 <= out[a]["confidence"] <= 1.0
    assert out[a]["why"]                      # still non-empty
    assert "no signal evidence" in out[a]["why"]


def test_confidence_bounded_and_why_clean_across_a_grid(store):
    # Property check: over a grid of ages x gaps x signal combos, confidence is
    # never outside [0,1] and why is always non-empty with no newline/markdown.
    ages = [NOW + 10 * DAY, NOW, NOW - 10 * DAY, NOW - 400 * DAY, 1_000_000_000]
    ids = [_put(store, f"grid atom {i}", occurredAt=age) for i, age in enumerate(ages)]
    signalCombos = [
        set(), {"bm25"}, {"dense"}, {"facet"},
        {"bm25", "dense"}, {"bm25", "facet"}, {"bm25", "dense", "facet"},
    ]
    scoreGrids = [
        [5.0, 4.9, 4.8, 4.7, 4.6],     # all tight
        [5.0, 1.0, 0.9, 0.8, 0.7],     # decisive top gap
        [5.0, 5.0, 5.0, 5.0, 5.0],     # exact ties
        [-8.0, -9.0, -10.0, -11.0, -12.0],  # negative logits, still ordered
    ]
    markdown = set("*_#`[]")
    for scores in scoreGrids:
        reranked = list(zip(ids, scores))
        for combo in signalCombos:
            signalHits = {a: set(combo) for a in ids}
            out = assessTrust(reranked, signalHits, store, NOW)
            assert len(out) == len(ids)        # all live -> none dropped
            for row in out:
                assert 0.0 <= row["confidence"] <= 1.0
                assert isinstance(row["shouldTrust"], bool)
                assert row["why"]
                assert "\n" not in row["why"]
                assert not (markdown & set(row["why"]))


def test_superseded_confidence_never_reaches_the_trust_floor(store):
    # Even a superseded atom with maximal live evidence stays below TRUST_FLOOR:
    # the cap makes a historical fact untrustworthy regardless of its signals.
    old = _put(store, "old but strongly matched", occurredAt=NOW)
    new = _put(store, "the successor")
    supersede(store, old, new, {"source": "claude-code"})

    out = assessTrust(
        [(old, 9.0), (new, 1.0)],              # decisive gap, both signals, recent
        {old: {"bm25", "dense", "facet"}, new: {"bm25"}},
        store,
        NOW,
    )
    row = _by_id(out)[old]
    assert row["confidence"] <= SUPERSEDED_CONF_CAP
    assert row["shouldTrust"] is False
