"""RRF fusion + two-sided time prior.

Two units under test, both pure Python on the fusion side:

- ``rrf(lists, k=60, weights=None)`` combines several best-first ranked lists by
  reciprocal-rank fusion. It looks ONLY at positions, never the input scores, so
  the tests hand it lists whose scores are deliberately arbitrary and assert on
  the exact ``1/(k+rank)`` arithmetic.

- ``applyPriors(fused, store, hints)`` multiplies fused scores by an importance
  factor and a two-sided time factor (or, under ``timeScope``, restricts to a
  window). The prior side needs a real store for importance/timestamps, so those
  tests build a live SQLite store by hand -- no model load, no embeddings.

The load-bearing property this file pins down: the time factor has a FLOOR, so a
high-importance old atom is never buried by a fresh trivial one (the "decades
outrank freshness" rule).
"""
import pytest

from recall.fusion import rrf, applyPriors, timeFactor, FLOOR, TAU
from store.store import openStore, putAtom

# A fixed reference "now" (2033) so ages are deterministic regardless of the wall
# clock. Atoms whose age matters set occurred_at explicitly, so COALESCE uses it
# and the test never depends on created_at (stamped to the real clock).
NOW = 2_000_000_000


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text="atom text", importance=0.0, occurredAt=None, project="aegis"):
    atomInput = {
        "text": text,
        "kind": "atom",
        "project": project,
        "importance": importance,
        "provenance": {"source": "claude-code"},
    }
    if occurredAt is not None:
        atomInput["occurredAt"] = occurredAt
    return putAtom(store, atomInput)


# --------------------------------------------------------------------------- #
# rrf -- reciprocal rank fusion                                               #
# --------------------------------------------------------------------------- #

def test_rrf_agreement_beats_single_strong_placement():
    # Brief Step 1: an atom ranked #1 by one signal and #3 by another BEATS an
    # atom ranked #2 by only one signal. RRF rewards agreement across signals.
    #   X: rank 1 in listA, rank 3 in listB  -> 1/61 + 1/63
    #   Y: rank 2 in listB only              -> 1/62
    listA = [("X", 9.9), ("a", 9.0), ("b", 8.0)]
    listB = [("c", 0.9), ("Y", 0.5), ("X", 0.1)]

    fused = rrf([listA, listB])
    ids = [atomId for atomId, _ in fused]
    byId = dict(fused)

    assert "X" in ids and "Y" in ids
    assert ids.index("X") < ids.index("Y")          # X ranked above Y
    assert byId["X"] > byId["Y"]
    assert byId["X"] == pytest.approx(1 / 61 + 1 / 63)
    assert byId["Y"] == pytest.approx(1 / 62)


def test_rrf_ignores_input_scores_only_positions_matter():
    # The RRF contract: input SCORES are discarded, only rank positions count.
    # Two lists with identical ordering but wildly different scores must fuse
    # identically.
    a = [("p", 1000.0), ("q", 0.001)]
    b = [("p", 0.001), ("q", 1000.0)]
    assert rrf([a]) == rrf([b])


def test_rrf_single_list_passes_through_in_rank_order():
    fused = rrf([[("a", 9), ("b", 8), ("c", 7)]])
    assert [atomId for atomId, _ in fused] == ["a", "b", "c"]
    byId = dict(fused)
    assert byId["a"] == pytest.approx(1 / 61)
    assert byId["b"] == pytest.approx(1 / 62)
    assert byId["c"] == pytest.approx(1 / 63)


def test_rrf_duplicate_atom_sums_across_lists():
    # The agreement reward IS the sum: an atom at rank 1 in one list and rank 3
    # in another earns 1/(60+1) + 1/(60+3). Assert the exact arithmetic.
    listA = [("x", 9), ("a", 8), ("b", 7)]   # x rank 1
    listB = [("c", 9), ("d", 8), ("x", 7)]   # x rank 3
    byId = dict(rrf([listA, listB]))
    assert byId["x"] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_empty_input_is_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_weights_change_ordering():
    # The tuning lever Task 11's eval harness owns: with uniform weights two
    # single-placement atoms tie; weighting one list up must break the tie and
    # flip ordering. Prove the parameter actually moves results.
    listA = [("a", 0.9)]
    listB = [("b", 0.9)]

    uniform = dict(rrf([listA, listB]))
    assert uniform["a"] == pytest.approx(uniform["b"])   # tie under uniform

    weighted = rrf([listA, listB], weights=[3.0, 1.0])
    ids = [atomId for atomId, _ in weighted]
    byId = dict(weighted)
    assert ids[0] == "a"
    assert byId["a"] > byId["b"]
    assert byId["a"] == pytest.approx(3.0 / 61)
    assert byId["b"] == pytest.approx(1.0 / 61)


def test_rrf_weights_length_must_match_lists():
    with pytest.raises(ValueError):
        rrf([[("a", 1)], [("b", 1)]], weights=[1.0])


# --------------------------------------------------------------------------- #
# applyPriors -- importance factor + two-sided time prior                     #
# --------------------------------------------------------------------------- #

def test_time_factor_floor_property_holds_for_any_age():
    # The load-bearing property, asserted directly on the factor:
    #   timeFactor(0) == 1.0            (freshest possible boost)
    #   timeFactor(huge age) == FLOOR   (never decays below the floor)
    #   FLOOR <= timeFactor(age) <= 1.0 for every non-negative age
    assert timeFactor(0) == pytest.approx(1.0)
    assert timeFactor(10 ** 12) == FLOOR          # underflows to exactly FLOOR
    assert timeFactor(TAU) == pytest.approx(FLOOR + (1.0 - FLOOR) * (1 / 2.718281828))
    for age in (0, 3600, TAU, 10 * TAU, 100 * TAU, 10 ** 12):
        assert FLOOR <= timeFactor(age) <= 1.0


def test_old_important_atom_is_not_buried_by_fresh_trivial(store):
    # Brief Step 5a: two atoms EQUAL on fusion. One is old (31 years) but
    # maximally important; the other is brand new but trivial. The floor
    # guarantees the old important atom is not buried:
    #   old:  fused 1.0 * (1+min(1,1)=2.0) * timeFactor(huge)=FLOOR  = 2.0*0.5 = 1.0
    #   new:  fused 1.0 * (1+0=1.0)         * timeFactor(0)=1.0       = 1.0
    old = _put(store, "decades-old but important", importance=1.0, occurredAt=1_000_000_000)
    new = _put(store, "fresh but trivial", importance=0.0, occurredAt=NOW)

    fused = [(old, 1.0), (new, 1.0)]
    scored = dict(applyPriors(fused, store, {"now": NOW}))

    assert scored[old] >= scored[new]                 # NOT buried
    assert scored[old] == pytest.approx(1.0)
    assert scored[new] == pytest.approx(1.0)


def test_importance_factor_is_capped_at_one(store):
    # importanceFactor = 1 + min(importance, 1.0): earned importance above 1.0
    # cannot run away. An atom with importance 5.0 scores the same as one at 1.0.
    huge = _put(store, "importance five", importance=5.0, occurredAt=NOW)
    capped = _put(store, "importance one", importance=1.0, occurredAt=NOW)

    scored = dict(applyPriors([(huge, 1.0), (capped, 1.0)], store, {"now": NOW}))
    assert scored[huge] == pytest.approx(scored[capped])
    assert scored[huge] == pytest.approx(2.0)         # 1.0 * 2.0 * timeFactor(0)=1.0


def test_apply_priors_sorted_best_first(store):
    # Output is best-first by final score. A high-importance atom outranks a
    # trivial one at equal fusion and freshness.
    important = _put(store, "important", importance=1.0, occurredAt=NOW)
    trivial = _put(store, "trivial", importance=0.0, occurredAt=NOW)

    out = applyPriors([(trivial, 1.0), (important, 1.0)], store, {"now": NOW})
    assert out[0][0] == important
    assert out[0][1] > out[1][1]


def test_time_scope_restricts_to_window_and_skips_time_prior(store):
    # Brief Step 5b: timeScope=(2027 window) returns ONLY 2027-era atoms. The
    # time prior is NOT applied inside the window (importance factor still is).
    era2027 = _put(store, "happened in 2027", occurredAt=1_800_000_000)   # ~2027-01
    era2020 = _put(store, "happened in 2020", occurredAt=1_600_000_000)   # ~2020-09

    window = (1_798_761_600, 1_830_297_600)   # 2027-01-01 .. 2027-12-31
    out = applyPriors(
        [(era2027, 1.0), (era2020, 1.0)], store, {"now": NOW, "timeScope": window}
    )
    ids = [atomId for atomId, _ in out]
    assert ids == [era2027]                    # 2020 atom dropped
    # inside the window there is no time decay: fused 1.0 * importanceFactor 1.0.
    assert dict(out)[era2027] == pytest.approx(1.0)


def test_time_scope_still_applies_importance(store):
    # Inside a window, the importance factor still ranks: a high-importance atom
    # beats a trivial one, but no time decay separates equally-dated atoms.
    important = _put(store, "important 2027", importance=1.0, occurredAt=1_800_000_000)
    trivial = _put(store, "trivial 2027", importance=0.0, occurredAt=1_800_100_000)

    window = (1_798_761_600, 1_830_297_600)
    out = applyPriors(
        [(important, 1.0), (trivial, 1.0)], store, {"timeScope": window}
    )
    assert out[0][0] == important              # importance still ranks in-window
    assert dict(out)[important] == pytest.approx(2.0)
    assert dict(out)[trivial] == pytest.approx(1.0)


def test_time_scope_does_not_require_now(store):
    # timeScope mode never reads hints["now"] -- it is not needed when the window
    # replaces the time prior. Omitting now must still work.
    inWindow = _put(store, "in 2027", occurredAt=1_800_000_000)
    window = (1_798_761_600, 1_830_297_600)
    out = applyPriors([(inWindow, 1.0)], store, {"timeScope": window})
    assert [a for a, _ in out] == [inWindow]


def test_time_scope_degenerate_window_is_empty(store):
    # start > end is an impossible window -> empty result (consistent with the
    # Task 6 facet time filter). Documented in applyPriors.
    a = _put(store, "some atom", occurredAt=1_800_000_000)
    out = applyPriors([(a, 1.0)], store, {"now": NOW, "timeScope": (5000, 1000)})
    assert out == []


def test_apply_priors_empty_fused_is_empty(store):
    assert applyPriors([], store, {"now": NOW}) == []


def test_apply_priors_unknown_atom_raises_loudly(store):
    # A fused candidate absent from the store means index/store desync. The
    # decades rule: fail loudly on inconsistency, naming the offending id --
    # never silently drop it.
    real = _put(store, "a real atom", occurredAt=NOW)
    with pytest.raises(ValueError) as exc:
        applyPriors([(real, 1.0), ("GHOST_NOT_IN_STORE", 1.0)], store, {"now": NOW})
    assert "GHOST_NOT_IN_STORE" in str(exc.value)
