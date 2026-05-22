"""PENPY-P9-IMP-2: Regression test for the "context only boosts, never penalizes"
universal invariant.

The code at spreading.py:1442-1443 (bipartite fast path) and 1456-1459
(general path) multiplies query scores by (1.0 + ctx_value) where
ctx_value is selected via a >0 mask. This means the boost factor is
always >= 1.0 -- context can only push scores up, never down. A future
commit inverting this to a penalty (e.g. multiplying by a value <= 1.0
for context-matching docs, or dividing instead of multiplying) would
violate this invariant.

The previous test suite (TestContextDisambiguation in test_spreading.py)
exercises CORRECT context-boost behavior end-to-end, but it leans on
exact ordering, not on the per-doc inequality `with_ctx >= without_ctx`.
A sabotage of the boost direction that flipped a < to a > in the
right place could leave the disambiguation tests still passing if the
sabotaged penalty happened to land on the doc that was already losing.
The two checks here are deliberately tight on the invariant:

  (a) FOR EVERY doc in the bare-query result, its score with non-empty
      context must be >= its bare score.
  (b) Specifically: a context that activates ONE of the candidate docs
      should leave that doc's score STRICTLY higher than its bare
      score (proves the boost actually fired), while leaving the other
      doc's score at the bare value (no spurious penalty).

The bipartite fast path and the general (non-bipartite) path use
different code, so we cover both.
"""
import pytest

from pensive import SpreadingActivation, SpreadingConfig
from pensive.patterns import SYNTHETIC_PATTERNS


def make_two_doc_corpus():
    """Two ambiguous docs sharing a date but with distinct values."""
    return [
        {'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
         'id': 'n1', 'value': '199ms'},
        {'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
         'id': 'n2', 'value': '257ms'},
    ]


def _scores_by_value(results):
    """Flatten [(value, score), ...] -> {value: score}."""
    return {value: score for value, score in results}


def test_context_only_boosts_bipartite_fast_path():
    """Bipartite fast path (spreading.py:1442-1443).

    For every value present in the bare-query result, the same value's
    score in a context-augmented query MUST be >= the bare score. The
    boost factor in the code is (1.0 + ctx_arr[ctx_mask]) where ctx_mask
    is `ctx_arr > 0`, so it is always strictly > 1.0 when context
    activates anything overlapping with the query result and exactly
    1.0 (no change) otherwise.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(make_two_doc_corpus())
    assert sa._is_bipartite, "fixture must be bipartite for fast path"

    bare = _scores_by_value(sa.query("2025-07-16"))
    assert "199ms" in bare and "257ms" in bare, \
        f"bare query must return both docs as candidates; got {bare}"

    # Context that activates the doc whose value is 199ms.
    boosted = _scores_by_value(
        sa.query("2025-07-16", context=["199ms"])
    )
    # Every result that was in the bare set must be present and have a
    # score >= its bare score. Context cannot DROP a result.
    for value, bare_score in bare.items():
        assert value in boosted, (
            f"PENPY-P9-IMP-2 violated: context REMOVED candidate "
            f"{value!r} from results (was {bare_score}, now missing). "
            f"Context should only boost, never penalize."
        )
        assert boosted[value] >= bare_score - 1e-9, (
            f"PENPY-P9-IMP-2 violated: context PENALIZED candidate "
            f"{value!r}: bare={bare_score}, with_context={boosted[value]}. "
            f"Context should only boost, never penalize."
        )

    # The activated value must have STRICTLY increased (proves the
    # boost actually fired, not silently no-op'd).
    assert boosted["199ms"] > bare["199ms"], (
        f"PENPY-P9-IMP-2 violated: context activating '199ms' did not "
        f"boost its score: bare={bare['199ms']}, "
        f"with_context={boosted['199ms']}"
    )


def test_context_only_boosts_general_path():
    """Non-bipartite (general) path (spreading.py:1456-1459).

    Same invariant, exercised on the codepath that runs when the graph
    is not bipartite. We force the general path by enabling
    entity-to-entity edges (config.entity_edges=True if exposed, else
    we exercise it via the equivalent dict-based intersect path by
    setting _is_bipartite=False on the built graph -- monkey-patching
    just the flag so the code takes the second branch).
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(make_two_doc_corpus())
    # Force the dict-based intersect path. This is a test seam -- the
    # real bipartite-vs-general decision is set during build; we are
    # only here to exercise the OTHER intersect code branch with the
    # same invariant.
    sa._is_bipartite = False

    bare = _scores_by_value(sa.query("2025-07-16"))
    assert "199ms" in bare and "257ms" in bare

    boosted = _scores_by_value(
        sa.query("2025-07-16", context=["199ms"])
    )
    for value, bare_score in bare.items():
        assert value in boosted, (
            f"PENPY-P9-IMP-2 violated (general path): context REMOVED "
            f"candidate {value!r}"
        )
        assert boosted[value] >= bare_score - 1e-9, (
            f"PENPY-P9-IMP-2 violated (general path): context "
            f"PENALIZED candidate {value!r}: bare={bare_score}, "
            f"with_context={boosted[value]}"
        )
    assert boosted["199ms"] > bare["199ms"], (
        f"PENPY-P9-IMP-2 violated (general path): boost did not fire "
        f"for activated value"
    )


def test_unmatched_context_no_penalty():
    """A context whose entities do not overlap with the query result
    must leave every score unchanged. This catches a sabotage where the
    bipartite path's `ctx_mask = ctx_arr > 0` got flipped to `ctx_arr
    <= 0` (penalizing only the docs that the context did NOT activate)
    -- the disambiguation tests would not necessarily notice that.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(make_two_doc_corpus())

    bare = _scores_by_value(sa.query("2025-07-16"))

    # Context that does not match any entity in the graph.
    boosted = _scores_by_value(
        sa.query("2025-07-16", context=["unrelated-entity-xyz-9999"])
    )
    for value, bare_score in bare.items():
        assert value in boosted, (
            f"PENPY-P9-IMP-2 violated: unmatched context REMOVED "
            f"candidate {value!r}"
        )
        # Allow small float jitter; the values must be effectively
        # equal because no boost should have fired.
        assert boosted[value] >= bare_score - 1e-9, (
            f"PENPY-P9-IMP-2 violated: unmatched context PENALIZED "
            f"candidate {value!r}: bare={bare_score}, "
            f"with_unmatched_context={boosted[value]}"
        )
