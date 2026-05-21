"""Coverage tests for L2Handler.batch_query / size / clear and
ParallelHybrid.shutdown / get_learned_entities.

PENPY-P6-MIN-3: pass-6 audit found that these public APIs had no direct
test coverage. batch_query is the multi-query encoding path that
ParallelHybrid bulk paths depend on; size and clear are state helpers;
shutdown cleans up the optional thread pool; get_learned_entities is
the accessor for PatternLearner state.

These are not exhaustive correctness tests; they verify the public
contract (input/output shapes, no crash, no hang) so a future contract
regression surfaces cleanly.
"""
import threading
import time

import pytest

# These tests touch L2 + faiss directly; skip cleanly when extras absent.
pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

from pensive.l2 import L2Handler, L2Config, L2Result
from pensive.parallel_hybrid import ParallelHybrid


# --------------------------------------------------------------------------
# Shared fixture: a populated L2Handler with 3 small docs.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_l2():
    """Build an L2Handler with 3 documents. Module-scoped to amortize
    the sentence-transformer model load across the file (saves ~3s per
    test on cold start)."""
    l2 = L2Handler(L2Config(max_results=5))
    docs = [
        {'id': 'doc1', 'content': 'The server latency dropped to 42ms after the cache fix.'},
        {'id': 'doc2', 'content': 'GPU temperature peaked at 82 degrees Celsius under load.'},
        {'id': 'doc3', 'content': 'The new release ships with FAISS IVF indexing.'},
    ]
    added = l2.add_documents(docs)
    assert added == 3, f"fixture setup failed: only {added}/3 docs added"
    return l2


# --------------------------------------------------------------------------
# L2Handler.batch_query
# --------------------------------------------------------------------------


def test_batch_query_returns_list_per_input(populated_l2):
    """batch_query must return one result list per input query, matching length."""
    queries = [
        "How fast is the server?",
        "What is the GPU temperature?",
        "What indexing does the release use?",
    ]
    results = populated_l2.batch_query(queries, top_k=3)

    assert len(results) == len(queries), (
        f"batch_query returned {len(results)} lists for {len(queries)} queries"
    )
    for i, sub in enumerate(results):
        assert isinstance(sub, list), (
            f"batch_query result {i} is {type(sub)}, expected list"
        )
        for r in sub:
            assert isinstance(r, L2Result), (
                f"batch_query result element is {type(r)}, expected L2Result"
            )


def test_batch_query_empty_input_returns_empty(populated_l2):
    """batch_query with no queries returns []."""
    assert populated_l2.batch_query([]) == []


def test_batch_query_single_input_returns_single_list(populated_l2):
    """batch_query with a single query still returns a list-of-list shape."""
    results = populated_l2.batch_query(["server latency"], top_k=2)
    assert len(results) == 1, (
        f"batch_query single-input returned {len(results)} lists, expected 1"
    )
    assert isinstance(results[0], list)


# --------------------------------------------------------------------------
# L2Handler.size / L2Handler.clear
# --------------------------------------------------------------------------


def test_size_zero_on_empty_handler():
    """A fresh L2Handler reports size 0."""
    l2 = L2Handler()
    assert l2.size == 0, f"empty L2Handler reports size={l2.size}, expected 0"


def test_size_matches_document_count():
    """size must match the number of documents added."""
    l2 = L2Handler()
    l2.add_documents([
        {'id': 'a', 'content': 'first document content'},
        {'id': 'b', 'content': 'second document content'},
    ])
    assert l2.size == 2, f"after adding 2 docs, size={l2.size}, expected 2"


def test_clear_empties_handler():
    """clear must reset size to 0 and forget all docs."""
    l2 = L2Handler()
    l2.add_documents([
        {'id': 'a', 'content': 'first document content'},
        {'id': 'b', 'content': 'second document content'},
    ])
    assert l2.size == 2

    l2.clear()
    assert l2.size == 0, f"after clear, size={l2.size}, expected 0"

    # Verify the handler is still usable after clear: add a new doc.
    l2.add_documents([{'id': 'c', 'content': 'third document content'}])
    assert l2.size == 1, f"after add post-clear, size={l2.size}, expected 1"


# --------------------------------------------------------------------------
# ParallelHybrid.shutdown
# --------------------------------------------------------------------------


def test_parallel_hybrid_shutdown_does_not_hang():
    """shutdown must complete promptly (the thread pool is lazy, so the
    no-pool case should be near-instant). Wrap in a watchdog thread that
    flips a flag, then assert the call finished before the watchdog."""
    ph = ParallelHybrid(spreading_activation=None, l2_handler=None,
                        enable_pattern_learning=False)

    # Force-create the executor by setting it directly: the no-op case
    # would skip the shutdown branch entirely.
    from concurrent.futures import ThreadPoolExecutor
    ph._executor = ThreadPoolExecutor(max_workers=2)

    done = threading.Event()

    def _run_shutdown():
        ph.shutdown()
        done.set()

    t = threading.Thread(target=_run_shutdown, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert done.is_set(), "ParallelHybrid.shutdown did not return within 5s"


def test_parallel_hybrid_shutdown_no_executor_safe():
    """shutdown must be safe when _executor was never created (the lazy
    initialization path leaves it as None until first parallel query)."""
    ph = ParallelHybrid(spreading_activation=None, l2_handler=None,
                        enable_pattern_learning=False)
    assert ph._executor is None
    # Must not raise even though there's no pool to shut down.
    ph.shutdown()


# --------------------------------------------------------------------------
# ParallelHybrid.get_learned_entities
# --------------------------------------------------------------------------


def test_get_learned_entities_empty_when_no_pattern_learner():
    """With enable_pattern_learning=False, get_learned_entities returns
    an empty set (the accessor must not crash on the None case)."""
    ph = ParallelHybrid(spreading_activation=None, l2_handler=None,
                        enable_pattern_learning=False)
    assert ph.pattern_learner is None
    result = ph.get_learned_entities()
    assert isinstance(result, (set, frozenset)), (
        f"get_learned_entities returned {type(result)}, expected set-like"
    )
    assert len(result) == 0, (
        f"no-pattern-learner case returned {len(result)} entities, expected 0"
    )


def test_get_learned_entities_returns_pattern_learner_terms():
    """With a PatternLearner attached, get_learned_entities exposes its
    learned terms. Manually add a term and verify it surfaces.

    PatternLearner.add_manual normalizes to lowercase; the assertion
    matches against the normalized form. The point of this test is to
    verify the ParallelHybrid accessor forwards to the learner, not to
    test the learner's casing behavior.
    """
    from pensive.pattern_learner import PatternLearner

    pl = PatternLearner()
    pl.add_manual("CUSTOM_ENTITY")

    ph = ParallelHybrid(spreading_activation=None, l2_handler=None,
                        enable_pattern_learning=True, pattern_learner=pl)

    terms = set(ph.get_learned_entities())
    assert "custom_entity" in terms, (
        f"manually added term not surfaced via get_learned_entities(): "
        f"got {terms!r}, expected to contain 'custom_entity' (normalized)"
    )
