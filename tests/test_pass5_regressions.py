"""Regression tests for Pass 5 fixes (PENPY-IMP-5, IMP-6, MIN-4).

Each test mirrors the auditor's reproduction from
/tmp/audit-pensive-engram/pensive-py-p3.json. Under sabotage (revert the
fix), each test MUST fail loudly with a clear message -- the discipline
pattern from PENPY-CRIT-2.
"""
import pytest

from pensive import SpreadingActivation
from pensive.boundary import analyze_boundary
from pensive.mega_extract import MegaExtractor
from pensive.patterns import SYNTHETIC_PATTERNS


# PENPY-IMP-5: boundary value-label cache stale after build()-rebuild


def test_imp5_boundary_cache_invalidates_on_rebuild_same_n_nodes():
    """A second build() with size-stable schema but different value labels
    must NOT serve a stale cached label-index from the first build.

    Reproduction from PENPY-IMP-5: build A with values 199ms/257ms,
    populate cache via analyze_boundary, rebuild with values 999ms/888ms
    (same n_nodes). Without the fix, analyze_boundary returns the stale
    199ms/257ms entity disambiguation context for B's queries -- the
    user-visible symptom is analysis.suggested_context coming back empty
    because the label-index lookup misses on B's new labels.
    """
    docs_a = [
        {'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
         'id': 'n1', 'value': '199ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
         'id': 'n2', 'value': '257ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
    ]
    docs_b = [
        {'content': 'System latency on 2025-07-16 was 999ms at the 99th percentile.',
         'id': 'n1', 'value': '999ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'System latency on 2025-07-16 was 888ms at the 99th percentile.',
         'id': 'n2', 'value': '888ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
    ]

    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)

    # Build A and prime the cache via an ambiguous query
    sa.build(docs_a)
    result_a = analyze_boundary(sa, "What was the P99 latency on 2025-07-16?")
    # Sanity: the ambiguous query on A must produce suggested_context
    # with A's value labels -- if this is empty even with build A,
    # the test isn't actually exercising the disambiguation path.
    assert set(result_a.analysis.suggested_context) >= {'199ms', '257ms'}, (
        "Sanity: build A's ambiguous query should produce "
        "suggested_context containing both A labels, got "
        f"{result_a.analysis.suggested_context}"
    )
    # Cache must now be populated with A's labels.
    cached_a = getattr(sa, '_boundary_value_label_index', None)
    assert cached_a is not None, (
        "Sanity: analyze_boundary on ambiguous query should populate "
        "the boundary value-label cache for PENPY-IMP-2."
    )

    # Rebuild with docs_b. n_nodes should match (same schema), but every
    # value label has changed. The stale cache key (n_nodes alone) would
    # let stale 199ms/257ms indices leak through.
    n_nodes_a = len(sa._idx_to_node)
    sa.build(docs_b)
    n_nodes_b = len(sa._idx_to_node)
    assert n_nodes_a == n_nodes_b, (
        "Reproduction precondition: docs_a and docs_b must produce the "
        f"same n_nodes (got {n_nodes_a} vs {n_nodes_b}). If schemas "
        "diverged, the cache would invalidate trivially and the test "
        "wouldn't exercise PENPY-IMP-5."
    )

    result_b = analyze_boundary(sa, "What was the P99 latency on 2025-07-16?")

    # The user-visible symptom: analysis.suggested_context returns []
    # instead of B's new labels because _resolve_value_node_idx hits
    # the stale A-cache, finds no entries for 999ms/888ms, and returns
    # None for both result indices -- which short-circuits the
    # differentiating-entities computation.
    suggested = set(result_b.analysis.suggested_context)
    assert {'999ms', '888ms'}.issubset(suggested), (
        f"PENPY-IMP-5 regression: rebuild with new value labels returned "
        f"suggested_context={result_b.analysis.suggested_context} -- "
        "expected B's labels (999ms, 888ms). The boundary value-label "
        "cache likely served stale entries from build A. Check "
        "_get_value_label_index cache key and _reset_graph_state "
        "invalidation."
    )
    assert '199ms' not in suggested and '257ms' not in suggested, (
        f"PENPY-IMP-5 regression: rebuild surfaced A's stale labels in "
        f"B's suggested_context: {result_b.analysis.suggested_context}"
    )


def test_imp5_generation_counter_advances_on_reset():
    """_graph_generation must monotonically advance on each rebuild so
    cache keys remain unique even at identical n_nodes.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    docs = [
        {'content': 'sample doc one', 'id': 'd1', 'value': 'v1'},
    ]
    sa.build(docs)
    gen1 = sa._graph_generation
    sa.build(docs)
    gen2 = sa._graph_generation
    assert gen2 > gen1, (
        "PENPY-IMP-5: graph generation must advance on every build() so "
        "the (n_nodes, generation) cache key changes even when n_nodes "
        f"is stable. Got gen1={gen1}, gen2={gen2}."
    )


def test_imp5_generation_counter_advances_on_add_documents():
    """add_documents() must also bump the generation so any consumer that
    caches across incremental adds invalidates.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([{'content': 'one', 'id': 'd1', 'value': 'v1'}])
    gen_before = sa._graph_generation
    sa.add_documents([{'content': 'two', 'id': 'd2', 'value': 'v2'}])
    gen_after = sa._graph_generation
    assert gen_after > gen_before, (
        "PENPY-IMP-5: add_documents() must advance _graph_generation, "
        f"got gen_before={gen_before}, gen_after={gen_after}."
    )


def test_imp5_reset_clears_cache_attribute_belt_and_suspenders():
    """Belt-and-suspenders: _reset_graph_state() must also drop the
    cache attribute directly so any consumer that didn't migrate to the
    generation-aware key still sees a clean slate.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([{'content': 'one', 'id': 'd1', 'value': 'v1'}])
    # Manually plant the cache attribute, then trigger reset:
    sa._boundary_value_label_index = ((0, 0), {'stale': [999]})
    sa._reset_graph_state()
    assert not hasattr(sa, '_boundary_value_label_index'), (
        "PENPY-IMP-5: _reset_graph_state must drop the cache attribute "
        "explicitly (belt-and-suspenders), but the attribute survived."
    )


# PENPY-IMP-6: query(top_k=0) returns all results in bipartite paths


@pytest.fixture
def sa_5_docs_shared_entity():
    """5 docs all sharing the same entity so any non-zero query hits all."""
    docs = [
        {'content': 'response time was 100ms today',
         'id': f'd{i}', 'value': f'V{i}',
         'query': '100ms latency'}
        for i in range(5)
    ]
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(docs)
    return sa


def test_imp6_query_top_k_zero_returns_empty(sa_5_docs_shared_entity):
    """sa.query(text, top_k=0) must return [] on the bipartite numpy
    fast-path. The pre-fix bug: `np.argpartition(scores, -0)[-0:]` is
    `argpartition(scores, 0)[0:]` which returns the FULL array.
    """
    sa = sa_5_docs_shared_entity
    results = sa.query('100ms', top_k=0)
    assert results == [], (
        f"PENPY-IMP-6 regression: sa.query(top_k=0) returned {results} "
        f"({len(results)} items) instead of []. The numpy fast-path's "
        "argpartition(scores, -0)[-0:] degenerates to returning ALL "
        "items when top_k=0. Add an early-return at the top of "
        "_spread_and_collect_bipartite / _collect_from_array."
    )


def test_imp6_query_with_doc_ids_top_k_zero_returns_empty(sa_5_docs_shared_entity):
    """Same as above for query_with_doc_ids -- exercises
    _spread_and_collect_bipartite_with_ids and _collect_from_array_with_ids.
    """
    sa = sa_5_docs_shared_entity
    results = sa.query_with_doc_ids('100ms', top_k=0)
    assert results == [], (
        f"PENPY-IMP-6 regression: sa.query_with_doc_ids(top_k=0) returned "
        f"{results} ({len(results)} items) instead of []."
    )


def test_imp6_query_analyzed_top_k_zero_returns_empty(sa_5_docs_shared_entity):
    """query_analyzed wraps query so the same fix must hold here."""
    sa = sa_5_docs_shared_entity
    result = sa.query_analyzed('100ms', top_k=0)
    assert result.results == [], (
        f"PENPY-IMP-6 regression: sa.query_analyzed(top_k=0) returned "
        f"{result.results} ({len(result.results)} items) instead of []."
    )


def test_imp6_query_top_k_zero_with_context_returns_empty(sa_5_docs_shared_entity):
    """Top_k=0 with context still hits _collect_from_array via the
    context branch -- must also short-circuit to [].
    """
    sa = sa_5_docs_shared_entity
    results = sa.query('100ms', top_k=0, context=['latency'])
    assert results == [], (
        f"PENPY-IMP-6 regression: sa.query(top_k=0, context=...) returned "
        f"{results} -- context branch missed the top_k=0 short-circuit."
    )


def test_imp6_query_top_k_negative_returns_empty(sa_5_docs_shared_entity):
    """Defensive: top_k<0 is nonsense input but must not leak results."""
    sa = sa_5_docs_shared_entity
    results = sa.query('100ms', top_k=-1)
    assert results == [], (
        f"PENPY-IMP-6: sa.query(top_k=-1) returned {results} -- the "
        "top_k<=0 guard must cover negatives too."
    )


# PENPY-MIN-4: MegaExtractor IndexError on named-group patterns


def test_min4_named_group_pattern_rejected_at_construction():
    """A pattern with a named capturing group has 1 group via re.groups
    but routes through m.group(m.lastindex) -- which conflicts with the
    etype_map indexing assumption when combined with other patterns.
    Validate at construction with a clear ValueError, not a runtime
    IndexError on the first match.
    """
    # 2 capturing groups (one named, one positional)
    bad_patterns = [(r'(?P<name>\d{4})-(?P<month>\d{2})', 'date', False)]
    with pytest.raises(ValueError) as exc_info:
        MegaExtractor(bad_patterns)
    msg = str(exc_info.value)
    assert 'capturing group' in msg.lower(), (
        f"PENPY-MIN-4: ValueError message must mention capturing groups, "
        f"got: {msg!r}"
    )
    assert '2' in msg, (
        f"PENPY-MIN-4: ValueError should report actual group count, "
        f"got: {msg!r}"
    )


def test_min4_zero_capturing_groups_rejected():
    """A pattern with no capturing group also violates the contract --
    extract() requires exactly one group containing the entity text.
    """
    bad_patterns = [(r'\bfoo\b', 'word', False)]
    with pytest.raises(ValueError) as exc_info:
        MegaExtractor(bad_patterns)
    assert 'capturing group' in str(exc_info.value).lower()


def test_min4_two_positional_groups_rejected():
    """Two positional capturing groups would shift the etype_map index."""
    bad_patterns = [(r'(\d{4})-(\d{2})', 'date', False)]
    with pytest.raises(ValueError) as exc_info:
        MegaExtractor(bad_patterns)
    assert 'capturing group' in str(exc_info.value).lower()
    assert '2' in str(exc_info.value)


def test_min4_valid_single_group_pattern_still_works():
    """Sanity: a properly-shaped single-group pattern must still construct
    and extract successfully -- the validation must not be over-eager.
    """
    good_patterns = [(r'\b(\d{4})\b', 'year', False)]
    extractor = MegaExtractor(good_patterns)
    results = extractor.extract("born in 1985 and graduated in 2003")
    labels = [label for label, _ in results]
    assert '1985' in labels and '2003' in labels, (
        f"valid single-group pattern broke after MIN-4 fix: got {labels}"
    )


def test_min4_non_capturing_groups_dont_count():
    """Non-capturing groups (?:...), lookaheads (?=...), lookbehinds (?<=...)
    must NOT trip the capturing-group counter -- they contribute 0 groups.
    """
    # Pattern with non-capturing group + lookahead + 1 real capturing group
    pattern = r'\b(?:foo|bar)(?=baz)(\w+)\b'
    extractor = MegaExtractor([(pattern, 'matched', False)])
    # Smoke test: extract from text where the pattern matches
    results = extractor.extract("foobaz123 barbaz456")
    # Should not raise -- the only capturing group is the real one
    assert isinstance(results, list)
