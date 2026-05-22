"""PENPY-P9-IMP-4: Regression test for the consumer-side cache key in
boundary._get_value_label_index().

Pass-7 fix-4 added PENPY-IMP-5: ``sa._graph_generation`` bumps on every
mutation that can change value labels (build, add_documents, reset).
The consumer-side cache key in ``boundary._get_value_label_index`` is
``(n_nodes, generation)`` rather than just ``n_nodes``, because a
second ``build()`` with the same schema can keep the node count
identical while changing every label.

The source-side generation bump IS tested (see
``tests/test_pass4_regressions.py::test_PENPY_IMP_5_*``). The
consumer-side use of that generation as part of the cache key is NOT
tested -- the P9 audit sabotage-gate confirmed that changing
``key = (n_nodes, generation)`` to ``key = (n_nodes,)`` passes all
existing tests.

This regression locks down the consumer-side key explicitly:

  1. Build SA with corpus A. Call _get_value_label_index -- this caches.
  2. Re-build SA with corpus B of the SAME node count but DIFFERENT
     labels. _graph_generation has now advanced.
  3. Call _get_value_label_index again. It MUST return the corpus-B
     mapping, not the cached corpus-A mapping.

Sabotaging the key to ``(n_nodes,)`` makes the cache hit on the
corpus-A entry whose labels are now wrong, and the test fails with a
direct "stale cache returned for new corpus" diagnostic.
"""
from pensive import SpreadingActivation
from pensive.boundary import _get_value_label_index
from pensive.patterns import SYNTHETIC_PATTERNS


def _make_corpus(labels):
    """Build a corpus where each doc has a value matching `labels[i]`.

    All docs share a date entity so the resulting graph has predictable
    node count: N value nodes + 1 shared date entity = N+1 nodes (if no
    other entities extract from the value strings themselves).
    """
    return [
        {
            'id': f'd{i}',
            'content': f'On 2023-05-15 the measurement was {label} as recorded.',
            'value': label,
        }
        for i, label in enumerate(labels)
    ]


def test_p9_imp4_consumer_cache_key_includes_generation():
    """Re-build() with same n_nodes but different labels must invalidate
    the boundary value-label cache, not return the stale corpus-A map.

    Sabotage gate: change boundary.py:395 key to ``(n_nodes,)`` -- the
    second _get_value_label_index call returns the corpus-A index even
    though the SA now holds corpus-B labels.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)

    # Corpus A: 3 labels with a known shape.
    labels_a = ['199ms', '257ms', '342ms']
    sa.build(_make_corpus(labels_a))

    # Force boundary cache population. _get_value_label_index returns
    # {label -> [value-node indices]}. After this call sa has the
    # cached attribute set to (key_A, index_A).
    index_a = _get_value_label_index(sa)
    cached_a = getattr(sa, '_boundary_value_label_index')
    n_nodes_a = len(sa._idx_to_node)
    assert set(labels_a).issubset(index_a.keys()), (
        f"corpus A index missing expected labels; got {list(index_a.keys())}"
    )

    # Corpus B: SAME number of labels (so n_nodes after rebuild matches),
    # but DIFFERENT labels. The PENPY-IMP-5 generation bump is what the
    # cache key must depend on.
    labels_b = ['401ms', '512ms', '619ms']
    sa.build(_make_corpus(labels_b))

    # If build() did not change node count, sabotaging the key to
    # (n_nodes,) would silently re-use the corpus-A cache.
    n_nodes_b = len(sa._idx_to_node)
    assert n_nodes_a == n_nodes_b, (
        f"fixture invariant: corpus A and corpus B must yield the same "
        f"node count ({n_nodes_a} vs {n_nodes_b}) -- otherwise the "
        f"sabotage shape (n_nodes,) would also miss for the wrong reason"
    )

    # Call _get_value_label_index again. With the (n_nodes, generation)
    # key the cache must miss because generation has advanced. With the
    # sabotaged (n_nodes,) key the cache wrongly hits.
    index_b = _get_value_label_index(sa)

    # Direct check: the returned index must contain corpus-B labels,
    # and must NOT contain corpus-A labels (which no longer exist in
    # the graph).
    assert set(labels_b).issubset(index_b.keys()), (
        f"PENPY-P9-IMP-4 violated: stale cache returned for new corpus. "
        f"After rebuild() the value-label index is missing the new "
        f"labels {set(labels_b) - set(index_b.keys())}. The cache key "
        f"(n_nodes, generation) at boundary.py:395 must depend on "
        f"_graph_generation; check that it has not been reduced to "
        f"(n_nodes,)."
    )
    assert not set(labels_a).intersection(index_b.keys()), (
        f"PENPY-P9-IMP-4 violated: stale cache leaked corpus-A labels "
        f"into corpus-B result: "
        f"{set(labels_a).intersection(index_b.keys())}. The cache key "
        f"must invalidate on _graph_generation bump."
    )

    # Belt-and-suspenders: the cached tuple's key half must have moved.
    cached_b = getattr(sa, '_boundary_value_label_index')
    assert cached_a[0] != cached_b[0], (
        f"PENPY-P9-IMP-4 violated: cache key did not change across "
        f"rebuild. Corpus-A key={cached_a[0]} corpus-B key={cached_b[0]}. "
        f"The generation component of the key is missing or stuck."
    )


def test_p9_imp4_add_documents_also_invalidates_consumer_cache():
    """add_documents() also bumps _graph_generation. The boundary cache
    must invalidate on that path too. Belt-and-suspenders coverage --
    same invariant, different mutation entrypoint.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(_make_corpus(['199ms', '257ms']))
    index_pre = _get_value_label_index(sa)
    cached_pre = getattr(sa, '_boundary_value_label_index')

    # Incremental add. n_nodes will grow, but we also assert the
    # generation bumped -- the key depends on BOTH.
    sa.add_documents([{
        'id': 'd-new',
        'content': 'On 2024-01-01 the measurement was 999ms exactly.',
        'value': '999ms',
    }])

    index_post = _get_value_label_index(sa)
    cached_post = getattr(sa, '_boundary_value_label_index')

    assert '999ms' in index_post, (
        f"PENPY-P9-IMP-4 violated: add_documents did not invalidate the "
        f"boundary cache; new label '999ms' missing from index."
    )
    assert cached_pre[0] != cached_post[0], (
        f"PENPY-P9-IMP-4 violated: cache key did not advance across "
        f"add_documents."
    )
