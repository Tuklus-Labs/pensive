"""REFACTOR-5: SpreadingActivation.compact().

The graph is append-only; long-lived ingesting daemons previously had to
hand-roll the documented serialize-discard-rebuild dance to reclaim the
COO build buffers and caches. compact() performs that round-trip in
place. Contract under test:

* query results are identical before and after compact()
* node/edge counts (stats) are preserved
* the COO buffers and _entity_edge_positions are actually reclaimed
* add_documents() after compact() works (CSR -> COO re-materialization)
* the graph generation bumps (consumer caches must invalidate)
* the context provider and the held lock survive the swap
* compact() before build() raises RuntimeError
* a compacted graph still save/load round-trips
* compact() is safe under concurrent reader load
"""
import threading

import pytest

from pensive import SpreadingActivation


def _doc(i: int) -> dict:
    """Doc with REAL_DATA_PATTERN entities (date + metric)."""
    return {
        'id': f'd-{i}',
        'content': (
            f'latency was {(i * 7) % 500 + 30}ms on 2025-07-{(i % 27) + 1:02d} '
            f'sarah chen reported issue'
        ),
        'value': f'val-{i}',
    }


def _built_sa(n: int = 60) -> SpreadingActivation:
    sa = SpreadingActivation()
    sa.build([_doc(i) for i in range(n)])
    # Mix in incremental adds so the COO buffers and
    # _entity_edge_positions have real growth to reclaim.
    sa.add_documents([_doc(i) for i in range(n, n + 40)])
    return sa


_QUERIES = ['2025-07-03', '2025-07-15', '142ms', 'sarah chen']


def test_compact_preserves_query_results():
    sa = _built_sa()
    before = {q: sa.query(q, top_k=20) for q in _QUERIES}
    sa.compact()
    after = {q: sa.query(q, top_k=20) for q in _QUERIES}
    assert before == after, (
        "compact() changed query results; the round-trip swap is not "
        "behavior-preserving"
    )


def test_compact_preserves_stats():
    sa = _built_sa()
    before = sa.stats()
    sa.compact()
    assert sa.stats() == before


def test_compact_reclaims_build_buffers():
    sa = _built_sa()
    # add_documents above populated the mutable-edge structures.
    assert len(sa._edge_src) > 0
    assert len(sa._entity_edge_positions) > 0
    sa.compact()
    assert len(sa._edge_src) == 0, "COO src buffer not reclaimed"
    assert len(sa._edge_dst) == 0, "COO dst buffer not reclaimed"
    assert len(sa._edge_weight) == 0, "COO weight buffer not reclaimed"
    assert len(sa._entity_edge_positions) == 0, (
        "_entity_edge_positions not reclaimed"
    )
    assert sa._substr_match_cache == {}
    assert not sa._dirty
    assert sa._adj is not None


def test_add_documents_after_compact():
    """Post-compact state equals loaded-from-disk state, so a later
    add_documents() must re-materialize COO from the CSR and keep
    serving correct results."""
    sa = _built_sa()
    sa.compact()
    sa.add_documents([{
        'id': 'post-compact',
        'content': 'spike of 999ms on 2025-08-01 after deploy',
        'value': 'post-compact-val',
    }])
    results = sa.query('999ms', top_k=5)
    assert any(v == 'post-compact-val' for v, _ in results), (
        f"doc added after compact() not retrievable: {results!r}"
    )
    # Pre-existing entities still answer.
    assert sa.query('2025-07-03', top_k=5)


def test_compact_bumps_generation():
    sa = _built_sa()
    gen = sa._graph_generation
    sa.compact()
    assert sa._graph_generation > gen, (
        "compact() must bump the graph generation so "
        "(n_nodes, generation)-keyed consumer caches invalidate"
    )


def test_compact_preserves_context_provider_and_lock():
    sa = _built_sa()
    provider = lambda q: []  # noqa: E731
    sa.set_context_provider(provider)
    lock = sa._build_lock
    sa.compact()
    assert sa._context_provider is provider
    assert sa._build_lock is lock, (
        "compact() must not replace the lock other threads may hold"
    )


def test_compact_unbuilt_raises():
    sa = SpreadingActivation()
    with pytest.raises(RuntimeError, match="requires a built graph"):
        sa.compact()


def test_compact_returns_self_for_chaining():
    sa = _built_sa()
    assert sa.compact() is sa


def test_compacted_graph_save_load_round_trips():
    sa = _built_sa()
    sa.compact()
    clone = SpreadingActivation.from_save_data(sa.get_save_data())
    for q in _QUERIES:
        assert clone.query(q, top_k=20) == sa.query(q, top_k=20)


def test_compact_under_concurrent_reader_load():
    """compact() is a writer: it must serialize behind _build_lock and
    never let a reader observe a torn graph (empty COO + unpublished
    CSR would surface as wrong/empty results or an exception)."""
    sa = _built_sa()
    expected = sa.query('2025-07-03', top_k=20)
    assert expected, "fixture must produce results for the probe query"

    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                got = sa.query('2025-07-03', top_k=20)
                if got != expected:
                    errors.append(f"torn read: {got!r}")
                    return
            except Exception as e:  # noqa: BLE001 -- test harness collector
                errors.append(repr(e))
                return

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(10):
            sa.compact()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
    assert not errors, f"reader saw inconsistent state during compact: {errors}"
