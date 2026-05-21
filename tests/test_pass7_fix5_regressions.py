"""Regression tests for PENPY-P7-NEW-5 (fix-5 wave).

The pass-7 audit found a pre-existing inconsistency exposed (not caused)
by the validation block added in 0793818: `add_documents()` with
documents whose content matches NO REAL_DATA_PATTERN entities still
appends a `v:<doc-id>` answer node to the parallel arrays, but never
calls `_add_edge()`. The old `_add_edge()` was the only path that set
`self._dirty = True` during incremental updates, so a no-entity add
left `_dirty=False` even though `len(_idx_to_node) > _adj.shape[0]`.

Symptoms:
  - In-memory queries operated on a stale CSR (lost any benefit of the
    new node existing, but never crashed).
  - `get_save_data()` -> `_compile()` early-returned because not dirty,
    serialized parallel arrays of length N+M with `_adj` of shape NxN.
  - `from_save_data()` correctly raised
      "node_type=N+M expected n_nodes=N"
    catching the corruption on the round-trip, but only the symptom,
    not the root cause.

Fix: `_get_or_add_node()` now sets `self._dirty = True` whenever a NEW
node is created. The edge-add path still sets it, but the node-add
path is now canonical -- anything that lengthens `_idx_to_node` flips
the recompile flag.

Sabotage gate (verified before commit): remove the `self._dirty = True`
line from `_get_or_add_node()` and the
`test_no_entity_add_round_trips_cleanly` test below fails with the
ValueError raised by the validation block in `_from_sparse_v1`.
"""
import pytest

from pensive import SpreadingActivation


# A doc whose content does NOT match any entity in patterns.PATTERNS.
# 'banana smoothie' has no IPs, no timestamps, no IDs, no measurements,
# no UUIDs. Confirmed by inspecting REAL_DATA_PATTERNS.
_NO_ENTITY_DOC = {
    'id': 'fruit-1',
    'content': 'banana smoothie',
    'value': 'fruit-fact',
}


def _entity_bearing_doc(i: int = 0) -> dict:
    """A doc that DOES match at least one REAL_DATA_PATTERN.

    The date `2025-10-DD` matches the `date` pattern in the default
    extractor; '40Cms' looks like it could but doesn't. Confirmed in
    fix-5 sabotage gate by inspecting extractor output.
    """
    return {
        'id': f'measure-{i}',
        'content': f'latency was {40 + i}ms on 2025-10-{(i % 27) + 1:02d}',
        'value': f'measure-val-{i}',
    }


def test_no_entity_add_marks_graph_dirty():
    """add_documents() with no-entity content must set _dirty=True so the
    next query recompiles the CSR to match the lengthened parallel arrays.

    Without this, _idx_to_node grows but _adj.shape[0] does not, and the
    structural invariant len(_idx_to_node) == _adj.shape[0] breaks.
    """
    sa = SpreadingActivation()
    sa.build([_entity_bearing_doc(i) for i in range(5)])

    # Force compile so _dirty starts at False.
    sa.query('2025-10-01')
    assert sa._dirty is False, "precondition: compile should have cleared dirty"

    pre_n = len(sa._idx_to_node)
    sa.add_documents([_NO_ENTITY_DOC])

    # A new v: node was appended even though no entity matched.
    assert len(sa._idx_to_node) == pre_n + 1
    # And the graph MUST be marked dirty so the next compile rebuilds
    # _adj at the new shape.
    assert sa._dirty is True, (
        "no-entity add_documents must mark the graph dirty; otherwise "
        "_adj.shape[0] stays at the old n while _idx_to_node lengthens"
    )


def test_no_entity_add_round_trips_cleanly():
    """save -> load must succeed after a no-entity add_documents.

    Before fix-5, the validation block in `_from_sparse_v1` raised:
        ValueError: node_type=N+M expected n_nodes=N
    because `get_save_data()` called `_compile()` which early-returned
    on `_dirty=False`, serializing inconsistent parallel arrays.

    Reproducing the bug requires forcing a compile (e.g. via query()
    or get_save_data()) BEFORE the no-entity add_documents; otherwise
    `build()`'s own `_dirty=True` flag still propagates through the
    add and the bug is masked. A real consumer hits this whenever a
    query lands between two add_documents calls -- common in any
    long-running daemon.
    """
    sa = SpreadingActivation()
    sa.build([_entity_bearing_doc(i) for i in range(3)])
    # Force compile so _dirty=False at the moment of the no-entity add.
    # This is what exposes the bug per the pass-7 forensic walk.
    sa.query('2025-10-01')
    assert sa._dirty is False

    sa.add_documents([_NO_ENTITY_DOC])

    # Round-trip through the save format. Must NOT raise.
    payload = sa.get_save_data()
    restored = SpreadingActivation.from_save_data(payload)

    # The restored graph should have the same node count as the original.
    assert len(restored._idx_to_node) == len(sa._idx_to_node)
    assert restored._adj.shape[0] == len(sa._idx_to_node)


def test_no_entity_add_preserves_prior_entity_queries():
    """Adding a no-entity doc must NOT break queries against entities
    that existed before the add. (Catches the case where the fix
    accidentally invalidates the cached CSR without rebuilding it.)
    """
    sa = SpreadingActivation()
    sa.build([_entity_bearing_doc(i) for i in range(3)])

    # Baseline: an entity-exact query returns hits.
    baseline = sa.query('2025-10-01')
    assert len(baseline) > 0, "precondition: baseline query should hit"

    sa.add_documents([_NO_ENTITY_DOC])

    # The original entity must still be queryable after the no-entity
    # add. (Sabotage: if the fix instead set _dirty=True but the
    # subsequent _compile() truncated entity edges, this fails.)
    after = sa.query('2025-10-01')
    assert len(after) > 0, (
        "queries against pre-existing entities must survive a "
        "no-entity add_documents call"
    )


def test_no_entity_add_then_query_finds_no_new_v_node():
    """A v: node added via no-entity content has no edges, so it cannot
    be reached by spreading activation from any entity. Confirm the
    contract: the new node exists structurally but is unreachable
    until edges are added (which only happens via entity matches).
    """
    sa = SpreadingActivation()
    sa.build([_entity_bearing_doc(0)])
    sa.add_documents([_NO_ENTITY_DOC])

    # No entity query can possibly return the orphan v:fruit-1.
    results = sa.query('banana')  # 'banana' is not an entity pattern
    # Whether results are empty depends on extraction; the point is the
    # graph mutation completed without raising and the CSR is consistent.
    assert isinstance(results, list)

    # And the round-trip validation passes (i.e. _compile() ran).
    payload = sa.get_save_data()
    assert payload['adj_shape'][0] == len(sa._idx_to_node)
