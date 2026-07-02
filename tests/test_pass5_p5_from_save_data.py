"""Regression tests for PENPY-P5-IMP-2.

The pass-5 deep audit found that SpreadingActivation.from_save_data()
trusts adj_indices / adj_indptr / adj_shape literally. A corrupted
(truncated, tampered, or half-written) save with adj_indices >= n_nodes
does not error at load time; the OOB index survives into the numba JIT
spread kernel and segfaults Python on the first query that seeds the
corresponding entity.

Fix: _from_sparse_v1 now calls scipy.sparse.csr_matrix.check_format(
full_check=True) immediately after constructing the matrix, and wraps
any scipy structural exception into ValueError so callers (load_graph
in particular) catch corruption uniformly. IngestPipeline.load_graph
was also broadened to catch ValueError/IndexError around the
from_save_data call.

Sabotage gate verified during fix-3: removing the check_format()
invocation from _from_sparse_v1 makes the OOB payload accepted
silently, and both tests below fail "DID NOT RAISE ValueError".

These tests intentionally use a malformed-but-non-segfaulting payload
(adj_indices points to an OOB column but adj_indptr is consistent with
a single edge from row 0 to that bogus column); check_format catches it
synchronously, so we don't need to risk a real segfault inside the
pytest worker.
"""
import array
import hashlib
import hmac
import pickle

import numpy as np
import pytest

from pensive import SpreadingActivation


def _oob_payload(n_nodes: int, oob_col: int) -> dict:
    """Build a sparse_v1 save payload with adj_indices outside [0, n_nodes)."""
    assert oob_col >= n_nodes, "oob_col must actually be out of bounds"
    return {
        'format': 'sparse_v1',
        'node_to_idx': {'e:tenant': 0, 'v:doc-0': 1},
        'idx_to_node': ['e:tenant', 'v:doc-0'],
        'node_type': array.array('b', [0, 1]),
        'node_label': ['tenant', 'doc-0'],
        'node_specificity': [1.0, 1.0],
        'adj_data': np.array([0.5], dtype=np.float32),
        'adj_indices': np.array([oob_col], dtype=np.int32),
        'adj_indptr': np.array([0, 1, 1], dtype=np.int32),
        'adj_shape': (n_nodes, n_nodes),
        'entity_freq': {'tenant': 1},
        'entity_index': {'tenant': ['e:tenant']},
        'is_bipartite': True,
    }


def test_p5_imp2_from_save_data_rejects_oob_adj_indices():
    """A save payload with adj_indices outside [0, n_nodes) must raise
    ValueError, not segfault and not silently accept bad data."""
    payload = _oob_payload(n_nodes=2, oob_col=999_999)

    with pytest.raises(ValueError, match=r'(?i)corrupt|invalid|index|bound'):
        SpreadingActivation.from_save_data(payload)


def test_p5_imp2_load_graph_wraps_oob_as_value_error(tmp_path):
    """IngestPipeline.load_graph must also surface the OOB rejection as
    a ValueError (not let the underlying scipy exception escape)."""
    from pensive.ingestion.pipeline import IngestPipeline
    from pensive.ingestion import pipeline as pipeline_mod

    # Save a real graph first so the on-disk file structure (signed
    # header + HMAC + payload) is valid; we then overwrite the payload
    # with our malformed one re-signed under the same key.
    sa = SpreadingActivation()
    sa.build([{'id': 'seed', 'content': 'placeholder', 'value': 'x'}])
    pipe = IngestPipeline(sa=sa)
    out = tmp_path / 'graph.pkl'
    pipe.save_graph(str(out))

    payload = _oob_payload(n_nodes=2, oob_col=42)
    payload['pipeline_stats'] = {}
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    key = pipeline_mod._load_or_create_key()
    mac = hmac.new(key, body, hashlib.sha256).digest()
    with open(out, 'wb') as fh:
        fh.write(pipeline_mod._SIGNED_MAGIC)
        fh.write(mac)
        fh.write(body)

    with pytest.raises(ValueError):
        IngestPipeline.load_graph(str(out))


# PENPY-P6-MIN-1: parallel-array length consistency on load
#
# check_format only validates the CSR adjacency. It does NOT verify that
# node_type / idx_to_node / node_label have the same length as n_nodes.
# A corrupted save with mismatched parallel arrays was previously
# accepted silently; subsequent queries returned empty or behaved
# inconsistently with no crash. _from_sparse_v1 now raises ValueError
# with a descriptive message when any of these length-mismatch.
# Sabotage gate: deleting the length-mismatch checks restores the
# silent-accept behavior and these tests fail "DID NOT RAISE".


def _valid_payload(n_nodes: int = 2) -> dict:
    """Build a valid sparse_v1 payload to mutate per-test.

    A self-loop on node 0 keeps adj_indices/adj_indptr structurally
    valid; mutating only the parallel arrays isolates the MIN-1 path.
    """
    return {
        'format': 'sparse_v1',
        'node_to_idx': {f'e:n{i}': i for i in range(n_nodes)},
        'idx_to_node': [f'e:n{i}' for i in range(n_nodes)],
        'node_type': array.array('b', [0] * n_nodes),
        'node_label': [f'n{i}' for i in range(n_nodes)],
        'node_specificity': [1.0] * n_nodes,
        'adj_data': np.array([0.5], dtype=np.float32),
        'adj_indices': np.array([0], dtype=np.int32),
        'adj_indptr': np.array([0, 1] + [1] * (n_nodes - 1), dtype=np.int32),
        'adj_shape': (n_nodes, n_nodes),
        'entity_freq': {},
        'entity_index': {},
        'is_bipartite': True,
    }


def test_p6_min1_rejects_node_type_length_mismatch():
    """node_type length != n_nodes must raise ValueError mentioning node_type."""
    payload = _valid_payload(n_nodes=2)
    payload['node_type'] = array.array('b', [0])  # length 1, expected 2

    with pytest.raises(ValueError, match=r'node_type=1 expected n_nodes=2'):
        SpreadingActivation.from_save_data(payload)


def test_p6_min1_rejects_idx_to_node_length_mismatch():
    """idx_to_node length != n_nodes must raise ValueError mentioning idx_to_node."""
    payload = _valid_payload(n_nodes=2)
    payload['idx_to_node'] = ['e:n0']  # length 1, expected 2

    with pytest.raises(ValueError, match=r'idx_to_node=1 expected n_nodes=2'):
        SpreadingActivation.from_save_data(payload)


def test_p6_min1_rejects_node_label_length_mismatch():
    """node_label length != n_nodes must raise ValueError mentioning node_label."""
    payload = _valid_payload(n_nodes=2)
    payload['node_label'] = ['n0']  # length 1, expected 2

    with pytest.raises(ValueError, match=r'node_label=1 expected n_nodes=2'):
        SpreadingActivation.from_save_data(payload)


def test_p6_min1_valid_payload_still_loads():
    """A correctly-shaped payload must still load successfully -- guard
    against an over-eager rejection breaking the happy path."""
    payload = _valid_payload(n_nodes=2)
    sa = SpreadingActivation.from_save_data(payload)
    assert sa._adj.shape == (2, 2)
    assert len(sa._node_type) == 2
    assert len(sa._idx_to_node) == 2
    assert len(sa._node_label) == 2
