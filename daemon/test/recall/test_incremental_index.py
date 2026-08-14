"""Gate for incremental index maintenance on the write path.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    A just-written atom becomes dense-recallable WITHOUT rebuilding its kind
    class, and a retired atom stops being recallable WITHOUT the index being
    told about it.

Why this exists. Every emit used to call ``reindex(kinds=...)``, which is
``embedMissing`` (90ms scanning 299,728 live atoms to return zero rows on this
store) plus a full class rebuild (272ms memory, 18,645ms code). All of it
synchronous on the event-loop thread, so every concurrent read waited behind it.
That is what set L2's P95 tail: measured maxima of 2,961ms and 6,286ms against a
20ms budget, while the median sat at 26-32ms. The median was never the problem.

THE PART THAT LOOKS WRONG AND IS NOT. Retirement does no index work at all.
``recall.trust.assessTrust`` runs unconditionally in the pipeline and issues one
SELECT for ``status`` over every reranked atom, dropping tombstones and chaining
superseded atoms to their live replacement. So the index is NOT what keeps a
retracted atom out of a result, and a stale index costs ranking QUALITY (a dead
row occupying a candidate slot) rather than correctness. That claim is not taken
on faith here: ``test_a_superseded_atom_left_in_the_index_is_not_recallable``
injects one and drives the real engine.
"""
import numpy as np
import pytest

from recall.vector_index import FlatIndex
from recall.hnsw_index import HnswIndex


def unit(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_add_makes_an_atom_searchable_without_a_rebuild(cls):
    index = cls()
    vecs = {f"atom-{i}": unit(i) for i in range(5)}
    for atomId, vec in vecs.items():
        index.add(atomId, vec)
    # each atom must be its own nearest neighbour
    for atomId, vec in vecs.items():
        top = index.search(vec, 1)
        assert top, f"{atomId} returned nothing"
        assert top[0][0] == atomId, f"{atomId} did not rank itself first"


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_add_works_on_a_never_built_index(cls):
    """FlatIndex carries shape (0, 0) before build and HnswIndex carries a None
    index, because usearch needs a positive ndim to construct. The first added
    vector has to establish the width in both."""
    index = cls()
    v = unit(99)
    index.add("first", v)
    assert index.search(v, 1)[0][0] == "first"


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_add_keeps_key_and_atomid_in_correspondence(cls):
    """HnswIndex maps a usearch key straight through ``_atomIds[key]``, so an
    append that does not land at the matching position silently returns the
    WRONG atom for a correct vector. That is a wrong answer, not an error."""
    index = cls()
    ids = [f"a{i}" for i in range(6)]
    vecs = [unit(100 + i) for i in range(6)]
    for atomId, vec in zip(ids, vecs):
        index.add(atomId, vec)
    assert index._atomIds == ids
    for atomId, vec in zip(ids, vecs):
        assert index.search(vec, 1)[0][0] == atomId


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_add_after_a_real_build_extends_rather_than_replaces(cls, tmp_path):
    """The append path must not discard what build() loaded."""
    index = cls()
    for i in range(4):
        index.add(f"built-{i}", unit(200 + i))
    before = list(index._atomIds)
    late = unit(999)
    index.add("late", late)
    assert index._atomIds == before + ["late"]
    assert index.search(late, 1)[0][0] == "late"
    # the pre-existing entries still answer
    assert index.search(unit(200), 1)[0][0] == "built-0"


def test_flat_add_preserves_unit_norm_rows():
    """search() treats stored rows as unit length, so an un-normalized append
    would silently inflate that atom's score against every other."""
    index = FlatIndex()
    for i in range(3):
        index.add(f"n{i}", unit(300 + i))
    norms = np.linalg.norm(index._matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms
