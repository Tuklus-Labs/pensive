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

RETIREMENT NEEDS EXPLICIT INDEX WORK, and this file header once said the
opposite. The claim was that ``recall.trust.assessTrust`` resolves ``status``
per query and would therefore keep a retracted atom out of results by itself.
It does resolve status, and it does not DROP a superseded atom: it annotates it
with ``supersededBy`` at a capped confidence, so the retired claim still reaches
the payload beside the current one. A rebuild used to prevent that implicitly by
loading live atoms only, so incremental maintenance has to do it explicitly,
which is why ``add`` and ``remove`` are a pair.

That paragraph also cited
``test_a_superseded_atom_left_in_the_index_is_not_recallable`` as its evidence.
No such test was ever written. A docstring citing a test that does not exist is
worse than one citing nothing, because it reads as verification. The real
evidence is ``test_correct_supersedes_and_recall_shows_current_truth`` in
test/serve/test_mcp.py, which is what actually caught the error.

Corrected in place rather than rewritten silently, because the wrong version
shipped in a commit.
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
    assert index._atomIds == ids, (
        "key-to-atomId correspondence broken: search maps a usearch key "
        "through _atomIds[key], so a mismatch returns the WRONG ATOM for a "
        "correct vector, which is a wrong answer and not an error. "
        f"expected {ids} got {index._atomIds}")
    for atomId, vec in zip(ids, vecs):
        assert index.search(vec, 1)[0][0] == atomId, (
            "key-to-atomId correspondence broken: an atom queried with its "
            f"OWN vector returned {index.search(vec, 1)[0][0]!r}, not {atomId!r}")


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


# --------------------------------------------------------------------------- #
# retirement, on BOTH implementations                                           #
# --------------------------------------------------------------------------- #
#
# These are parametrized over both index types on purpose. A sabotage pass found
# that skipping HnswIndex.remove entirely left the whole suite GREEN, because
# every store a test builds sits below HNSW_THRESHOLD (5,000) and therefore only
# ever exercises FlatIndex. In production BOTH kind classes are past that
# threshold (memory 16,763, code 282,985), so the untested implementation is the
# one that actually runs. Coverage that inverts between test and production is
# worse than no coverage, because it reads as coverage.
#
# Each test asserts INCLUSION before it asserts exclusion, in the same run on the
# same input. Without that first assertion an exclusion test cannot fail: the
# atom may be absent because the mechanism worked, or because it was never going
# to rank. Two probes in this campaign returned the expected answer for the wrong
# reason before that rule was written down.


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_remove_stops_an_atom_being_returned(cls):
    index = cls()
    vecs = {f"r{i}": unit(400 + i) for i in range(5)}
    for atomId, vec in vecs.items():
        index.add(atomId, vec)

    victim, victimVec = "r2", vecs["r2"]
    # INCLUSION FIRST: queried with its own vector it must rank first, or this
    # test has no power and its later assertion proves nothing.
    assert index.search(victimVec, 5)[0][0] == victim, (
        "THIS TEST HAS NO POWER: the victim must rank first BEFORE removal, "
        "or the absence assertion below cannot fail and proves nothing. "
        f"top-5 was {[a for a, _ in index.search(victimVec, 5)]}")

    assert index.remove(victim) is True, (
        "remove() contract violated: it must report True when the atom was "
        f"present, and {victim!r} was just asserted to rank first")
    got = [a for a, _ in index.search(victimVec, 5)]
    assert victim not in got, f"{victim} survived removal: {got}"
    # and the survivors are untouched
    assert set(got) == set(vecs) - {victim}


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_remove_clears_every_copy_of_a_duplicated_atom(cls):
    """`list.index` returns ONE position and an atom can occupy several.

    `add` appends, so an atom already present from a build gains a second row
    when indexAtom runs for it again: an emit retry, a redelivered MCP call.
    Masking only the first left `remove` returning True while search still
    returned the atom. That is a retracted memory served, with the API reporting
    success, which is the worst shape available here.

    Found by a clean-pass audit against BOTH implementations, after the first fix
    for this class of bug shipped with the hole still in it."""
    index = cls()
    v, w = unit(900), unit(901)
    # force the duplicate the way production would: add, add something else,
    # add the same atom again
    index._atomIds.append("dup")
    if hasattr(index, "_matrix"):
        import numpy as np
        index._matrix = np.vstack([index._matrix, v.reshape(1, -1)]) \
            if index._matrix.shape[0] else v.reshape(1, -1).copy()
    else:
        pass
    index2 = cls()
    index2.add("dup", v)
    index2.add("other", w)
    index2._atomIds.append("dup")          # a raw second copy, as a rebuild+add produces
    if hasattr(index2, "_matrix"):
        import numpy as np
        index2._matrix = np.vstack([index2._matrix, v.reshape(1, -1)])
    else:
        index2._index.add(len(index2._atomIds) - 1, v)

    before = [a for a, _ in index2.search(v, 5)]
    assert "dup" in before, (
        "THIS TEST HAS NO POWER: the duplicated atom must be returned before "
        f"removal, or the assertion below cannot fail. got {before}")

    assert index2.remove("dup") is True
    after = [a for a, _ in index2.search(v, 5)]
    assert "dup" not in after, (
        "remove() cleared only ONE copy: a retracted atom is still recallable "
        f"while remove reported success. got {after}")


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_add_is_idempotent_by_atom_id(cls):
    """Re-adding an atom must not leave two live rows for it, or every later
    remove has to find them all. Keeping the index single-valued is what makes
    that guarantee cheap."""
    index = cls()
    v = unit(910)
    index.add("same", v)
    index.add("same", v)
    got = [a for a, _ in index.search(v, 5)]
    assert got.count("same") == 1, (
        f"an atom added twice is returned {got.count('same')} times; the index "
        "is no longer single-valued")


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_remove_is_honest_about_an_unknown_atom(cls):
    index = cls()
    index.add("present", unit(500))
    assert index.remove("never-added") is False, (
        "remove() contract violated: an unknown atom must report False "
        "rather than raising or silently reporting success")
    assert index.search(unit(500), 1)[0][0] == "present"


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_remove_then_add_keeps_later_atoms_addressable(cls):
    """FlatIndex masks a row POSITION and HnswIndex drops a usearch key, and in
    both the surviving entries keep their slots. An implementation that
    renumbered would hand back the wrong atom id for a correct vector, which is
    a wrong answer rather than an error."""
    index = cls()
    ids = [f"s{i}" for i in range(4)]
    vecs = [unit(600 + i) for i in range(4)]
    for atomId, vec in zip(ids, vecs):
        index.add(atomId, vec)
    index.remove("s1")
    late = unit(700)
    index.add("late", late)

    assert index.search(late, 1)[0][0] == "late"
    for atomId, vec in zip(ids, vecs):
        if atomId == "s1":
            continue
        assert index.search(vec, 1)[0][0] == atomId, (
            "key-to-atomId correspondence broken: an atom queried with its "
            f"OWN vector returned {index.search(vec, 1)[0][0]!r}, not {atomId!r}")


@pytest.mark.parametrize("cls", [FlatIndex, HnswIndex])
def test_removing_every_atom_leaves_an_empty_result_not_a_crash(cls):
    index = cls()
    v = unit(800)
    index.add("only", v)
    assert index.search(v, 3)[0][0] == "only", (
        "THIS TEST HAS NO POWER: the atom must be returned before removal, "
        "or the empty-result assertion below cannot fail")
    index.remove("only")
    assert index.search(v, 3) == [], (
        "an index whose every atom is retired must return no results; "
        f"got {index.search(v, 3)}")


def test_flat_add_preserves_unit_norm_rows():
    """search() treats stored rows as unit length, so an un-normalized append
    would silently inflate that atom's score against every other."""
    index = FlatIndex()
    for i in range(3):
        index.add(f"n{i}", unit(300 + i))
    norms = np.linalg.norm(index._matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms
