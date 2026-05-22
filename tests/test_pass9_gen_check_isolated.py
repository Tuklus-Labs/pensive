"""PENPY-P9-IMP-3: isolation test for the generation-check inside
_compile()'s publish step.

Pass-8 fix-6 (e58e492) added two protections in the publish-step
critical section of _compile():

    with self._build_lock:
        if not self._dirty:        # GUARD A -- redundant "already compiled"
            return
        if self._graph_generation != snap_generation:  # GUARD B -- the fix
            continue
        # ... publish ...

The pass-8 reproducer (tests/test_pass8_fix6_compile_race.py) exercises
the T1-snapshot, T2-publish, T1-stale-publish interleaving. In that
interleaving T2 calls add_documents() AND _compile(), so by the time
T1 resumes _dirty has already been flipped to False by T2's publish.
Result: GUARD A catches the test even when GUARD B is sabotaged.

P9 audit confirmed by sabotage gating that reverting JUST the gen-check
guard (GUARD B) at spreading.py:463 to `if False:` leaves all 218 tests
green. The gen-check is load-bearing for a DIFFERENT interleaving that
no existing test exercises:

    T1: snapshots gen=0 under lock, releases, slow CSR build
    T2: calls add_documents() -- bumps _graph_generation to 1 and sets
        _dirty=True, but does NOT call _compile(). _dirty STAYS True.
    T1: finishes stale CSR. Re-acquires lock. _dirty=True so GUARD A
        does NOT short-circuit. Only GUARD B (gen=1 != snap_generation=0)
        keeps T1 from publishing the stale CSR over T2's writes.

This test reproduces that interleaving deterministically. When GUARD B
is in place T1 must discard its stale CSR and retry against the newer
graph (which then includes T2's docs). When GUARD B is sabotaged T1
publishes the stale shape and queries for T2-only entities return [].
"""
import threading

import pytest
import scipy.sparse

from pensive import SpreadingActivation


def _doc_with_date(i: int, ent_idx: int) -> dict:
    month = (ent_idx % 12) + 1
    day = (ent_idx % 28) + 1
    return {
        "id": f"d{i}",
        "content": (
            f"Acme Corporation incident report dated 2098-{month:02d}-"
            f"{day:02d} reference {i}"
        ),
        "value": f"answer_{i}",
    }


def test_p9_imp3_gen_check_catches_concurrent_add_without_publish():
    """Interleaving:
       T1 snapshots, releases lock, starts slow CSR.
       T2 calls add_documents() ONLY (no _compile -- _dirty stays True).
       T1 resumes: GUARD A (_dirty short-circuit) does NOT save us
       because _dirty is True. Only GUARD B (generation check) prevents
       publishing the stale CSR over T2's writes.

    Sabotaging GUARD B at spreading.py:463 to `if False:` will make T1
    publish the stale shape and the T2-only-entity query returns [].
    """
    sa = SpreadingActivation()
    sa.add_documents([_doc_with_date(i, i) for i in range(5)])
    sa._compile()
    assert sa._adj.shape[0] == len(sa._idx_to_node)
    assert sa._dirty is False

    n_nodes_pre_race = len(sa._idx_to_node)

    import pensive.spreading as sp_mod
    original_csr = scipy.sparse.csr_matrix
    add_done_gate = threading.Event()
    t1_release_gate = threading.Event()

    def slow_csr_for_t1(*args, **kwargs):
        result = original_csr(*args, **kwargs)
        if threading.current_thread().name == "T1-genguard":
            # Signal T2 that T1 is about to re-acquire the lock for
            # publish. T2 has only enqueued add_documents() which bumps
            # the generation; it does NOT call _compile, so _dirty
            # stays True and only GUARD B can prevent the stale
            # publish.
            add_done_gate.set()
            t1_release_gate.wait(timeout=15)
        return result

    scipy.sparse.csr_matrix = slow_csr_for_t1
    sp_mod.scipy.sparse.csr_matrix = slow_csr_for_t1

    def t1_run():
        # T1 needs the graph dirty when it enters _compile so it takes
        # the snapshot + heavy-CSR-build path. add_documents leaves it
        # dirty.
        sa.add_documents([_doc_with_date(100, 100)])
        sa._compile()

    def t2_run():
        # Wait for T1 to have snapshotted and entered the slow CSR
        # build. Then ONLY add_documents -- do NOT call _compile.
        # The graph stays dirty, but _graph_generation has bumped.
        add_done_gate.wait(timeout=15)
        sa.add_documents([_doc_with_date(200 + i, 200 + i) for i in range(5)])
        # Release T1 so it tries to publish. Under the fix it must
        # observe the gen mismatch and retry; the retry rebuilds
        # against the now-larger graph and publishes the correct
        # shape.
        scipy.sparse.csr_matrix = original_csr
        sp_mod.scipy.sparse.csr_matrix = original_csr
        t1_release_gate.set()

    t1 = threading.Thread(target=t1_run, name="T1-genguard")
    t2 = threading.Thread(target=t2_run, name="T2-genguard")
    try:
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert not t1.is_alive(), "T1 deadlocked"
        assert not t2.is_alive(), "T2 deadlocked"
    finally:
        scipy.sparse.csr_matrix = original_csr
        sp_mod.scipy.sparse.csr_matrix = original_csr

    # Sanity: the in-memory graph reflects both T1's and T2's add_documents
    # calls (1 + 5 = 6 new docs on top of the original 5).
    assert len(sa._idx_to_node) > n_nodes_pre_race, (
        "the race fixture did not actually mutate the node table; the "
        "interleaving did not fire as expected"
    )

    # Invariant: the published CSR must reflect ALL writes (T1's +
    # T2's). The stale-publish (sabotaged-guard) failure mode is that
    # T1 publishes its CSR built against the smaller pre-T2 snapshot.
    # Then sa._adj.shape[0] < len(sa._idx_to_node) by exactly the count
    # of T2's new entity nodes. (sa._compile() at this point is a no-op
    # because _dirty=False from T1's stale publish.)
    assert sa._adj is not None
    assert sa._adj.shape[0] == len(sa._idx_to_node), (
        f"PENPY-P9-IMP-3 violated: GUARD B (generation check) failed "
        f"the gen-bump-without-publish interleaving. T1 published a "
        f"stale CSR built before T2's add_documents. "
        f"adj.shape={sa._adj.shape} vs n_nodes={len(sa._idx_to_node)}. "
        f"Confirm spreading.py:463 `if self._graph_generation != "
        f"snap_generation: continue` is intact in the publish step."
    )

    # Invariant: entities only T2 added are reachable through the
    # published CSR. This is the user-visible symptom of the sabotage.
    t2_month = (200 % 12) + 1
    t2_day = (200 % 28) + 1
    t2_date = f"2098-{t2_month:02d}-{t2_day:02d}"
    hits = sa.query(t2_date, top_k=10)
    assert hits, (
        f"PENPY-P9-IMP-3 violated: entity {t2_date!r} (only added by "
        f"T2) is unreachable through the published CSR -- T1's stale "
        f"shape clobbered T2's writes."
    )
