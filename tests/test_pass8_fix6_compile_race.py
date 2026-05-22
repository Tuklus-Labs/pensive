"""Regression test for PENPY-P8-CRIT-1: _compile() publish-outside-lock race.

The pass-7 fix-5 audit explicitly deferred this race as a pass-8
candidate (see tests/test_pass7_fix5_concurrency_mixed.py module
docstring). Pass-8 confirmed the race deterministically using a
monkey-patched scipy.sparse.csr_matrix that gates one writer's publish
until another writer has completed a faster compile.

Race shape:

    T1: enters _compile(), takes snapshot under _build_lock,
        releases lock, starts building CSR (slow).
    T2: calls add_documents() (bumps _graph_generation),
        enters _compile(), takes a newer snapshot, builds CSR,
        publishes (_adj reflects current state, _dirty=False).
    T1: finishes its (now stale) CSR build, publishes UNCONDITIONALLY,
        clobbering T2's fresher _adj and re-setting _dirty=False.

Observed impacts (all CRITICAL): silent wrong query results (entities
present in _idx_to_node but unreachable through the stale _adj),
save/load ValueError on the n_nodes invariant, and permanently-stuck
state because _dirty=False short-circuits the next _compile() at the
top.

Fix (fix-6): snapshot _graph_generation alongside the COO buffers.
Build the CSR outside the lock. Re-acquire the lock at publish time
and verify _graph_generation hasn't moved. If it has, discard our
stale CSR and retry. After a bounded number of optimistic retries (8),
fall back to a synchronous build+publish under the lock to guarantee
forward progress for readers (preventing reader starvation under
sustained writer contention).

Sabotage gates verified during fix-6:

1. Revert the generation check at publish time -- the deterministic
   reproducer fires SHAPE MISMATCH + STATE PERMANENTLY STUCK within
   one iteration.
2. Drop the synchronous fallback (always retry-and-give-up) -- the
   pass-5 starvation test fails with "readers stalled (queries=0)"
   because _adj never gets initialized under aggressive writer load.
"""
import os
import threading
import time

import numpy as np
import pytest
import scipy.sparse

from pensive import SpreadingActivation


def _doc(i: int, ent_idx: int = 0) -> dict:
    # Use a fully-extractable date entity that varies per doc so we can
    # query for an entity that ONLY exists in T2's writes. Pattern is
    # YYYY-MM-DD which the default REAL_DATA_PATTERNS matches verbatim.
    # Months stay in 01-12 and days in 01-28 so the date is always valid
    # and recognized by the date pattern.
    month = (ent_idx % 12) + 1
    day = (ent_idx % 28) + 1
    return {
        "id": f"d{i}",
        "content": (
            f"Acme Corporation incident report dated 2099-{month:02d}-"
            f"{day:02d} reference {i}"
        ),
        "value": f"answer_{i}",
    }


def test_p8_crit1_compile_race_deterministic_reproducer():
    """Force the exact T1-snapshot, T2-publish, T1-stale-publish
    interleaving with a monkey-patched CSR constructor that gates T1
    until T2 has completed. The fix must:

      - leave _adj.shape consistent with len(_idx_to_node)
      - leave _dirty correctly reflecting in-memory state
      - return correct query results for entities only T2 added
      - allow save/load round-trip
    """
    sa = SpreadingActivation()
    sa.add_documents([_doc(i, i) for i in range(5)])
    sa._compile()
    assert sa._adj.shape[0] == len(sa._idx_to_node)
    assert sa._dirty is False

    import pensive.spreading as sp_mod
    original_csr = scipy.sparse.csr_matrix
    publish_gate = threading.Event()
    release_gate = threading.Event()

    def slow_csr_for_t1(*args, **kwargs):
        result = original_csr(*args, **kwargs)
        if threading.current_thread().name == "T1":
            publish_gate.set()
            # Bounded wait: if the fix re-enters _compile via the
            # fallback path, the test could deadlock without a timeout.
            release_gate.wait(timeout=15)
        return result

    # Patch both the module-level reference (which spreading.py
    # uses via `import scipy`) and the package attribute.
    scipy.sparse.csr_matrix = slow_csr_for_t1
    sp_mod.scipy.sparse.csr_matrix = slow_csr_for_t1

    def t1_run():
        sa.add_documents([_doc(100, 100)])
        sa._compile()

    def t2_run():
        publish_gate.wait(timeout=15)
        # Restore real CSR for T2 so its build doesn't block.
        sp_mod.scipy.sparse.csr_matrix = original_csr
        scipy.sparse.csr_matrix = original_csr
        sa.add_documents([_doc(200 + i, 200 + i) for i in range(5)])
        sa._compile()
        # Reinstate the gating wrapper for T1's resumption so T1's
        # publish path exercises the generation check on the wrapper
        # too (not strictly necessary, but matches the reproducer).
        sp_mod.scipy.sparse.csr_matrix = slow_csr_for_t1
        scipy.sparse.csr_matrix = slow_csr_for_t1
        release_gate.set()

    t1 = threading.Thread(target=t1_run, name="T1")
    t2 = threading.Thread(target=t2_run, name="T2")
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

    # Invariant 1: the published CSR shape matches the node count.
    # Pre-fix this was the diagnostic that fired: shape (8, 8) vs
    # n_nodes 13.
    assert sa._adj is not None, (
        "race left _adj=None; the synchronous-fallback path didn't fire"
    )
    assert sa._adj.shape[0] == len(sa._idx_to_node), (
        f"_compile race regression: adj.shape={sa._adj.shape} vs "
        f"n_nodes={len(sa._idx_to_node)} -- a stale CSR was published "
        f"over a fresher one"
    )

    # Invariant 2: queries for date entities only T2 added must find
    # them. T2 added docs with ent_idx in {200..204}, which produce
    # date strings like 2099-09-25 that only T2's docs reference. T1's
    # docs use ent_idx={0..4, 100}, producing different dates. Pre-fix
    # this returned [] because the CSR was the (smaller) stale shape
    # T1 published, missing the rows/cols for T2's nodes.
    t2_month = (200 % 12) + 1
    t2_day = (200 % 28) + 1
    t2_date = f"2099-{t2_month:02d}-{t2_day:02d}"
    hits = sa.query(t2_date, top_k=10)
    assert hits, (
        f"query for entity {t2_date!r} (only added by T2) returned "
        f"[] -- the stale CSR was published, hiding T2's writes"
    )

    # Invariant 3: save/load round-trip succeeds. The n_nodes
    # invariant in _from_sparse_v1 fires loudly when the CSR is stale.
    data = sa.get_save_data()
    sa2 = SpreadingActivation.from_save_data(data)
    assert sa2._adj.shape[0] == len(sa2._idx_to_node)


def test_p8_crit1_race_does_not_starve_readers():
    """Pass-5 starvation test recurrence guard: the fix-6 generation
    check must not block readers when writers are aggressive.

    Configuration matches tests/test_pass5_p5_concurrency.py: 4
    writers + 4 readers for 5 seconds. If the synchronous-fallback
    path is broken or omitted, readers see _adj=None and crash on
    .indptr -- or the optimistic retry never converges and read_counter
    stays at zero.
    """
    sa = SpreadingActivation()

    def _make_doc(i):
        return {
            'id': f'd-{i}',
            'content': (
                f'latency on 2025-07-{(i % 27) + 1:02d} was '
                f'{(i * 13) % 500 + 50}ms for TENANT-{i % 32}'
            ),
            'value': f'{(i * 13) % 500 + 50}ms',
            'query': f'P99 latency TENANT-{i % 32}',
        }

    sa.build([_make_doc(i) for i in range(50)])

    stop = threading.Event()
    errors = []
    errors_lock = threading.Lock()
    write_counter = 0
    write_lock = threading.Lock()
    read_counter = 0
    read_lock = threading.Lock()

    def writer(worker_id):
        nonlocal write_counter
        i = 1_000 + worker_id * 100_000
        try:
            while not stop.is_set():
                sa.add_documents([_make_doc(i + j) for j in range(25)])
                i += 25
                with write_lock:
                    write_counter += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(('writer', e))

    def reader(worker_id):
        nonlocal read_counter
        q = f'P99 latency TENANT-{worker_id}'
        try:
            while not stop.is_set():
                sa.query(q, top_k=10)
                with read_lock:
                    read_counter += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(('reader', e))

    threads = []
    for w in range(4):
        t = threading.Thread(target=writer, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for r in range(4):
        t = threading.Thread(target=reader, args=(r,), daemon=True)
        t.start()
        threads.append(t)

    duration = float(os.environ.get('PENSIVE_CONCURRENCY_DURATION', '5.0'))
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=15.0)

    assert write_counter > 5, f'writers stalled: {write_counter}'
    assert read_counter > 5, (
        f'readers starved: {read_counter} (the fix-6 generation check '
        f'lost forward progress)'
    )
    if errors:
        origin, first = errors[0]
        raise AssertionError(
            f'unexpected error in {origin}: '
            f'{type(first).__name__}: {first}'
        ) from first

    # Final invariant: state is consistent and adj is populated.
    sa._compile()
    assert sa._adj is not None
    assert sa._adj.shape[0] == len(sa._idx_to_node)
    assert sa._dirty is False
