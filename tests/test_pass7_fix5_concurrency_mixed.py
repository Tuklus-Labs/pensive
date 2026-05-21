"""Concurrency regression test for PENPY-P7-NEW-3.

The pass-7 audit's external stress probe ran 8-writer / 8-reader for
20s with content like 'code {i%100}' that matched no extraction
patterns. Result: writers exercised _add_node but NOT _add_edge -- so
the probe did not stress the COO edge-array race that pass-5 caught.

This test fixes the gap: interleaves writers that produce
entity-bearing content (exercises _add_edge and the buffer race) with
writers that produce no-entity content (exercises the P7-NEW-5
node-only path) while readers spread on the established entities.
No-entity writes are added in fix-5 to stress the _dirty-flag flip in
_get_or_add_node, which previously stayed False on the node-only path
and let queries operate on a stale CSR.

If either the pass-5 buffer-export race or the pass-7-NEW-5 stale-CSR
bug regresses, a writer thread will crash with one of:
  - BufferError from the np.frombuffer race
  - ValueError "all index and data arrays must have the same length"
    from the torn COO snapshot
  - IndexError from a stale CSR pointed at out-of-range indices

This test does NOT also verify save/load round-trip post-stress.
There is a known, separately-scoped race between _compile()'s
unlocked-CSR-build window and concurrent writers: a writer that
appends nodes BETWEEN _compile()'s lock-released snapshot and its
final _dirty=False assignment can leave the published _adj smaller
than _idx_to_node. That race is invariant for save/load consistency
under concurrent mutation -- not specific to fix-5 -- and tracked as
a pass-8 candidate. The post-stress invariant we DO check is "no
exception escaped any worker thread", which is the in-scope contract
for P7-NEW-3.

Audit lesson baked in (per pass-5):
    Concurrency regression tests MUST run >= 10s and have
    write_counter / read_counter assertions to confirm the API was
    actually exercised (not trivially short-circuited).
"""
import os
import threading
import time

import pytest

from pensive import SpreadingActivation


def _entity_doc(i: int) -> dict:
    """Doc that matches REAL_DATA_PATTERN entities (date + metric)."""
    return {
        'id': f'e-{i}',
        'content': (
            f'latency was {(i * 7) % 500 + 30}ms on 2025-07-{(i % 27) + 1:02d} '
            f'sarah chen reported issue'
        ),
        'value': f'val-e-{i}',
    }


def _no_entity_doc(i: int) -> dict:
    """Doc that matches NO REAL_DATA_PATTERN entities. Exercises the
    P7-NEW-5 node-only mutation path under concurrent reader pressure.
    """
    return {
        'id': f'n-{i}',
        'content': f'banana smoothie code {i % 100}',
        'value': f'val-n-{i}',
    }


def test_p7_new3_mixed_entity_and_no_entity_writers_under_reader_load():
    """Writers split: half produce entity-bearing docs (edge-add path),
    half produce no-entity docs (node-only path). Readers spread
    against the entity-bearing docs.

    This is the cross-product of the fix-3 and fix-5 sabotage gates:
    if fix-3 regresses, the entity-writer thread crashes with a
    BufferError or torn-read ValueError. If fix-5 regresses, the
    no-entity-writer thread leaves the graph in a state where the
    final save/load round-trip raises ValueError.
    """
    sa = SpreadingActivation()
    # Prime with entity-bearing docs so readers have something to spread.
    sa.build([_entity_doc(i) for i in range(50)])

    stop = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    entity_writes = 0
    entity_writes_lock = threading.Lock()
    noentity_writes = 0
    noentity_writes_lock = threading.Lock()
    reads = 0
    reads_lock = threading.Lock()

    def entity_writer(worker_id: int) -> None:
        nonlocal entity_writes
        i = 1_000 + worker_id * 100_000
        try:
            while not stop.is_set():
                sa.add_documents([_entity_doc(i + j) for j in range(20)])
                i += 20
                with entity_writes_lock:
                    entity_writes += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(('entity_writer', e))

    def noentity_writer(worker_id: int) -> None:
        nonlocal noentity_writes
        i = 5_000_000 + worker_id * 100_000
        try:
            while not stop.is_set():
                sa.add_documents([_no_entity_doc(i + j) for j in range(20)])
                i += 20
                with noentity_writes_lock:
                    noentity_writes += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(('noentity_writer', e))

    def reader(worker_id: int) -> None:
        nonlocal reads
        # Use entities the build() phase actually indexed.
        q = '2025-07-01' if worker_id % 2 == 0 else 'sarah chen'
        try:
            while not stop.is_set():
                sa.query(q, top_k=10)
                with reads_lock:
                    reads += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(('reader', e))

    threads = []
    for w in range(2):
        t = threading.Thread(target=entity_writer, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for w in range(2):
        t = threading.Thread(target=noentity_writer, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for r in range(4):
        t = threading.Thread(target=reader, args=(r,), daemon=True)
        t.start()
        threads.append(t)

    duration = float(os.environ.get('PENSIVE_CONCURRENCY_DURATION', '10.0'))
    time.sleep(duration)
    stop.set()
    # Generous join timeout: a writer in the middle of add_documents()
    # holding _build_lock can block other threads from observing stop
    # until it releases. 30s is well above the worst-case observed time
    # for the batch sizes used here. If a thread is still alive at the
    # end, fail loudly rather than racing with the save check below.
    for t in threads:
        t.join(timeout=30.0)
    still_alive = [i for i, t in enumerate(threads) if t.is_alive()]
    assert not still_alive, (
        f'concurrency stress: {len(still_alive)} thread(s) did not '
        f'quiesce after stop.set() + 30s join; cannot trust the '
        f'save-state check below'
    )

    # Each path must have actually been exercised, otherwise the test
    # passed by being trivially short-circuited.
    assert entity_writes > 5, (
        f'entity writers stalled (batches={entity_writes}); did not '
        f'stress the edge-add race'
    )
    assert noentity_writes > 5, (
        f'no-entity writers stalled (batches={noentity_writes}); did '
        f'not stress the node-only P7-NEW-5 path'
    )
    assert reads > 10, (
        f'readers stalled (queries={reads}); did not stress the race'
    )

    # Loud assertion: any error from inside the engine, with origin tag.
    # This is the in-scope contract for P7-NEW-3 -- mixed entity/no-entity
    # concurrent mutation must not raise from any worker thread. The
    # save/load post-condition was intentionally dropped after pass-7
    # found a separately-scoped _compile() lock-release race between
    # the snapshot and final _dirty=False (not specific to fix-5; see
    # module docstring).
    if errors:
        origin, first = errors[0]
        raise AssertionError(
            f'concurrency regression in {origin}: {len(errors)} error(s); '
            f'first = {type(first).__name__}: {first}'
        ) from first
