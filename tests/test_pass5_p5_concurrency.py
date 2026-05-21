"""Concurrency regression tests for PENPY-P5-CRIT-1.

Audit lesson baked into this file (pass-4 vs pass-5 disagreement):
    Pass-4's concurrency test was insufficient -- it ran a short window
    that didn't reliably hit the buffer-export race window. Pass-5's
    longer/wider test caught it within seconds.

    Therefore: concurrency regression tests in this codebase MUST run
    for >= 10 seconds with multiple writers + multiple readers. A 1-2
    second window can pass by chance and a fix-verification on that
    window proves nothing.

PENPY-P5-CRIT-1: add_documents + concurrent query crashes writer with
BufferError exported by reader-side np.frombuffer over the array.array
COO storage, plus a follow-on torn-read race where the three
_edge_src/_edge_dst/_edge_weight arrays are copied one at a time and
a writer's append between copies leaves them at different lengths.

Sabotage gates verified during fix-3:

1. Revert _compile() to use np.frombuffer over self._edge_* (live
   array.array buffers): BufferError fires within seconds.
2. Drop the build_lock acquisition in add_documents() Phase 2: torn-read
   ValueError ("all index and data arrays must have the same length")
   fires within seconds.

Both sabotages were run and confirmed before this file was committed.
"""
import os
import threading
import time

import pytest

from pensive import SpreadingActivation


def _make_doc(i: int) -> dict:
    """Make a small doc with enough entity surface to keep extractor busy."""
    return {
        'id': f'd-{i}',
        'content': (
            f'System latency on 2025-07-{(i % 27) + 1:02d} was '
            f'{(i * 13) % 500 + 50}ms at the 99th percentile. '
            f'Tenant TENANT-{i % 32} reported timeout '
            f'after {i % 9}s on endpoint /api/v{i % 4}/data.'
        ),
        'value': f'{(i * 13) % 500 + 50}ms',
        'query': f'P99 latency on tenant TENANT-{i % 32}',
    }


def test_p5_crit1_concurrent_writers_readers_no_buffer_error():
    """4 writers + 4 readers for >= 10s with no BufferError, no crash.

    Reproduces the pass-5 stress that exposed the np.frombuffer race
    where reader-side _compile() exported buffer views over the
    array.array storage that writers were trying to append to.
    """
    sa = SpreadingActivation()
    # Prime the graph so query() has something to spread over.
    sa.build([_make_doc(i) for i in range(50)])

    stop = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    write_counter = 0
    write_counter_lock = threading.Lock()
    read_counter = 0
    read_counter_lock = threading.Lock()

    def writer(worker_id: int) -> None:
        nonlocal write_counter
        i = 1_000 + worker_id * 100_000
        try:
            while not stop.is_set():
                batch = [_make_doc(i + j) for j in range(25)]
                i += 25
                sa.add_documents(batch)
                with write_counter_lock:
                    write_counter += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(e)

    def reader(worker_id: int) -> None:
        nonlocal read_counter
        q = f'P99 latency tenant TENANT-{worker_id}'
        try:
            while not stop.is_set():
                sa.query(q, top_k=10)
                with read_counter_lock:
                    read_counter += 1
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(e)

    threads = []
    for w in range(4):
        t = threading.Thread(target=writer, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for r in range(4):
        t = threading.Thread(target=reader, args=(r,), daemon=True)
        t.start()
        threads.append(t)

    # Audit-mandated minimum: >= 10 seconds with multiple writers + readers.
    duration = float(os.environ.get('PENSIVE_CONCURRENCY_DURATION', '10.0'))
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)

    # Each side must have actually exercised the API; otherwise the test
    # passed by being trivially short-circuited.
    assert write_counter > 10, (
        f'writers stalled (batches={write_counter}); test did not exercise '
        f'the race window'
    )
    assert read_counter > 10, (
        f'readers stalled (queries={read_counter}); test did not exercise '
        f'the race window'
    )

    # Loud assertion: any BufferError or crash from inside the engine
    # fails the test with the original exception attached.
    if errors:
        first = errors[0]
        raise AssertionError(
            f'concurrency regression: {len(errors)} error(s) in '
            f'{duration}s; first = {type(first).__name__}: {first}'
        ) from first


def test_p5_crit1_single_writer_many_readers_no_buffer_error():
    """1 writer + 7 readers config from the pass-5 finding.

    Pass-4 explicitly claimed this configuration "empirically passes";
    pass-5 proved it does not. This test runs the exact failing config
    long enough to surface the race deterministically.
    """
    sa = SpreadingActivation()
    sa.build([_make_doc(i) for i in range(50)])

    stop = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def writer() -> None:
        i = 5_000
        try:
            while not stop.is_set():
                sa.add_documents([_make_doc(i + j) for j in range(10)])
                i += 10
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(e)

    def reader(worker_id: int) -> None:
        q = f'tenant TENANT-{worker_id} latency'
        try:
            while not stop.is_set():
                sa.query(q, top_k=10)
        except BaseException as e:  # noqa: BLE001
            with errors_lock:
                errors.append(e)

    threads = [threading.Thread(target=writer, daemon=True)]
    threads[0].start()
    for r in range(7):
        t = threading.Thread(target=reader, args=(r,), daemon=True)
        t.start()
        threads.append(t)

    duration = float(os.environ.get('PENSIVE_CONCURRENCY_DURATION', '10.0'))
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)

    if errors:
        first = errors[0]
        raise AssertionError(
            f'1-writer/7-reader race fired: {len(errors)} error(s); '
            f'first = {type(first).__name__}: {first}'
        ) from first
