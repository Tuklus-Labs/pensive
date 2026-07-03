"""Association experiment tests (Task 21).

Risk model:

- Invariants:
  - row: two-hop-only - assocSignal returns second-hop candidates, not seeds or
    first-hop through nodes. Covered by
    test_assoc_signal_finds_specific_two_hop_neighbor.
  - row: specificity-damping - paths through higher-degree nodes must score
    below otherwise comparable specific paths. Covered by
    test_assoc_signal_damps_hub_paths_below_specific_paths.
  - row: deterministic-order - equal inputs produce byte-for-byte identical
    ranked tuples. Covered by test_assoc_signal_is_deterministic.
- State transitions: N/A - assocSignal is a read-only query over a fixed store.
- Boundaries:
  - row: k-limit - output length is capped after deterministic ranking. Covered
    by test_assoc_signal_respects_k_after_ranking.
  - row: empty-store - no atoms or edges returns an empty list. Covered by
    test_assoc_signal_handles_empty_store.
  - row: no-edges - seeds without incident edges return an empty list. Covered by
    test_assoc_signal_handles_seed_with_no_edges.
- Malformed inputs: N/A - this harness helper is internal; callers pass store
  and atom ids from the gate.
- Concurrency: N/A - no mutation, no shared caches, no sampling.
- Persistence: N/A - no writes; SQLite persistence behavior is covered by store
  tests.
- Integration contracts:
  - row: harness-runner - the experiment runner compares baseline and
    assoc-augmented metrics using gate.py's metric path without touching serving
    recall/fusion. Covered by test_compare_metrics_reports_deltas_and_subset.
- Regression traps:
  - boundary: empty collection treated as missing collection - covered by empty
    and no-edge tests.
  - concurrency: N/A - deterministic single-threaded calculation.
  - contract: API returns null where caller expects empty collection - covered by
    empty and no-edge tests returning [].
  - encoding: N/A - no serialization in assocSignal.
  - framework: N/A - no framework parser behavior.
  - io: N/A - no filesystem/network/device IO.
  - persistence: N/A - read-only store queries.
  - resource: N/A - no external handles beyond caller-owned store.
  - state: N/A - no lifecycle state.

Coverage matrix:
- two-hop-only -> test_assoc_signal_finds_specific_two_hop_neighbor
- specificity-damping -> test_assoc_signal_damps_hub_paths_below_specific_paths
- deterministic-order -> test_assoc_signal_is_deterministic
- k-limit -> test_assoc_signal_respects_k_after_ranking
- empty-store -> test_assoc_signal_handles_empty_store
- no-edges -> test_assoc_signal_handles_seed_with_no_edges
- harness-runner -> test_compare_metrics_reports_deltas_and_subset

Sabotage log:
- finds_specific_two_hop_neighbor: Mutating the walk to return hop-1 nodes would
  fail the candidate assertion; weakening the expected candidate assertion would
  let a through-node through, so the assertion names both excluded ids.
- damps_hub_paths: Removing degree division would make the hub candidate tie or
  beat the specific candidate; weakening the order assertion would miss the hub
  regression, so it asserts both order and numeric relation.
- respects_k: Removing the final slice would fail the exact length/id assertion;
  weakening to truthiness would miss over-returning.
- no_edges/empty_store: Returning None instead of [] would fail exact equality;
  weakening to falsy would miss the API contract.
- deterministic: Sorting only by score can drift on equal-score insert order;
  the exact tuple-list comparison catches that.
- compare_metrics: Dropping the assoc path or subset filter changes the exact
  delta/subset fields; weakening to key-presence would miss wrong metrics.

Loudness audit: every assertion below includes a rule-naming failure message.
"""

from store.store import addEdge, openStore, putAtom

from eval.assoc_experiment import assocSignal, compareMetrics


def _put(store, text):
    return putAtom(
        store,
        {"text": text, "kind": "atom", "provenance": {"source": "test"}},
    )


def _open(tmp_path):
    return openStore(tmp_path / "assoc.db")


def test_assoc_signal_finds_specific_two_hop_neighbor(tmp_path):
    # risk: two-hop-only
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "rare shared facet")
        candidate = _put(store, "associated neighbor")
        addEdge(store, {"src": seed, "dst": through, "type": "facet"})
        addEdge(store, {"src": through, "dst": candidate, "type": "facet"})

        ranked = assocSignal(store, [seed], k=10)

        assert ranked == [(candidate, 0.5)], (
            "two-hop-only invariant violated: expected only the second-hop "
            f"candidate={candidate}, got ranked={ranked}, seed={seed}, "
            f"through={through}"
        )
    finally:
        store.close()


def test_assoc_signal_damps_hub_paths_below_specific_paths(tmp_path):
    # risk: specificity-damping
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        rare = _put(store, "rare through")
        hub = _put(store, "hub through")
        specific_candidate = _put(store, "specific candidate")
        hub_candidate = _put(store, "hub candidate")
        addEdge(store, {"src": seed, "dst": rare, "type": "facet"})
        addEdge(store, {"src": rare, "dst": specific_candidate, "type": "facet"})
        addEdge(store, {"src": seed, "dst": hub, "type": "facet"})
        addEdge(store, {"src": hub, "dst": hub_candidate, "type": "facet"})
        for i in range(6):
            addEdge(store, {"src": hub, "dst": _put(store, f"hub spoke {i}"), "type": "facet"})

        ranked = assocSignal(store, [seed], k=10)
        scores = dict(ranked)

        assert ranked[0][0] == specific_candidate, (
            "specificity-damping invariant violated: rare-through candidate "
            f"must outrank hub-through candidate, ranked={ranked}, hub={hub}, "
            f"rare={rare}"
        )
        assert scores[specific_candidate] > scores[hub_candidate], (
            "specificity-damping score rule violated: expected "
            f"specific={scores[specific_candidate]} > hub={scores[hub_candidate]} "
            f"with ranked={ranked}"
        )
    finally:
        store.close()


def test_assoc_signal_respects_k_after_ranking(tmp_path):
    # risk: k-limit
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "through")
        addEdge(store, {"src": seed, "dst": through, "type": "facet"})
        candidates = []
        for i in range(4):
            candidate = _put(store, f"candidate {i}")
            candidates.append(candidate)
            addEdge(store, {"src": through, "dst": candidate, "type": "facet"})

        ranked = assocSignal(store, [seed], k=2)

        assert len(ranked) == 2, (
            "k-limit boundary violated: expected exactly 2 results, "
            f"got ranked={ranked}"
        )
        assert [atom_id for atom_id, _ in ranked] == sorted(candidates)[:2], (
            "deterministic k-limit tie-break violated: expected atom-id order "
            f"for equal scores, got ranked={ranked}, candidates={candidates}"
        )
    finally:
        store.close()


def test_assoc_signal_handles_seed_with_no_edges(tmp_path):
    # risk: no-edges
    store = _open(tmp_path)
    try:
        seed = _put(store, "lonely seed")

        ranked = assocSignal(store, [seed], k=10)

        assert ranked == [], (
            "no-edges boundary violated: seed with no incident edges must "
            f"return [], got ranked={ranked}"
        )
    finally:
        store.close()


def test_assoc_signal_handles_empty_store(tmp_path):
    # risk: empty-store
    store = _open(tmp_path)
    try:
        ranked = assocSignal(store, ["missing"], k=10)

        assert ranked == [], (
            "empty-store contract violated: missing seeds in empty graph must "
            f"return [], got ranked={ranked}"
        )
    finally:
        store.close()


def test_assoc_signal_is_deterministic(tmp_path):
    # risk: deterministic-order
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "through")
        addEdge(store, {"src": seed, "dst": through, "type": "facet"})
        for label in ("c", "a", "b"):
            addEdge(store, {"src": through, "dst": _put(store, label), "type": "facet"})

        first = assocSignal(store, [seed], k=10)
        second = assocSignal(store, [seed], k=10)

        assert first == second, (
            "deterministic-order invariant violated: repeated runs differed, "
            f"first={first}, second={second}"
        )
    finally:
        store.close()


def test_compare_metrics_reports_deltas_and_subset():
    # risk: harness-runner
    queries = [
        {"query": "temporal one", "relevant": {"a"}, "own": set(), "metadata": {"kind": "temporal-neighborhood"}},
        {"query": "ordinary two", "relevant": {"b"}, "own": set(), "metadata": {"kind": "ordinary"}},
    ]
    baseline_rankings = [["x", "a"], ["b"]]
    assoc_rankings = [["a"], ["x", "b"]]

    report = compareMetrics(queries, baseline_rankings, assoc_rankings)

    assert report["baseline"]["r_at_1"] == 0.5, (
        "harness-runner baseline metric violated: expected one of two R@1 hits, "
        f"report={report}"
    )
    assert report["assoc"]["r_at_1"] == 0.5, (
        "harness-runner assoc metric violated: expected one of two R@1 hits, "
        f"report={report}"
    )
    assert report["delta"]["r_at_10"] == 0.0, (
        "harness-runner delta metric violated: both rankings hit within 10, "
        f"report={report}"
    )
    assert report["temporal_neighborhood"]["delta"]["mrr_at_10"] == 0.5, (
        "temporal-neighborhood subset violated: metadata kind should isolate "
        f"the first query and improve MRR by 0.5, report={report}"
    )
