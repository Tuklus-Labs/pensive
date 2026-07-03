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

from eval.assoc_experiment import (
    assocSignal,
    compareMetrics,
    gateWithAssoc,
    _experimentReport,
)


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


def test_experiment_report_uses_captured_temporal_subset_rankings():
    # risk: temporal-subset-real-rankings
    queries = [
        {"query": "temporal one", "relevant": {"a"}, "own": set(), "metadata": {"kind": "temporal-neighborhood"}},
        {"query": "ordinary two", "relevant": {"b"}, "own": set(), "metadata": {"kind": "ordinary"}},
    ]
    baseline = {"r_at_1": 0.5, "r_at_5": 1.0, "r_at_10": 1.0, "r_at_20": 1.0, "mrr_at_10": 0.75, "n": 2}
    assoc = {"r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "r_at_20": 1.0, "mrr_at_10": 1.0, "n": 2}
    bm25 = {"r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0, "r_at_20": 0.0, "mrr_at_10": 0.0, "n": 2}
    baseline_rankings = [["x", "a"], ["b"]]
    assoc_rankings = [["a"], ["b"]]

    report = _experimentReport(
        2, {"ingested": 2}, baseline, assoc, bm25, queries,
        baseline_rankings, assoc_rankings,
    )

    assert report["temporal_neighborhood"]["baseline"]["mrr_at_10"] == 0.5, (
        "temporal-subset-real-rankings violated: subset baseline must come "
        f"from captured rankings, report={report}"
    )
    assert report["temporal_neighborhood"]["delta"]["mrr_at_10"] == 0.5, (
        "temporal-subset-real-rankings violated: subset delta must be "
        f"hand-computed from real rankings, report={report}"
    )


def test_experiment_report_marks_unannotated_temporal_subset_unavailable():
    # risk: temporal-subset-unavailable
    queries = [
        {"query": "ordinary one", "relevant": {"a"}, "own": set()},
    ]
    metrics = {"r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "r_at_20": 1.0, "mrr_at_10": 1.0, "n": 1}

    report = _experimentReport(
        1, {"ingested": 1}, metrics, metrics, metrics, queries, [["a"]], [["a"]]
    )

    subset = report["temporal_neighborhood"]
    assert subset["status"] == "UNAVAILABLE", (
        "temporal-subset-unavailable violated: unannotated runs must not "
        f"fabricate zero metrics, subset={subset}"
    )
    assert subset["baseline"] is None and subset["assoc"] is None and subset["delta"] is None, (
        "temporal-subset-unavailable violated: unavailable subset must use "
        f"None metric fields instead of zeros, subset={subset}"
    )


def test_assoc_signal_never_emits_dead_neighbors(tmp_path):
    # risk: live-only-emit
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "through")
        dead_candidate = _put(store, "dead candidate")
        store._conn.execute("UPDATE atoms SET status = 'tombstone' WHERE id = ?", (dead_candidate,))
        store._conn.commit()
        addEdge(store, {"src": seed, "dst": through, "type": "facet"})
        addEdge(store, {"src": through, "dst": dead_candidate, "type": "facet"})

        ranked = assocSignal(store, [seed], k=10)

        assert dead_candidate not in [atom_id for atom_id, _score in ranked], (
            "live-only-emit violated: tombstoned candidate surfaced in assoc "
            f"results, ranked={ranked}, dead_candidate={dead_candidate}"
        )
    finally:
        store.close()


def test_assoc_signal_dead_through_node_still_conducts_live_neighbors(tmp_path):
    # risk: conduct-but-not-emit
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "dead through")
        live_candidate = _put(store, "live candidate")
        store._conn.execute("UPDATE atoms SET status = 'superseded' WHERE id = ?", (through,))
        store._conn.commit()
        addEdge(store, {"src": seed, "dst": through, "type": "facet"})
        addEdge(store, {"src": through, "dst": live_candidate, "type": "facet"})

        ranked = assocSignal(store, [seed], k=10)

        assert ranked == [(live_candidate, 0.5)], (
            "conduct-but-not-emit violated: dead through-node should conduct "
            f"to live second-hop candidate without being emitted, ranked={ranked}"
        )
    finally:
        store.close()


def test_assoc_signal_mirrored_edges_contribute_once_with_max_weight(tmp_path):
    # risk: logical-edge-dedupe
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "through")
        candidate = _put(store, "candidate")
        addEdge(store, {"src": seed, "dst": through, "type": "facet", "weight": 0.25})
        addEdge(store, {"src": through, "dst": seed, "type": "facet", "weight": 0.75})
        addEdge(store, {"src": through, "dst": candidate, "type": "facet", "weight": 1.0})

        ranked = assocSignal(store, [seed], k=10)

        assert ranked == [(candidate, 0.375)], (
            "logical-edge-dedupe violated: mirrored seed-through rows should "
            f"contribute once at max weight 0.75 over degree 2, ranked={ranked}"
        )
    finally:
        store.close()


def test_assoc_signal_distinct_edge_types_between_same_nodes_count_separately(tmp_path):
    # risk: logical-edge-type-separation
    store = _open(tmp_path)
    try:
        seed = _put(store, "seed")
        through = _put(store, "through")
        candidate = _put(store, "candidate")
        addEdge(store, {"src": seed, "dst": through, "type": "facet", "weight": 1.0})
        addEdge(store, {"src": seed, "dst": through, "type": "causes", "weight": 1.0})
        addEdge(store, {"src": through, "dst": candidate, "type": "facet", "weight": 1.0})

        ranked = assocSignal(store, [seed], k=10)

        assert ranked == [(candidate, 1.0)], (
            "logical-edge-type-separation violated: two genuine edge types "
            f"between the same endpoints should both contribute, ranked={ranked}"
        )
    finally:
        store.close()


def test_gate_with_assoc_passes_when_without_arm_matches_gate_baseline(monkeypatch):
    # risk: arm-parity-correct-clone
    baseline = {"r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "r_at_20": 1.0, "mrr_at_10": 1.0, "n": 1, "low_confidence_rate": 0.0}
    assoc = {"r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0, "r_at_20": 0.0, "mrr_at_10": 0.0, "n": 1, "low_confidence_rate": 0.0}

    monkeypatch.setattr("eval.assoc_experiment.gate_mod.gate", lambda store, index, embedder, queries: baseline)
    monkeypatch.setattr(
        "eval.assoc_experiment._baselineWithRankings",
        lambda store, index, embedder, queries, recallK=None: (baseline, [["a"]]),
    )
    monkeypatch.setattr(
        "eval.assoc_experiment._assocWithRankings",
        lambda store, index, embedder, queries, recallK=None: (assoc, [["b"]]),
    )

    result = gateWithAssoc(object(), object(), object(), [{"query": "q", "relevant": {"a"}, "own": set()}])

    assert result == (assoc, [["b"]]), (
        "arm-parity-correct-clone violated: matching without-arm clone should "
        f"allow assoc run through, result={result}"
    )


def test_gate_with_assoc_raises_when_without_arm_diverges_from_gate_baseline(monkeypatch):
    # risk: arm-parity-divergent-clone
    baseline = {"r_at_1": 1.0, "r_at_5": 1.0, "r_at_10": 1.0, "r_at_20": 1.0, "mrr_at_10": 1.0, "n": 1, "low_confidence_rate": 0.0}
    divergent = dict(baseline, r_at_1=0.0)
    assoc = dict(baseline)

    monkeypatch.setattr("eval.assoc_experiment.gate_mod.gate", lambda store, index, embedder, queries: baseline)
    monkeypatch.setattr(
        "eval.assoc_experiment._baselineWithRankings",
        lambda store, index, embedder, queries, recallK=None: (divergent, [["x"]]),
    )
    monkeypatch.setattr(
        "eval.assoc_experiment._assocWithRankings",
        lambda store, index, embedder, queries, recallK=None: (assoc, [["a"]]),
    )

    try:
        gateWithAssoc(object(), object(), object(), [{"query": "q", "relevant": {"a"}, "own": set()}])
    except RuntimeError as exc:
        assert "gate.py baseline" in str(exc) and "experiment clone" in str(exc), (
            "arm-parity-divergent-clone violated: mismatch error must name "
            f"both metric sets, error={exc}"
        )
    else:
        raise AssertionError(
            "arm-parity-divergent-clone violated: gateWithAssoc must abort "
            "when the without-arm clone diverges from gate.py baseline"
        )
