"""Regression tests for candidate-scoped facet boosts."""

import recall.engine as engine
import recall.signals as signals
from recall.engine import recall
from recall.payload import SENTINEL_LOW_CONFIDENCE
from store.store import addFacet, openStore, putAtom, supersede


NOW = 2_000_000_000


def _put(store, text, *, project="pensive"):
    return putAtom(store, {
        "text": text,
        "kind": "atom",
        "project": project,
        "occurredAt": NOW - 60,
        "provenance": {"source": "test", "agent": "codex"},
    })


def test_candidate_scope_equals_global_intersection_for_live_candidates(tmp_path):
    # I1/I3/I4/S1/B2/M1/P2: candidate membership is the exact old intersection.
    store = openStore(tmp_path / "facets.db")
    try:
        inside = _put(store, "candidate about pensive")
        outside = _put(store, "noncandidate about pensive")
        retired = _put(store, "retired candidate about pensive")
        successor = _put(store, "replacement without that entity")
        for atom_id in (inside, outside, retired):
            addFacet(store, atom_id, "entity", "pensive")
        supersede(store, retired, successor, {"source": "test"})

        global_result = signals.facetSignal(store, {"query": "pensive recall"})
        candidate_ids = [inside, retired, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", inside]
        scoped_result = signals.facetSignal(
            store, {"query": "pensive recall", "candidateIds": candidate_ids})

        expected = global_result["boostSet"].intersection(candidate_ids)
        assert scoped_result["boostSet"] == expected == {inside}, (
            "candidate-facet intersection invariant violated: "
            f"global={sorted(global_result['boostSet'])} candidates={candidate_ids} "
            f"scoped={sorted(scoped_result['boostSet'])} expected={[inside]}"
        )
        assert outside not in scoped_result["boostSet"], (
            "facet candidate-boundary invariant violated: a matching noncandidate "
            f"was materialized, outside={outside} scoped={sorted(scoped_result['boostSet'])}"
        )
        assert retired not in scoped_result["boostSet"], (
            "facet liveness invariant violated: superseded candidate was boosted, "
            f"retired={retired} scoped={sorted(scoped_result['boostSet'])}"
        )
    finally:
        store.close()


def test_omitted_candidate_scope_preserves_global_facet_contract(tmp_path):
    # B1/C1: None/omitted is legacy global behavior, unlike an explicit empty pool.
    store = openStore(tmp_path / "legacy.db")
    try:
        atom_id = _put(store, "global pensive note")
        addFacet(store, atom_id, "entity", "pensive")

        omitted = signals.facetSignal(store, {"query": "pensive"})
        explicit_none = signals.facetSignal(
            store, {"query": "pensive", "candidateIds": None})

        assert omitted == explicit_none, (
            "legacy facet-signal contract violated: omitted and None candidate "
            f"scope diverged, omitted={omitted} explicit_none={explicit_none}"
        )
        assert omitted["boostSet"] == {atom_id}, (
            "legacy global facet lookup invariant violated: matching live atom "
            f"was absent, atom={atom_id} boost={sorted(omitted['boostSet'])}"
        )
    finally:
        store.close()


def test_empty_candidate_scope_skips_entity_extraction(tmp_path, monkeypatch):
    # S2/B1/resource: an empty pool is a real empty scope and performs no query work.
    store = openStore(tmp_path / "empty.db")
    try:
        def extractor_bomb():
            raise AssertionError("entity extractor must not run for an empty candidate pool")

        monkeypatch.setattr(signals, "_getExtractor", extractor_bomb)
        result = signals.facetSignal(
            store, {"query": "pensive", "candidateIds": []})

        assert result == {"boostSet": set(), "filterSet": None}, (
            "empty candidate-scope invariant violated: expected an empty boost "
            f"without a filter, got={result}"
        )
    finally:
        store.close()


def test_candidate_scope_forces_atom_first_facet_index(tmp_path):
    # C3/framework: pin the query-plan instruction that bounds common-label work.
    store = openStore(tmp_path / "plan.db")
    statements = []
    try:
        atom_id = _put(store, "candidate pensive note")
        addFacet(store, atom_id, "entity", "pensive")
        store._conn.set_trace_callback(statements.append)

        result = signals.facetSignal(
            store, {"query": "pensive", "candidateIds": [atom_id]})

        facet_selects = [sql for sql in statements if "FROM facets" in sql]
        assert result["boostSet"] == {atom_id}, (
            "facet plan-test setup invariant violated: candidate did not match, "
            f"atom={atom_id} result={result}"
        )
        assert any("INDEXED BY sqlite_autoindex_facets_1" in sql for sql in facet_selects), (
            "atom-first facet-plan invariant violated: candidate lookup did not "
            f"force the facets primary key, statements={facet_selects}"
        )
    finally:
        store._conn.set_trace_callback(None)
        store.close()


def test_recall_candidate_scoped_facets_match_frozen_clock_reference(
        tmp_path, monkeypatch):
    # I1/I2/C2/C4/state: new stage placement is byte-identical at one clock.
    store = openStore(tmp_path / "engine.db")
    try:
        plain = _put(store, "plain candidate")
        boosted = _put(store, "pensive candidate")
        outside = _put(store, "pensive noncandidate")
        addFacet(store, boosted, "entity", "pensive")
        addFacet(store, outside, "entity", "pensive")
        lexical = [(plain, 0.9), (boosted, 0.8)]

        monkeypatch.setattr(engine.time, "time", lambda: NOW)
        monkeypatch.setattr(
            engine, "bm25",
            lambda store, query, k, kinds=None, agent=None, project=None,
                   timeScope=None: list(lexical),
        )
        captured = {}
        real_facet = signals.facetSignal

        def scoped_spy(store_arg, hints):
            captured.update(hints)
            return real_facet(store_arg, hints)

        monkeypatch.setattr(engine, "facetSignal", scoped_spy)
        candidate_result = recall(
            store, {"memory": None, "code": None}, object(),
            "pensive", tier="L2", k=10, tokenBudget=1500)

        candidate_ids = [row["atomId"] for row in candidate_result["results"]]
        assert candidate_ids == [boosted, plain], (
            "facet boost-order invariant violated: the matching lower-ranked "
            f"candidate did not move first, ids={candidate_ids} "
            f"boosted={boosted} plain={plain}"
        )
        assert "facet match" in candidate_result["results"][0]["why"], (
            "facet trust-evidence invariant violated: boosted result omitted its "
            f"facet evidence, result={candidate_result['results'][0]}"
        )
        assert set(captured["candidateIds"]) == {plain, boosted}, (
            "facet stage-order invariant violated: lookup did not receive the "
            f"complete fused pool, captured={captured} expected={[plain, boosted]}"
        )
        scoped = real_facet(store, captured)
        global_result = real_facet(store, {"query": "pensive"})
        assert scoped["boostSet"] == global_result["boostSet"].intersection(
                captured["candidateIds"]), (
            "engine facet-membership invariant violated: scoped lookup differs "
            f"from global intersection, scoped={sorted(scoped['boostSet'])} "
            f"global={sorted(global_result['boostSet'])} candidates={captured['candidateIds']}"
        )

        def old_global_reference(store_arg, hints):
            return real_facet(store_arg, {"query": hints["query"]})

        monkeypatch.setattr(engine, "facetSignal", old_global_reference)
        reference_result = recall(
            store, {"memory": None, "code": None}, object(),
            "pensive", tier="L2", k=10, tokenBudget=1500)

        assert candidate_result["results"] == reference_result["results"], (
            "frozen-clock recall-order invariant violated: candidate-scoped "
            f"results={candidate_result['results']} reference={reference_result['results']}"
        )
        assert candidate_result["payload"] == reference_result["payload"], (
            "frozen-clock payload invariant violated: candidate-scoped and global "
            f"payloads differ, candidate={candidate_result['payload']!r} "
            f"reference={reference_result['payload']!r}"
        )
        assert candidate_result["tokensUsed"] == reference_result["tokensUsed"], (
            "frozen-clock token-accounting invariant violated: "
            f"candidate={candidate_result['tokensUsed']} "
            f"reference={reference_result['tokensUsed']}"
        )
        assert candidate_result["lowConfidence"] == reference_result["lowConfidence"], (
            "frozen-clock confidence-state invariant violated: "
            f"candidate={candidate_result['lowConfidence']} "
            f"reference={reference_result['lowConfidence']}"
        )
    finally:
        store.close()


def test_empty_fused_pool_skips_facet_work(tmp_path, monkeypatch):
    # S2/C2/resource: no candidates means no entity extraction or facet SQL.
    store = openStore(tmp_path / "no-candidates.db")
    try:
        monkeypatch.setattr(
            engine, "bm25",
            lambda store, query, k, kinds=None, agent=None, project=None,
                   timeScope=None: [],
        )

        def facet_bomb(*_args, **_kwargs):
            raise AssertionError("facetSignal must not run before candidates exist")

        monkeypatch.setattr(engine, "facetSignal", facet_bomb)
        result = recall(
            store, {"memory": None, "code": None}, object(),
            "pensive", tier="L2")

        assert result["results"] == [], (
            "empty-fusion result invariant violated: no candidates produced "
            f"results={result['results']}"
        )
        assert result["payload"] == SENTINEL_LOW_CONFIDENCE, (
            "empty-fusion payload invariant violated: expected low-confidence "
            f"sentinel, payload={result['payload']!r}"
        )
    finally:
        store.close()
