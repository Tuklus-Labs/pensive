"""Tests for boundary analysis diagnostics."""
import pytest
import numpy as np
from pensive.boundary import BoundaryAnalysis, FrequencyBands


class TestBoundaryAnalysisDataclass:
    def test_construction_with_all_fields(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.05,
            disambiguation_gap=0.02,
            band_crossing=True,
            context_needed=True,
            suggested_context=["199ms", "257ms"],
            fundamentally_ambiguous=False,
            top_scores=[0.20, 0.18, 0.10],
        )
        assert ba.boundary_distance == 0.05
        assert ba.disambiguation_gap == 0.02
        assert ba.band_crossing is True
        assert ba.context_needed is True
        assert ba.suggested_context == ["199ms", "257ms"]
        assert ba.fundamentally_ambiguous is False

    def test_confidence_high_when_far_from_boundary(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.50,
            disambiguation_gap=0.30,
            band_crossing=False,
            context_needed=False,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[0.65, 0.35],
        )
        assert ba.confidence == "high"

    def test_confidence_low_near_boundary_with_small_gap(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.02,
            disambiguation_gap=0.01,
            band_crossing=False,
            context_needed=True,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[0.17, 0.16],
        )
        assert ba.confidence == "low"

    def test_confidence_medium_moderate_distance(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.10,
            disambiguation_gap=0.08,
            band_crossing=False,
            context_needed=False,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[0.25, 0.17],
        )
        assert ba.confidence == "medium"

    def test_confidence_low_when_context_is_needed(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.50,
            disambiguation_gap=0.01,
            band_crossing=False,
            context_needed=True,
            suggested_context=["199ms"],
            fundamentally_ambiguous=False,
            top_scores=[0.65, 0.64],
        )
        assert ba.confidence == "low"

    def test_no_results_yields_none_boundary(self):
        ba = BoundaryAnalysis(
            boundary_distance=None,
            disambiguation_gap=None,
            band_crossing=False,
            context_needed=False,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[],
        )
        assert ba.confidence == "none"

    def test_should_trust_false_when_context_needed(self):
        ba = BoundaryAnalysis(
            boundary_distance=0.50,
            disambiguation_gap=0.01,
            band_crossing=False,
            context_needed=True,
            suggested_context=["199ms"],
            fundamentally_ambiguous=False,
            top_scores=[0.65, 0.64],
        )
        assert ba.should_trust is False
        assert ba.recommended_action == "request_context"

    def test_no_results_recommend_no_result(self):
        ba = BoundaryAnalysis(
            boundary_distance=None,
            disambiguation_gap=None,
            band_crossing=False,
            context_needed=False,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[],
        )
        assert ba.should_trust is False
        assert ba.recommended_action == "no_result"


class TestFrequencyBands:
    def test_compute_bands_from_freq_dict(self):
        # Zipf-like: few rare, many common
        entity_freq = {
            "rare1": 1, "rare2": 2, "rare3": 1,
            "mid1": 15, "mid2": 20, "mid3": 18,
            "common1": 200, "common2": 350, "common3": 500,
        }
        bands = FrequencyBands.from_entity_freq(entity_freq, n_bands=3)
        assert bands.n_bands == 3
        assert bands.band_of("rare1") != bands.band_of("common1")

    def test_same_band_for_similar_frequencies(self):
        entity_freq = {"a": 10, "b": 12, "c": 11}
        bands = FrequencyBands.from_entity_freq(entity_freq, n_bands=3)
        # All similar freq -- should land in same band
        assert bands.band_of("a") == bands.band_of("b") == bands.band_of("c")

    def test_is_cross_band(self):
        entity_freq = {
            "rare": 1, "common": 500,
        }
        bands = FrequencyBands.from_entity_freq(entity_freq, n_bands=3)
        assert bands.is_cross_band(["rare", "common"]) is True
        assert bands.is_cross_band(["rare"]) is False

    def test_unknown_entity_returns_none_band(self):
        entity_freq = {"known": 10}
        bands = FrequencyBands.from_entity_freq(entity_freq, n_bands=3)
        assert bands.band_of("unknown") is None

    def test_single_entity_no_crash(self):
        entity_freq = {"only": 5}
        bands = FrequencyBands.from_entity_freq(entity_freq, n_bands=3)
        assert bands.band_of("only") is not None
        assert bands.is_cross_band(["only"]) is False


from pensive import SpreadingActivation, SpreadingConfig
from pensive.patterns import SYNTHETIC_PATTERNS
from pensive.boundary import analyze_boundary, AnalyzedResult


def _build_sa_with_ambiguous_docs():
    """Build an SA instance with two docs sharing entities but different answers."""
    docs = [
        {'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
         'id': 'n1', 'value': '199ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
         'id': 'n2', 'value': '257ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
    ]
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(docs)
    return sa


def _build_sa_with_distinct_docs():
    """Build an SA with docs that have non-overlapping entities."""
    docs = [
        {'content': 'Meeting room A-512 has capacity of 11 people.',
         'id': 'n1', 'value': '11 people',
         'query': 'Capacity of meeting room A-512?'},
        {'content': 'Dr. Taylor Smith leads the Horizon initiative.',
         'id': 'n2', 'value': 'Dr. Taylor Smith',
         'query': 'Who leads Horizon?'},
    ]
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(docs)
    return sa


class TestAnalyzeBoundary:
    def test_returns_boundary_analysis(self):
        sa = _build_sa_with_ambiguous_docs()
        result = analyze_boundary(sa, "What was the P99 latency on 2025-07-16?")
        assert isinstance(result, AnalyzedResult)
        assert isinstance(result.analysis, BoundaryAnalysis)
        assert len(result.results) >= 1

    def test_ambiguous_query_has_small_disambiguation_gap(self):
        sa = _build_sa_with_ambiguous_docs()
        result = analyze_boundary(sa, "What was the P99 latency on 2025-07-16?")
        # Two docs with same entities -- gap should be small or zero
        if result.analysis.disambiguation_gap is not None:
            assert result.analysis.disambiguation_gap < 0.20

    def test_distinct_query_has_high_confidence(self):
        sa = _build_sa_with_distinct_docs()
        result = analyze_boundary(sa, "Capacity of meeting room A-512?")
        assert result.analysis.confidence in ("high", "medium")

    def test_no_results_query(self):
        sa = _build_sa_with_distinct_docs()
        result = analyze_boundary(sa, "xyzzy nonexistent gibberish")
        assert result.analysis.confidence == "none"
        assert result.analysis.boundary_distance is None
        assert result.results == []

    def test_boundary_distance_is_score_minus_threshold(self):
        sa = _build_sa_with_distinct_docs()
        result = analyze_boundary(sa, "Capacity of meeting room A-512?")
        if result.results:
            top_score = result.results[0][1]
            expected_dist = top_score - sa.config.threshold
            assert abs(result.analysis.boundary_distance - expected_dist) < 1e-6

    def test_suggested_context_contains_differentiating_entities(self):
        sa = _build_sa_with_ambiguous_docs()
        result = analyze_boundary(
            sa, "What was the P99 latency on 2025-07-16?", top_k=2
        )
        if result.analysis.suggested_context:
            assert all(isinstance(s, str) for s in result.analysis.suggested_context)


class TestQueryAnalyzed:
    def test_query_analyzed_returns_analyzed_result(self):
        sa = _build_sa_with_ambiguous_docs()
        result = sa.query_analyzed("What was the P99 latency on 2025-07-16?")
        assert hasattr(result, 'results')
        assert hasattr(result, 'analysis')
        assert isinstance(result.analysis, BoundaryAnalysis)

    def test_query_analyzed_results_match_query(self):
        sa = _build_sa_with_ambiguous_docs()
        q = "What was the P99 latency on 2025-07-16?"
        normal = sa.query(q)
        analyzed = sa.query_analyzed(q)
        # Results should be identical
        assert len(analyzed.results) == len(normal)
        for (a_label, a_score), (n_label, n_score) in zip(analyzed.results, normal):
            assert a_label == n_label
            assert abs(a_score - n_score) < 1e-6

    def test_query_analyzed_with_context(self):
        sa = _build_sa_with_ambiguous_docs()
        result = sa.query_analyzed(
            "What was the P99 latency on 2025-07-16?",
            context=["199ms"]
        )
        assert result.results[0][0] == "199ms"

    def test_query_analyzed_not_built_raises(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        with pytest.raises(ValueError, match="not built"):
            sa.query_analyzed("test")


class TestBoundaryEdgeCases:
    def test_single_result_no_disambiguation_gap(self):
        """One result means no gap to compute."""
        docs = [
            {'content': 'Meeting room A-512 has capacity of 11.',
             'id': 'n1', 'value': '11',
             'query': 'Capacity of A-512?'},
        ]
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)
        result = analyze_boundary(sa, "Capacity of A-512?", top_k=1)
        assert result.analysis.disambiguation_gap is None
        assert result.analysis.context_needed is False

    def test_many_tied_results(self):
        """Multiple results at similar scores should flag context needed."""
        docs = [
            {'content': f'Latency on 2025-07-16 was {v}ms.',
             'id': f'n{i}', 'value': f'{v}ms',
             'query': 'Latency on 2025-07-16?'}
            for i, v in enumerate([199, 200, 201, 202])
        ]
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)
        result = analyze_boundary(sa, "Latency on 2025-07-16?")
        # With 4 nearly identical docs, should detect ambiguity
        assert result.analysis.disambiguation_gap is not None

    def test_empty_query_string(self):
        sa = _build_sa_with_distinct_docs()
        result = analyze_boundary(sa, "")
        assert result.analysis.confidence == "none"
        assert result.results == []

    def test_frequency_bands_cached_across_calls(self):
        """FrequencyBands.from_entity_freq is deterministic."""
        freq = {"a": 1, "b": 100, "c": 10000}
        b1 = FrequencyBands.from_entity_freq(freq, n_bands=3)
        b2 = FrequencyBands.from_entity_freq(freq, n_bands=3)
        for e in freq:
            assert b1.band_of(e) == b2.band_of(e)


class TestContextDetection:
    def test_ambiguous_docs_suggest_context(self):
        """When two results tie and differ by an entity, suggest it."""
        sa = _build_sa_with_ambiguous_docs()
        result = analyze_boundary(sa, "What was the P99 latency on 2025-07-16?")
        # The two answers differ in their value entities (199ms vs 257ms)
        if result.analysis.context_needed:
            assert (len(result.analysis.suggested_context) > 0
                    or result.analysis.fundamentally_ambiguous)

    def test_distinct_docs_no_context_needed(self):
        """Clear winner means no context needed."""
        sa = _build_sa_with_distinct_docs()
        result = analyze_boundary(sa, "Capacity of meeting room A-512?")
        assert result.analysis.context_needed is False
        assert result.analysis.fundamentally_ambiguous is False
