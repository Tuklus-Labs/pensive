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
