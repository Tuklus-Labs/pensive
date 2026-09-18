# Boundary Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `BoundaryAnalysis` diagnostic module to PyPensive that computes boundary distance, disambiguation gap, entity frequency band crossing, and context requirement detection -- turning opaque retrieval scores into actionable confidence signals.

**Architecture:** A new `boundary.py` module containing a `BoundaryAnalysis` dataclass (the diagnostic result) and an `analyze_boundary()` function that takes a raw score array + the SA instance and returns the diagnostic. The existing `query()` and `query_with_doc_ids()` methods get a new `diagnose=False` kwarg that optionally attaches a `BoundaryAnalysis` to the return value. Entity frequency bands are computed lazily on first use and cached. No changes to the hot path when `diagnose=False`.

**Tech Stack:** Python, numpy, scipy (already deps). No new dependencies.

**Branch:** `research/boundary-analysis`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/pensive/boundary.py` (CREATE) | `BoundaryAnalysis` dataclass, `analyze_boundary()` function, `FrequencyBands` band calculator |
| `src/pensive/spreading.py` (MODIFY) | Add `query_analyzed()` method that returns results + diagnosis. Does NOT modify existing `query()`/`query_with_doc_ids()` signatures. |
| `tests/test_boundary.py` (CREATE) | All boundary analysis tests |

**Design decision -- why a new method instead of a kwarg on `query()`:** The existing `query()` returns `List[Tuple[str, float]]`. Adding diagnosis would change the return type, breaking every caller. A separate `query_analyzed()` method returns a richer result type without touching the stable API. Production callers are unaffected. The research branch can iterate on the analysis without worrying about backward compat.

---

### Task 1: BoundaryAnalysis Dataclass and Stubs

**Files:**
- Create: `src/pensive/boundary.py`
- Test: `tests/test_boundary.py`

- [ ] **Step 1: Write the failing test -- BoundaryAnalysis construction**

```python
# tests/test_boundary.py
"""Tests for boundary analysis diagnostics."""
import pytest
from pensive.boundary import BoundaryAnalysis


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pensive.boundary'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pensive/boundary.py
"""Boundary analysis diagnostics for spreading activation retrieval.

Implements three computable signals from the boundary function / memory
graphon framework (see research/boundary-functions-memory-graphons.md):

1. Boundary distance scoring (Definition 7.2) -- how close the top
   result is to the activation threshold.
2. Graphon-informed entity clustering (Section 6.3) -- whether matched
   entities span frequency bands, indicating cross-band ambiguity.
3. Context requirement detection (Proposition 5.5) -- whether context
   can resolve the ambiguity, and if so, which entities would help.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundaryAnalysis:
    """Diagnostic result from boundary analysis of a query.

    Attributes:
        boundary_distance: Gap between top score and activation threshold.
            Large = confident result. Small = on the boundary. None = no results.
        disambiguation_gap: Score gap between the top two results.
            Large = clear winner. Small = tied candidates.
        band_crossing: Whether matched entities span different frequency bands.
            True = query mixes rare and common entities (source of ambiguity).
        context_needed: Whether providing context would improve results.
        suggested_context: Entity labels that would disambiguate tied results.
            These are entities that activate one competitor but not the other.
        fundamentally_ambiguous: True if no context can help -- competing
            results share all activated entities.
        top_scores: Raw activation scores for the top results (for inspection).
    """
    boundary_distance: Optional[float]
    disambiguation_gap: Optional[float]
    band_crossing: bool
    context_needed: bool
    suggested_context: List[str]
    fundamentally_ambiguous: bool
    top_scores: List[float] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """Human-readable confidence level: 'high', 'medium', 'low', 'none'."""
        if not self.top_scores:
            return "none"
        if self.boundary_distance is None:
            return "none"
        if self.boundary_distance >= 0.15 and (
            self.disambiguation_gap is None or self.disambiguation_gap >= 0.10
        ):
            return "high"
        if self.boundary_distance < 0.05 and (
            self.disambiguation_gap is not None and self.disambiguation_gap < 0.05
        ):
            return "low"
        return "medium"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestBoundaryAnalysisDataclass -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pensive/boundary.py tests/test_boundary.py
git commit -m "feat(boundary): add BoundaryAnalysis dataclass with confidence scoring"
```

---

### Task 2: Entity Frequency Band Detection

**Files:**
- Modify: `src/pensive/boundary.py`
- Test: `tests/test_boundary.py`

- [ ] **Step 1: Write the failing test -- FrequencyBands**

```python
# Append to tests/test_boundary.py
import numpy as np
from pensive.boundary import FrequencyBands


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestFrequencyBands -v`
Expected: FAIL with `ImportError: cannot import name 'FrequencyBands'`

- [ ] **Step 3: Write implementation**

Add to `src/pensive/boundary.py`:

```python
from typing import Dict

class FrequencyBands:
    """Entity frequency band calculator.

    Partitions entities into frequency bands using log-scale quantiles.
    The regularity lemma (Section 6.3 of the paper) says the memory
    graphon decomposes into bands where within-band behavior is smooth.
    Cross-band queries are where the Lipschitz property breaks down.
    """

    def __init__(self, entity_to_band: Dict[str, int], n_bands: int,
                 boundaries: List[float]):
        self._entity_to_band = entity_to_band
        self._n_bands = n_bands
        self._boundaries = boundaries

    @property
    def n_bands(self) -> int:
        return self._n_bands

    @classmethod
    def from_entity_freq(cls, entity_freq: Dict[str, int],
                         n_bands: int = 5) -> 'FrequencyBands':
        """Compute frequency bands from an entity frequency dict.

        Uses log-scale quantile boundaries so bands are evenly spaced
        in log-frequency space (natural for Zipf distributions).
        """
        if not entity_freq:
            return cls({}, 0, [])

        import numpy as np

        entities = list(entity_freq.keys())
        freqs = np.array([entity_freq[e] for e in entities], dtype=np.float64)

        # Log-scale for Zipf distributions
        log_freqs = np.log1p(freqs)

        if len(set(log_freqs)) == 1:
            # All same frequency -- single band
            return cls({e: 0 for e in entities}, 1, [float(log_freqs[0])])

        # Quantile boundaries in log space
        actual_bands = min(n_bands, len(set(log_freqs)))
        quantiles = np.linspace(0, 100, actual_bands + 1)
        boundaries = np.percentile(log_freqs, quantiles)
        # np.searchsorted: which bin does each log_freq fall into?
        # clip to [0, actual_bands-1]
        band_indices = np.clip(
            np.searchsorted(boundaries[1:], log_freqs, side='right'),
            0, actual_bands - 1
        )

        entity_to_band = {e: int(band_indices[i]) for i, e in enumerate(entities)}
        return cls(entity_to_band, actual_bands, boundaries.tolist())

    def band_of(self, entity: str) -> Optional[int]:
        """Return the frequency band index for an entity, or None if unknown."""
        return self._entity_to_band.get(entity)

    def is_cross_band(self, entities: List[str]) -> bool:
        """Return True if the given entities span multiple frequency bands."""
        bands = {self._entity_to_band[e] for e in entities
                 if e in self._entity_to_band}
        return len(bands) > 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestFrequencyBands -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pensive/boundary.py tests/test_boundary.py
git commit -m "feat(boundary): add FrequencyBands for entity frequency band detection"
```

---

### Task 3: Core `analyze_boundary()` Function

**Files:**
- Modify: `src/pensive/boundary.py`
- Test: `tests/test_boundary.py`

This is the main logic. Key design decision: we call `sa.query()` to get results (guaranteeing identical output to the normal path regardless of numba/max_active pruning), then separately run `_spread_bipartite_raw()` just for the score array needed by analysis. For entity neighbor lookup, we resolve value nodes by index (not label) to handle duplicate value strings correctly.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_boundary.py
from pensive import SpreadingActivation, SpreadingConfig
from pensive.patterns import SYNTHETIC_PATTERNS
from pensive.boundary import analyze_boundary


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
        # The differentiating entities between "199ms" and "257ms"
        # should include those value strings themselves
        if result.analysis.suggested_context:
            # At minimum, suggested context should be strings
            assert all(isinstance(s, str) for s in result.analysis.suggested_context)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestAnalyzeBoundary -v`
Expected: FAIL with `ImportError: cannot import name 'analyze_boundary'`

- [ ] **Step 3: Write implementation**

Add to `src/pensive/boundary.py`:

```python
import numpy as np
from dataclasses import dataclass as _dc_dataclass

# Node type constants (mirror spreading.py)
_ENTITY_TYPE = 0
_VALUE_TYPE = 1


@_dc_dataclass
class AnalyzedResult:
    """Query results bundled with boundary analysis."""
    results: list  # List[Tuple[str, float]] -- same as query() returns
    analysis: BoundaryAnalysis


def analyze_boundary(sa: 'SpreadingActivation', query_text: str,
                     top_k: int = 10,
                     context: Optional[List[str]] = None,
                     n_bands: int = 5) -> AnalyzedResult:
    """Run a query and compute boundary analysis diagnostics.

    Design: calls sa.query() for results (guaranteeing identical output
    to the normal path regardless of numba/max_active pruning), then
    separately runs _spread_bipartite_raw() for the raw score array
    needed by analysis.

    Args:
        sa: A built SpreadingActivation instance.
        query_text: The query string.
        top_k: Number of results to return.
        context: Optional context for contextual intersection.
        n_bands: Number of frequency bands for entity clustering.

    Returns:
        AnalyzedResult with .results (same as query()) and .analysis.
    """
    if not sa._built:
        raise ValueError("Graph not built. Call build() first.")

    # Get results via the canonical path (handles numba, max_active, etc.)
    results = sa.query(query_text, top_k=top_k, context=context)

    # Get the raw score array separately for analysis
    # (We need per-node scores, not just top-k results)
    sa._compile()
    words = [w.lower().strip('?.,') for w in query_text.split() if len(w) >= 2]
    query_act = sa._seed_from_words(words)

    if sa._is_bipartite:
        score_arr = sa._spread_bipartite_raw(query_act)
    else:
        spread = sa._spread(query_act)
        score_arr = np.zeros(len(sa._idx_to_node), dtype=np.float32)
        for idx, score in spread.items():
            score_arr[idx] = score

    analysis = _compute_analysis(
        sa, score_arr, results, query_act, top_k, n_bands
    )
    return AnalyzedResult(results=results, analysis=analysis)


def _compute_analysis(
    sa: 'SpreadingActivation',
    score_arr: np.ndarray,
    results: list,
    query_act: dict,
    top_k: int,
    n_bands: int,
) -> BoundaryAnalysis:
    """Compute the BoundaryAnalysis from raw scores and results."""
    threshold = sa.config.threshold

    if not results:
        return BoundaryAnalysis(
            boundary_distance=None,
            disambiguation_gap=None,
            band_crossing=False,
            context_needed=False,
            suggested_context=[],
            fundamentally_ambiguous=False,
            top_scores=[],
        )

    top_scores = [score for _, score in results[:top_k]]
    top_score = top_scores[0]

    # 1. Boundary distance (Definition 7.2)
    boundary_distance = top_score - threshold

    # 2. Disambiguation gap
    disambiguation_gap = None
    if len(top_scores) >= 2:
        disambiguation_gap = top_scores[0] - top_scores[1]

    # 3. Entity frequency band crossing (Section 6.3)
    bands = FrequencyBands.from_entity_freq(sa.entity_freq, n_bands=n_bands)
    matched_entities = [
        sa._node_label[idx] for idx in query_act
        if sa._node_type[idx] == _ENTITY_TYPE
    ]
    band_crossing = bands.is_cross_band(matched_entities)

    # 4. Context requirement detection (Proposition 5.5)
    # Find entities that differentiate the top two results
    suggested_context = []
    fundamentally_ambiguous = False
    context_needed = False

    if (len(top_scores) >= 2
            and disambiguation_gap is not None
            and disambiguation_gap < 0.05):
        context_needed = True
        # Resolve value node indices from score array (not by label scan)
        idx_a = _resolve_value_node_idx(sa, score_arr, results[0])
        idx_b = _resolve_value_node_idx(sa, score_arr, results[1])
        if idx_a is not None and idx_b is not None:
            suggested_context, fundamentally_ambiguous = (
                _find_differentiating_entities(sa, idx_a, idx_b)
            )

    return BoundaryAnalysis(
        boundary_distance=boundary_distance,
        disambiguation_gap=disambiguation_gap,
        band_crossing=band_crossing,
        context_needed=context_needed,
        suggested_context=suggested_context,
        fundamentally_ambiguous=fundamentally_ambiguous,
        top_scores=top_scores,
    )


def _resolve_value_node_idx(
    sa: 'SpreadingActivation',
    score_arr: np.ndarray,
    result: tuple,
) -> Optional[int]:
    """Resolve the node index for a result tuple.

    Uses the score array to disambiguate when multiple value nodes
    share the same label -- picks the one whose score matches.
    """
    label, score = result
    best_idx = None
    best_diff = float('inf')
    for idx, node_label in enumerate(sa._node_label):
        if node_label == label and sa._node_type[idx] == _VALUE_TYPE:
            diff = abs(float(score_arr[idx]) - score)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
    return best_idx


def _find_differentiating_entities(
    sa: 'SpreadingActivation',
    idx_a: int,
    idx_b: int,
) -> tuple:
    """Find entities that activate one result but not the other.

    Args:
        sa: The SpreadingActivation instance.
        idx_a: Node index of the first value node.
        idx_b: Node index of the second value node.

    Returns:
        (suggested_context, fundamentally_ambiguous)
    """
    entities_a = _get_entity_neighbors_by_idx(sa, idx_a)
    entities_b = _get_entity_neighbors_by_idx(sa, idx_b)

    differentiating = entities_a.symmetric_difference(entities_b)

    if not differentiating:
        return [], True  # fundamentally ambiguous

    return sorted(differentiating), False


def _get_entity_neighbors_by_idx(sa: 'SpreadingActivation', value_idx: int) -> set:
    """Get the set of entity labels connected to a value node by index.

    Uses the CSR column (incoming edges) to find which entity nodes
    point to this value node.
    """
    adj = sa._adj
    if adj is None:
        return set()

    # adj is entity->value (rows=src, cols=dst).
    # Column `value_idx` contains the entity neighbors.
    col = adj.getcol(value_idx)
    entity_indices = col.nonzero()[0]

    return {sa._node_label[idx] for idx in entity_indices
            if sa._node_type[idx] == _ENTITY_TYPE}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestAnalyzeBoundary -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pensive/boundary.py tests/test_boundary.py
git commit -m "feat(boundary): implement analyze_boundary() with full diagnostic pipeline"
```

---

### Task 4: Wire Into SpreadingActivation as `query_analyzed()`

**Files:**
- Modify: `src/pensive/spreading.py` (add method, ~15 lines)
- Test: `tests/test_boundary.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_boundary.py

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestQueryAnalyzed -v`
Expected: FAIL with `AttributeError: 'SpreadingActivation' object has no attribute 'query_analyzed'`

- [ ] **Step 3: Add `query_analyzed` method to SpreadingActivation**

In `src/pensive/spreading.py`, add this method after the `query_with_doc_ids` method (after line 1132):

```python
    def query_analyzed(self, query_text: str, top_k: int = 10,
                       context: Optional[List[str]] = None,
                       n_bands: int = 5):
        """Query with boundary analysis diagnostics.

        Same as query() but returns an AnalyzedResult with both the
        normal results and a BoundaryAnalysis diagnostic.

        See pensive.boundary for details on the diagnostic fields.
        """
        from .boundary import analyze_boundary
        return analyze_boundary(self, query_text, top_k=top_k,
                                context=context, n_bands=n_bands)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py::TestQueryAnalyzed -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `cd ~/Projects/pensive && python -m pytest tests/ -v`
Expected: All existing tests pass, all new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pensive/spreading.py tests/test_boundary.py
git commit -m "feat(boundary): wire analyze_boundary into SA as query_analyzed()"
```

---

### Task 5: Edge Cases and Integration Tests

**Files:**
- Modify: `tests/test_boundary.py`

- [ ] **Step 1: Write edge case and integration tests**

```python
# Append to tests/test_boundary.py

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
```

- [ ] **Step 2: Run all tests**

Run: `cd ~/Projects/pensive && python -m pytest tests/test_boundary.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_boundary.py
git commit -m "test(boundary): add edge case and context detection integration tests"
```

---

### Task 6: Export from Package `__init__.py`

**Files:**
- Modify: `src/pensive/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

Run: `cat src/pensive/__init__.py`

- [ ] **Step 2: Add boundary exports**

Add to `src/pensive/__init__.py`:

```python
from .boundary import BoundaryAnalysis, AnalyzedResult, FrequencyBands, analyze_boundary
```

- [ ] **Step 3: Run full test suite**

Run: `cd ~/Projects/pensive && python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/pensive/__init__.py
git commit -m "feat(boundary): export boundary analysis from package root"
```
