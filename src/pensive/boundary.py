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
from typing import Dict, List, Optional

import numpy as np

# Node type constants (mirror spreading.py)
_ENTITY_TYPE = 0
_VALUE_TYPE = 1


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
        if self.context_needed:
            return "low"
        if self.boundary_distance >= 0.15 and (
            self.disambiguation_gap is None or self.disambiguation_gap >= 0.10
        ):
            return "high"
        if self.boundary_distance < 0.05 and (
            self.disambiguation_gap is not None and self.disambiguation_gap < 0.05
        ):
            return "low"
        return "medium"

    @property
    def should_trust(self) -> bool:
        """Research helper: whether the current result looks safe to trust."""
        if self.confidence == "none":
            return False
        if self.context_needed:
            return False
        return self.confidence in {"high", "medium"}

    @property
    def recommended_action(self) -> str:
        """Research helper for the next retrieval action to take."""
        if self.confidence == "none":
            return "no_result"
        if self.context_needed and not self.fundamentally_ambiguous:
            return "request_context"
        if self.context_needed:
            return "warn_ambiguous"
        if not self.should_trust:
            return "warn_low_confidence"
        return "trust"


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

        # Determine how many bands the data actually supports.
        # A meaningful band should span at least ~0.5 in log1p space,
        # which corresponds to roughly a 1.65x frequency ratio.
        # Tightly clustered frequencies (e.g. 10, 11, 12) collapse
        # to a single band rather than creating artificial splits.
        log_range = float(log_freqs.max() - log_freqs.min())
        min_band_width = 0.5
        range_bands = max(1, int(log_range / min_band_width))
        actual_bands = min(n_bands, len(set(log_freqs)), range_bands)

        if actual_bands <= 1:
            return cls({e: 0 for e in entities}, 1,
                       [float(log_freqs.min()), float(log_freqs.max())])

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


@dataclass
class AnalyzedResult:
    """Query results bundled with boundary analysis."""
    results: list  # List[Tuple[str, float]] -- same as query() returns
    analysis: BoundaryAnalysis


def analyze_boundary_results(sa, query_text: str,
                             results: list,
                             n_bands: int = 5) -> BoundaryAnalysis:
    """Compute boundary analysis for an existing SA result set.

    Accepts either `query()` results (`[(value, score), ...]`) or
    `query_with_doc_ids()` results (`[(doc_id, value, score), ...]`).
    This lets callers reuse already-fetched SA hits when they only need
    the diagnostic.
    """
    if not sa._built:
        raise ValueError("Graph not built. Call build() first.")

    # Mirror SpreadingActivation.query()'s falsy-query short-circuit
    # (PENPY-IMP-3). Without this, _compute_score_arr() would call
    # `query_text.split()` on None and raise AttributeError; an empty
    # string would propagate to an empty seed set with no useful
    # analysis. Either way, the analysis collapses to the no-results
    # branch of _compute_analysis, so return that directly.
    if not query_text:
        return _compute_analysis(
            sa,
            score_arr=np.zeros(0, dtype=np.float32),
            results=[],
            query_act={},
            top_k=0,
            n_bands=n_bands,
        )

    score_arr, query_act = _compute_score_arr(sa, query_text)
    normalized_results = _normalize_results(results)
    return _compute_analysis(
        sa, score_arr, normalized_results, query_act, len(normalized_results),
        n_bands,
    )


def analyze_boundary(sa, query_text: str,
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
    analysis = analyze_boundary_results(sa, query_text, results, n_bands=n_bands)
    return AnalyzedResult(results=results, analysis=analysis)


def _compute_score_arr(sa, query_text: str) -> tuple:
    """Return the raw score array and seeded query activations."""
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

    return score_arr, query_act


def _normalize_results(results: list) -> list:
    """Normalize SA results to `(label, score)` tuples."""
    normalized = []
    for result in results:
        if len(result) == 2:
            label, score = result
        elif len(result) >= 3:
            label, score = result[1], result[-1]
        else:
            raise ValueError(f"Unsupported SA result tuple: {result!r}")
        normalized.append((label, float(score)))
    return normalized


def _compute_analysis(
    sa,
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


def _resolve_value_node_idx(sa, score_arr: np.ndarray, result: tuple) -> Optional[int]:
    """Resolve the node index for a result tuple.

    Uses the score array to disambiguate when multiple value nodes
    share the same label -- picks the one whose score matches.

    Uses a per-SA cached label->[value-idx,...] inverse index instead of
    a linear scan over all nodes. At 164K-doc corpora the linear scan
    used to dominate per-query cost; the cache is built once on the
    first ambiguous-query analysis and reused thereafter. The cache is
    invalidated if the SA grows (node count changes); the next call
    rebuilds it lazily. (PENPY-IMP-2 perf fix.)
    """
    label, score = result
    index = _get_value_label_index(sa)
    matches = index.get(label)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    best_idx = None
    best_diff = float('inf')
    for idx in matches:
        diff = abs(float(score_arr[idx]) - score)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def _get_value_label_index(sa) -> Dict[str, List[int]]:
    """Return (cached) {label -> [value-node indices]} dict for the SA.

    Lazily built on first call; invalidated when the SA's node count
    changes (the only kind of growth this module observes). Caching is
    keyed on a private attribute on the SA itself so callers don't need
    to thread state through.
    """
    cache_attr = '_boundary_value_label_index'
    n_nodes = len(sa._idx_to_node)
    cached = getattr(sa, cache_attr, None)
    if cached is not None and cached[0] == n_nodes:
        return cached[1]

    index: Dict[str, List[int]] = {}
    # Prefer the precomputed _value_indices numpy array if available; it
    # already filters to value nodes, avoiding the per-node type check.
    value_indices = getattr(sa, '_value_indices', None)
    if value_indices is not None:
        node_label = sa._node_label
        for idx in value_indices:
            idx_int = int(idx)
            index.setdefault(node_label[idx_int], []).append(idx_int)
    else:
        node_type = sa._node_type
        for idx, node_label in enumerate(sa._node_label):
            if node_type[idx] == _VALUE_TYPE:
                index.setdefault(node_label, []).append(idx)

    try:
        setattr(sa, cache_attr, (n_nodes, index))
    except (AttributeError, TypeError):
        # SA may use __slots__ in some future version; fall back to
        # returning the freshly-built index uncached.
        pass
    return index


def _find_differentiating_entities(sa, idx_a: int, idx_b: int) -> tuple:
    """Find entities that activate one result but not the other.

    Returns:
        (suggested_context, fundamentally_ambiguous)
    """
    entities_a = _get_entity_neighbors_by_idx(sa, idx_a)
    entities_b = _get_entity_neighbors_by_idx(sa, idx_b)

    differentiating = entities_a.symmetric_difference(entities_b)

    if not differentiating:
        return [], True  # fundamentally ambiguous

    return sorted(differentiating), False


def _get_entity_neighbors_by_idx(sa, value_idx: int) -> set:
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
