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
