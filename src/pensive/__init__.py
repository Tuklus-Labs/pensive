"""Pensive - Spreading activation retrieval for document collections.

Fast, sub-millisecond entity-graph retrieval that scales to 50M+ documents.
Uses regex-based entity extraction and scipy sparse matrices for memory
efficiency.

Quickstart:
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08', 'value': '42ms'},
        {'id': '2', 'content': 'GPU temp hit 82C during training run', 'value': '82C'},
    ])

    # Queries are entity-exact -- pass the entity surface form, NOT a
    # natural-language question. See README "Quickstart" + the
    # "Natural-language queries" section for the entity-extraction
    # pattern that maps a question like "What was the P99 latency?"
    # onto a query the engine understands.
    results = sa.query("42ms")
    # [('42ms', 4.5)]
"""
from .spreading import SpreadingActivation, SpreadingConfig
from .mega_extract import MegaExtractor
from .patterns import REAL_DATA_PATTERNS, ALL_PATTERNS, build_pattern_set
from .boundary import BoundaryAnalysis, AnalyzedResult, FrequencyBands, analyze_boundary
from .ingestion.pipeline import IngestPipeline

__version__ = "0.2.0"

__all__ = [
    'SpreadingActivation',
    'SpreadingConfig',
    'MegaExtractor',
    'IngestPipeline',
    'REAL_DATA_PATTERNS',
    'ALL_PATTERNS',
    'build_pattern_set',
    'BoundaryAnalysis',
    'AnalyzedResult',
    'FrequencyBands',
    'analyze_boundary',
]
