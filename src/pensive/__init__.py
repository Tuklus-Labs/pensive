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
    results = sa.query("What was the P99 latency?")
"""
from .spreading import SpreadingActivation, SpreadingConfig
from .mega_extract import MegaExtractor
from .patterns import REAL_DATA_PATTERNS, ALL_PATTERNS, build_pattern_set

__version__ = "0.1.1"

__all__ = [
    'SpreadingActivation',
    'SpreadingConfig',
    'MegaExtractor',
    'REAL_DATA_PATTERNS',
    'ALL_PATTERNS',
    'build_pattern_set',
]
