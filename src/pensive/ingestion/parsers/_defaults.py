"""Shared default instances for the parsers.

SentenceAwareChunker and QueryGenerator hold no per-parser state, so every
parser reuses one instance of each instead of constructing its own.
"""
from ..chunker import SentenceAwareChunker
from ..query_gen import QueryGenerator

DEFAULT_CHUNKER = SentenceAwareChunker()
DEFAULT_QUERY_GEN = QueryGenerator()
