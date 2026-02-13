"""Ingestion pipeline for building SA graphs from data exports."""
from .base import SADocument, BaseParser
from .chunker import SentenceAwareChunker


def __getattr__(name):
    if name == 'IngestPipeline':
        from .pipeline import IngestPipeline
        return IngestPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
