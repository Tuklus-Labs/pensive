"""Recall engine: turning atom text into vectors and searching them.

Phase 2 opener. ``Embedder`` + ``embedMissing`` fill the ``embeddings`` table;
``VectorIndex`` / ``FlatIndex`` search it.
"""
from recall.embedder import Embedder, embedMissing, vecToBlob, blobToVec
from recall.vector_index import VectorIndex, FlatIndex

__all__ = [
    "Embedder",
    "embedMissing",
    "vecToBlob",
    "blobToVec",
    "VectorIndex",
    "FlatIndex",
]
