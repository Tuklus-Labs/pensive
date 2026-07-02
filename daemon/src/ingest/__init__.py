"""Ingest package: bringing an existing corpus into the v3 store.

``backfill`` is the one-time migration path (Task 11): it reads a read-only
export of the live corpus and writes atoms, provenance, and facets through the
canonical store API, ready for ``embedMissing`` to vectorize.
"""
from ingest.backfill import backfill

__all__ = ["backfill"]
