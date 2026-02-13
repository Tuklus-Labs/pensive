"""Ingestion pipeline orchestrator for spreading activation.

Coordinates parsing from multiple data sources, builds the SA graph
incrementally, and provides pickle-based serialization for persistence.
"""
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..spreading import SpreadingActivation, SpreadingConfig
from ..patterns import REAL_DATA_PATTERNS
from .base import BaseParser

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrate ingestion from all sources into a SpreadingActivation graph."""

    def __init__(
        self,
        sa: Optional[SpreadingActivation] = None,
        batch_size: int = 1000,
        progress_callback: Optional[Callable] = None,
    ):
        self.sa = sa or SpreadingActivation(
            config=SpreadingConfig(
                max_hops=2,
                max_active=200,
            ),
            patterns=REAL_DATA_PATTERNS,
        )
        self.batch_size = batch_size
        self.progress = progress_callback or (lambda *a: None)
        self.stats: Dict[str, int] = defaultdict(int)

    def ingest_source(self, parser: BaseParser) -> int:
        """Ingest all documents from a single source."""
        name = parser.source_name()
        count = 0
        batch = []
        t0 = time.time()

        for doc in parser.parse():
            batch.append(doc.to_sa_dict())
            count += 1
            if len(batch) >= self.batch_size:
                self.sa.add_documents(batch)
                batch = []
                self.progress(name, count)

        if batch:
            self.sa.add_documents(batch)

        elapsed = time.time() - t0
        self.stats[name] = count
        logger.info("%s: ingested %d docs in %.1fs", name, count, elapsed)
        return count

    def ingest_all(self, parsers: List[BaseParser]) -> Dict[str, int]:
        """Ingest all sources sequentially."""
        for parser in parsers:
            name = parser.source_name()
            self.progress(f"Starting {name}", 0)
            n = self.ingest_source(parser)
            self.progress(f"Finished {name}", n)
        return dict(self.stats)

    def save_graph(self, path: str) -> None:
        """Serialize the SA graph to disk."""
        data = self.sa.get_save_data()
        data['pipeline_stats'] = dict(self.stats)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        logger.info("Graph saved to %s (%.1f MB)", path, size_mb)

    @classmethod
    def load_graph(cls, path: str) -> 'IngestPipeline':
        """Load a previously built graph."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sa = SpreadingActivation.from_save_data(data)
        pipe = cls(sa=sa)
        pipe.stats = data.get('pipeline_stats', data.get('stats', {}))
        return pipe
