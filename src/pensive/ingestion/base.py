"""Base types and interfaces for the ingestion pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Any


@dataclass
class SADocument:
    """Document ready for SpreadingActivation.add_document()."""
    content: str
    doc_id: str
    value: str
    query: str = ''
    source: str = ''
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sa_dict(self) -> dict:
        d = {'content': self.content, 'id': self.doc_id, 'value': self.value}
        if self.query:
            d['query'] = self.query
        return d


class BaseParser(ABC):
    """Base class for data source parsers."""

    @abstractmethod
    def parse(self) -> Iterator[SADocument]:
        """Yield SADocuments from this data source."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Return a short identifier for this source."""
        ...
