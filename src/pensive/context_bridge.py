"""
L1 Context Bridge for Spreading Activation.

Extracts recent entity context from L1 cache to feed into
contextual spreading activation queries. This bridges the gap
between Pensive's memory state and the graph-based retrieval.

Sources of context:
1. Recent query responses (entities mentioned in last N responses)
2. Entity index (recently accessed entities)
3. Memory graph neighbors (entities associated with query terms)
"""
import re
from collections import defaultdict
from typing import List, Optional, Protocol, runtime_checkable

try:
    from .patterns import REAL_DATA_PATTERNS
except ImportError:
    REAL_DATA_PATTERNS = None


@runtime_checkable
class L1CacheLike(Protocol):
    """Minimal interface for L1 cache context extraction."""
    entity_index: dict
    recent_queries: dict

    def get_associations(self, entity: str, max_hops: int = 2,
                        min_weight: float = 0.1,
                        max_paths: int = 10) -> list:
        ...


def _extract_entities_from_text(text: str) -> List[str]:
    """Extract entity strings from free text using the pattern registry."""
    patterns = REAL_DATA_PATTERNS
    if patterns is None:
        # Fallback if import failed
        patterns = [
            (r'\b(\d{4}-\d{2}-\d{2})\b', 'date', True),
            (r'\b(\d+(?:ms|MB|GB|%|people))\b', 'metric', True),
            (r'\b([A-Z]-\d{2,4})\b', 'room', False),
            (r'\b(prod-[a-z0-9-]+|gateway-[a-z0-9-]+)\b', 'server', True),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\b', 'person', False),
        ]
    entities = set()
    for pat, _etype, ci in patterns:
        flags = re.IGNORECASE if ci else 0
        for m in re.findall(pat, text, flags):
            entity = (m if isinstance(m, str) else m[0]).lower()
            if len(entity) >= 2:
                entities.add(entity)
    return list(entities)


class L1ContextBridge:
    """Bridges L1 cache entity state into spreading activation context.

    Extracts recent entities from:
    1. L1 recent_queries cache (entities in recent responses)
    2. L1 entity_index (recently accessed entities)
    3. L1 memory_graph neighbors (associations with query terms)
    """

    def __init__(self, l1_cache, max_context_entities: int = 20,
                 recent_query_window: int = 5):
        """
        Args:
            l1_cache: L1Cache instance (or anything matching L1CacheLike)
            max_context_entities: Max entities to return per query
            recent_query_window: How many recent queries to scan
        """
        self.l1 = l1_cache
        self.max_entities = max_context_entities
        self.recent_window = recent_query_window

    def get_context(self, query_text: str) -> List[str]:
        """Get context entities relevant to this query.

        Pulls from three sources in priority order:
        1. Recent query responses (most relevant - what user just discussed)
        2. Graph neighbors of query entities (associated concepts)
        3. Active entity index entries (recently accessed)

        Returns:
            List of entity strings, deduplicated, capped at max_entities.
            Entities that appear in the query itself are excluded
            (they're already seeded by the primary activation).
        """
        context = []
        query_lower = query_text.lower()

        # Source 1: Entities from recent query responses
        context.extend(self._from_recent_queries())

        # Source 2: Graph neighbors of query terms
        context.extend(self._from_graph_associations(query_text))

        # Source 3: Recently active entities from index
        context.extend(self._from_entity_index())

        # Deduplicate and filter out entities already in the query
        seen = set()
        filtered = []
        for entity in context:
            e_lower = entity.lower()
            if e_lower not in seen and e_lower not in query_lower:
                seen.add(e_lower)
                filtered.append(e_lower)
                if len(filtered) >= self.max_entities:
                    break

        return filtered

    def _from_recent_queries(self) -> List[str]:
        """Extract entities from recent query responses."""
        entities = []
        try:
            recent = self.l1.recent_queries
            # OrderedDict - iterate in reverse (most recent first)
            items = list(recent.values())[-self.recent_window:]
            for entry in reversed(items):
                response_text = ''
                if isinstance(entry, dict):
                    response_text = entry.get('response', '')
                elif isinstance(entry, str):
                    response_text = entry
                if response_text:
                    entities.extend(_extract_entities_from_text(response_text))
        except (AttributeError, TypeError):
            pass
        return entities

    def _from_graph_associations(self, query_text: str) -> List[str]:
        """Get associated entities from L1's memory graph."""
        entities = []
        try:
            # Extract entities from the query
            query_entities = _extract_entities_from_text(query_text)
            for qe in query_entities[:3]:  # Limit to avoid explosion
                paths = self.l1.get_associations(qe, max_hops=1,
                                                  min_weight=0.3,
                                                  max_paths=5)
                for path in paths:
                    for node_label, _, _ in path:
                        entities.append(node_label.lower())
        except (AttributeError, TypeError):
            pass
        return entities

    def _from_entity_index(self) -> List[str]:
        """Get recently active entities from L1's entity index."""
        entities = []
        try:
            # Entity index: entity_name -> [summaries]
            # We want entities with recent activity
            for entity_name in self.l1.entity_index:
                entities.append(str(entity_name).lower())
        except (AttributeError, TypeError):
            pass
        return entities
