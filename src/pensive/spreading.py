"""
Spreading activation retrieval with specificity weighting.

Uses scipy.sparse CSR matrices instead of networkx for ~90% memory
reduction at scale (23M edges: 19.5GB -> ~2GB).

Usage:
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build(documents)  # List of dicts with 'content', 'id', 'value' keys
    results = sa.query("What was the P99 latency on 2025-10-08?")
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .mega_extract import MegaExtractor
from .patterns import REAL_DATA_PATTERNS, SYNTHETIC_PATTERNS

import numpy as np
import scipy.sparse

# Node type constants
_ENTITY_TYPE = 0
_VALUE_TYPE = 1

# Common English words that should not be indexed as partial entity matches.
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'must',
    'not', 'no', 'nor', 'so', 'if', 'when', 'what', 'which', 'who', 'whom',
    'how', 'where', 'why', 'that', 'this', 'these', 'those', 'it', 'its',
    'he', 'she', 'they', 'we', 'you', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'any', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'also', 'now', 'here', 'there', 'then', 'once', 'much', 'many',
    'well', 'back', 'even', 'still', 'new', 'old', 'long', 'big', 'good',
    'bad', 'great', 'right', 'left', 'high', 'low', 'part', 'first', 'last',
    'made', 'make', 'like', 'get', 'got', 'set', 'put', 'see', 'say', 'said',
    'know', 'take', 'come', 'think', 'look', 'want', 'give', 'use', 'find',
    'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep',
    'let', 'begin', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
    'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
    'pay', 'meet', 'include', 'continue', 'end', 'start', 'turn', 'help',
    'way', 'yes', 'using', 'going', 'able', 'may', 'might',
})


@dataclass
class SpreadingConfig:
    """Configuration for spreading activation."""
    # Specificity weighting
    spec_power: float = 0.2  # Lower = milder penalty for common entities
    edge_weight: float = 5.0  # Entity -> answer edge weight multiplier

    # Activation boosting
    exact_boost: float = 1.5   # Exact word match
    partial_boost: float = 0.8  # Word in multi-word label
    substr_boost: float = 0.5   # Substring match (4+ chars)

    # Spreading parameters
    decay: float = 0.6
    threshold: float = 0.15
    max_hops: int = 4
    max_active: int = 50


# Default patterns for real conversational data
PATTERNS = REAL_DATA_PATTERNS


_worker_extractor = None


def _init_worker(extractor):
    """Set the shared extractor in each forked worker process."""
    global _worker_extractor
    _worker_extractor = extractor


def _extract_chunk(chunk):
    """Worker function for multiprocessing entity extraction."""
    return [(doc, _worker_extractor.extract(
        doc['content'] + (f" {doc['query']}" if 'query' in doc else "")
    )) for doc in chunk]


class SpreadingActivation:
    """
    Spreading activation retrieval with specificity weighting.

    Key insight: Weight entities by inverse frequency - rare entities
    are more specific and discriminative.

    Uses scipy.sparse CSR matrices internally for ~90% memory reduction
    vs networkx at scale.
    """

    def __init__(self, config: Optional[SpreadingConfig] = None,
                 patterns: Optional[List[Tuple]] = None):
        self.config = config or SpreadingConfig()
        self.patterns = patterns or PATTERNS

        # Node storage (parallel arrays)
        self._node_to_idx: Dict[str, int] = {}
        self._idx_to_node: List[str] = []
        self._node_type: List[int] = []
        self._node_label: List[str] = []
        self._node_specificity: List[float] = []

        # Edge storage (COO during build, CSR for queries)
        self._edge_src: List[int] = []
        self._edge_dst: List[int] = []
        self._edge_weight: List[float] = []
        self._adj: Optional[scipy.sparse.csr_matrix] = None
        self._dirty = True

        self.entity_freq: Dict[str, int] = defaultdict(int)
        self._entity_index: Dict[str, List[str]] = defaultdict(list)
        self._built = False
        self._context_provider = None
        self._extractor = MegaExtractor(self.patterns)

    def _get_or_add_node(self, node_id: str, node_type: int,
                         label: str, specificity: float) -> int:
        """Get existing node index or add a new node. Returns the index."""
        idx = self._node_to_idx.get(node_id)
        if idx is not None:
            self._node_specificity[idx] = specificity
            return idx
        idx = len(self._idx_to_node)
        self._node_to_idx[node_id] = idx
        self._idx_to_node.append(node_id)
        self._node_type.append(node_type)
        self._node_label.append(label)
        self._node_specificity.append(specificity)
        return idx

    def _add_edge(self, src_idx: int, dst_idx: int, weight: float) -> None:
        """Add an edge (COO format). Marks graph as dirty."""
        self._edge_src.append(src_idx)
        self._edge_dst.append(dst_idx)
        self._edge_weight.append(weight)
        self._dirty = True

    def _compile(self) -> None:
        """Convert COO edge lists to CSR matrix for fast neighbor iteration."""
        if not self._dirty:
            return
        n = len(self._idx_to_node)
        if not self._edge_src:
            self._adj = scipy.sparse.csr_matrix((n, n), dtype=np.float32)
        else:
            self._adj = scipy.sparse.csr_matrix(
                (np.array(self._edge_weight, dtype=np.float32),
                 (np.array(self._edge_src, dtype=np.int32),
                  np.array(self._edge_dst, dtype=np.int32))),
                shape=(n, n),
            )
        self._dirty = False

    def set_context_provider(self, provider):
        """Set a callable that provides context entities for each query.

        The provider receives the query text and returns a list of
        entity strings from the current conversation/episode.

        Args:
            provider: Callable[[str], List[str]]
        """
        self._context_provider = provider

    def build(self, documents: List[Dict]) -> 'SpreadingActivation':
        """
        Build the graph from documents.

        Each document should have:
        - 'content': str - The text content
        - 'id': str - Unique identifier
        - 'value': str - The answer/value to retrieve
        - 'query': str (optional) - Query text to include in entity extraction
        """
        self._node_to_idx = {}
        self._idx_to_node = []
        self._node_type = []
        self._node_label = []
        self._node_specificity = []
        self._edge_src = []
        self._edge_dst = []
        self._edge_weight = []
        self._adj = None
        self._dirty = True
        self.entity_freq = defaultdict(int)
        self._entity_index = defaultdict(list)

        # Single extraction pass
        extracted = []
        for doc in documents:
            text = doc['content']
            if 'query' in doc:
                text = f"{text} {doc['query']}"
            extracted.append((doc, self._extractor.extract(text)))

        # Count entity frequencies
        for _, entities in extracted:
            for entity, _ in entities:
                self.entity_freq[entity] += 1

        # Build graph with specificity weights
        for doc, entities in extracted:
            answer_node = f"v:{doc['id']}"
            ans_idx = self._get_or_add_node(
                answer_node, _VALUE_TYPE, doc['value'], 0.0
            )

            doc_entities = {}
            for entity, etype in entities:
                if len(entity) < 2:
                    continue
                node_id = f"e:{etype}:{entity}"
                specificity = 1.0 / (self.entity_freq[entity] ** self.config.spec_power)
                doc_entities[node_id] = (entity, etype, specificity)

            for node_id, (entity, etype, specificity) in doc_entities.items():
                is_new = node_id not in self._node_to_idx
                ent_idx = self._get_or_add_node(
                    node_id, _ENTITY_TYPE, entity, specificity
                )

                if is_new:
                    self._entity_index[entity].append(node_id)
                    for word in entity.split():
                        if word != entity and word not in _STOPWORDS:
                            self._entity_index[word].append(node_id)

                self._add_edge(
                    ent_idx, ans_idx,
                    specificity * self.config.edge_weight
                )

        self._built = True
        return self

    def add_document(self, doc: Dict) -> None:
        """Incrementally add a single document to the graph."""
        text = doc['content']
        if 'query' in doc:
            text = f"{text} {doc['query']}"

        doc_entities = {}
        for entity, etype in self._extractor.extract(text):
            if len(entity) < 2:
                continue
            self.entity_freq[entity] += 1
            node_id = f"e:{etype}:{entity}"
            doc_entities[node_id] = (entity, etype)

        answer_node = f"v:{doc['id']}"
        ans_idx = self._get_or_add_node(
            answer_node, _VALUE_TYPE, doc['value'], 0.0
        )

        for node_id, (entity, etype) in doc_entities.items():
            specificity = 1.0 / (self.entity_freq[entity] ** self.config.spec_power)

            is_new = node_id not in self._node_to_idx
            ent_idx = self._get_or_add_node(
                node_id, _ENTITY_TYPE, entity, specificity
            )

            if is_new:
                self._entity_index[entity].append(node_id)
                for word in entity.split():
                    if word != entity and word not in _STOPWORDS:
                        self._entity_index[word].append(node_id)

            self._add_edge(
                ent_idx, ans_idx,
                specificity * self.config.edge_weight
            )

        self._built = True

    def add_documents(self, documents: List[Dict]) -> None:
        """Incrementally add multiple documents to the graph."""
        for doc in documents:
            self.add_document(doc)

    def build_parallel(self, documents: List[Dict],
                       workers: Optional[int] = None) -> 'SpreadingActivation':
        """Build graph using multiprocessing for entity extraction.

        Entity extraction (regex) is CPU-bound and embarrassingly parallel.
        Graph construction (node/edge creation) remains sequential since it
        mutates shared state, but is fast compared to extraction.

        Args:
            documents: Same format as build()
            workers: Number of worker processes. Defaults to min(cpu_count, 16).
        """
        import multiprocessing as mp

        if workers is None:
            workers = min(mp.cpu_count(), 16)

        if workers <= 1 or len(documents) < 10_000:
            return self.build(documents)

        # Reset
        self._node_to_idx = {}
        self._idx_to_node = []
        self._node_type = []
        self._node_label = []
        self._node_specificity = []
        self._edge_src = []
        self._edge_dst = []
        self._edge_weight = []
        self._adj = None
        self._dirty = True
        self.entity_freq = defaultdict(int)
        self._entity_index = defaultdict(list)

        # Chunk documents for workers
        chunk_size = max(1000, len(documents) // workers)
        chunks = [documents[i:i + chunk_size]
                  for i in range(0, len(documents), chunk_size)]

        # Use fork context to share parent's compiled regexes with workers
        ctx = mp.get_context('fork')
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(self._extractor,)) as pool:
            chunk_results = pool.map(_extract_chunk, chunks)

        extracted = [item for chunk in chunk_results for item in chunk]

        for _, entities in extracted:
            for entity, _ in entities:
                self.entity_freq[entity] += 1

        for doc, entities in extracted:
            answer_node = f"v:{doc['id']}"
            ans_idx = self._get_or_add_node(
                answer_node, _VALUE_TYPE, doc['value'], 0.0
            )

            doc_entities = {}
            for entity, etype in entities:
                if len(entity) < 2:
                    continue
                node_id = f"e:{etype}:{entity}"
                specificity = 1.0 / (self.entity_freq[entity] ** self.config.spec_power)
                doc_entities[node_id] = (entity, etype, specificity)

            for node_id, (entity, etype, specificity) in doc_entities.items():
                is_new = node_id not in self._node_to_idx
                ent_idx = self._get_or_add_node(
                    node_id, _ENTITY_TYPE, entity, specificity
                )

                if is_new:
                    self._entity_index[entity].append(node_id)
                    for word in entity.split():
                        if word != entity and word not in _STOPWORDS:
                            self._entity_index[word].append(node_id)

                self._add_edge(
                    ent_idx, ans_idx,
                    specificity * self.config.edge_weight
                )

        self._built = True
        return self

    def _seed_from_words(self, words: List[str]) -> Dict[int, float]:
        """Seed activation from a list of lowercase words."""
        activations: Dict[int, float] = {}
        do_substr = len(self._entity_index) <= 10_000

        for word in words:
            if word in _STOPWORDS:
                continue
            for node_id in self._entity_index.get(word, []):
                idx = self._node_to_idx.get(node_id)
                if idx is None:
                    continue
                spec = self._node_specificity[idx]
                label = self._node_label[idx]
                if word == label:
                    activations[idx] = max(
                        activations.get(idx, 0),
                        self.config.exact_boost * spec
                    )
                else:
                    activations[idx] = max(
                        activations.get(idx, 0),
                        self.config.partial_boost * spec
                    )

            if do_substr and len(word) >= 4:
                for label, node_ids in self._entity_index.items():
                    if word != label and word in label:
                        for node_id in node_ids:
                            idx = self._node_to_idx.get(node_id)
                            if idx is None:
                                continue
                            if idx not in activations:
                                spec = self._node_specificity[idx]
                                activations[idx] = max(
                                    activations.get(idx, 0),
                                    self.config.substr_boost * spec
                                )
        return activations

    def _spread(self, activations: Dict[int, float],
                hops: Optional[int] = None) -> Dict[int, float]:
        """Run spreading activation for N hops."""
        self._compile()

        hops = hops if hops is not None else self.config.max_hops
        indptr = self._adj.indptr
        indices = self._adj.indices
        data = self._adj.data

        for _ in range(hops):
            new_act: Dict[int, float] = defaultdict(float)

            for node_idx, act in activations.items():
                new_act[node_idx] = max(new_act[node_idx], act * self.config.decay)

                if act < self.config.threshold:
                    continue

                row_start = indptr[node_idx]
                row_end = indptr[node_idx + 1]
                for j in range(row_start, row_end):
                    neighbor = indices[j]
                    weight = data[j]
                    new_act[neighbor] = max(
                        new_act[neighbor],
                        act * self.config.decay * weight
                    )

            if len(new_act) > self.config.max_active:
                new_act = dict(
                    sorted(new_act.items(), key=lambda x: -x[1])[:self.config.max_active]
                )

            activations = {k: v for k, v in new_act.items()
                          if v >= self.config.threshold}

        return activations

    def _collect_results(self, activations: Dict[int, float],
                         top_k: int) -> List[Tuple[str, float]]:
        """Collect value nodes from activation map."""
        results = []
        for idx, score in sorted(activations.items(), key=lambda x: -x[1]):
            if self._node_type[idx] == _VALUE_TYPE:
                label = self._node_label[idx]
                results.append((label, score))
                if len(results) >= top_k:
                    break
        return results

    def _collect_results_with_ids(self, activations: Dict[int, float],
                                   top_k: int) -> List[Tuple[str, str, float]]:
        """Collect value nodes with doc IDs from activation map."""
        results = []
        for idx, score in sorted(activations.items(), key=lambda x: -x[1]):
            if self._node_type[idx] == _VALUE_TYPE:
                node_id = self._idx_to_node[idx]
                doc_id = node_id[2:] if node_id.startswith('v:') else node_id
                label = self._node_label[idx]
                results.append((doc_id, label, score))
                if len(results) >= top_k:
                    break
        return results

    def query(self, query_text: str, top_k: int = 10,
              context: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Query the graph using spreading activation with optional
        contextual intersection.

        Args:
            query_text: The natural language query
            top_k: Number of results to return
            context: Optional list of entity strings from conversation
                     context. When provided, these seed a second
                     activation pass that intersects with the query
                     activation - disambiguating candidates that the
                     query alone can't distinguish.

        Returns:
            List of (answer, activation_score) tuples
        """
        if not self._built:
            raise ValueError("Graph not built. Call build() first.")

        if context is None and self._context_provider is not None:
            context = self._context_provider(query_text)

        words = [w.lower().strip('?.,') for w in query_text.split()
                 if len(w) >= 2]
        query_act = self._seed_from_words(words)
        query_act = self._spread(query_act)

        if not context:
            return self._collect_results(query_act, top_k)

        ctx_words = [w.lower().strip('?.,') for w in context
                     if len(w) > 1]
        ctx_act = self._seed_from_words(ctx_words)
        ctx_act = self._spread(ctx_act, hops=1)

        intersected = {}
        for idx, q_score in query_act.items():
            if idx in ctx_act:
                intersected[idx] = q_score * (1.0 + ctx_act[idx])
            else:
                intersected[idx] = q_score

        return self._collect_results(intersected, top_k)

    def query_with_doc_ids(self, query_text: str, top_k: int = 50,
                           context: Optional[List[str]] = None
                           ) -> List[Tuple[str, str, float]]:
        """Query returning doc IDs for hybrid retrieval integration.

        Same as query() but returns (doc_id, value, score) tuples
        for use as a first-stage retriever before L2 reranking.
        """
        if not self._built:
            raise ValueError("Graph not built. Call build() first.")

        if context is None and self._context_provider is not None:
            context = self._context_provider(query_text)

        words = [w.lower().strip('?.,') for w in query_text.split()
                 if len(w) >= 2]
        query_act = self._seed_from_words(words)
        query_act = self._spread(query_act)

        if not context:
            return self._collect_results_with_ids(query_act, top_k)

        ctx_words = [w.lower().strip('?.,') for w in context if len(w) > 1]
        ctx_act = self._seed_from_words(ctx_words)
        ctx_act = self._spread(ctx_act, hops=1)

        intersected = {}
        for idx, q_score in query_act.items():
            if idx in ctx_act:
                intersected[idx] = q_score * (1.0 + ctx_act[idx])
            else:
                intersected[idx] = q_score

        return self._collect_results_with_ids(intersected, top_k)

    def stats(self) -> Dict:
        """Return graph statistics."""
        n_nodes = len(self._idx_to_node)
        n_entity = sum(1 for t in self._node_type if t == _ENTITY_TYPE)
        n_value = n_nodes - n_entity
        if self._dirty or self._adj is None:
            n_edges = len(self._edge_src)
        else:
            n_edges = self._adj.nnz
        return {
            'nodes': n_nodes,
            'edges': n_edges,
            'entity_nodes': n_entity,
            'value_nodes': n_value,
            'unique_entities': len(self.entity_freq),
        }

    def get_save_data(self) -> Dict:
        """Return all state needed to serialize this graph."""
        self._compile()
        return {
            'format': 'sparse_v1',
            'node_to_idx': self._node_to_idx,
            'idx_to_node': self._idx_to_node,
            'node_type': self._node_type,
            'node_label': self._node_label,
            'node_specificity': self._node_specificity,
            'adj_data': self._adj.data,
            'adj_indices': self._adj.indices,
            'adj_indptr': self._adj.indptr,
            'adj_shape': self._adj.shape,
            'entity_freq': dict(self.entity_freq),
            'entity_index': dict(self._entity_index),
            'config': self.config,
            'patterns': self.patterns,
        }

    @classmethod
    def from_save_data(cls, data: Dict) -> 'SpreadingActivation':
        """Restore a SpreadingActivation from saved data.

        Handles both the new sparse_v1 format and legacy networkx pickles.
        """
        if data.get('format') == 'sparse_v1':
            return cls._from_sparse_v1(data)
        else:
            return cls._from_legacy_networkx(data)

    @classmethod
    def _from_sparse_v1(cls, data: Dict) -> 'SpreadingActivation':
        """Restore from sparse_v1 format."""
        config = data.get('config', SpreadingConfig())
        patterns = data.get('patterns', PATTERNS)
        sa = cls(config=config, patterns=patterns)

        sa._node_to_idx = data['node_to_idx']
        sa._idx_to_node = data['idx_to_node']
        sa._node_type = data['node_type']
        sa._node_label = data['node_label']
        sa._node_specificity = data['node_specificity']

        sa._adj = scipy.sparse.csr_matrix(
            (data['adj_data'], data['adj_indices'], data['adj_indptr']),
            shape=data['adj_shape'],
        )
        sa._edge_src = []
        sa._edge_dst = []
        sa._edge_weight = []
        sa._dirty = False

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._built = True
        return sa

    @classmethod
    def _from_legacy_networkx(cls, data: Dict) -> 'SpreadingActivation':
        """Restore from legacy networkx pickle format."""
        import networkx as nx

        config = data.get('config', SpreadingConfig())
        patterns = data.get('patterns', PATTERNS)
        sa = cls(config=config, patterns=patterns)

        graph = data['graph']
        for node_id in graph.nodes():
            nd = graph.nodes[node_id]
            ntype = _ENTITY_TYPE if nd.get('type') == 'entity' else _VALUE_TYPE
            label = nd.get('label', '')
            spec = nd.get('specificity', 0.0)
            sa._get_or_add_node(node_id, ntype, label, spec)

        for src, dst, ed in graph.edges(data=True):
            src_idx = sa._node_to_idx[src]
            dst_idx = sa._node_to_idx[dst]
            weight = ed.get('weight', 1.0)
            sa._edge_src.append(src_idx)
            sa._edge_dst.append(dst_idx)
            sa._edge_weight.append(weight)

        sa._dirty = True
        sa._compile()

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._built = True
        return sa
