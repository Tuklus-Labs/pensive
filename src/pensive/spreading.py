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
import array
import heapq
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .mega_extract import MegaExtractor
from .patterns import REAL_DATA_PATTERNS, SYNTHETIC_PATTERNS

import numpy as np
import scipy.sparse

# Optional numba JIT for ~38x faster spread kernel
try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

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

# Split entity labels into tokens on non-alphanumeric boundaries
_TOKEN_RE = re.compile(r'[a-z0-9]+', re.IGNORECASE)


# Numba-JIT spread kernel: eliminates Python loop overhead for ~38x speedup.
# Falls back to numpy path if numba is not installed.
if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _numba_spread_bipartite(act_indices, act_scores, indptr, indices,
                                 data, decay, threshold, n):
        """JIT-compiled bipartite spread: scatter-max via CSR adjacency."""
        result = np.zeros(n, dtype=np.float32)
        for k in range(len(act_indices)):
            idx = act_indices[k]
            score = act_scores[k]
            decayed = score * decay
            if decayed > result[idx]:
                result[idx] = decayed
            if score < threshold:
                continue
            row_start = indptr[idx]
            row_end = indptr[idx + 1]
            for j in range(row_start, row_end):
                val = score * decay * data[j]
                nbr = indices[j]
                if val > result[nbr]:
                    result[nbr] = val
        return result
else:
    _numba_spread_bipartite = None


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
        # array.array enables zero-copy np.frombuffer in _compile()
        self._edge_src = array.array('i')
        self._edge_dst = array.array('i')
        self._edge_weight = array.array('f')
        self._entity_edge_positions: Dict[int, List[int]] = defaultdict(list)
        self._adj: Optional[scipy.sparse.csr_matrix] = None
        self._dirty = True

        self.entity_freq: Dict[str, int] = defaultdict(int)
        self._entity_index: Dict[str, List[str]] = defaultdict(list)
        self._exact_entities = set()
        self._entity_terms: List[str] = []
        self._token_index: Dict[str, List[str]] = defaultdict(list)  # token -> entity labels
        self._substr_match_cache: Dict[str, List[str]] = {}
        self._built = False
        self._is_bipartite = True  # True until proven otherwise
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
        pos = len(self._edge_src)
        self._edge_src.append(src_idx)
        self._edge_dst.append(dst_idx)
        self._edge_weight.append(weight)
        if self._node_type[src_idx] == _ENTITY_TYPE:
            self._entity_edge_positions[src_idx].append(pos)
        self._dirty = True

    def _compile(self) -> None:
        """Convert COO edge lists to CSR matrix for fast neighbor iteration.

        Uses np.frombuffer for zero-copy views of the array.array buffers.
        scipy internally copies during CSR construction so the view is safe.
        Also caches numpy arrays for node_type and node_specificity.
        """
        if not self._dirty:
            return
        n = len(self._idx_to_node)
        if not self._edge_src:
            self._adj = scipy.sparse.csr_matrix((n, n), dtype=np.float32)
        else:
            weights = np.frombuffer(self._edge_weight, dtype=np.float32)
            rows = np.frombuffer(self._edge_src, dtype=np.int32)
            cols = np.frombuffer(self._edge_dst, dtype=np.int32)
            self._adj = scipy.sparse.csr_matrix(
                (weights, (rows, cols)), shape=(n, n),
            )
        # Cache numpy arrays for vectorized operations in _collect_results
        self._node_type_arr = np.array(self._node_type, dtype=np.int8)
        # Pre-compute value node indices for fast collection
        self._value_indices = np.flatnonzero(self._node_type_arr == _VALUE_TYPE)
        self._dirty = False

    def _reset_graph_state(self) -> None:
        """Reset graph state before a full rebuild."""
        self._node_to_idx = {}
        self._idx_to_node = []
        self._node_type = []
        self._node_label = []
        self._node_specificity = []
        self._edge_src = array.array('i')    # int32 COO row indices
        self._edge_dst = array.array('i')    # int32 COO col indices
        self._edge_weight = array.array('f') # float32 COO values
        self._entity_edge_positions = defaultdict(list)
        self._adj = None
        self._dirty = True
        self._is_bipartite = True
        self.entity_freq = defaultdict(int)
        self._entity_index = defaultdict(list)
        self._exact_entities = set()
        self._entity_terms = []
        self._substr_match_cache = {}
        self._token_index = defaultdict(list)

    def _index_entity_node(self, entity: str, node_id: str) -> None:
        """Index a new entity node for exact, partial, and substring seeding."""
        self._entity_index[entity].append(node_id)

        if entity not in self._exact_entities:
            self._exact_entities.add(entity)
            self._entity_terms.append(entity)
            self._substr_match_cache.clear()

            # Build token-level index for fast substring matching.
            # Tokens are alphanumeric runs from the entity label.
            for tok in _TOKEN_RE.findall(entity):
                tok_lower = tok.lower()
                if tok_lower != entity and len(tok_lower) >= 2:
                    self._token_index[tok_lower].append(entity)

        for word in entity.split():
            if word != entity and word not in _STOPWORDS:
                self._entity_index[word].append(node_id)

    def _refresh_specificity_for_entities(self, entities: Iterable[str]) -> None:
        """Recompute specificity and edge weights for entities whose freq changed."""
        touched = set(entities)
        if not touched:
            return

        for entity in touched:
            freq = self.entity_freq.get(entity, 0)
            if freq <= 0:
                continue
            specificity = 1.0 / (freq ** self.config.spec_power)
            weight = specificity * self.config.edge_weight

            for node_id in self._entity_index.get(entity, []):
                idx = self._node_to_idx.get(node_id)
                if idx is None or self._node_type[idx] != _ENTITY_TYPE:
                    continue
                self._node_specificity[idx] = specificity
                for edge_pos in self._entity_edge_positions.get(idx, []):
                    self._edge_weight[edge_pos] = weight

        # If we touched existing edges, the CSR matrix must be rebuilt.
        self._dirty = True

    def _ensure_mutable_edges(self) -> None:
        """Materialize COO edge lists from CSR if this graph was loaded from disk."""
        if self._edge_src:
            return
        if self._adj is None or self._adj.nnz == 0:
            return
        if self._dirty:
            return

        coo = self._adj.tocoo(copy=True)
        self._edge_src = array.array('i', coo.row.astype(np.int32))
        self._edge_dst = array.array('i', coo.col.astype(np.int32))
        self._edge_weight = array.array('f', coo.data.astype(np.float32))
        self._entity_edge_positions = defaultdict(list)

        for pos, src_idx in enumerate(self._edge_src):
            if self._node_type[src_idx] == _ENTITY_TYPE:
                self._entity_edge_positions[src_idx].append(pos)

    def _substring_seed_nodes(self, word: str) -> List[str]:
        """Return entity node IDs whose exact labels contain the query term.

        Uses token index for O(1) lookup of word-boundary matches.
        This covers the vast majority of useful substring hits (date
        components, name parts, compound terms). True arbitrary
        substring matches (e.g. "loss" in "dataloss") are not indexed
        but are rare in practice.
        """
        cached = self._substr_match_cache.get(word)
        if cached is not None:
            return cached

        matches = []
        seen = set()
        for entity in self._token_index.get(word, []):
            for node_id in self._entity_index.get(entity, []):
                if node_id not in seen:
                    seen.add(node_id)
                    matches.append(node_id)

        self._substr_match_cache[word] = matches
        return matches

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
        self._reset_graph_state()

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

        # Build graph with specificity weights (single pass per doc)
        spec_power = self.config.spec_power
        edge_weight = self.config.edge_weight
        entity_freq = self.entity_freq
        node_to_idx = self._node_to_idx

        for doc, entities in extracted:
            answer_node = f"v:{doc['id']}"
            ans_idx = self._get_or_add_node(
                answer_node, _VALUE_TYPE, doc['value'], 0.0
            )

            seen_in_doc = set()
            for entity, etype in entities:
                if len(entity) < 2:
                    continue
                node_id = f"e:{etype}:{entity}"
                if node_id in seen_in_doc:
                    continue
                seen_in_doc.add(node_id)

                specificity = 1.0 / (entity_freq[entity] ** spec_power)
                is_new = node_id not in node_to_idx
                ent_idx = self._get_or_add_node(
                    node_id, _ENTITY_TYPE, entity, specificity
                )

                if is_new:
                    self._index_entity_node(entity, node_id)

                self._add_edge(
                    ent_idx, ans_idx,
                    specificity * edge_weight
                )

        self._built = True
        return self

    def add_document(self, doc: Dict) -> None:
        """Incrementally add a single document to the graph."""
        self.add_documents([doc])

    def add_documents(self, documents: List[Dict]) -> None:
        """Incrementally add multiple documents to the graph.

        Uses a two-pass update to keep specificity weights stable:
        1) count all entity-frequency deltas in the incoming batch
        2) update existing entity weights once
        3) add batch edges with post-update specificity
        """
        if not documents:
            return

        self._ensure_mutable_edges()

        extracted = []
        batch_counts: Dict[str, int] = defaultdict(int)
        for doc in documents:
            text = doc['content']
            if 'query' in doc:
                text = f"{text} {doc['query']}"
            entities = self._extractor.extract(text)
            extracted.append((doc, entities))
            for entity, _ in entities:
                if len(entity) >= 2:
                    batch_counts[entity] += 1

        for entity, count in batch_counts.items():
            self.entity_freq[entity] += count

        # Keep old edges consistent with the updated frequencies.
        self._refresh_specificity_for_entities(batch_counts.keys())

        spec_power = self.config.spec_power
        edge_weight = self.config.edge_weight
        entity_freq = self.entity_freq
        node_to_idx = self._node_to_idx

        for doc, entities in extracted:
            answer_node = f"v:{doc['id']}"
            ans_idx = self._get_or_add_node(
                answer_node, _VALUE_TYPE, doc['value'], 0.0
            )

            seen_in_doc = set()
            for entity, etype in entities:
                if len(entity) < 2:
                    continue
                node_id = f"e:{etype}:{entity}"
                if node_id in seen_in_doc:
                    continue
                seen_in_doc.add(node_id)

                specificity = 1.0 / (entity_freq[entity] ** spec_power)
                is_new = node_id not in node_to_idx
                ent_idx = self._get_or_add_node(
                    node_id, _ENTITY_TYPE, entity, specificity
                )

                if is_new:
                    self._index_entity_node(entity, node_id)

                self._add_edge(
                    ent_idx, ans_idx,
                    specificity * edge_weight
                )

        self._built = True

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

        if workers <= 1 or len(documents) < 2_000:
            return self.build(documents)

        # Reset
        self._reset_graph_state()

        # Scale workers to doc count to avoid fork overhead dominating
        effective_workers = min(workers, max(2, len(documents) // 1000))
        chunk_size = max(500, len(documents) // effective_workers)
        chunks = [documents[i:i + chunk_size]
                  for i in range(0, len(documents), chunk_size)]

        # Use fork context to share parent's compiled regexes with workers
        ctx = mp.get_context('fork')
        with ctx.Pool(effective_workers, initializer=_init_worker,
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
                    self._index_entity_node(entity, node_id)

                self._add_edge(
                    ent_idx, ans_idx,
                    specificity * self.config.edge_weight
                )

        self._built = True
        return self

    def _seed_from_words(self, words: List[str]) -> Dict[int, float]:
        """Seed activation from a list of lowercase words."""
        activations: Dict[int, float] = {}
        do_substr = len(self._entity_terms) <= 10_000

        for word in dict.fromkeys(words):
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
                for node_id in self._substring_seed_nodes(word):
                    idx = self._node_to_idx.get(node_id)
                    if idx is None:
                        continue
                    if idx in activations:
                        continue
                    spec = self._node_specificity[idx]
                    activations[idx] = max(
                        activations.get(idx, 0),
                        self.config.substr_boost * spec
                    )
        return activations

    def _spread(self, activations: Dict[int, float],
                hops: Optional[int] = None) -> Dict[int, float]:
        """Run spreading activation for N hops.

        For bipartite graphs (entity->value only, which is the default
        build() structure), uses a vectorized single-hop path since
        hops 2+ just apply uniform decay without changing rankings.
        """
        self._compile()

        hops = hops if hops is not None else self.config.max_hops

        if self._is_bipartite and hops >= 1:
            return self._spread_bipartite(activations)

        return self._spread_general(activations, hops)

    def _spread_bipartite(self, activations: Dict[int, float]) -> Dict[int, float]:
        """Vectorized single-hop spread for bipartite entity->value graphs.

        Since value nodes have zero outgoing edges, only hop 1 produces
        new activations. Extra hops just multiply all scores by decay,
        preserving relative ordering.

        Uses numba JIT kernel when available (~38x faster than numpy),
        falls back to numpy scatter-max otherwise.
        """
        indptr = self._adj.indptr
        indices = self._adj.indices
        data = self._adj.data
        decay = self.config.decay
        threshold = self.config.threshold
        max_active = self.config.max_active

        n = len(self._idx_to_node)

        # Convert activations dict to arrays for the kernel
        act_indices = np.array(list(activations.keys()), dtype=np.int64)
        act_scores = np.array(list(activations.values()), dtype=np.float32)

        if _numba_spread_bipartite is not None:
            result = _numba_spread_bipartite(
                act_indices, act_scores, indptr, indices, data,
                np.float32(decay), np.float32(threshold), n,
            )
        else:
            result = np.zeros(n, dtype=np.float32)
            for idx, score in activations.items():
                decayed = score * decay
                if decayed > result[idx]:
                    result[idx] = decayed
                if score < threshold:
                    continue
                row_start = indptr[idx]
                row_end = indptr[idx + 1]
                if row_start == row_end:
                    continue
                nbrs = indices[row_start:row_end]
                weights = data[row_start:row_end]
                np.maximum.at(result, nbrs, score * decay * weights)

        # Threshold filter + top-k pruning
        active_mask = result >= threshold
        final_idx = np.flatnonzero(active_mask)

        if len(final_idx) == 0:
            return {}

        final_scores = result[final_idx]

        if len(final_idx) > max_active:
            top_k = np.argpartition(final_scores, -max_active)[-max_active:]
            final_idx = final_idx[top_k]
            final_scores = final_scores[top_k]

        return {int(i): float(s) for i, s in zip(final_idx, final_scores)}

    def _spread_general(self, activations: Dict[int, float],
                        hops: int) -> Dict[int, float]:
        """General multi-hop spreading for non-bipartite graphs."""
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
                    heapq.nlargest(self.config.max_active, new_act.items(), key=lambda x: x[1])
                )

            activations = {k: v for k, v in new_act.items()
                          if v >= self.config.threshold}

        return activations

    def _collect_results(self, activations: Dict[int, float],
                         top_k: int) -> List[Tuple[str, float]]:
        """Collect value nodes from activation map.

        Uses heapq.nlargest for O(n log k) instead of full sort O(n log n).
        """
        value_acts = [
            (score, idx) for idx, score in activations.items()
            if self._node_type[idx] == _VALUE_TYPE
        ]
        if not value_acts:
            return []
        top = heapq.nlargest(top_k, value_acts)
        return [(self._node_label[idx], score) for score, idx in top]

    def _collect_results_with_ids(self, activations: Dict[int, float],
                                   top_k: int) -> List[Tuple[str, str, float]]:
        """Collect value nodes with doc IDs from activation map.

        Uses heapq.nlargest for O(n log k) instead of full sort O(n log n).
        """
        value_acts = [
            (score, idx) for idx, score in activations.items()
            if self._node_type[idx] == _VALUE_TYPE
        ]
        if not value_acts:
            return []
        top = heapq.nlargest(top_k, value_acts)
        results = []
        for score, idx in top:
            node_id = self._idx_to_node[idx]
            doc_id = node_id[2:] if node_id.startswith('v:') else node_id
            results.append((doc_id, self._node_label[idx], score))
        return results

    def _spread_bipartite_raw(self, activations: Dict[int, float]) -> np.ndarray:
        """Like _spread_bipartite but returns raw numpy array instead of dict.

        Avoids the array->dict->array round-trip when collecting results.
        """
        indptr = self._adj.indptr
        indices = self._adj.indices
        data = self._adj.data
        decay = self.config.decay
        threshold = self.config.threshold
        n = len(self._idx_to_node)

        act_indices = np.array(list(activations.keys()), dtype=np.int64)
        act_scores = np.array(list(activations.values()), dtype=np.float32)

        if _numba_spread_bipartite is not None:
            return _numba_spread_bipartite(
                act_indices, act_scores, indptr, indices, data,
                np.float32(decay), np.float32(threshold), n,
            )

        result = np.zeros(n, dtype=np.float32)
        for idx, score in activations.items():
            decayed = score * decay
            if decayed > result[idx]:
                result[idx] = decayed
            if score < threshold:
                continue
            row_start = indptr[idx]
            row_end = indptr[idx + 1]
            if row_start == row_end:
                continue
            nbrs = indices[row_start:row_end]
            weights = data[row_start:row_end]
            np.maximum.at(result, nbrs, score * decay * weights)
        return result

    def _collect_from_array(self, result: np.ndarray, top_k: int
                            ) -> List[Tuple[str, float]]:
        """Collect top-k value nodes directly from a numpy score array.

        Uses pre-computed _value_indices to skip entity nodes without
        scanning the full array.
        """
        vi = self._value_indices
        scores = result[vi]

        # Threshold filter
        above = scores >= self.config.threshold
        if not above.any():
            return []

        nz_local = np.flatnonzero(above)
        nz_scores = scores[nz_local]
        nz_global = vi[nz_local]

        k = min(top_k, len(nz_local))
        if k >= len(nz_local):
            top_idx = np.argsort(nz_scores)[::-1]
        else:
            top_idx = np.argpartition(nz_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(nz_scores[top_idx])[::-1]]

        return [(self._node_label[nz_global[i]], float(nz_scores[i])) for i in top_idx]

    def _collect_from_array_with_ids(self, result: np.ndarray, top_k: int
                                      ) -> List[Tuple[str, str, float]]:
        """Like _collect_from_array but returns (doc_id, value, score)."""
        vi = self._value_indices
        scores = result[vi]

        above = scores >= self.config.threshold
        if not above.any():
            return []

        nz_local = np.flatnonzero(above)
        nz_scores = scores[nz_local]
        nz_global = vi[nz_local]

        k = min(top_k, len(nz_local))
        if k >= len(nz_local):
            top_idx = np.argsort(nz_scores)[::-1]
        else:
            top_idx = np.argpartition(nz_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(nz_scores[top_idx])[::-1]]

        results = []
        for i in top_idx:
            idx = nz_global[i]
            node_id = self._idx_to_node[idx]
            doc_id = node_id[2:] if node_id.startswith('v:') else node_id
            results.append((doc_id, self._node_label[idx], float(nz_scores[i])))
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

        # Fast path: bipartite without context skips dict round-trip
        if self._is_bipartite and not context and self.config.max_hops >= 1:
            self._compile()
            result_arr = self._spread_bipartite_raw(query_act)
            return self._collect_from_array(result_arr, top_k)

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

        # Fast path: bipartite without context skips dict round-trip
        if self._is_bipartite and not context and self.config.max_hops >= 1:
            self._compile()
            result_arr = self._spread_bipartite_raw(query_act)
            return self._collect_from_array_with_ids(result_arr, top_k)

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
            'is_bipartite': self._is_bipartite,
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
        sa._edge_src = array.array('i')
        sa._edge_dst = array.array('i')
        sa._edge_weight = array.array('f')
        sa._entity_edge_positions = defaultdict(list)
        sa._dirty = False
        sa._node_type_arr = np.array(sa._node_type, dtype=np.int8)
        sa._value_indices = np.flatnonzero(sa._node_type_arr == _VALUE_TYPE)

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._exact_entities = set(sa.entity_freq.keys())
        sa._entity_terms = list(sa._exact_entities)
        sa._substr_match_cache = {}
        sa._token_index = defaultdict(list)
        for entity in sa._entity_terms:
            for tok in _TOKEN_RE.findall(entity):
                tok_lower = tok.lower()
                if tok_lower != entity and len(tok_lower) >= 2:
                    sa._token_index[tok_lower].append(entity)
        sa._is_bipartite = data.get('is_bipartite', True)
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
            sa._add_edge(src_idx, dst_idx, weight)

        sa._compile()

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._exact_entities = set(sa.entity_freq.keys())
        sa._entity_terms = list(sa._exact_entities)
        sa._substr_match_cache = {}
        sa._token_index = defaultdict(list)
        for entity in sa._entity_terms:
            for tok in _TOKEN_RE.findall(entity):
                tok_lower = tok.lower()
                if tok_lower != entity and len(tok_lower) >= 2:
                    sa._token_index[tok_lower].append(entity)
        # Legacy networkx graphs may not be bipartite -- check
        sa._is_bipartite = all(
            sa._adj.indptr[idx] == sa._adj.indptr[idx + 1]
            for idx in range(len(sa._idx_to_node))
            if sa._node_type[idx] == _VALUE_TYPE
        ) if sa._adj is not None and sa._adj.nnz > 0 else True
        sa._built = True
        return sa
