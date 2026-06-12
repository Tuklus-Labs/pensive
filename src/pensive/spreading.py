"""
Spreading activation retrieval with specificity weighting.

Uses scipy.sparse CSR matrices instead of networkx for ~90% memory
reduction at scale (23M edges: 19.5GB -> ~2GB).

Usage:
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build(documents)  # List of dicts with 'content', 'id', 'value' keys

    # Entity-exact: pass the entity surface form (a date, metric, ID,
    # name, ...), NOT a natural-language question. See README
    # "Quickstart" + the "Natural-language queries" section for the
    # extractor-then-query pattern.
    results = sa.query("2025-10-08")
"""
import array
import heapq
import re
import threading
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

# Shared empty results for the collect/top-k helpers (never mutated)
_EMPTY_IDX = np.empty(0, dtype=np.int64)
_EMPTY_SCORES = np.empty(0, dtype=np.float32)

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

    @numba.njit(cache=True)
    def _numba_spread_and_collect(act_indices, act_scores, indptr, indices,
                                   data, decay, threshold, n, node_type_arr,
                                   value_type):
        """Spread + collect value nodes in one kernel (avoids numpy round-trip)."""
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
        # Single-pass collect: pre-allocate max-size buffer to avoid
        # a counting pass over all n nodes (eliminates second O(n) scan).
        nz_idx = np.empty(n, dtype=np.int64)
        nz_scores = np.empty(n, dtype=np.float32)
        pos = 0
        for i in range(n):
            if result[i] >= threshold and node_type_arr[i] == value_type:
                nz_idx[pos] = i
                nz_scores[pos] = result[i]
                pos += 1
        return nz_idx[:pos], nz_scores[:pos]
else:
    _numba_spread_bipartite = None
    _numba_spread_and_collect = None


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

    def __post_init__(self):
        """Reject configurations that would produce nonsense activations.

        Each rejected condition has a specific failure mode:

        * ``decay >= 1.0`` -- multi-hop ``_spread_general`` multiplies
          activation by ``decay`` each hop. With decay >= 1 activation
          grows without bound; the threshold filter never fires and
          ranking collapses.
        * ``decay < 0`` -- flips the sign of every propagated score on
          alternating hops; top-k loses meaning.
        * ``spec_power < 0`` -- inverts the inverse-frequency weighting,
          rewarding common entities and burying rare/discriminative ones.
        * ``threshold < 0`` -- the activation filter ``score >= threshold``
          stops filtering anything; ``max_active`` becomes the only cap
          and queries explode in cost.
        * ``max_hops < 0`` -- ``range(max_hops)`` skips the spread loop
          entirely and the query returns only the seed activations.
        """
        if self.decay >= 1.0:
            raise ValueError(
                f"SpreadingConfig.decay must be < 1.0 (got {self.decay}); "
                "values >= 1 cause unbounded amplification across hops."
            )
        if self.decay < 0:
            raise ValueError(
                f"SpreadingConfig.decay must be >= 0 (got {self.decay}); "
                "negative decay flips activation sign per hop."
            )
        if self.spec_power < 0:
            raise ValueError(
                f"SpreadingConfig.spec_power must be >= 0 (got "
                f"{self.spec_power}); negative power inverts the "
                "inverse-frequency weighting."
            )
        if self.threshold < 0:
            raise ValueError(
                f"SpreadingConfig.threshold must be >= 0 (got "
                f"{self.threshold}); negative threshold disables filtering."
            )
        if self.max_hops < 0:
            raise ValueError(
                f"SpreadingConfig.max_hops must be >= 0 (got {self.max_hops})."
            )


# Default patterns for real conversational data
PATTERNS = REAL_DATA_PATTERNS


_worker_extractor = None


def _init_worker(extractor):
    """Set the shared extractor in each forked worker process."""
    global _worker_extractor
    _worker_extractor = extractor


def _extract_chunk(chunk):
    """Worker function for multiprocessing entity extraction."""
    extract = _worker_extractor.extract
    results = []
    for doc in chunk:
        text = doc['content']
        query = doc.get('query')
        if query:
            text = text + ' ' + query
        results.append((doc, extract(text)))
    return results


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
        self._node_type = array.array('b')  # int8, zero-copy to numpy in _compile
        self._node_label: List[str] = []
        self._node_specificity: List[float] = []

        # Edge storage (COO during build, CSR for queries).
        # array.array gives compact typed storage (8 bytes per int edge,
        # 4 bytes per float weight) and amortized-O(1) append. _compile()
        # copies these into fresh numpy arrays -- see PENPY-P5-CRIT-1 in
        # _compile() for why we do not use np.frombuffer here.
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
        self._do_substr = True  # cached at compile time
        self._built = False
        self._is_bipartite = True  # True until proven otherwise

        # Monotonic generation counter. Incremented on every graph mutation
        # boundary (reset, build, add_documents) so downstream caches
        # (e.g. boundary._get_value_label_index) can invalidate on rebuild
        # even when the new graph happens to have the same node count.
        # See PENPY-IMP-5: keying caches on len(_idx_to_node) alone was
        # unsafe for build()-then-rebuild with size-stable schemas.
        self._graph_generation: int = 0

        # Guards build_parallel's worker-thread fanout, where the workers
        # concurrently mutate entity_freq which is also read by
        # _get_specificity during query. Callers that mix parallel build
        # with live queries should hold this lock.
        #
        # Note (PENPY-P5-CRIT-1): the BufferError race between
        # add_documents() and concurrent query() is NOT covered by this
        # lock. That race was solved structurally in _compile() by
        # copying array.array buffers into fresh numpy arrays instead of
        # exporting views via np.frombuffer(). add_documents + query is
        # therefore safe to call concurrently without holding this lock,
        # subject to the usual rule that mid-update reads can see a
        # consistent-but-stale graph.
        self._build_lock = threading.RLock()
        self._context_provider = None
        self._extractor = MegaExtractor(self.patterns)

    def _get_or_add_node(self, node_id: str, node_type: int,
                         label: str, specificity: float) -> int:
        """Get existing node index or add a new node. Returns the index.

        PENPY-P7-NEW-5: marks the graph dirty on new-node creation. The
        edge-add path also sets _dirty=True, but add_documents() can grow
        _idx_to_node without ever calling _add_edge() if a document
        contains no entities matched by the patterns (the v: answer node
        is still appended). Without this flag, _adj.shape[0] stays at the
        old node count while _idx_to_node lengthens, so queries operate
        on a stale CSR and save/load round-trips fail the n_nodes
        validation. Setting _dirty here is canonical: anything that
        appends to _idx_to_node must trigger a recompile.
        """
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
        self._dirty = True
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

    def _add_edge_fast(self, src_idx: int, dst_idx: int, weight: float) -> None:
        """Add an edge during initial build (skips position tracking)."""
        self._edge_src.append(src_idx)
        self._edge_dst.append(dst_idx)
        self._edge_weight.append(weight)

    def _compile(self) -> None:
        """Convert COO edge lists to CSR matrix for fast neighbor iteration.

        Concurrency model (PENPY-P5-CRIT-1, PENPY-P8-CRIT-1):

        Readers (query()) and writers (add_documents()) both eventually
        call _compile(). Four hazards we have to defend against:

          1. Buffer-export race -- an earlier version used np.frombuffer()
             over the array.array storage, which kept a live buffer view
             while the CSR was being built. A writer's _add_edge.append()
             during that window raised BufferError. Fix: copy into fresh
             numpy arrays (np.array(arr)) so no buffer view escapes.

          2. Torn-read race -- copying _edge_src, _edge_dst, _edge_weight
             one at a time is NOT atomic. A writer that appends between
             the rows-copy and the cols-copy yields three arrays with
             mismatched lengths, which scipy rejects with
             "all index and data arrays must have the same length".
             Fix: snapshot len() once under the build lock, then copy
             exactly that many elements from each array.

          3. CSR-publish race -- a reader mid-_compile sees a partially
             constructed self._adj. Fix: build the new csr_matrix into a
             local, then assign self._adj in one statement at the end.
             CPython attribute writes are atomic under the GIL, so other
             readers either see the old _adj or the new one, never an
             intermediate.

          4. Publish-outside-lock race (PENPY-P8-CRIT-1) -- thread A
             snapshots state under the lock, releases, builds CSR
             slowly. Thread B interleaves: acquires lock, mutates the
             graph (add_documents bumps _graph_generation), compiles,
             publishes. Then A publishes its STALE CSR over B's fresher
             one and unconditionally sets _dirty=False -- so the next
             _compile() short-circuits at the top, leaving the graph
             permanently stuck with _adj.shape[0] < len(_idx_to_node).
             Symptoms: silent wrong query results (entities present but
             unreachable through CSR), save/load ValueError on the
             n_nodes invariant, and persistent stuck state.

             Fix: snapshot _graph_generation under the lock alongside
             the COO copies. After building CSR outside the lock,
             re-acquire the lock and check the generation hasn't moved.
             If it has, discard our work and retry with the newer
             snapshot. The publish itself happens INSIDE the lock so a
             concurrent writer cannot interleave between our generation
             check and the _dirty=False assignment.

        We hold self._build_lock for the length snapshot + array copies
        and for the generation-checked publish. The actual CSR
        construction happens outside the lock so concurrent queries
        don't serialize behind a single _compile().

        Also caches numpy arrays for node_type and node_specificity.
        """
        if not self._dirty:
            return

        # Bounded optimistic retry: if a concurrent writer beats us to
        # publish, we rebuild against the newer snapshot. Each iteration
        # builds the CSR outside the lock to preserve the read
        # parallelism that was the original point of releasing the lock
        # in pass-5 fix-3.
        #
        # If we exhaust the retry budget under pathological writer
        # contention (writers bumping _graph_generation faster than we
        # can build a CSR), we fall through to a synchronous publish:
        # snapshot + build + publish all under the lock. That guarantees
        # forward progress and a non-None _adj even when readers cannot
        # find an optimistic gap. The synchronous path serializes
        # writers behind us for the CSR-build duration, which is the
        # legitimate cost of guaranteeing the reader sees a current
        # graph. Without this fallback, readers can be perpetually
        # starved -- _adj stays None and queries crash on .indptr.
        for _attempt in range(8):
            # Snapshot the COO buffers and the graph generation under
            # the lock so writers can't tear the read between len() and
            # the copy. Fast pure-Python critical section.
            with self._build_lock:
                if not self._dirty:
                    # Another thread already compiled while we waited.
                    return
                snap_generation = self._graph_generation
                edge_count = len(self._edge_src)
                n = len(self._idx_to_node)
                # Snapshot node_type + n_terms under the lock so the
                # publish step doesn't re-read live (and possibly newer)
                # state when the generation check passes. This also
                # fixes a latent inconsistency where the edge_count == 0
                # path was using bytes(self._node_type) at publish time,
                # observing a different node_type than the one this
                # _compile() was supposed to publish.
                node_type_snap = bytes(self._node_type)
                n_terms = len(self._entity_terms)
                if edge_count == 0:
                    src_snap = dst_snap = w_snap = None
                else:
                    # np.array(arr.array) does a clean typed copy with
                    # no buffer-export refcount on the source.
                    # array.array slicing copies, not views.
                    src_snap = self._edge_src[:edge_count]
                    dst_snap = self._edge_dst[:edge_count]
                    w_snap = self._edge_weight[:edge_count]
                # End of snapshot critical section -- the heavy CSR
                # construction happens outside the lock.

            new_adj = self._build_csr(
                edge_count, n, src_snap, dst_snap, w_snap,
            )
            new_node_type_arr = np.frombuffer(
                node_type_snap, dtype=np.int8,
            ).copy()

            # Publish-with-generation-check (PENPY-P8-CRIT-1). Re-acquire
            # the lock and verify no writer has touched the graph since
            # our snapshot. If the generation moved, our CSR is stale --
            # discard it and retry. If the generation matches, the
            # publish is atomic with respect to all writers (they all
            # mutate _graph_generation under the same _build_lock).
            with self._build_lock:
                if not self._dirty:
                    # Another thread already published while we were
                    # building. Their publish is current; ours is stale.
                    return
                if self._graph_generation != snap_generation:
                    # A concurrent writer raced ahead during our CSR
                    # build. Retry with the newer snapshot. We do NOT
                    # touch _adj or _dirty here, so the writer's
                    # subsequent publish (or our next attempt) wins.
                    continue
                # Generation matches: our snapshot is still current.
                self._adj = new_adj
                self._node_type_arr = new_node_type_arr
                self._value_indices = np.flatnonzero(
                    new_node_type_arr == _VALUE_TYPE
                )
                self._do_substr = n_terms <= 10_000
                self._dirty = False
                return

        # Optimistic retry exhausted under contention. Fall back to a
        # synchronous build+publish that holds the lock across the CSR
        # construction. This guarantees forward progress: writers serialize
        # behind us for the build duration but the reader is guaranteed
        # to see a non-stale, non-None _adj on return. Pathological
        # writer storms are the only path here in practice.
        with self._build_lock:
            if not self._dirty:
                return
            edge_count = len(self._edge_src)
            n = len(self._idx_to_node)
            node_type_snap = bytes(self._node_type)
            n_terms = len(self._entity_terms)
            if edge_count == 0:
                src_snap = dst_snap = w_snap = None
            else:
                src_snap = self._edge_src[:edge_count]
                dst_snap = self._edge_dst[:edge_count]
                w_snap = self._edge_weight[:edge_count]
            new_adj = self._build_csr(
                edge_count, n, src_snap, dst_snap, w_snap,
            )
            new_node_type_arr = np.frombuffer(
                node_type_snap, dtype=np.int8,
            ).copy()
            self._adj = new_adj
            self._node_type_arr = new_node_type_arr
            self._value_indices = np.flatnonzero(
                new_node_type_arr == _VALUE_TYPE
            )
            self._do_substr = n_terms <= 10_000
            self._dirty = False

    @staticmethod
    def _build_csr(edge_count, n, src_snap, dst_snap, w_snap):
        """Build the CSR adjacency matrix from a snapshot of COO buffers.

        Pure function over the snapshots -- safe to call outside any
        lock. The buffers must already be disconnected from the live
        array.array storage (slicing array.array yields a copy).
        """
        if edge_count == 0:
            return scipy.sparse.csr_matrix((n, n), dtype=np.float32)
        weights = np.array(w_snap, dtype=np.float32)
        rows = np.array(src_snap, dtype=np.int32)
        cols = np.array(dst_snap, dtype=np.int32)
        return scipy.sparse.csr_matrix(
            (weights, (rows, cols)), shape=(n, n),
        )

    def _reset_graph_state(self) -> None:
        """Reset graph state before a full rebuild."""
        self._node_to_idx = {}
        self._idx_to_node = []
        self._node_type = array.array('b')
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

        # PENPY-IMP-5: bump generation so downstream caches keyed on
        # (n_nodes, generation) invalidate on rebuild even when the new
        # graph has the same node count. Belt-and-suspenders: also drop
        # the boundary value-label cache directly, in case any consumer
        # didn't migrate to the generation-aware key.
        self._graph_generation += 1
        if hasattr(self, '_boundary_value_label_index'):
            try:
                delattr(self, '_boundary_value_label_index')
            except AttributeError:
                pass

    def _index_entity_node(self, entity: str, node_id: str) -> None:
        """Index a new entity node for exact, partial, and substring seeding."""
        self._entity_index[entity].append(node_id)

        if entity not in self._exact_entities:
            self._exact_entities.add(entity)
            self._entity_terms.append(entity)
            if self._substr_match_cache:
                self._substr_match_cache.clear()

            # Build token-level index for fast substring matching.
            # Tokens are alphanumeric runs from the entity label.
            for tok in _TOKEN_RE.findall(entity):
                tok_lower = tok.lower()
                if tok_lower != entity and len(tok_lower) >= 2:
                    self._token_index[tok_lower].append(entity)

        # Multi-word entities get word-level indexing for partial matching.
        # Single-word entities skip: split returns [entity] and word==entity filters it.
        if ' ' in entity:
            for word in entity.split():
                if word not in _STOPWORDS:
                    self._entity_index[word].append(node_id)

    def _rebuild_token_index(self) -> None:
        """Rebuild the token-level inverted index from _entity_terms."""
        idx = defaultdict(list)
        for entity in self._entity_terms:
            for tok in _TOKEN_RE.findall(entity):
                tok_lower = tok.lower()
                if tok_lower != entity and len(tok_lower) >= 2:
                    idx[tok_lower].append(entity)
        self._token_index = idx

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

    _SUBSTR_CACHE_MAX = 10_000

    def _substring_seed_nodes(self, word: str) -> List[str]:
        """Return entity node IDs whose exact labels contain the query term.

        Uses token index for O(1) lookup of word-boundary matches.
        This covers the vast majority of useful substring hits (date
        components, name parts, compound terms). True arbitrary
        substring matches (e.g. "loss" in "dataloss") are not indexed
        but are rare in practice.

        Cache is evicted when it exceeds _SUBSTR_CACHE_MAX entries to
        prevent unbounded memory growth in long-running sessions.
        """
        cached = self._substr_match_cache.get(word)
        if cached is not None:
            return cached

        node_ids = set()
        for entity in self._token_index.get(word, []):
            node_ids.update(self._entity_index.get(entity, ()))

        matches = list(node_ids)
        if len(self._substr_match_cache) >= self._SUBSTR_CACHE_MAX:
            self._substr_match_cache.clear()
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

    def _ingest_extracted(
        self,
        extracted: List[Tuple[Dict, List[Tuple[str, str]]]],
        *,
        fast_edges: bool,
    ) -> None:
        """Shared per-document graph-construction body for the three
        ingest paths (_build_locked, add_documents,
        _build_parallel_locked).

        Caller holds self._build_lock and has already folded the batch's
        entity counts into self.entity_freq, so specificity computed
        here reflects post-update frequencies. For each (doc, entities)
        pair: create the v:<id> value node, dedup entity nodes within
        the doc, skip entities shorter than 2 chars, index newly created
        entity nodes, and add the entity->value edge.

        fast_edges=True uses _add_edge_fast (initial build into fresh
        buffers; per-entity edge positions are backfilled later by
        _ensure_mutable_edges). fast_edges=False uses _add_edge, which
        maintains _entity_edge_positions so a later
        _refresh_specificity_for_entities can rewrite old edge weights.
        """
        spec_power = self.config.spec_power
        edge_weight = self.config.edge_weight
        entity_freq = self.entity_freq
        node_to_idx = self._node_to_idx
        add_edge = self._add_edge_fast if fast_edges else self._add_edge

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

                # Defense-in-depth: clamp freq to >= 1. The vanilla
                # ingest paths fold batch counts into entity_freq
                # before this runs, so freq should always be >= 1
                # here, but a subclass or future caller could pass
                # `extracted` without updating `entity_freq` and a
                # freq of 0 would blow up with ZeroDivisionError
                # under spec_power > 0.
                freq = max(entity_freq[entity], 1)
                specificity = 1.0 / (freq ** spec_power)
                is_new = node_id not in node_to_idx
                ent_idx = self._get_or_add_node(
                    node_id, _ENTITY_TYPE, entity, specificity
                )

                if is_new:
                    self._index_entity_node(entity, node_id)

                add_edge(
                    ent_idx, ans_idx,
                    specificity * edge_weight
                )

    def build(self, documents: List[Dict]) -> 'SpreadingActivation':
        """
        Build the graph from documents.

        Each document should have:
        - 'content': str - The text content
        - 'id': str - Unique identifier
        - 'value': str - The answer/value to retrieve
        - 'query': str (optional) - Query text to include in entity extraction

        Holds self._build_lock across the mutation phase so concurrent
        query threads see a consistent graph. If build raises mid-way, the
        graph is reset and _built stays False so subsequent queries fail
        fast rather than returning stale results.
        """
        with self._build_lock:
            try:
                return self._build_locked(documents)
            except BaseException:
                self._reset_graph_state()
                self._built = False
                raise

    def _build_locked(self, documents: List[Dict]) -> 'SpreadingActivation':
        """Caller holds self._build_lock."""
        self._reset_graph_state()

        # Extraction + frequency counting in one pass (avoids second iteration)
        extracted = []
        entity_freq = self.entity_freq
        extractor_extract = self._extractor.extract
        for doc in documents:
            text = doc['content']
            query = doc.get('query')
            if query:
                text = text + ' ' + query
            entities = extractor_extract(text)
            extracted.append((doc, entities))
            for entity, _ in entities:
                entity_freq[entity] += 1

        # Build graph with specificity weights
        self._ingest_extracted(extracted, fast_edges=True)

        self._dirty = True
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

        PENPY-P5-CRIT-1: the mutation phase (everything that touches
        _edge_* / _node_* / entity_freq) runs under _build_lock so a
        concurrent reader's _compile() cannot observe a torn snapshot
        where, e.g., _edge_src has 1000 entries but _edge_dst has 1001.
        The regex extraction phase above does not touch shared state, so
        it stays outside the lock.
        """
        if not documents:
            return

        # Phase 1: pure extraction, no shared-state mutation. Safe
        # outside the lock so multiple writers can extract in parallel.
        extractor_extract = self._extractor.extract
        extracted = []
        batch_counts: Dict[str, int] = defaultdict(int)
        for doc in documents:
            text = doc['content']
            query = doc.get('query')
            if query:
                text = text + ' ' + query
            entities = extractor_extract(text)
            extracted.append((doc, entities))
            for entity, _ in entities:
                if len(entity) >= 2:
                    batch_counts[entity] += 1

        # Phase 2: graph mutation. Single-writer guard so readers in
        # _compile() see either the pre-batch or post-batch graph but
        # not an intermediate where the COO arrays have mismatched
        # lengths.
        with self._build_lock:
            self._ensure_mutable_edges()

            # PENPY-IMP-5: bump generation so caches keyed on
            # (n_nodes, generation) invalidate. add_documents() typically
            # increases n_nodes, but a batch of all-duplicate documents
            # could leave node count unchanged while still mutating edges
            # and frequencies.
            self._graph_generation += 1
            if hasattr(self, '_boundary_value_label_index'):
                try:
                    delattr(self, '_boundary_value_label_index')
                except AttributeError:
                    pass

            for entity, count in batch_counts.items():
                self.entity_freq[entity] += count

            # Keep old edges consistent with the updated frequencies.
            self._refresh_specificity_for_entities(batch_counts.keys())

            self._ingest_extracted(extracted, fast_edges=False)

            self._built = True

    def build_parallel(self, documents: List[Dict],
                       workers: Optional[int] = None) -> 'SpreadingActivation':
        """Build graph using multiprocessing for entity extraction.

        Entity extraction (regex) is CPU-bound and embarrassingly parallel.
        Graph construction (node/edge creation) remains sequential since it
        mutates shared state, but is fast compared to extraction.

        Holds ``self._build_lock`` across the whole mutation phase so
        concurrent query threads (which acquire the same lock when reading
        ``entity_freq`` / the CSR) see a consistent graph.

        Args:
            documents: Same format as build()
            workers: Number of worker processes. Defaults to min(cpu_count, 16).
        """
        import multiprocessing as mp

        if workers is None:
            workers = min(mp.cpu_count(), 16)

        if workers <= 1 or len(documents) < 2_000:
            return self.build(documents)

        with self._build_lock:
            try:
                return self._build_parallel_locked(documents, workers, mp)
            except BaseException:
                # Half-built graph is worse than no graph. Reset state and
                # mark unbuilt so queries fail fast instead of returning
                # garbage from a partially-populated adjacency matrix.
                self._reset_graph_state()
                self._built = False
                raise

    def _build_parallel_locked(self, documents, workers, mp):
        """Internal: caller holds self._build_lock."""
        # Reset
        self._reset_graph_state()

        # Scale workers to doc count to avoid fork overhead dominating
        effective_workers = min(workers, max(2, len(documents) // 1000))
        chunk_size = max(500, len(documents) // effective_workers)
        chunks = [documents[i:i + chunk_size]
                  for i in range(0, len(documents), chunk_size)]

        # Use fork context to share parent's compiled regexes with workers
        # (fork is faster than spawn on Linux/macOS; falls back to spawn on
        # Windows where fork is not available).
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context('spawn')
        with ctx.Pool(effective_workers, initializer=_init_worker,
                      initargs=(self._extractor,)) as pool:
            chunk_results = pool.map(_extract_chunk, chunks)

        extracted = [item for chunk in chunk_results for item in chunk]

        entity_freq = self.entity_freq
        for _, entities in extracted:
            for entity, _ in entities:
                entity_freq[entity] += 1

        self._ingest_extracted(extracted, fast_edges=True)

        self._dirty = True
        self._built = True
        return self

    def _seed_from_words(self, words: List[str]) -> Dict[int, float]:
        """Seed activation from a list of lowercase words."""
        activations: Dict[int, float] = {}
        do_substr = self._do_substr
        entity_index = self._entity_index
        node_to_idx = self._node_to_idx
        node_specificity = self._node_specificity
        node_label = self._node_label
        exact_boost = self.config.exact_boost
        partial_boost = self.config.partial_boost
        substr_boost = self.config.substr_boost

        for word in dict.fromkeys(words):
            if word in _STOPWORDS:
                continue
            node_ids = entity_index.get(word)
            if node_ids:
                for node_id in node_ids:
                    idx = node_to_idx.get(node_id)
                    if idx is None:
                        continue
                    spec = node_specificity[idx]
                    if word == node_label[idx]:
                        activations[idx] = max(
                            activations.get(idx, 0),
                            exact_boost * spec
                        )
                    else:
                        activations[idx] = max(
                            activations.get(idx, 0),
                            partial_boost * spec
                        )

            if do_substr and len(word) >= 4:
                for node_id in self._substring_seed_nodes(word):
                    idx = node_to_idx.get(node_id)
                    if idx is None:
                        continue
                    if idx in activations:
                        continue
                    spec = node_specificity[idx]
                    activations[idx] = max(
                        activations.get(idx, 0),
                        substr_boost * spec
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

        if _numba_spread_bipartite is not None:
            # Convert activations dict to arrays for the JIT kernel.
            # np.fromiter with count avoids intermediate Python list objects.
            n_act = len(activations)
            act_indices = np.fromiter(activations.keys(), dtype=np.int64, count=n_act)
            act_scores = np.fromiter(activations.values(), dtype=np.float32, count=n_act)
            result = _numba_spread_bipartite(
                act_indices, act_scores, indptr, indices, data,
                np.float32(decay), np.float32(threshold), n,
            )
        else:
            # Fallback: iterate dict directly, no array overhead
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

        # Threshold filter + top-k pruning (skip intermediate bool mask)
        final_idx = np.flatnonzero(result >= threshold)

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
        decay = self.config.decay
        threshold = self.config.threshold
        max_active = self.config.max_active

        for _ in range(hops):
            new_act: Dict[int, float] = defaultdict(float)

            for node_idx, act in activations.items():
                new_act[node_idx] = max(new_act[node_idx], act * decay)

                if act < threshold:
                    continue

                row_start = indptr[node_idx]
                row_end = indptr[node_idx + 1]
                for j in range(row_start, row_end):
                    neighbor = indices[j]
                    weight = data[j]
                    new_act[neighbor] = max(
                        new_act[neighbor],
                        act * decay * weight
                    )

            if len(new_act) > max_active:
                new_act = dict(
                    heapq.nlargest(max_active, new_act.items(), key=lambda x: x[1])
                )

            activations = {k: v for k, v in new_act.items()
                          if v >= threshold}

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

        if _numba_spread_bipartite is not None:
            n_act = len(activations)
            act_indices = np.fromiter(activations.keys(), dtype=np.int64, count=n_act)
            act_scores = np.fromiter(activations.values(), dtype=np.float32, count=n_act)
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

    def _collect_value_candidates(
        self, result: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Threshold + value-node filter over a dense activation array.

        Operates on the sparse non-zero entries of the result array
        rather than indexing all value nodes. Since spreading produces
        sparse results (~0.5-1% density), this avoids O(n) fancy
        indexing on the full value node array.

        Returns (global_indices, scores), unordered; both empty when
        nothing clears the threshold.
        """
        nz_all = np.flatnonzero(result >= self.config.threshold)
        if len(nz_all) == 0:
            return _EMPTY_IDX, _EMPTY_SCORES
        is_value = self._node_type_arr[nz_all] == _VALUE_TYPE
        nz_global = nz_all[is_value]
        if len(nz_global) == 0:
            return _EMPTY_IDX, _EMPTY_SCORES
        return nz_global, result[nz_global]

    def _order_topk(
        self, nz_global: np.ndarray, nz_scores: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Descending top-k selection over candidate (indices, scores).

        PENPY-IMP-6: top_k <= 0 must return empty here. The numpy
        fast-path's np.argpartition(scores, -0)[-0:] degenerates to
        argpartition(scores, 0)[0:] = the full array, leaking ALL value
        nodes back to the caller (the general non-numpy path handles 0
        correctly via heapq.nlargest(0, ...) = []). Negatives are
        nonsense input but the guard covers them too. Callers also guard
        early to skip the spread work; this is the backstop so no future
        caller can reach argpartition with k=0.
        """
        if top_k <= 0 or len(nz_global) == 0:
            return _EMPTY_IDX, _EMPTY_SCORES

        k = min(top_k, len(nz_global))
        if k >= len(nz_global):
            top_idx = np.argsort(nz_scores)[::-1]
        else:
            top_idx = np.argpartition(nz_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(nz_scores[top_idx])[::-1]]

        return nz_global[top_idx], nz_scores[top_idx]

    def _spread_candidates(
        self, activations: Dict[int, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bipartite spread returning thresholded value-node candidates.

        When numba is available, runs the spread and value-node
        collection in a single JIT kernel, avoiding the numpy round-trip
        (~20% faster). Otherwise falls back to the raw numpy spread plus
        _collect_value_candidates. Returns (global_indices, scores),
        unordered.
        """
        if _numba_spread_and_collect is not None:
            n_act = len(activations)
            act_indices = np.fromiter(activations.keys(), dtype=np.int64, count=n_act)
            act_scores = np.fromiter(activations.values(), dtype=np.float32, count=n_act)
            return _numba_spread_and_collect(
                act_indices, act_scores,
                self._adj.indptr, self._adj.indices, self._adj.data,
                np.float32(self.config.decay), np.float32(self.config.threshold),
                len(self._idx_to_node),
                self._node_type_arr, np.int8(_VALUE_TYPE),
            )
        return self._collect_value_candidates(self._spread_bipartite_raw(activations))

    def _format_value_rows(
        self, ordered_idx: np.ndarray, ordered_scores: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Format ordered candidates as (value_label, score) rows."""
        node_label = self._node_label
        return [(node_label[g], float(s))
                for g, s in zip(ordered_idx, ordered_scores)]

    def _format_value_rows_with_ids(
        self, ordered_idx: np.ndarray, ordered_scores: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        """Format ordered candidates as (doc_id, value_label, score) rows."""
        results = []
        for g, s in zip(ordered_idx, ordered_scores):
            node_id = self._idx_to_node[g]
            doc_id = node_id[2:] if node_id.startswith('v:') else node_id
            results.append((doc_id, self._node_label[g], float(s)))
        return results

    def _spread_and_collect_bipartite(self, activations: Dict[int, float],
                                       top_k: int) -> List[Tuple[str, float]]:
        """Combined spread + collect for bipartite graphs."""
        # PENPY-IMP-6: early-out before spending the spread (see
        # _order_topk for the argpartition degeneracy this defends).
        if top_k <= 0:
            return []
        nz_global, nz_scores = self._spread_candidates(activations)
        return self._format_value_rows(
            *self._order_topk(nz_global, nz_scores, top_k))

    def _spread_and_collect_bipartite_with_ids(
        self, activations: Dict[int, float], top_k: int
    ) -> List[Tuple[str, str, float]]:
        """Like _spread_and_collect_bipartite but returns (doc_id, value, score)."""
        # PENPY-IMP-6: see _order_topk.
        if top_k <= 0:
            return []
        nz_global, nz_scores = self._spread_candidates(activations)
        return self._format_value_rows_with_ids(
            *self._order_topk(nz_global, nz_scores, top_k))

    def _collect_from_array(self, result: np.ndarray, top_k: int
                            ) -> List[Tuple[str, float]]:
        """Collect top-k value nodes directly from a numpy score array.

        Reached via the context+bipartite branch in query() and
        query_with_doc_ids().
        """
        # PENPY-IMP-6: see _order_topk.
        if top_k <= 0:
            return []
        nz_global, nz_scores = self._collect_value_candidates(result)
        return self._format_value_rows(
            *self._order_topk(nz_global, nz_scores, top_k))

    def _collect_from_array_with_ids(self, result: np.ndarray, top_k: int
                                      ) -> List[Tuple[str, str, float]]:
        """Like _collect_from_array but returns (doc_id, value, score)."""
        # PENPY-IMP-6: see _order_topk.
        if top_k <= 0:
            return []
        nz_global, nz_scores = self._collect_value_candidates(result)
        return self._format_value_rows_with_ids(
            *self._order_topk(nz_global, nz_scores, top_k))

    def query(self, query_text: str, top_k: int = 10,
              context: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Query the graph using spreading activation with optional
        contextual intersection.

        Args:
            query_text: The entity surface form to look up (e.g.
                     ``"42ms"``, ``"2025-10-08"``, ``"sarah chen"``).
                     This is NOT a natural-language question -- pensive
                     does no NL parsing. A query like "What was the P99
                     latency?" returns [] because none of those tokens
                     are recognized entities in the graph. See README
                     "Quickstart" and the "Natural-language queries"
                     section for the extractor-then-query pattern that
                     maps questions onto entity queries.
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

        # Empty / None / whitespace query short-circuits to no results.
        # The downstream word-tokenization would yield an empty seed
        # set anyway, but the upstream fast-path branches still run a
        # numpy spread over an empty dict and an empty context branch
        # would crash on `for w in None`.
        # (PENPY-MIN-1: also catch whitespace-only queries so the
        # comment-vs-code contract holds.)
        if not (query_text and query_text.strip()):
            return []

        if context is None and self._context_provider is not None:
            context = self._context_provider(query_text)

        words = [w.lower().strip('?.,') for w in query_text.split()
                 if len(w) >= 2]
        query_act = self._seed_from_words(words)

        # Fast path: bipartite graphs use vectorized numpy kernels
        if self._is_bipartite and self.config.max_hops >= 1:
            self._compile()
            if not context:
                return self._spread_and_collect_bipartite(query_act, top_k)
            # Context + bipartite: spread both on numpy arrays, intersect
            query_arr = self._spread_bipartite_raw(query_act)
            ctx_words = [w.lower().strip('?.,') for w in context
                         if len(w) > 1]
            ctx_act = self._seed_from_words(ctx_words)
            ctx_arr = self._spread_bipartite_raw(ctx_act)
            # Boost query scores where context also activated
            ctx_mask = ctx_arr > 0
            query_arr[ctx_mask] *= (1.0 + ctx_arr[ctx_mask])
            return self._collect_from_array(query_arr, top_k)

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
        Entity-exact -- see query() and README "Quickstart" for the
        contract; natural-language questions do not work here.
        """
        if not self._built:
            raise ValueError("Graph not built. Call build() first.")

        # Empty / None / whitespace query short-circuits. Mirrors
        # query(); see there for rationale. (PENPY-MIN-1.)
        if not (query_text and query_text.strip()):
            return []

        if context is None and self._context_provider is not None:
            context = self._context_provider(query_text)

        words = [w.lower().strip('?.,') for w in query_text.split()
                 if len(w) >= 2]
        query_act = self._seed_from_words(words)

        # Fast path: bipartite graphs use vectorized numpy kernels
        if self._is_bipartite and self.config.max_hops >= 1:
            self._compile()
            if not context:
                return self._spread_and_collect_bipartite_with_ids(query_act, top_k)
            # Context + bipartite: spread both on numpy arrays, intersect
            query_arr = self._spread_bipartite_raw(query_act)
            ctx_words = [w.lower().strip('?.,') for w in context
                         if len(w) > 1]
            ctx_act = self._seed_from_words(ctx_words)
            ctx_arr = self._spread_bipartite_raw(ctx_act)
            ctx_mask = ctx_arr > 0
            query_arr[ctx_mask] *= (1.0 + ctx_arr[ctx_mask])
            return self._collect_from_array_with_ids(query_arr, top_k)

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

    def query_analyzed(self, query_text: str, top_k: int = 10,
                       context: Optional[List[str]] = None,
                       n_bands: int = 5):
        """Query with boundary analysis diagnostics.

        Same as query() but returns an AnalyzedResult with both the
        normal results and a BoundaryAnalysis diagnostic.

        See pensive.boundary for details on the diagnostic fields.
        """
        from .boundary import analyze_boundary
        return analyze_boundary(self, query_text, top_k=top_k,
                                context=context, n_bands=n_bands)

    def stats(self) -> Dict:
        """Return graph statistics."""
        n_nodes = len(self._idx_to_node)
        # Use cached _value_indices if compiled, otherwise count from list
        if hasattr(self, '_value_indices') and not self._dirty:
            n_value = len(self._value_indices)
        else:
            n_value = sum(1 for t in self._node_type if t == _VALUE_TYPE)
        n_entity = n_nodes - n_value
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

    def compact(self) -> 'SpreadingActivation':
        """Reclaim the append-only build structures in place.

        The graph is append-only: add_documents() grows the COO edge
        buffers, _entity_edge_positions, and assorted caches without
        ever shrinking them (structural RSS growth; see the README
        "Long-lived processes" section). compact() performs the
        previously-documented get_save_data() -> from_save_data()
        round-trip internally and swaps the rebuilt minimal state into
        this instance, so a long-lived ingesting daemon no longer needs
        to hand-roll serialize-discard-rebuild. After compacting, the
        instance is in the same state as one freshly loaded from disk:
        empty COO buffers, a compiled CSR, cold caches. A later
        add_documents() re-materializes mutable edges via
        _ensure_mutable_edges(), exactly as for a loaded graph.

        Query results are identical before and after; node and edge
        counts are preserved (duplicate COO entries, if any, are summed
        into the CSR, matching what the manual round-trip produced).

        Concurrency: the swap runs under _build_lock, so it serializes
        with build()/add_documents() like any other writer, and the
        compile inside get_save_data() publishes under the same
        generation contract as always. The graph generation is bumped
        afterwards so (n_nodes, generation)-keyed consumer caches drop
        references to the old buffers (PENPY-IMP-5 family).

        Returns:
            self, for chaining.

        Raises:
            RuntimeError: if called before the graph is built.
        """
        with self._build_lock:
            if not self._built:
                raise RuntimeError(
                    "compact() requires a built graph; call build() or "
                    "add_documents() first"
                )
            fresh = type(self).from_save_data(self.get_save_data())

            # Preserve this instance's identity-bearing state: the lock
            # we are holding right now, the caller-installed context
            # provider, the extractor, and the monotonic generation
            # counter. Everything graph-shaped comes from the rebuilt
            # instance (node storage is shared by reference via
            # get_save_data; the win is dropping the COO buffers,
            # _entity_edge_positions, and stale caches).
            preserve = {
                '_build_lock', '_context_provider', '_extractor',
                '_graph_generation',
            }
            for name, value in fresh.__dict__.items():
                if name not in preserve:
                    setattr(self, name, value)

            # Buffer swap is a mutation boundary: bump the generation
            # and drop the boundary value-label cache, mirroring
            # _reset_graph_state / add_documents (PENPY-IMP-5).
            self._graph_generation += 1
            if hasattr(self, '_boundary_value_label_index'):
                try:
                    delattr(self, '_boundary_value_label_index')
                except AttributeError:
                    pass
        return self

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
            'token_index': dict(self._token_index),
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
        nt = data['node_type']
        if not isinstance(nt, array.array):
            nt = array.array('b', nt)
        sa._node_type = nt
        sa._node_label = data['node_label']
        sa._node_specificity = data['node_specificity']

        sa._adj = scipy.sparse.csr_matrix(
            (data['adj_data'], data['adj_indices'], data['adj_indptr']),
            shape=data['adj_shape'],
        )
        # PENPY-P5-IMP-2: from_save_data() previously trusted adj_indices
        # / adj_indptr / adj_shape literally. A corrupted (truncated or
        # tampered) save with adj_indices >= n_nodes would not error at
        # load time -- the OOB index would survive into the numba JIT
        # kernel and segfault Python on the first query that seeded the
        # corresponding entity. check_format(full_check=True) validates
        # that all indices are in range and indptr is monotone, and
        # raises ValueError on violation. Wrap any scipy structural
        # exception into ValueError so callers (e.g. load_graph) catch
        # corruption uniformly instead of letting it surface as a
        # delayed segfault.
        try:
            sa._adj.check_format(full_check=True)
        except (ValueError, TypeError, IndexError) as e:
            raise ValueError(
                f"from_save_data: corrupted adjacency matrix: {e}"
            ) from e

        # PENPY-P6-MIN-1: check_format only validates the CSR structural
        # invariants; it does NOT verify that the parallel node arrays
        # (node_type / idx_to_node / node_label) have the same length as
        # n_nodes. A corrupted save where these arrays disagree would be
        # silently accepted; subsequent queries would return empty or
        # behave inconsistently with no crash. Defense-in-depth: catch
        # length mismatches explicitly and raise ValueError so callers
        # (load_graph) reject the save uniformly. Covered by
        # tests/test_pass4_pass6_dep1_regression.py and
        # tests/test_pass5_p5_from_save_data.py extensions.
        n_nodes = sa._adj.shape[0]
        if len(sa._node_type) != n_nodes:
            raise ValueError(
                f"_from_sparse_v1: corrupted save data: "
                f"node_type={len(sa._node_type)} expected n_nodes={n_nodes}"
            )
        if len(sa._idx_to_node) != n_nodes:
            raise ValueError(
                f"_from_sparse_v1: corrupted save data: "
                f"idx_to_node={len(sa._idx_to_node)} expected n_nodes={n_nodes}"
            )
        if len(sa._node_label) != n_nodes:
            raise ValueError(
                f"_from_sparse_v1: corrupted save data: "
                f"node_label={len(sa._node_label)} expected n_nodes={n_nodes}"
            )
        sa._edge_src = array.array('i')
        sa._edge_dst = array.array('i')
        sa._edge_weight = array.array('f')
        sa._entity_edge_positions = defaultdict(list)
        sa._dirty = False
        sa._node_type_arr = np.frombuffer(sa._node_type, dtype=np.int8).copy()
        sa._value_indices = np.flatnonzero(sa._node_type_arr == _VALUE_TYPE)

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._exact_entities = set(sa.entity_freq.keys())
        sa._entity_terms = list(sa._exact_entities)
        sa._substr_match_cache = {}
        sa._do_substr = len(sa._entity_terms) <= 10_000
        # Use serialized token index if available (avoids O(n) rebuild)
        saved_token_idx = data.get('token_index')
        if saved_token_idx is not None:
            sa._token_index = defaultdict(list, saved_token_idx)
        else:
            sa._rebuild_token_index()
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
            sa._add_edge_fast(src_idx, dst_idx, weight)

        sa._dirty = True
        sa._compile()

        sa.entity_freq = defaultdict(int, data.get('entity_freq', {}))
        sa._entity_index = defaultdict(list, data.get('entity_index', {}))
        sa._exact_entities = set(sa.entity_freq.keys())
        sa._entity_terms = list(sa._exact_entities)
        sa._substr_match_cache = {}
        sa._do_substr = len(sa._entity_terms) <= 10_000
        sa._rebuild_token_index()
        # Legacy networkx graphs may not be bipartite -- check
        sa._is_bipartite = all(
            sa._adj.indptr[idx] == sa._adj.indptr[idx + 1]
            for idx in range(len(sa._idx_to_node))
            if sa._node_type[idx] == _VALUE_TYPE
        ) if sa._adj is not None and sa._adj.nnz > 0 else True
        sa._built = True
        return sa
