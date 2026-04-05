"""
Adaptive pattern learning for spreading activation.

Watches query/result pairs and learns new entity patterns when:
1. SA returns 0 results but L2 finds relevant content
2. Query terms appear frequently in successful L2 results

Learned entities are stored with the graph and used in future queries.
"""
import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Common words that should never be learned as entities
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'not', 'no', 'so', 'if', 'when', 'what', 'which', 'who', 'whom', 'how',
    'where', 'why', 'that', 'this', 'it', 'he', 'she', 'they', 'we', 'you',
    'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them',
    'all', 'some', 'any', 'most', 'other', 'such', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'well', 'back', 'even', 'still', 'get', 'got', 'like', 'know', 'think',
    'want', 'see', 'look', 'make', 'go', 'come', 'take', 'use', 'find',
    'give', 'tell', 'say', 'said', 'ask', 'try', 'need', 'feel', 'become',
    'leave', 'put', 'mean', 'keep', 'let', 'begin', 'seem', 'help', 'show',
    'hear', 'play', 'run', 'move', 'live', 'believe', 'hold', 'bring',
    'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet',
    'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
    'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add',
    'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
    'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
    'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'killed',
})


@dataclass
class LearnedEntity:
    """A dynamically learned entity."""
    term: str
    source: str  # 'query_gap', 'l2_frequent', 'manual'
    frequency: int = 1
    first_seen_query: str = ''


@dataclass
class PatternLearnerConfig:
    """Configuration for pattern learning."""
    # Minimum term length to consider learning
    min_term_length: int = 3
    # Minimum L2 result score to consider the result "good"
    min_l2_score: float = 0.5
    # How many times a term must appear before being learned
    frequency_threshold: int = 1  # Learn immediately on first gap
    # Maximum learned entities to keep
    max_learned: int = 10000


class PatternLearner:
    """
    Learns new entity patterns from query/result gaps.

    When SA returns nothing but L2 finds good results, extracts
    candidate entities from the query and learns them for future use.
    """

    def __init__(self, config: Optional[PatternLearnerConfig] = None):
        self.config = config or PatternLearnerConfig()
        self.learned_entities: Dict[str, LearnedEntity] = {}
        self.candidate_counts: Dict[str, int] = defaultdict(int)
        self._callbacks: List[callable] = []

    def on_learn(self, callback: callable):
        """Register callback for when new entity is learned."""
        self._callbacks.append(callback)

    def observe(self, query: str, sa_results: List, l2_results: List) -> List[str]:
        """
        Observe a query and its results, learning from gaps.

        Args:
            query: The original query string
            sa_results: Results from spreading activation (list of (value, score))
            l2_results: Results from L2 semantic search

        Returns:
            List of newly learned entity terms
        """
        newly_learned = []

        # Only learn when SA fails but L2 succeeds
        sa_found = len(sa_results) > 0
        l2_found = len(l2_results) > 0

        if sa_found or not l2_found:
            return newly_learned

        # Extract candidate terms from query
        candidates = self._extract_candidates(query)

        # Also extract from top L2 results to find common terms
        for result in l2_results[:3]:
            text = getattr(result, 'summary', str(result))
            result_terms = self._extract_candidates(text)
            # Terms that appear in both query and results are good candidates
            # Fast path: exact set intersection covers most matches
            exact_hits = candidates & result_terms
            for term in exact_hits:
                self.candidate_counts[term] += 1
            # Slow path: substring check only for remaining candidates
            remaining = candidates - exact_hits
            if remaining and result_terms:
                for term in remaining:
                    if any(term in rt for rt in result_terms):
                        self.candidate_counts[term] += 1

        # Learn candidates that meet threshold
        for term in candidates:
            self.candidate_counts[term] += 1

            if (self.candidate_counts[term] >= self.config.frequency_threshold
                and term not in self.learned_entities
                and len(self.learned_entities) < self.config.max_learned):

                entity = LearnedEntity(
                    term=term,
                    source='query_gap',
                    frequency=self.candidate_counts[term],
                    first_seen_query=query,
                )
                self.learned_entities[term] = entity
                newly_learned.append(term)
                logger.info(f"Learned new entity: '{term}' from query: '{query}'")

                # Notify callbacks
                for cb in self._callbacks:
                    try:
                        cb(term, entity)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

        return newly_learned

    _WORD_RE = re.compile(r'[a-zA-Z]+')

    def _extract_candidates(self, text: str) -> Set[str]:
        """Extract candidate entity terms from text in a single pass.

        Uses finditer to avoid materializing a full word list, and
        lowercases per-word instead of copying the entire input string.
        """
        candidates = set()
        min_len = self.config.min_term_length
        min_bigram = min_len * 2
        stopwords = _STOPWORDS

        prev_word = None
        prev_ok = False
        for m in self._WORD_RE.finditer(text):
            word = m.group().lower()
            ok = len(word) >= min_len and word not in stopwords
            if ok:
                candidates.add(word)
            if prev_ok and ok:
                phrase = f"{prev_word} {word}"
                if len(phrase) >= min_bigram:
                    candidates.add(phrase)
            prev_word = word
            prev_ok = ok

        return candidates

    def get_learned_terms(self) -> Set[str]:
        """Get all learned entity terms.

        Returns a view-like set backed by dict keys. Callers should not
        mutate the returned set.
        """
        return self.learned_entities.keys()

    def add_manual(self, term: str, source: str = 'manual'):
        """Manually add a learned entity."""
        term = term.lower().strip()
        if term and term not in self.learned_entities:
            self.learned_entities[term] = LearnedEntity(
                term=term,
                source=source,
                frequency=1,
            )
            logger.info(f"Manually added entity: '{term}'")

    def to_dict(self) -> dict:
        """Serialize for saving with graph."""
        return {
            'learned_entities': {
                k: {
                    'term': v.term,
                    'source': v.source,
                    'frequency': v.frequency,
                    'first_seen_query': v.first_seen_query,
                }
                for k, v in self.learned_entities.items()
            },
            'candidate_counts': dict(self.candidate_counts),
        }

    @classmethod
    def from_dict(cls, data: dict, config: Optional[PatternLearnerConfig] = None) -> 'PatternLearner':
        """Deserialize from saved data."""
        learner = cls(config=config)

        for k, v in data.get('learned_entities', {}).items():
            learner.learned_entities[k] = LearnedEntity(
                term=v['term'],
                source=v['source'],
                frequency=v.get('frequency', 1),
                first_seen_query=v.get('first_seen_query', ''),
            )

        learner.candidate_counts = defaultdict(int, data.get('candidate_counts', {}))
        return learner


def integrate_with_sa(sa, learner: PatternLearner):
    """
    Integrate pattern learner with SpreadingActivation instance.

    Patches both _seed_from_words and query_with_doc_ids to handle
    learned entities that aren't in the graph as entity nodes.

    Key insight: Value nodes seeded directly won't survive spreading
    (they have no neighbors, just decay below threshold). So we track
    them separately and inject them into final results.
    """
    original_seed = sa._seed_from_words
    original_query = sa.query_with_doc_ids

    # Lazy index: term → list of value node IDs
    _learned_term_index: Dict[str, List[str]] = {}

    # Track directly seeded value nodes per query (cleared each query)
    _direct_value_seeds: Dict[str, float] = {}

    # Pre-build value node lists once (cached for all future _build_term_index calls)
    _cached_value_labels: List[str] = []
    _cached_value_nodes: List[str] = []
    if hasattr(sa, '_idx_to_node') and hasattr(sa, '_node_type'):
        for idx, node_id in enumerate(sa._idx_to_node):
            if sa._node_type[idx] != 1:
                continue
            _cached_value_labels.append(sa._node_label[idx].lower())
            _cached_value_nodes.append(node_id)

        for term in learner.learned_entities:
            matches = [node_id for label, node_id in zip(_cached_value_labels, _cached_value_nodes)
                       if term in label]
            _learned_term_index[term] = matches

    def _build_term_index(term: str) -> List[str]:
        """Build index for a learned term (one-time scan per term).

        Uses cached value node lists instead of rescanning all graph nodes.
        """
        if term in _learned_term_index:
            return _learned_term_index[term]

        matches = [node_id for label, node_id in zip(_cached_value_labels, _cached_value_nodes)
                   if term in label]
        _learned_term_index[term] = matches
        return matches

    def patched_seed(words: List[str]) -> Dict[int, float]:
        # Clear direct seeds from previous query
        _direct_value_seeds.clear()

        # Get original seeds (entity nodes)
        seeds = original_seed(words)

        # Check learned entities against query words
        query_words = set(words)

        for term in learner.get_learned_terms():
            term_words = set(term.split())
            if term in query_words or term_words & query_words:
                matching_nodes = _build_term_index(term)
                for node in matching_nodes:
                    # Track direct value node seeds separately
                    # (they won't survive spreading)
                    _direct_value_seeds[node] = sa.config.substr_boost

        return seeds

    def patched_query(query_text: str, top_k: int = 50,
                      context: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """Query with learned entity support."""
        # Call original query (which uses patched _seed_from_words)
        results = original_query(query_text, top_k, context)

        # Inject directly seeded value nodes that didn't survive spreading
        if _direct_value_seeds:
            result_ids = {r[0] for r in results}
            additional = []

            for node_id, score in sorted(_direct_value_seeds.items(),
                                         key=lambda x: -x[1]):
                doc_id = node_id[2:] if node_id.startswith('v:') else node_id
                if doc_id not in result_ids:
                    if hasattr(sa, '_node_to_idx') and hasattr(sa, '_node_label'):
                        idx = sa._node_to_idx.get(node_id)
                        if idx is None:
                            continue
                        label = sa._node_label[idx]
                    else:
                        label = sa.graph.nodes[node_id].get('label', '')
                    additional.append((doc_id, label, score))

            if additional:
                # Merge and re-sort
                results = results + additional
                results.sort(key=lambda x: -x[2])
                results = results[:top_k]

        return results

    sa._seed_from_words = patched_seed
    sa.query_with_doc_ids = patched_query
    logger.info(f"Integrated pattern learner with SA ({len(learner.learned_entities)} learned entities)")
