"""Query generation for different data sources.

Produces natural-language queries that match how a user would ask about
the content, bridging the Q<->A asymmetry gap in retrieval.
"""
import re
from typing import List


_WORD_RE = re.compile(r'[A-Za-z0-9]+')

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'shall', 'must', 'need',
    'this', 'that', 'these', 'those', 'from', 'with', 'into', 'onto',
    'upon', 'about', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'some', 'any', 'each', 'every', 'other', 'such',
    'than', 'also', 'then', 'more', 'very', 'just', 'only', 'still',
    'even', 'both', 'well', 'here', 'there', 'where', 'when', 'what',
    'which', 'while', 'because', 'since', 'until', 'like', 'your',
    'their', 'they', 'them', 'said', 'says', 'went', 'been', 'being',
    'back', 'over', 'most', 'much', 'many', 'made', 'make', 'know',
})


class QueryGenerator:
    """Generate retrieval queries tailored to each data source."""

    def for_chatgpt(self, title: str, role: str, content: str) -> str:
        """Generate a query for a ChatGPT conversation message.

        Args:
            title: Conversation title.
            role: 'user' or 'assistant'.
            content: The message text.
        """
        key_terms = self._extract_key_terms(content)
        terms_str = ' '.join(key_terms)
        if role == 'user':
            return f"What did I discuss about {title}? {terms_str}"
        return f"What was the response about {title}? {terms_str}"

    def for_facebook(self, sender: str, content: str) -> str:
        """Generate a query for a Facebook message."""
        key_terms = self._extract_key_terms(content)
        terms_str = ' '.join(key_terms)
        return f"What did {sender} say? {terms_str}"

    def for_calendar(self, summary: str, date: str) -> str:
        """Generate a query for a calendar event."""
        return f"What was scheduled for {date}? {summary}"

    def for_session(self, project: str, content: str) -> str:
        """Generate a query for a Claude session entry."""
        key_terms = self._extract_key_terms(content)
        terms_str = ' '.join(key_terms)
        return f"What happened in {project}? {terms_str}"

    def _extract_key_terms(self, text: str, max_terms: int = 3) -> List[str]:
        """Extract the most distinctive words from text.

        Scoring heuristic:
        - Base score = word length (longer words are more distinctive)
        - +3 if the word starts with a capital letter (proper nouns, acronyms)
        - +2 if the word contains any digit (versions, IDs, dates)
        """
        words = _WORD_RE.findall(text)
        # Filter: 4+ chars, not a stopword
        candidates = [w for w in words if len(w) >= 4 and w.lower() not in STOPWORDS]

        # Deduplicate while preserving first-occurrence order
        seen = set()
        unique: List[str] = []
        for w in candidates:
            low = w.lower()
            if low not in seen:
                seen.add(low)
                unique.append(w)

        unique.sort(key=_score_term, reverse=True)
        return unique[:max_terms]


def _score_term(word: str) -> int:
    """Score a word for distinctiveness. Module-level to avoid closure per call."""
    s = len(word)
    if word[0].isupper():
        s += 3
    # Single-pass digit check without generator overhead
    for c in word:
        if '0' <= c <= '9':
            s += 2
            break
    return s
