"""Mega-regex entity extractor for spreading activation.

Combines N individual regex patterns into 2 compiled mega-regexes
(case-insensitive + case-sensitive), reducing per-document text
scanning from N passes to 2 passes.

Literal alternation patterns (e.g. \\b(word1|word2|...)\\b where all
alternatives are alphanumeric) are extracted into frozenset lookups
for ~20% faster extraction vs regex.

This class is a pure function (text in, entities out) with no graph
state, making it safe for multiprocessing workers.
"""
import re
from typing import Dict, FrozenSet, List, Optional, Tuple


_WORD_RE = re.compile(r'[a-zA-Z0-9]+')


class MegaExtractor:
    """Combine regex patterns into mega-regexes + literal sets for fast extraction.

    All patterns must have exactly 1 capturing group containing the entity text.
    Patterns are split by case sensitivity and joined with | alternation.
    Pure literal alternation patterns are further extracted into frozenset
    lookups to avoid regex overhead for simple word matching.
    """

    __slots__ = (
        '_ci_regex', '_ci_etype_map',
        '_cs_regex', '_cs_etype_map',
        '_ci_literal_words', '_cs_literal_words',
    )

    def __init__(self, patterns: List[Tuple[str, str, bool]]):
        """Build mega-regexes from pattern list.

        Args:
            patterns: List of (regex_str, entity_type, case_insensitive) tuples.
                      Each regex must have exactly 1 capturing group.
        """
        ci_pats = [(r, et) for r, et, ci in patterns if ci]
        cs_pats = [(r, et) for r, et, ci in patterns if not ci]

        ci_regex_pats, ci_literals = _split_literal_patterns(ci_pats)
        cs_regex_pats, cs_literals = _split_literal_patterns(cs_pats)

        self._ci_regex, self._ci_etype_map = _build_mega(ci_regex_pats, re.IGNORECASE)
        self._cs_regex, self._cs_etype_map = _build_mega(cs_regex_pats, 0)
        self._ci_literal_words = ci_literals  # {word_lower: entity_type}
        self._cs_literal_words = cs_literals  # {word: entity_type}

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Extract all entities from text.

        Returns:
            List of (entity_label_lowercase, entity_type) tuples.
            May contain duplicates if the same text matches multiple patterns.
        """
        results = []

        # Fast path: literal word lookups (single scan for both CI and CS)
        ci_literals = self._ci_literal_words
        cs_literals = self._cs_literal_words
        if ci_literals or cs_literals:
            for m in _WORD_RE.finditer(text):
                w = m.group()
                wl = w.lower()
                if ci_literals:
                    etype = ci_literals.get(wl)
                    if etype is not None:
                        results.append((wl, etype))
                if cs_literals:
                    etype = cs_literals.get(w)
                    if etype is not None:
                        # Return lowercase for consistency with extract()'s
                        # documented contract (all other paths in this
                        # function also lowercase). Callers that need the
                        # original case should use extract_with_raw().
                        results.append((wl, etype))

        # Regex path for complex patterns
        ci_regex = self._ci_regex
        if ci_regex is not None:
            ci_etype = self._ci_etype_map
            for m in ci_regex.finditer(text):
                gi = m.lastindex
                results.append((m.group(gi).lower(), ci_etype[gi - 1]))

        cs_regex = self._cs_regex
        if cs_regex is not None:
            cs_etype = self._cs_etype_map
            for m in cs_regex.finditer(text):
                gi = m.lastindex
                results.append((m.group(gi).lower(), cs_etype[gi - 1]))

        return results

    def extract_with_raw(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities, preserving original case.

        Returns:
            List of (entity_lowercase, entity_type, entity_raw) tuples.
        """
        results = []

        ci_literals = self._ci_literal_words
        cs_literals = self._cs_literal_words
        if ci_literals or cs_literals:
            for m in _WORD_RE.finditer(text):
                w = m.group()
                wl = w.lower()
                if ci_literals:
                    etype = ci_literals.get(wl)
                    if etype is not None:
                        results.append((wl, etype, w))
                if cs_literals:
                    etype = cs_literals.get(w)
                    if etype is not None:
                        results.append((wl, etype, w))

        for regex, etype_map in ((self._ci_regex, self._ci_etype_map),
                                  (self._cs_regex, self._cs_etype_map)):
            if regex is None:
                continue
            for m in regex.finditer(text):
                gi = m.lastindex
                g = m.group(gi)
                results.append((g.lower(), etype_map[gi - 1], g))
        return results

    def extract_with_spans(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Extract entities with character-level span positions.

        Returns:
            List of (start_char, end_char, entity_lowercase, entity_type) tuples,
            sorted by start position. Includes all occurrences of each entity.
        """
        # Collect all spans from all sources, then sort once.
        all_spans = []

        ci_literals = self._ci_literal_words
        cs_literals = self._cs_literal_words
        if ci_literals or cs_literals:
            for m in _WORD_RE.finditer(text):
                w = m.group()
                wl = w.lower()
                start = m.start()
                end = m.end()
                if ci_literals:
                    etype = ci_literals.get(wl)
                    if etype is not None:
                        all_spans.append((start, end, wl, etype))
                if cs_literals:
                    etype = cs_literals.get(w)
                    if etype is not None:
                        all_spans.append((start, end, wl, etype))

        # finditer returns matches in text order, so each list is pre-sorted.
        # Merge two sorted lists in O(n) instead of sorting O(n log n).
        lists = []
        for regex, etype_map in ((self._ci_regex, self._ci_etype_map),
                                  (self._cs_regex, self._cs_etype_map)):
            if regex is None:
                continue
            spans = []
            for m in regex.finditer(text):
                gi = m.lastindex
                g = m.group(gi)
                spans.append((m.start(gi), m.end(gi), g.lower(), etype_map[gi - 1]))
            lists.append(spans)

        if all_spans:
            all_spans.sort(key=lambda x: x[0])
            lists.insert(0, all_spans)

        if not lists:
            return []
        if len(lists) == 1:
            return lists[0]

        # Merge two sorted span lists
        a, b = lists[0], lists[1]
        merged = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i][0] <= b[j][0]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        if i < len(a):
            merged.extend(a[i:])
        else:
            merged.extend(b[j:])

        # If there's a third list, merge again
        if len(lists) > 2:
            a, b = merged, lists[2]
            merged = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i][0] <= b[j][0]:
                    merged.append(a[i])
                    i += 1
                else:
                    merged.append(b[j])
                    j += 1
            if i < len(a):
                merged.extend(a[i:])
            else:
                merged.extend(b[j:])

        return merged


def _split_literal_patterns(
    patterns: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Split patterns into regex patterns and literal word lookups.

    A pattern qualifies as "literal" if it matches the form:
        \\b(word1|word2|...)\\b
    where every alternative is purely alphanumeric [a-zA-Z0-9]+.

    Returns:
        (regex_patterns, literal_dict) where literal_dict maps word -> etype.
    """
    regex_pats = []
    literals: Dict[str, str] = {}

    for regex_str, etype in patterns:
        # Check for \b(...)\b wrapper with flat alternation (no nested groups)
        if not (regex_str.startswith(r'\b(') and regex_str.endswith(r')\b')):
            regex_pats.append((regex_str, etype))
            continue

        inner = regex_str[3:-3]  # strip \b( and )\b

        # Reject if inner content has nested groups - the | split
        # would incorrectly break apart group contents
        if '(' in inner or ')' in inner:
            regex_pats.append((regex_str, etype))
            continue

        alternatives = inner.split('|')

        # Split into literal (pure alphanumeric) and complex alternatives
        literal_alts = []
        complex_alts = []
        for alt in alternatives:
            if re.fullmatch(r'[a-zA-Z0-9]+', alt):
                literal_alts.append(alt)
            else:
                complex_alts.append(alt)

        # Add literal words to the dict
        for word in literal_alts:
            literals[word.lower()] = etype

        # Keep complex alternatives in the regex
        if complex_alts:
            remaining = r'\b(' + '|'.join(complex_alts) + r')\b'
            regex_pats.append((remaining, etype))

    return regex_pats, literals


def _build_mega(
    patterns: List[Tuple[str, str]],
    flags: int,
) -> Tuple[Optional[re.Pattern], List[str]]:
    """Build a compiled mega-regex and group->etype mapping.

    Each pattern is wrapped in (?:...) to form one alternative in the
    mega-regex. Since each pattern has exactly 1 capturing group,
    pattern N's group is at index N in match.groups().

    Args:
        patterns: List of (regex_str, entity_type) tuples.
        flags: re flags (e.g. re.IGNORECASE).

    Returns:
        (compiled_regex, etype_list) where etype_list[i] is the entity
        type for the i-th capturing group. Returns (None, []) if patterns
        is empty.
    """
    if not patterns:
        return None, []
    mega = '|'.join(f'(?:{r})' for r, _ in patterns)
    return re.compile(mega, flags), [et for _, et in patterns]
