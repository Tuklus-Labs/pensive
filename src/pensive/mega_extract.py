"""Mega-regex entity extractor for spreading activation.

Combines N individual regex patterns into 2 compiled mega-regexes
(case-insensitive + case-sensitive), reducing per-document text
scanning from N passes to 2 passes.

This class is a pure function (text in, entities out) with no graph
state, making it safe for multiprocessing workers.
"""
import re
from typing import List, Optional, Tuple


class MegaExtractor:
    """Combine regex patterns into 2 mega-regexes for fast entity extraction.

    All patterns must have exactly 1 capturing group containing the entity text.
    Patterns are split by case sensitivity and joined with | alternation.
    """

    __slots__ = ('_ci_regex', '_ci_etype_map', '_cs_regex', '_cs_etype_map')

    def __init__(self, patterns: List[Tuple[str, str, bool]]):
        """Build mega-regexes from pattern list.

        Args:
            patterns: List of (regex_str, entity_type, case_insensitive) tuples.
                      Each regex must have exactly 1 capturing group.
        """
        ci_pats = [(r, et) for r, et, ci in patterns if ci]
        cs_pats = [(r, et) for r, et, ci in patterns if not ci]
        self._ci_regex, self._ci_etype_map = _build_mega(ci_pats, re.IGNORECASE)
        self._cs_regex, self._cs_etype_map = _build_mega(cs_pats, 0)

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Extract all entities from text.

        Returns:
            List of (entity_label_lowercase, entity_type) tuples.
            May contain duplicates if the same text matches multiple patterns.
        """
        results = []
        for regex, etype_map in ((self._ci_regex, self._ci_etype_map),
                                  (self._cs_regex, self._cs_etype_map)):
            if regex is None:
                continue
            for m in regex.finditer(text):
                gi = m.lastindex
                results.append((m.group(gi).lower(), etype_map[gi - 1]))
        return results

    def extract_with_raw(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities, preserving original case.

        Returns:
            List of (entity_lowercase, entity_type, entity_raw) tuples.
        """
        results = []
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
        return merged


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
