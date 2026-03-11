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
                groups = m.groups()
                for i, g in enumerate(groups):
                    if g is not None:
                        results.append((g.lower(), etype_map[i]))
                        break
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
                groups = m.groups()
                for i, g in enumerate(groups):
                    if g is not None:
                        results.append((g.lower(), etype_map[i], g))
                        break
        return results

    def extract_with_spans(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Extract entities with character-level span positions.

        Returns:
            List of (start_char, end_char, entity_lowercase, entity_type) tuples,
            sorted by start position. Includes all occurrences of each entity.
        """
        results = []
        for regex, etype_map in ((self._ci_regex, self._ci_etype_map),
                                  (self._cs_regex, self._cs_etype_map)):
            if regex is None:
                continue
            for m in regex.finditer(text):
                groups = m.groups()
                for i, g in enumerate(groups):
                    if g is not None:
                        start = m.start(i + 1)
                        end = m.end(i + 1)
                        results.append((start, end, g.lower(), etype_map[i]))
                        break
        results.sort(key=lambda r: r[0])
        return results


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
