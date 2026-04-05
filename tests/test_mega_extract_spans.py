from pensive.mega_extract import MegaExtractor, _split_literal_patterns
from pensive.patterns import REAL_DATA_PATTERNS, NL_PATTERNS, BASE_PATTERNS

def test_extract_with_spans_returns_positions():
    ext = MegaExtractor(REAL_DATA_PATTERNS)
    results = ext.extract_with_spans("Gary built Aegis on 2025-07-16.")
    assert len(results) > 0
    for start, end, entity_lower, entity_type in results:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end
        assert isinstance(entity_lower, str)
        assert isinstance(entity_type, str)

def test_extract_with_spans_positions_match_text():
    ext = MegaExtractor(REAL_DATA_PATTERNS)
    text = "The 7900 XTX runs PyTorch on ROCm."
    results = ext.extract_with_spans(text)
    for start, end, entity_lower, entity_type in results:
        assert text[start:end].lower() == entity_lower

def test_extract_with_spans_finds_all_occurrences():
    ext = MegaExtractor(REAL_DATA_PATTERNS)
    text = "PyTorch helps PyTorch users."
    results = ext.extract_with_spans(text)
    pytorch_spans = [(s, e) for s, e, el, et in results if el == "pytorch"]
    assert len(pytorch_spans) >= 2
    assert pytorch_spans[0][0] != pytorch_spans[1][0]

def test_extract_with_spans_sorted_by_position():
    ext = MegaExtractor(REAL_DATA_PATTERNS)
    results = ext.extract_with_spans("Aegis uses PyTorch with ROCm on 2025-01-01.")
    starts = [s for s, e, el, et in results]
    assert starts == sorted(starts)


# --- Literal pattern extraction tests ---

def test_split_literal_patterns_extracts_simple_words():
    """Pure alphanumeric alternation patterns get extracted as literals."""
    patterns = [
        (r'\b(foo|bar|baz)\b', 'test_type'),
    ]
    regex_pats, literals = _split_literal_patterns(patterns)
    assert len(regex_pats) == 0
    assert literals == {'foo': 'test_type', 'bar': 'test_type', 'baz': 'test_type'}


def test_split_literal_patterns_keeps_complex_patterns():
    """Patterns with regex metacharacters stay in the regex group."""
    patterns = [
        (r'\b(flash\s+attention|kv[\s_-]?cache)\b', 'concept'),
    ]
    regex_pats, literals = _split_literal_patterns(patterns)
    assert len(regex_pats) == 1
    assert len(literals) == 0


def test_split_literal_patterns_splits_mixed():
    """Mixed patterns split correctly: literals extracted, complex kept."""
    patterns = [
        (r'\b(aegis|pensive|mud[\s-]?puppy)\b', 'project'),
    ]
    regex_pats, literals = _split_literal_patterns(patterns)
    assert literals == {'aegis': 'project', 'pensive': 'project'}
    assert len(regex_pats) == 1
    assert 'mud' in regex_pats[0][0]


def test_split_literal_patterns_rejects_nested_groups():
    """Patterns with nested groups (parentheses in inner content) stay as regex."""
    patterns = [
        (r'\b((?:Jan|Feb|Mar)\w*\s+\d{1,2})\b', 'date'),
    ]
    regex_pats, literals = _split_literal_patterns(patterns)
    assert len(regex_pats) == 1
    assert len(literals) == 0


def test_literal_extraction_matches_regex_behavior():
    """Literal extraction produces the same results as full regex."""
    text = "We deployed pytorch and faiss on rocm with docker and kubernetes"

    # Regex-only extractor (no literal optimization)
    class RegexOnlyExtractor(MegaExtractor):
        def __init__(self, patterns):
            # Skip literal extraction by passing through directly
            import re
            ci_pats = [(r, et) for r, et, ci in patterns if ci]
            cs_pats = [(r, et) for r, et, ci in patterns if not ci]
            from pensive.mega_extract import _build_mega
            self._ci_regex, self._ci_etype_map = _build_mega(ci_pats, re.IGNORECASE)
            self._cs_regex, self._cs_etype_map = _build_mega(cs_pats, 0)
            self._ci_literal_words = {}
            self._cs_literal_words = {}

    ext_optimized = MegaExtractor(REAL_DATA_PATTERNS)
    ext_baseline = RegexOnlyExtractor(REAL_DATA_PATTERNS)

    results_opt = set(ext_optimized.extract(text))
    results_base = set(ext_baseline.extract(text))

    # The optimized version should find at least everything the baseline finds
    # (it may find more if literal matching is less restrictive than \b)
    for entity, etype in results_base:
        assert (entity, etype) in results_opt, f"Missing: ({entity}, {etype})"


def test_extract_with_raw_includes_literal_results():
    """extract_with_raw includes results from literal word lookups."""
    ext = MegaExtractor(REAL_DATA_PATTERNS)
    results = ext.extract_with_raw("pytorch and rocm")
    entities = {e for e, _, _ in results}
    assert 'pytorch' in entities
    assert 'rocm' in entities
