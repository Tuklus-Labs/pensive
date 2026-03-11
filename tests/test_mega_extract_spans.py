from pensive.mega_extract import MegaExtractor
from pensive.patterns import REAL_DATA_PATTERNS

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
