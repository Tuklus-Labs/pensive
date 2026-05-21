"""Regression tests for Pass 4 fixes: CLI --trusted flag and case-sensitive extraction."""
import subprocess
import sys

from pensive.mega_extract import MegaExtractor


def test_extract_returns_lowercase_per_contract():
    """extract() is documented to return lowercase labels. extract_with_raw()
    is for original-case needs. Pin this contract so we don't regress.
    """
    # Pattern format: (regex_str, entity_type, case_insensitive)
    # A case-SENSITIVE literal pattern 'PyPI' must still emit lowercase 'pypi'
    # from extract(). The separate extract_with_raw() preserves case.
    # Regex patterns must have a capturing group (enforced by build_mega).
    patterns = [(r"(PyPI)", "product", False)]
    extractor = MegaExtractor(patterns)
    got = extractor.extract("I published a package to PyPI today.")
    labels = [label for label, _ in got]
    assert "pypi" in labels, (
        f"extract() should always lowercase per docstring, got {labels}"
    )
    # And the raw version preserves case:
    got_raw = extractor.extract_with_raw("I published a package to PyPI today.")
    raws = [raw for _, _, raw in got_raw]
    assert "PyPI" in raws, (
        f"extract_with_raw() should preserve case, got raws {raws}"
    )


def test_cli_has_trusted_flag():
    """The `pensive` CLI must accept --trusted on query and stats."""
    help_out = subprocess.run(
        [sys.executable, "-m", "pensive.ingestion.cli", "query", "--help"],
        capture_output=True, text=True,
    )
    assert "--trusted" in help_out.stdout, (
        f"query subcommand missing --trusted flag. stdout:\n{help_out.stdout}"
    )

    help_out = subprocess.run(
        [sys.executable, "-m", "pensive.ingestion.cli", "stats", "--help"],
        capture_output=True, text=True,
    )
    assert "--trusted" in help_out.stdout, (
        f"stats subcommand missing --trusted flag. stdout:\n{help_out.stdout}"
    )


def test_case_sensitive_literal_alternation_matches_original_case():
    """PENPY-CRIT-1 regression: a literal alternation pattern registered
    with case_insensitive=False must match ONLY the original-case forms.

    The fast-literal optimization in _split_literal_patterns used to
    unconditionally lowercase keys, which silently broke case-sensitive
    literal patterns: the dict stored 'foo'/'bar' but extract() looked
    up 'Foo'/'Bar', so no matches ever fired.
    """
    # Case-sensitive literal alternation
    patterns = [(r"\b(Foo|Bar)\b", "thing", False)]
    extractor = MegaExtractor(patterns)

    # Original case: must match
    got = extractor.extract("Look at Foo and Bar over there")
    types_emitted = [etype for _, etype in got]
    assert types_emitted.count("thing") == 2, (
        f"Case-sensitive literals 'Foo' and 'Bar' should each match, got {got}"
    )

    # Lowercase: must NOT match (case-sensitive pattern)
    got_lower = extractor.extract("look at foo and bar over there")
    types_lower = [etype for _, etype in got_lower]
    assert "thing" not in types_lower, (
        f"Case-sensitive pattern must NOT match lowercase variants, "
        f"got {got_lower}"
    )

    # Mixed: only the matching-case word should fire
    got_mixed = extractor.extract("Foo and bar — only Foo should match")
    types_mixed = [etype for _, etype in got_mixed]
    assert types_mixed.count("thing") == 2, (
        f"Two case-correct 'Foo' should match, got {got_mixed}"
    )


def test_case_insensitive_literal_alternation_still_works():
    """Counterpart to PENPY-CRIT-1: ci=True literals must still match
    regardless of case (don't break the original behaviour while fixing
    the ci=False bug).
    """
    patterns = [(r"\b(Foo|Bar)\b", "thing", True)]
    extractor = MegaExtractor(patterns)

    for text in ("foo bar", "FOO BAR", "Foo Bar", "fOo bAr"):
        got = extractor.extract(text)
        types_emitted = [etype for _, etype in got]
        assert types_emitted.count("thing") == 2, (
            f"ci=True should match all cases for {text!r}, got {got}"
        )
