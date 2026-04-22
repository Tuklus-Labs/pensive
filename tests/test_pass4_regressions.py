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
