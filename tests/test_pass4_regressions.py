"""Regression tests for Pass 4 fixes: CLI --trusted flag and case-sensitive extraction."""
import subprocess
import sys

from pensive.mega_extract import MegaExtractor
from pensive.pattern_learner import PatternLearner, integrate_with_sa
from pensive.patterns import SYNTHETIC_PATTERNS
from pensive.spreading import SpreadingActivation


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


def test_pattern_learner_direct_seeds_survive_context_seeding():
    """PENPY-CRIT-2 regression: when SpreadingActivation.query() is invoked
    with a context, _seed_from_words runs TWICE inside a single query
    (once for the query terms, once for the context terms). The old
    patched_seed unconditionally cleared _direct_value_seeds on every
    call, so the second invocation (for context) wiped out the
    learned-term hits established by the first invocation (for the
    query). The fix moves the per-query clear to patched_query() and
    accumulates inside patched_seed().

    Sabotage check: reverting the fix (re-adding _direct_value_seeds.clear()
    inside patched_seed) makes this test fail because the query-side
    learned-entity hit ('alpha') is dropped before the result-merge step.
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc-alpha", "content": "alpha discussion notes",
         "value": "alpha discussion notes",
         "query": "what about alpha?"},
        {"id": "doc-bravo", "content": "bravo briefing",
         "value": "bravo briefing",
         "query": "what about bravo?"},
    ])

    learner = PatternLearner()
    # Add BOTH terms as learned entities so that one is hit from the
    # query and one from the context.
    learner.add_manual("alpha")
    learner.add_manual("bravo")
    integrate_with_sa(sa, learner)

    # Query mentions only "alpha"; context mentions only "bravo".
    # Both learned-entity hits must surface in the final result list.
    results = sa.query_with_doc_ids(
        "alpha report", top_k=20, context=["bravo"]
    )
    found_ids = {r[0] for r in results}

    assert "doc-alpha" in found_ids, (
        f"Query-side learned-term 'alpha' was lost after context seeding "
        f"cleared the direct-seed map. Got results: {results!r}"
    )
    assert "doc-bravo" in found_ids, (
        f"Context-side learned-term 'bravo' was not seeded. "
        f"Got results: {results!r}"
    )


def test_pattern_learner_clears_seeds_between_separate_queries():
    """Companion to PENPY-CRIT-2: the clear() must still happen between
    successive query() calls, otherwise stale learned-term hits from a
    prior query would leak into the next one's results.

    Sabotage check: removing the _direct_value_seeds.clear() from
    patched_query() entirely would make this test fail by surfacing
    doc-alpha in the second query (which only asks about bravo).
    """
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc-alpha", "content": "alpha discussion notes",
         "value": "alpha discussion notes",
         "query": "what about alpha?"},
        {"id": "doc-bravo", "content": "bravo briefing",
         "value": "bravo briefing",
         "query": "what about bravo?"},
    ])

    learner = PatternLearner()
    learner.add_manual("alpha")
    learner.add_manual("bravo")
    integrate_with_sa(sa, learner)

    # First query primes _direct_value_seeds with 'alpha' hits.
    first = sa.query_with_doc_ids("alpha report", top_k=20)
    assert any(r[0] == "doc-alpha" for r in first), (
        f"sanity: first query should match doc-alpha, got {first!r}"
    )

    # Second query asks only about bravo; doc-alpha must NOT leak through
    # via stale direct-seed state.
    second = sa.query_with_doc_ids("bravo report", top_k=20)
    assert any(r[0] == "doc-bravo" for r in second), (
        f"sanity: second query should match doc-bravo, got {second!r}"
    )
    # The leak signature would be: alpha appearing in the second query's
    # results purely from stale _direct_value_seeds carryover. Note: SA
    # spreading itself could still hit doc-alpha via shared structure;
    # what we're guarding against here is the specific direct-injection
    # path. The test would have failed pre-fix because the second query
    # would clear-and-repopulate seeds with only bravo, but the previous
    # version's behaviour after the fix is: clear at query entry, then
    # populate only for the current query's terms.
    # (We assert that bravo dominates, not that alpha is absent, since
    # alpha may legitimately co-activate via the graph.)
    bravo_rank = next(i for i, r in enumerate(second) if r[0] == "doc-bravo")
    alpha_in_second = [i for i, r in enumerate(second) if r[0] == "doc-alpha"]
    if alpha_in_second:
        assert bravo_rank < alpha_in_second[0], (
            f"For a bravo-only query, doc-bravo must outrank doc-alpha. "
            f"Got second={second!r}"
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
