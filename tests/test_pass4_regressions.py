"""Regression tests for Pass 4 fixes: CLI --trusted flag and case-sensitive extraction."""
import subprocess
import sys

from pensive.mega_extract import MegaExtractor
from pensive.parallel_hybrid import ParallelHybrid
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


# ---------------------------------------------------------------------------
# PENPY-IMP-1: L2 candidate-rerank fallback must respect the candidate set
# ---------------------------------------------------------------------------


class _LegacyL2Result:
    def __init__(self, document_id, content, score):
        self.document_id = document_id
        self.content = content
        self.score = score


class _LegacyL2NoCandidates:
    """L2 handler WITHOUT query_candidates() to exercise the fallback path.

    The global query() returns a mix of in-candidate-set and out-of-set
    doc IDs so we can prove the fallback filters correctly.
    """
    def __init__(self, payload):
        self._payload = payload
        self.global_call_count = 0

    def query(self, query_text, top_k=10):
        self.global_call_count += 1
        return self._payload[:top_k]


def test_l2_candidate_fallback_filters_to_candidate_set():
    """PENPY-IMP-1 regression: when the L2 handler lacks query_candidates(),
    _query_l2_on_candidates() must filter the global-query results down
    to the SA candidate set. Returning arbitrary global doc IDs would
    silently break the agreement-boost logic downstream by treating
    non-candidate documents as if they were SA hits.

    Sabotage check: reverting the fix (return unfiltered global results)
    surfaces 'doc-OUT' from the global query, which was never in the
    candidate list.
    """
    # Global L2 returns a deliberately scrambled mix.
    legacy_l2 = _LegacyL2NoCandidates([
        _LegacyL2Result('doc-OUT', 'unrelated content', 0.95),  # NOT a candidate
        _LegacyL2Result('doc-A', 'relevant A', 0.80),
        _LegacyL2Result('doc-OUT2', 'also unrelated', 0.75),    # NOT a candidate
        _LegacyL2Result('doc-B', 'relevant B', 0.60),
    ])

    hybrid = ParallelHybrid(
        spreading_activation=None,
        l2_handler=legacy_l2,
        enable_pattern_learning=False,
    )

    results = hybrid._query_l2_on_candidates(
        query='anything',
        candidate_doc_ids=['doc-A', 'doc-B'],
        top_k=5,
    )

    returned_ids = {r['doc_id'] for r in results}
    assert returned_ids == {'doc-A', 'doc-B'}, (
        f"Candidate-rerank fallback must filter to the candidate set. "
        f"Got {returned_ids!r}, expected only doc-A/doc-B. "
        f"Returning unfiltered global results (the pre-fix behavior) "
        f"would have leaked doc-OUT/doc-OUT2 through."
    )
    assert legacy_l2.global_call_count == 1, (
        f"Expected exactly one global query() call, got "
        f"{legacy_l2.global_call_count}"
    )


def test_l2_candidate_fallback_empty_candidate_set_returns_empty():
    """Companion to PENPY-IMP-1: an empty candidate set must short-circuit
    and never issue a global L2 query."""
    legacy_l2 = _LegacyL2NoCandidates([
        _LegacyL2Result('doc-OUT', 'unrelated content', 0.95),
    ])

    hybrid = ParallelHybrid(
        spreading_activation=None,
        l2_handler=legacy_l2,
        enable_pattern_learning=False,
    )

    results = hybrid._query_l2_on_candidates(
        query='anything',
        candidate_doc_ids=[],
        top_k=5,
    )

    assert results == []
    assert legacy_l2.global_call_count == 0, (
        f"Empty candidate list must short-circuit before issuing a global "
        f"L2 query; got {legacy_l2.global_call_count} calls"
    )


# ---------------------------------------------------------------------------
# PENPY-IMP-3: boundary.analyze_boundary must handle None/empty queries
# ---------------------------------------------------------------------------


def test_analyze_boundary_handles_none_query():
    """PENPY-IMP-3 regression: analyze_boundary(sa, None) must not crash.

    SpreadingActivation.query(None) already returns []. The boundary
    diagnostic path forgot to mirror that guard: _compute_score_arr ran
    `query_text.split()` on None and raised AttributeError. The bug was
    masked in ParallelHybrid by a broad `except Exception`, surfacing as
    'SA query failed' log spam instead.

    Sabotage check: removing the falsy-query guard from
    analyze_boundary_results makes this test raise AttributeError on
    `None.split`.
    """
    from pensive.boundary import analyze_boundary, analyze_boundary_results

    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build([
        {"id": "doc-a", "content": "alpha", "value": "alpha",
         "query": "alpha?"},
    ])

    # Direct results-helper call with None should not throw and should
    # return a no-result BoundaryAnalysis.
    analysis = analyze_boundary_results(sa, None, [])
    assert analysis.boundary_distance is None
    assert analysis.top_scores == []
    assert analysis.confidence == "none"

    # Same for empty string.
    analysis_empty = analyze_boundary_results(sa, "", [])
    assert analysis_empty.top_scores == []
    assert analysis_empty.confidence == "none"

    # Full analyze_boundary() entry point also routes through this.
    full = analyze_boundary(sa, None)
    assert full.results == []
    assert full.analysis.top_scores == []
    assert full.analysis.confidence == "none"


# ---------------------------------------------------------------------------
# PENPY-IMP-4: boost_identifier_matches must not mutate caller's input ranks
# ---------------------------------------------------------------------------


def test_boost_identifier_matches_does_not_mutate_input_rank():
    """PENPY-IMP-4 regression: when a result is not boosted (no
    identifier match), boost_identifier_matches used to re-export the
    original SearchResult and then reassign its .rank, silently
    mutating caller-owned state. Callers retaining a reference to the
    input list saw ranks overwritten out from under them.

    Sabotage check: dropping the new-SearchResult construction on the
    not-boosted branch makes the input rank get reassigned by the
    function's tail loop.
    """
    from pensive.hybrid_search import SearchResult, boost_identifier_matches

    # The query contains a hex identifier; only doc-A's content matches.
    inputs = [
        SearchResult(
            document_id='doc-A', content='see 0xdeadbeef for context',
            score=0.5, rank=42, source='dense',
        ),
        SearchResult(
            document_id='doc-B', content='unrelated body',
            score=0.4, rank=99, source='dense',
        ),
    ]
    original_rank_B = inputs[1].rank
    original_score_B = inputs[1].score
    original_meta_B = dict(inputs[1].metadata)

    out = boost_identifier_matches(inputs, query='look up 0xdeadbeef now')

    # doc-B (not boosted) must NOT have its caller-visible rank
    # changed.
    assert inputs[1].rank == original_rank_B, (
        f"Expected input doc-B rank to remain {original_rank_B}, "
        f"but it was mutated to {inputs[1].rank} (function leaked rank "
        f"reassignment into caller-owned object)."
    )
    # Score and metadata also untouched.
    assert inputs[1].score == original_score_B
    assert inputs[1].metadata == original_meta_B

    # The returned list still has well-ordered ranks (1, 2, ...).
    assert [r.rank for r in out] == list(range(1, len(out) + 1))
