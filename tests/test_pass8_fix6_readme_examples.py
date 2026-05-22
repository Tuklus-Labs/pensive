"""Regression test for PENPY-P8-IMP-1: stale natural-language examples
in README sections beyond Quickstart.

Fix-5 (a6c126e) rewrote ONLY the Quickstart to use entity-exact
queries. Pass-8 found three more examples still using natural-language
queries that silently return []:

  - README Ingestion section: pipe.sa.query("What did we talk about last week?")
  - README CLI example:        pensive query "What was the deployment date?"
  - README Contextual Disambig: sa.query("What was the temperature?", context=...)
  - README Boundary Analysis:  sa.query_analyzed("What was the P99 latency on 2025-07-16?")
    (this one was structurally valid -- did not raise -- but
     returned confidence='none', should_trust=False, recommended_action='no_result',
     which silently misleads readers into thinking that's the default
     for a well-formed query.)

Fix-6 rewrites each example to use entity-exact queries plus a note
that the user should extract entities from natural-language inputs
before calling sa.query() / pensive query --analyze.

Sabotage gate (verified before commit):
  - Restore any of the four broken NL queries to the README; the
    corresponding test below fires loudly.
"""
import re
from pathlib import Path


README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def _readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_no_known_broken_nl_queries():
    """Forbid every natural-language query phrasing that pass-8 found
    silently returning []. Each is an exact-string check so a future
    edit that re-introduces one of these fails loudly here instead of
    misleading users at onboarding.
    """
    text = _readme()
    forbidden = [
        'sa.query("What did we talk about last week?")',
        "sa.query('What did we talk about last week?')",
        'sa.query("What was the deployment date?")',
        "sa.query('What was the deployment date?')",
        'pensive query --graph graph.pkl "What was the deployment date?"',
        "pensive query --graph graph.pkl 'What was the deployment date?'",
        'sa.query(\n    "What was the temperature?"',
        "sa.query(\n    'What was the temperature?'",
        'sa.query_analyzed("What was the P99 latency on 2025-07-16?")',
        "sa.query_analyzed('What was the P99 latency on 2025-07-16?')",
        'pensive query --graph graph.pkl --analyze "What was the P99 latency on 2025-07-16?"',
    ]
    for phrase in forbidden:
        assert phrase not in text, (
            f"README re-introduced a known-broken NL example: {phrase!r}. "
            f"Pass-8 verified this returns [] silently (or "
            f"confidence='none' for query_analyzed). Use an entity-exact "
            f"query instead -- see Quickstart for the MegaExtractor "
            f"pattern that maps NL questions to entities."
        )


def test_readme_ingest_section_query_works():
    """Run the documented Ingestion-section query against the same
    fixture build set used by the Quickstart test. If the README's
    Ingestion query line returns [], the example will mislead users.
    """
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08',
         'value': '42ms on 2025-10-08'},
        {'id': '2', 'content': 'Build 1234 completed in 320s with rss 18GB',
         'value': '320s build, 18GB rss'},
        {'id': '3',
         'content': 'Meeting with Sarah Chen about Project Atlas budget',
         'value': 'Atlas budget meeting'},
    ])
    # The fixed README Ingestion example queries '2025-10-08' and
    # 'project atlas'. Both must return at least one hit.
    assert sa.query("2025-10-08"), (
        "README Ingestion example 'sa.query(\"2025-10-08\")' returns []"
    )
    assert sa.query("project atlas"), (
        "README Ingestion example 'sa.query(\"project atlas\")' returns []"
    )


def test_readme_cli_example_entity_resolves():
    """The README CLI section now documents `pensive query ... '2025-10-08'`.
    The shell wrapper calls sa.query() under the hood, so the underlying
    query must work against a comparable fixture.
    """
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08',
         'value': '42ms on 2025-10-08'},
    ])
    assert sa.query("2025-10-08"), (
        "README CLI example query '2025-10-08' returns []"
    )


def test_readme_context_disambiguation_example_works():
    """The Contextual Disambiguation section's in-section fixture and
    query must both produce a hit. The README's exact docs are:

      - Doc 1: GPU memory bandwidth was 199GB on Project Atlas
      - Doc 2: Project Atlas API P99 latency was 199ms

    The query is 'atlas' with context=['GPU', 'training run'].
    """
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1',
         'content': 'GPU memory bandwidth was 199GB on Project Atlas',
         'value': 'Atlas GPU 199GB bandwidth'},
        {'id': '2',
         'content': 'Project Atlas API P99 latency was 199ms',
         'value': 'Atlas API 199ms latency'},
    ])
    results = sa.query("atlas", context=["GPU", "training run"])
    assert results, (
        "README Contextual Disambiguation example returns [] -- the "
        "in-section fixture must yield hits for the documented query"
    )


def test_readme_query_analyzed_example_produces_non_none_confidence():
    """The Boundary Analysis section's query_analyzed example must
    return a non-'none' confidence. The pre-fix-6 form passed a
    natural-language string which gave confidence='none' and
    should_trust=False, misleading readers into thinking that's the
    expected default for a well-formed query.
    """
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1',
         'content': 'P99 latency was 199ms on 2025-07-16',
         'value': '199ms on 2025-07-16'},
    ])
    diagnosed = sa.query_analyzed("2025-07-16")
    assert diagnosed.analysis.confidence != "none", (
        f"README query_analyzed example yields confidence='none' "
        f"(should_trust={diagnosed.analysis.should_trust}); the "
        f"documented behavior is supposed to show a successful "
        f"diagnosis, not a no-result envelope"
    )


def test_readme_quickstart_examples_still_work():
    """Belt-and-suspenders: fix-5's Quickstart test
    (test_pass7_fix5_readme_quickstart.py) covers this, but a second
    assertion here means a future edit that breaks both files at
    once still fires here.
    """
    from pensive import SpreadingActivation

    sa = SpreadingActivation()
    sa.build([
        {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08',
         'value': '42ms on 2025-10-08'},
        {'id': '2', 'content': 'Build 1234 completed in 320s with rss 18GB',
         'value': '320s build, 18GB rss'},
        {'id': '3',
         'content': 'Meeting with Sarah Chen about Project Atlas budget',
         'value': 'Atlas budget meeting'},
    ])
    for q in ("42ms", "sarah chen", "2025-10-08"):
        assert sa.query(q), f"Quickstart query {q!r} returns []"


def test_readme_documents_entity_exact_semantics_outside_quickstart():
    """Beyond the Quickstart, every section that shows a sa.query()
    call should either (a) use a clearly-entity argument or (b)
    explain that the natural-looking string is itself an entity.
    Loose proxy: in the Ingestion, CLI, and Boundary Analysis
    sections, the only query forms should be plain entity strings or
    explicitly-extracted ones, and 'entity-exact' (or equivalent)
    should appear near at least one example outside Quickstart.
    """
    text = _readme()
    # Splits by H2 so we can scan section-by-section.
    sections = re.split(r'^## ', text, flags=re.MULTILINE)
    quickstart_idx = None
    for i, sec in enumerate(sections):
        if sec.startswith("Quickstart"):
            quickstart_idx = i
            break
    assert quickstart_idx is not None, "README must have a Quickstart section"

    other_sections = [s for i, s in enumerate(sections) if i != quickstart_idx]
    # At least one mention of "entity-exact" (or close variant) somewhere
    # outside the Quickstart, so users skimming any later section get the
    # signal that NL queries don't work.
    blob = "\n## ".join(other_sections).lower()
    assert (
        "entity-exact" in blob
        or "entity exact" in blob
        or "not parse natural language" in blob
        or "not a natural-language" in blob
    ), (
        "README sections beyond Quickstart must remind the reader that "
        "queries are entity-exact, since users skim and may not read "
        "Quickstart before the section they're looking at."
    )
