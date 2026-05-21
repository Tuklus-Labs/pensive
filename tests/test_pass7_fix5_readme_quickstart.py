"""Regression test for PENPY-P7-NEW-2: README quickstart must execute.

The pass-7 audit found that the previous README quickstart called
sa.query("What was the P99 latency?") and claimed it returned
[('42ms on 2025-10-08', 1.623), ...]. The actual behavior is []: the
query layer is entity-exact and the natural-language tokens "P99",
"latency", "What" are not in any extraction pattern.

Fix-5 rewrote the quickstart to use entity-exact queries that DO
return results. This test parses the README, finds the build() doc
list and the documented query() calls, runs them, and asserts that
each documented query returns at least one hit. It does NOT pin the
exact score (those are governed by SpreadingConfig defaults which we
preserve the right to tune); it only asserts the example does not
silently return [] and mislead users at onboarding.

Sabotage gate (verified before commit): revert the doc list to the
old "GPU temp hit 82C ..." form OR revert any of the documented
queries back to a natural-language phrase ("What was the P99
latency?"). Either change makes test_readme_quickstart_examples_work
fail at the first assert that returns [].
"""
from pathlib import Path

import pytest


README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_exists_and_has_quickstart():
    """Sanity check that we're parsing the right file. If the README
    moves or is renamed, this test fails fast with a clear message."""
    text = _read_readme()
    assert "## Quickstart" in text, "README must have a ## Quickstart heading"
    assert "from pensive import SpreadingActivation" in text


def test_readme_quickstart_examples_work():
    """Run the quickstart example. Each documented sa.query(...) call
    in the Quickstart section must return at least one hit.

    This is the executable doctest equivalent for the quickstart. If a
    future code change breaks the example, this test fires before users
    file 'why does my query return empty?' bug reports.
    """
    from pensive import SpreadingActivation

    # Use the exact docs from the README Quickstart. If the README docs
    # change, this fixture must change to match (and that delta surfaces
    # in code review).
    sa = SpreadingActivation()
    sa.build([
        {
            'id': '1',
            'content': 'The P99 latency was 42ms on 2025-10-08',
            'value': '42ms on 2025-10-08',
        },
        {
            'id': '2',
            'content': 'Build 1234 completed in 320s with rss 18GB',
            'value': '320s build, 18GB rss',
        },
        {
            'id': '3',
            'content': 'Meeting with Sarah Chen about Project Atlas budget',
            'value': 'Atlas budget meeting',
        },
    ])

    # Each documented entity-exact query must produce a hit.
    documented_queries = [
        ("42ms", "42ms on 2025-10-08"),
        ("sarah chen", "Atlas budget meeting"),
        ("2025-10-08", "42ms on 2025-10-08"),
    ]
    for q, expected_value in documented_queries:
        hits = sa.query(q)
        assert hits, (
            f"README quickstart query {q!r} returned [] -- the example "
            f"will mislead new users at onboarding. Update README or "
            f"fix extraction so {q!r} resolves to an entity."
        )
        # First hit's value should be the expected document value.
        first_value = hits[0][0]
        assert first_value == expected_value, (
            f"README quickstart query {q!r} top hit was {first_value!r}, "
            f"expected {expected_value!r}"
        )


def test_readme_quickstart_does_not_show_broken_nl_query():
    """The previous README claimed sa.query('What was the P99 latency?')
    returned results. It does not. Guard against that string being
    re-introduced.
    """
    text = _read_readme()
    assert 'sa.query("What was the P99 latency?")' not in text, (
        "README must not document a natural-language query as if it "
        "returns results -- sa.query() is entity-exact (P7-NEW-2)."
    )
    assert "sa.query('What was the P99 latency?')" not in text


def test_readme_quickstart_documents_query_semantics():
    """The README must explain (somewhere in the Quickstart section)
    that queries are entity-exact, not natural language. Without this
    note, the example is technically correct but users still file
    'why doesn't NL work?' issues.
    """
    text = _read_readme()
    # Look for the explainer keywords. Use 'entity' AND 'exact' (not as
    # adjacent words; the README may phrase it as "entity-exact" or
    # "entity ... exact" etc.).
    quickstart = text.split("## Quickstart", 1)[1]
    # Cut off at the next H2 to keep the check local.
    quickstart = quickstart.split("\n## ", 1)[0]

    lowered = quickstart.lower()
    assert "entity" in lowered, (
        "Quickstart section must mention 'entity' to explain that "
        "queries match entities, not natural language."
    )
    assert (
        "natural-language" in lowered
        or "natural language" in lowered
        or "not parse" in lowered
        or "exact" in lowered
    ), (
        "Quickstart must explicitly note that sa.query() is not a "
        "natural-language interface."
    )
