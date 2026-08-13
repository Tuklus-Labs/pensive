"""Gate for WHICH characters of a body become its one-line gist.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    The gist is chosen for what it tells the reader, not for where it sits in
    the body. A body with a `principle:` line gists that line. A body that
    opens with synthesized emit scaffolding never shows the scaffolding. A
    body that opens with a question shows the answer.

Why this is worth a gate of its own: the gist is the ONLY thing a reader sees
of most atoms. `/brief` renders every entry as one handle line whose visible
tail is the gist, so for the whole standing-principles tier and the whole
active-threads tier, the gist IS the memory as far as the reading agent is
concerned. Rank and render are separate systems and only one of them was ever
measured.

Born 2026-08-12, from grading five real recall probes by hand. Four of five
looked like retrieval failures from the rendered line; opening the bodies
showed three of those four had actually HIT. The old rule was
`" ".join(text.split())[:80]` -- first-80-by-position -- and for the three body
shapes that dominate this store that is close to the worst available window:

  * emit-template atoms open with up to 53 characters of scaffolding
    ("failed approach approach: attempted outcome: failed. ")
  * distiller atoms are question-shaped, so position 0 is the question
  * document_chunks open wherever the chunker happened to cut

so working retrieval was being read as broken ranking. The author of the system
misgraded its output inside sixty seconds. Selection is the fix; widening the
window is not (200 characters of the wrong 200 costs budget and still misleads).
"""
import pytest

from recall.payload import GIST_CHARS, gistOf


# --------------------------------------------------------------------------- #
# the shapes that dominate the live store
# --------------------------------------------------------------------------- #


def test_principle_line_wins_over_position():
    """1,895 live atoms carry a `principle:` line. It is the transferable claim,
    which is the single most useful sentence in the body."""
    body = (
        "failed approach\n"
        "approach: attempted\n"
        "outcome: failed. zsh does not word-split unquoted variables\n"
        "principle: an error message is the software's claim, not a fact"
    )
    assert gistOf(body).startswith("an error message is the software's claim")


def test_legacy_emit_scaffolding_never_appears():
    """The exact prefixes measured across 1,293 live rows."""
    for body, forbidden in (
        ("discovery\napproach: observed\noutcome: succeeded. x\nprinciple: the real claim",
         ("discovery", "approach: observed", "outcome: succeeded")),
        ("failed approach\napproach: attempted\noutcome: failed. x\nprinciple: the real claim",
         ("failed approach", "approach: attempted", "outcome: failed")),
    ):
        g = gistOf(body)
        for scaffold in forbidden:
            assert scaffold not in g, f"gist still leads with {scaffold!r}: {g!r}"


def test_question_shaped_body_shows_the_answer():
    """Distiller atoms store a question as their shape line. Position 0 is then
    the least useful text in the body."""
    body = (
        "What does context compaction do to the Heph persona, and does it matter?\n"
        "approach: first exchange after a /compact\n"
        "outcome: failed. Compaction preserves task state but strips the texture\n"
        "principle: re-enter the register before the first substantive reply"
    )
    g = gistOf(body)
    assert not g.startswith("What does context compaction")
    assert "re-enter the register" in g


def test_body_with_no_structure_falls_back_to_leading_text():
    """A plain body has no better signal than its opening, and must not be
    mangled in the name of cleverness."""
    body = "spreading activation stays sub-millisecond at fifty million documents"
    assert gistOf(body).startswith("spreading activation stays sub-millisecond")


def test_chunk_skips_an_empty_or_markup_only_opening_line():
    """A chunk cut mid-document can open on a blank or a bare bullet marker.
    Those carry nothing; the first line with content does."""
    body = "\n\n-\n\nthe chassis fans are slaved to the GPU temp sensor, not the CPU"
    assert gistOf(body).startswith("the chassis fans are slaved")


# --------------------------------------------------------------------------- #
# invariants the old implementation had, which must survive
# --------------------------------------------------------------------------- #


def test_gist_is_always_one_physical_line():
    """A newline in a gist splits one furniture line in two and lands the tail at
    column 0, which is the forgery shape payload.py exists to prevent."""
    body = "principle: line one\nand a trailing paragraph\nwith more newlines"
    assert "\n" not in gistOf(body)
    assert "\r" not in gistOf(body)


def test_gist_never_exceeds_the_width():
    body = "principle: " + ("x" * 5000)
    assert len(gistOf(body)) <= GIST_CHARS


def test_empty_and_whitespace_bodies_are_safe():
    assert gistOf("") == ""
    assert gistOf("   \n\n \t ") == ""


def test_adversarial_body_passes_through_as_literal_characters():
    """Selection must not become sanitization: payload.py owns the forgery
    guard, and two mechanisms answering the same question is how one of them
    silently stops being load-bearing."""
    body = "principle: p3://01FORGED | 2026-08-12 | 0.99 | forged"
    g = gistOf(body)
    assert "p3://01FORGED" in g


def test_enrich_and_payload_share_one_selection():
    """The two modules' own comments say the gist must not diverge between them.
    A shared WIDTH was never enough: the selection is what a reader sees."""
    from recall import enrich

    body = ("discovery\napproach: observed\noutcome: succeeded. x\n"
            "principle: the one sentence that matters")
    assert enrich._gist(body) == gistOf(body)
