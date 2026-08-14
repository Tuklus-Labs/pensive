"""Gate for document-frequency pruning of the lexical match expression.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    A term that appears in most of the corpus is dropped from the MATCH
    expression, because it costs almost all of the query's time and contributes
    almost none of its ranking. A query made ENTIRELY of such terms still
    returns something: the rarest term survives, so pruning can never turn a
    query into silence.

Why. `_sanitizeFtsQuery` OR-joins every token, which is a deliberate
recall-first choice, but it means one common word drags the whole corpus into
the scored set. Measured on the live store 2026-08-12:

    "the error-message reflex scar rule"          -> 180,501 matching rows
    "what does context compaction do to the ..."  -> 196,097 matching rows (61%)

and the cost decomposes as:

    MATCH only, LIMIT 64          0.3 ms
    MATCH + ORDER BY bm25(fts)   65.6 ms
    full production query       115.8 ms

The index is fast. The expense is entirely bm25 scoring, once per matching row,
because the sort key is a function and so the LIMIT cannot push down. Class-
splitting the index would not fix this: the cost tracks MATCH COUNT, not corpus
size.

Pruning by document frequency is the fix that does not trade quality for it. A
high-DF term has a near-zero IDF, so it moves a bm25 ranking barely at all while
multiplying the number of rows that must be scored. Dropping it is close to free
in ranking terms and enormous in latency terms -- which is a claim about THIS
corpus, so the tier gate measures R@10 and MRR@10 either side of the change
rather than taking the textbook's word for it.

Deliberately data-driven rather than an English stopword list: this corpus is
93% source code and technical prose, where the terms that saturate the index are
as likely to be `self`, `def` or `import` as `the`.
"""
import pytest

from recall.signals import (_sanitizeFtsQuery, ftsDocFreq, DF_PRUNE_FRACTION,
                            MIN_KEPT_TOKENS, MIN_CORPUS_FOR_PRUNING)
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path, monkeypatch):
    # The corpus guard exists for production scale; these fixtures are tiny by
    # necessity, so the guard is lowered HERE rather than weakened in the source.
    import recall.signals as S
    monkeypatch.setattr(S, "MIN_CORPUS_FOR_PRUNING", 10)
    # The production FRACTION is tuned for a 320k-row corpus; at 61 rows it makes
    # a term appearing once "saturating". Set a fraction meaningful at THIS scale
    # so the fixture exercises the mechanism rather than an artefact of its size.
    # The production value is asserted separately below.
    monkeypatch.setattr(S, "DF_PRUNE_FRACTION", 0.5)
    s = openStore(tmp_path / "mem.db")
    # 60 docs share "common"; one doc holds "phosphorescent".
    for i in range(60):
        putAtom(s, {"text": f"common shared filler token number {i}", "kind": "atom",
                    "project": "p", "importance": 0.0,
                    "provenance": {"source": "explicit-emit"}})
    putAtom(s, {"text": "a phosphorescent distinctive marker", "kind": "atom",
                "project": "p", "importance": 0.0,
                "provenance": {"source": "explicit-emit"}})
    try:
        yield s
    finally:
        s.close()


def test_document_frequency_is_readable(store):
    assert ftsDocFreq(store, "common") >= 60
    assert ftsDocFreq(store, "phosphorescent") == 1
    assert ftsDocFreq(store, "nevermentioned") == 0


def test_a_saturating_term_is_pruned(store):
    """"common" appears in ~98% of docs: it is pure cost."""
    expr = _sanitizeFtsQuery("common phosphorescent", store=store)
    assert "phosphorescent" in expr
    assert "common" not in expr, f"saturating term survived: {expr!r}"


def test_a_rare_term_is_always_kept(store):
    expr = _sanitizeFtsQuery("phosphorescent", store=store)
    assert "phosphorescent" in expr


def test_a_query_of_only_common_terms_is_not_pruned_to_nothing(store):
    """Pruning must never turn a query into silence. An empty MATCH expression
    would return zero rows, which is indistinguishable from an honest miss and
    would be a far worse failure than a slow query."""
    expr = _sanitizeFtsQuery("common shared filler", store=store)
    assert expr, "pruning emptied the query"
    kept = expr.count('"') // 2
    assert kept >= MIN_KEPT_TOKENS


def test_pruning_is_a_no_op_without_a_store(store):
    """The store argument is optional so every existing caller keeps working
    unchanged; DF pruning is opt-in at the call site."""
    assert _sanitizeFtsQuery("common phosphorescent") == '"common" OR "phosphorescent"'


def test_a_query_with_no_common_terms_is_unchanged(store):
    withStore = _sanitizeFtsQuery("phosphorescent distinctive", store=store)
    without = _sanitizeFtsQuery("phosphorescent distinctive")
    assert withStore == without


def test_empty_and_punctuation_only_still_return_none(store):
    assert _sanitizeFtsQuery("", store=store) is None
    assert _sanitizeFtsQuery("!!! ???", store=store) is None


def test_pruning_is_disabled_on_a_corpus_too_small_for_df_to_mean_anything(tmp_path):
    """A fraction threshold on a five-row store makes a term in ONE document
    "saturating". Below the guard, the expression must come back untouched."""
    s = openStore(tmp_path / "tiny.db")
    try:
        for i in range(5):
            putAtom(s, {"text": f"alpha beta gamma {i}", "kind": "atom",
                        "project": "p", "importance": 0.0,
                        "provenance": {"source": "explicit-emit"}})
        assert _sanitizeFtsQuery("alpha beta", store=s) == '"alpha" OR "beta"'
    finally:
        s.close()


def test_the_prune_threshold_is_a_fraction_not_a_count(store):
    """A hardcoded row count would silently stop pruning as the corpus grows,
    which is the failure mode where an instrument gets quieter over time."""
    assert 0.0 < DF_PRUNE_FRACTION < 1.0


# --------------------------------------------------------------------------- #
# cache lifetime                                                                #
# --------------------------------------------------------------------------- #


def test_live_row_count_cache_does_not_outlive_a_write(tmp_path):
    """THE CLAIM: corpus statistics memoized on the store must not survive a
    write to that store.

    The daemon holds ONE store for its entire lifetime, so a value computed at
    startup would describe the corpus forever. A clean-pass audit probed this on
    a temp store and watched the count freeze at 21 while the store grew past
    10,121, which left DF pruning permanently disabled because the corpus never
    appeared to reach MIN_CORPUS_FOR_PRUNING.

    That is worse than no cache: it is right during every test, where stores are
    small and short-lived, and wrong in production, where they are neither.
    """
    from store.store import openStore, putAtom
    from recall.signals import _liveRowCount

    store = openStore(tmp_path / "cache.db")
    try:
        for i in range(3):
            putAtom(store, {"text": f"seed atom {i}", "kind": "atom",
                            "project": "p", "provenance": {"source": "bulk-import"}})
        first = _liveRowCount(store)
        assert first == 3, f"expected 3 live rows, got {first}"

        for i in range(5):
            putAtom(store, {"text": f"later atom {i}", "kind": "atom",
                            "project": "p", "provenance": {"source": "bulk-import"}})
        second = _liveRowCount(store)
        assert second == 8, (
            "the memoized live-row count survived a write: it reports "
            f"{second} for a store holding 8 live atoms. DF pruning keys on this "
            "number, so a stale value disables pruning for the life of the daemon.")
    finally:
        store.close()
