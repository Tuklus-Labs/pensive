"""Three parallel recall signals: lexical (BM25), dense (cosine), and facet/temporal.

Task 7 fuses these with Reciprocal Rank Fusion; this module only *produces* the
per-signal rankings and the facet hint sets. No fusion, no reranking here.

Two conventions hold across all three signals:

- **Higher = better, always.** SQLite's FTS5 ``bm25()`` is a *cost* (lower is a
  better match); we negate it so lexical, dense (cosine, already higher-better),
  and any downstream trust layer read one sign convention.
- **Live atoms only.** ``FlatIndex`` filters to ``status='live'`` at build time,
  so the lexical signal must agree on the same candidate universe or the two
  signals disagree about which atoms even exist. ``bm25`` joins the FTS rowid
  back to ``atoms`` and filters ``status='live'`` *at query time* -- an atom
  superseded after it was indexed is excluded even though its text is still in
  the FTS index. Task 9's trust layer handles superseded-atom chaining from the
  edge graph; the signals never surface a non-live atom.

FTS5 query safety (correctness requirement): the recall engine feeds raw user
text straight into an FTS5 ``MATCH``. Unescaped quotes, hyphens, parens, a lone
``*``, and the operator words ``AND``/``OR``/``NOT``/``NEAR`` are all FTS5 query
grammar -- raw text containing them is a *syntax error*, not a no-op. ``bm25``
must never throw on arbitrary text, so it tokenizes the query to word runs,
quotes each token as an FTS5 phrase literal (neutralizing every operator), and
joins them with ``OR``. OR (disjunction) is deliberate: it matches BM25's usual
bag-of-words semantics and maximizes *recall* at this candidate-generation
stage, where downstream fusion (Task 7) and reranking (Task 8) supply precision.

Facet entity value format: entity facets are stored as ``(key='entity',
value=<lowercase surface form>)`` -- the label ``MegaExtractor.extract`` returns
in position 0. ``facetSignal`` extracts entities from the query the same way and
matches on that value, so the write side and the read side share one format.
"""
import re
import sys
from pathlib import Path

__all__ = ["bm25", "dense", "facetSignal"]

# The v2 entity extractor lives in THIS repo's local tree at <repo>/src/pensive,
# not in the daemon package. Import the local copy, never a pip-installed
# pypensive that may have drifted from the vendored source. The test conftest
# puts daemon/src on sys.path; we prepend <repo>/src here so `import pensive`
# resolves to the local tree first. parents: recall -> src -> daemon -> <repo>.
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from pensive.mega_extract import MegaExtractor  # noqa: E402  (path set above)
from pensive.patterns import REAL_DATA_PATTERNS, build_pattern_set  # noqa: E402
from recall.strata import kindInClause  # noqa: E402  (path set above)

# Default recall breadth: the plan's recall stage takes the top 200 per signal.
_DEFAULT_K = 200

# FTS5's unicode61 tokenizer splits on non-alphanumerics; matching that with a
# word-run regex keeps our phrase tokens aligned with how the index tokenized
# the stored text. Unicode-aware by default for str patterns (matches café).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Cap the sanitized MATCH expression at the first N tokens. Task 18's drift
# watcher posts conversation TAILS into recall as raw queries, so multi-thousand-
# token inputs are a designed-in case, not an edge. An unbounded "t" OR "t" ...
# expression is the one remaining hole in the never-throws guarantee: FTS5 caps
# the phrase-term count per MATCH expression (build/config dependent -- this box
# tolerates 100k, a stricter build does not), and thousands of OR terms are
# pointless for candidate-stage recall regardless. Take the FIRST N tokens (order
# preserved) so the leading/early tokens still drive the match.
_MAX_QUERY_TOKENS = 64

# Pattern compilation is not free, so build the extractor once, lazily -- Task 7
# imports this module for bm25/dense without ever touching facetSignal, and
# should not pay the compile cost on import.
_extractor = None


def _getExtractor():
    global _extractor
    if _extractor is None:
        _extractor = MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))
    return _extractor


# A term appearing in more than this FRACTION of live rows is dropped from the
# MATCH expression. A fraction, not a row count, because a hardcoded count stops
# pruning as the corpus grows -- the failure mode where an instrument silently
# gets quieter over time.
#
# Measured on the live store 2026-08-12: "the" matched 180,501 of ~320k rows and
# "what"/"do"/"to" pushed one query to 196,097 (61% of the store). The cost of a
# lexical query tracks MATCH COUNT, because ORDER BY bm25(fts) scores every
# matching row (the sort key is a function, so LIMIT cannot push down): MATCH
# alone is 0.3ms, MATCH + ORDER BY bm25 is 65.6ms.
DF_PRUNE_FRACTION = 0.01

# Never prune below this many tokens. A query made entirely of saturating terms
# must still ask something: an empty MATCH expression returns zero rows, which is
# indistinguishable from an honest miss and is a worse failure than a slow query.
MIN_KEPT_TOKENS = 1

# Below this many live rows, pruning is DISABLED entirely.
#
# A fraction threshold is meaningless on a small corpus: in a five-atom store,
# 1% is 0.05, so a term appearing in ONE document is "saturating" and every
# query collapses to a single token. Six existing tests caught exactly that.
# The guard is not a workaround for the tests, it is the honest scope of the
# optimization: bm25 over a few thousand rows is already milliseconds, so there
# is no cost here to remove and no DF signal worth trusting.
MIN_CORPUS_FOR_PRUNING = 10_000


def _ensureVocab(store):
    """Create the fts5vocab shadow table ONCE per store connection.

    It was created per lookup, i.e. once per token per query, and a schema
    statement on every token put a ~10ms floor under every lexical query --
    an optimization paying more than the cost it removed. Guarded by an
    attribute on the store so the statement runs once and never again.
    """
    if getattr(store, "_ftsVocabReady", False):
        return True
    try:
        store._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_vocab USING fts5vocab('fts','row')"
        )
    except Exception:
        store._ftsVocabReady = False
        return False
    store._ftsVocabReady = True
    return True


def ftsDocFreq(store, term):
    """Document frequency of ``term`` in the FTS index, or 0 if absent.

    Backed by an ``fts5vocab`` shadow table and memoized per store: term
    frequencies move only on write, and a query-time pruning decision that costs
    a query defeats itself. The whole point is to spend microseconds deciding
    not to spend milliseconds.
    """
    if not _ensureVocab(store):
        # A vocab table we cannot build or read means we cannot prune. Fail
        # toward the SLOW-but-correct query rather than toward a pruned one we
        # cannot justify: an optimization that cannot verify its own premise
        # must not fire.
        return 0
    cache = getattr(store, "_dfCache", None)
    if cache is None:
        cache = {}
        store._dfCache = cache
    key = term.lower()
    if key in cache:
        return cache[key]
    try:
        row = store._conn.execute(
            "SELECT doc FROM fts_vocab WHERE term = ?", (key,)
        ).fetchone()
    except Exception:
        return 0
    value = row[0] if row else 0
    cache[key] = value
    return value


def invalidateSignalCaches(store):
    """Drop the memoized DF and live-row-count caches for ``store``.

    Called by the write paths. WITHOUT THIS the caches never expire: the daemon
    holds one store for its entire lifetime, so a value computed at startup
    describes the corpus forever. Probed on a temp store during a clean-pass
    audit, the count froze at 21 while the store grew past 10,121, which left DF
    pruning permanently disabled because the corpus never appeared to reach
    MIN_CORPUS_FOR_PRUNING. A cache that is correct only until the first write is
    worse than no cache, because it is right during every test and wrong in
    production.
    """
    for attr in ("_dfCache", "_liveRowCountCache"):
        if hasattr(store, attr):
            delattr(store, attr)


def _liveRowCount(store):
    """Live row count, memoized per store: a COUNT(*) over 320k rows on every
    query is the same self-defeating shape as the schema statement was.

    Invalidated by :func:`invalidateSignalCaches` on write."""
    cached = getattr(store, "_liveRowCountCache", None)
    if cached is not None:
        return cached
    row = store._conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE status = 'live'"
    ).fetchone()
    value = row[0] if row else 0
    store._liveRowCountCache = value
    return value


def _pruneSaturatingTokens(store, tokens):
    """Drop tokens whose document frequency exceeds the prune fraction.

    Keeps the ``MIN_KEPT_TOKENS`` rarest tokens no matter what, so a query of
    entirely common words still asks a question. Returns the tokens in their
    ORIGINAL order, because FTS5 phrase order is part of the expression and
    reordering it would be a silent semantic change on top of a performance one.
    """
    total = _liveRowCount(store)
    if total < MIN_CORPUS_FOR_PRUNING:
        return tokens
    ceiling = total * DF_PRUNE_FRACTION
    freqs = {t: ftsDocFreq(store, t) for t in set(tokens)}
    kept = [t for t in tokens if freqs.get(t, 0) <= ceiling]
    if len(kept) >= MIN_KEPT_TOKENS:
        return kept
    # Everything saturates: keep the rarest, preserving original order.
    rarest = sorted(set(tokens), key=lambda t: freqs.get(t, 0))[:MIN_KEPT_TOKENS]
    return [t for t in tokens if t in set(rarest)][:MIN_KEPT_TOKENS]


def _sanitizeFtsQuery(query, store=None):
    """Turn arbitrary text into a safe FTS5 MATCH expression, or None if empty.

    Tokenizes to word runs and wraps each in double quotes (an FTS5 phrase
    literal), doubling any embedded quote for safety even though ``\\w+`` never
    yields one. Operators (AND/OR/NOT/NEAR), hyphens, parens, and ``*`` inside a
    quoted phrase are literal text, so the result can never be an FTS5 syntax
    error. Tokens are joined with ``OR`` (see module docstring: recall-first) and
    capped at the first ``_MAX_QUERY_TOKENS`` so a huge paste cannot blow the
    FTS5 expression limit. Returns None when there is no token at all (empty,
    whitespace, or punctuation/emoji-only input) so the caller returns [] without
    querying.
    """
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return None
    if len(tokens) > _MAX_QUERY_TOKENS:
        tokens = tokens[:_MAX_QUERY_TOKENS]
    # DF pruning is opt-in at the call site: a caller with no store keeps the
    # previous behaviour byte for byte.
    if store is not None:
        tokens = _pruneSaturatingTokens(store, tokens)
        if not tokens:
            return None
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " OR ".join(quoted)


def bm25(store, query, k=_DEFAULT_K, kinds=None, agent=None):
    """Lexical signal: FTS5 BM25 over live atom text -> ``[(atomId, score)]``.

    ``score`` is the negated ``bm25()`` cost, so higher = better and the list is
    already best-first. ``kinds`` (when a non-empty iterable) restricts the result
    to atoms of those kinds via an ``AND a.kind IN (...)`` clause, so the engine
    can pull a separate per-class candidate list; ``kinds=None`` is unrestricted
    (every live atom). ``agent`` (str or sequence) restricts to atoms that carry
    at least one matching provenance.agent row, so the top-k is drawn FROM that
    agent's universe rather than post-filtered from a global 200. Returns [] for
    ``k <= 0`` or a query with no searchable token (parity with the vector-index
    contract; never raises on raw text).
    """
    if k <= 0:
        return []
    match = _sanitizeFtsQuery(query, store=store)
    if match is None:
        return []
    kindClause, kindParams = kindInClause(kinds, alias="a")
    agentClause, agentParams = "", ()
    if agent:
        if isinstance(agent, str):
            wanted = (agent,)
        else:
            wanted = tuple(a for a in agent if a)
        if wanted:
            ph = ",".join("?" for _ in wanted)
            agentClause = (
                " AND a.id IN (SELECT atom_id FROM provenance "
                f"WHERE agent IN ({ph}))"
            )
            agentParams = wanted
    rows = store._conn.execute(
        "SELECT a.id, -bm25(fts) AS score "
        "FROM fts JOIN atoms a ON a.rowid = fts.rowid "
        "WHERE fts MATCH ? AND a.status = 'live'" + kindClause + agentClause + " "
        "ORDER BY bm25(fts) "
        "LIMIT ?",
        (match, *kindParams, *agentParams, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def dense(index, embedder, query, k=_DEFAULT_K):
    """Dense signal: embed the query once, delegate to ``index.search(vec, k)``.

    The embedding cost is one ``embed`` call per query; ranking and the
    live-only candidate universe are the index's responsibility (FlatIndex here,
    the Task 15 HNSW index later), so this is a thin, index-agnostic seam.
    """
    vecs = embedder.embed([query])
    if not vecs:
        return []
    return index.search(vecs[0], k)


def facetSignal(store, hints):
    """Structured hints -> ``{boostSet, filterSet}`` over live atoms.

    ``hints`` is a dict; every key is optional:

    - ``project`` (str): restrict recall to that project's live atoms.
    - ``timeRange`` ((start, end) unix seconds): restrict to live atoms whose
      ``COALESCE(occurred_at, created_at)`` falls in the inclusive window --
      occurred_at (when the remembered thing happened) wins, created_at is the
      fallback. A degenerate window with ``start > end`` matches nothing and
      yields an *empty* filterSet (not None): an impossible window is a real
      "filter to nothing", distinct from "no filter at all".
    - ``query`` (str): raw query text; entities extracted from it via the v2
      ``MegaExtractor`` boost the live atoms carrying a matching entity facet.

    ``filterSet`` is the intersection of the project and time constraints, or
    ``None`` when neither is given (meaning "do not filter"). ``boostSet`` is a
    set of atom ids to weight up; empty when the query yields no entity -- the
    common case (~83% of real queries have none), and never an error.

    Both sets contain live atoms only, keeping parity with bm25/dense.
    """
    hints = hints or {}
    conn = store._conn

    # filterSet: intersection of the constraints that are present.
    constraints = []

    project = hints.get("project")
    if project is not None:
        rows = conn.execute(
            "SELECT id FROM atoms WHERE project = ? AND status = 'live'",
            (project,),
        ).fetchall()
        constraints.append({r[0] for r in rows})

    timeRange = hints.get("timeRange")
    if timeRange is not None:
        start, end = timeRange
        rows = conn.execute(
            "SELECT id FROM atoms "
            "WHERE COALESCE(occurred_at, created_at) BETWEEN ? AND ? "
            "AND status = 'live'",
            (start, end),
        ).fetchall()
        constraints.append({r[0] for r in rows})

    filterSet = set.intersection(*constraints) if constraints else None

    # boostSet: live atoms whose entity facets match entities in the query text.
    boostSet = set()
    query = hints.get("query")
    if query:
        labels = {label for label, _etype in _getExtractor().extract(query)}
        if labels:
            placeholders = ",".join("?" for _ in labels)
            rows = conn.execute(
                "SELECT DISTINCT f.atom_id FROM facets f "
                "JOIN atoms a ON a.id = f.atom_id "
                "WHERE f.key = 'entity' AND a.status = 'live' "
                f"AND f.value IN ({placeholders})",
                tuple(labels),
            ).fetchall()
            boostSet = {r[0] for r in rows}

    return {"boostSet": boostSet, "filterSet": filterSet}
