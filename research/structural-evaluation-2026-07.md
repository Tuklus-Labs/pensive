# Pensive Structural Evaluation: Is Spreading Activation Load-Bearing?

Date: 2026-07-01. Method: first-hand kernel analysis, three evidence sweeps
(in-repo empirical claims, live serving path on Aegis, eval-corpus recon),
ranking ablations on the bench corpus, and a head-to-head retrieval eval on
the real ChatGPT export. Question under test: is spreading activation the
most effective retrieval core for Pensive, or a projection of how its
author thinks onto the graph?

## 1. What the shipped L1 actually computes

Strip the vocabulary and the bipartite "spread" is one algebraic statement:

    score(doc) = max over matched entities e of
                 boost_class(e) * specificity(e)^2 * decay * edge_weight

- specificity = 1/freq^0.2, an IDF analogue. It enters twice (seed and
  edge), so ranking is IDF-squared of the single rarest matched entity.
- max-pooling (numba kernel spreading.py:96-99, numpy fallback
  np.maximum.at :1040, :1165): a doc matching three query entities scores
  the same as a doc matching only the rarest one.
- decay multiplies every score uniformly in the single-hop bipartite case:
  zero effect on ranking, only on the threshold cutoff.
- hops 2+ are a documented no-op ("just apply uniform decay without
  changing rankings", spreading.py docstring).
- per-doc term frequency is deduplicated away at build (seen_in_doc,
  spreading.py:717-724). No TF signal exists.
- no recency: parsers capture a timestamp (ingestion/base.py:15) that
  never reaches the graph or the ranking. (AEGIS v1 had temporal half-life
  edge decay; the standalone extraction dropped it.)

Measured consequences (10k-doc bench corpus):

- Rank-equivalence: a 15-line posting-list scorer using only the entity
  vocabulary and per-entity constants reproduces query() output on
  168/200 entity queries; the 32 differences are tie-break arbitrariness
  under the internal max_active=50 cap, not ranking signal.
- Tie saturation: on single-entity queries (the documented primary use
  case), 30/30 sampled queries returned a top-10 in which every result
  had the same score. Within a posting list the engine has no opinion;
  result order is argpartition order.
- Discarded coordination: on 3-entity self-retrieval, sum-pooling beats
  the shipped max-pooling 169 wins to 0 (131 exact ties), MRR 0.982 vs
  0.503. The coordination signal exists in the graph and the kernel
  throws it away.

Conclusion of the math: the L1 is an exact-entity inverted index with
IDF^2-of-best-term scoring. "Activation" is a score, "spreading" is a
posting-list union, and the two biologically-motivated parameters (decay,
hops) are inert in the shipped configuration.

## 2. The parts that are NOT the inverted index

- context= intersection (spreading.py:1357-1365): docs activated by both
  query and context entities get multiplied by (1 + ctx score). This is
  the one real coordination mechanism, it does break the tie problem, and
  it is the basis of the 70.6% -> 97.9% claim. It is also plain
  posting-list math (soft AND), implementable without any graph.
- Boundary/confidence layer (boundary.py): boundary_distance = top score
  minus threshold; disambiguation_gap = top1 minus top2; band_crossing =
  IDF-quantile banding of matched entities; suggested_context = symmetric
  difference of the top-2 docs' entity sets. Genuinely novel as a
  trust layer; every metric is a posting-list or forward-index statistic.
  None requires a graph.
- L2 (FAISS + MiniLM) and BM25 hybrid: a different paradigm entirely,
  already in the package, already positioned above SA by the package's
  own fusion weights.

## 3. What the repo's own evidence says

- No in-repo evaluation compares SA against BM25, TF-IDF, dense, or a
  plain inverted index. None. The only "baseline" in boundary_bench is
  SA-without-context vs SA-with-context.
- The headline quality number (70.6% -> 97.9%) in the research paper
  cites the package's own PyPI page. No dataset, no query set, no
  definition of accuracy. The Dell benchmarks (n=4 and n=12, cases mined
  from Pensive's own output) measure the trust layer, not ranking, and
  say so themselves.
- test_performance.py asserts latency only. bench_spreading.py measures
  latency only. 103 commits of disciplined hardening; zero commits of
  retrieval-method experimentation.
- parallel_hybrid's fusion weights: AGREEMENT=100 > L2_ONLY=50 >
  SA_ONLY=30, L2 reranks SA candidates by default, and SA's own low
  confidence triggers abandoning SA candidates for global semantic
  search. The package already treats SA as its weakest signal.

## 4. What production on Aegis says

The live memory stack does not execute spreading activation at all.
pensive_recall = MiniLM-384 embed -> FAISS flat inner-product -> Python
substring post-filter (Engram/tools/pensive-mcp-server:271-329,
vector_service.py:699-771). The SA-importing modules (query_router,
ui_server) are not running; l1_queries has 0 rows; live analytics report
zero SA queries. The standalone pypensive library's SA core is imported
in production by nothing except its own benchmark; the only production
import of the package is the regex entity extractor, used at ingest.

The real query distribution (34 sampled invocations + corpus stats):
project-slug briefings and single-entity or short natural-language
semantic lookups. Nothing multi-hop, nothing SA-shaped.

Measured on 300 real human turns from the ChatGPT export: only 19% yield
ANY extractable entity under REAL_DATA_PATTERNS (mean 0.62, median 0).
The entity-exact contract caps L1 recall on the real query mix at ~19%
before ranking even starts.

Also: the synthetic-query pipeline is circular. Generated queries are the
doc's own highest-IDF tokens, and the pipeline concatenates the query
into the document text before extraction (spreading.py:782-783), so the
query's entities are indexed as edges to the target by construction. Any
eval using generated queries is tautological for L1 and inflated for
BM25. (The head-to-head below uses real human turns instead.)

## 5. Head-to-head on the real corpus

Corpus: the full ChatGPT export parsed by the production parser: 97,779
chunk docs (user and assistant turns), production graph build (211,059
nodes, generated queries included, which adds indexed entities and helps
L1). Queries: 1,500 real human turns sampled from 19,724 available
(question, next-assistant-answer) pairs, seed 42. Ground truth: the next
assistant turn's chunks. The query turn's own chunks are excluded from
every system's ranking. Dense = all-MiniLM-L6-v2, the production model.

Full sample, n=1500 (the real query distribution):

| system    |   R@1 |   R@5 |  R@10 |  R@20 | MRR@10 |
|-----------|-------|-------|-------|-------|--------|
| L1 direct | 0.007 | 0.022 | 0.033 | 0.053 |  0.013 |
| L1 readme | 0.004 | 0.013 | 0.017 | 0.026 |  0.008 |
| BM25      | 0.358 | 0.528 | 0.583 | 0.637 |  0.432 |
| dense     | 0.301 | 0.507 | 0.575 | 0.631 |  0.387 |

Answerable subset, n=252 (queries containing at least one extractable
entity; 16.8% of the sample, consistent with the 19% measured
independently on n=300):

| system    |   R@1 |   R@5 |  R@10 |  R@20 | MRR@10 |
|-----------|-------|-------|-------|-------|--------|
| L1 direct | 0.016 | 0.036 | 0.036 | 0.052 |  0.023 |
| L1 readme | 0.024 | 0.079 | 0.103 | 0.155 |  0.047 |
| BM25      | 0.381 | 0.544 | 0.635 | 0.714 |  0.459 |
| dense     | 0.325 | 0.540 | 0.639 | 0.718 |  0.423 |

Reading:

- On the real query mix, BM25 retrieves the answer in its top-10 for
  58% of questions; the spreading-activation engine manages 3.3%. That
  is a 17x gap, and it is mostly the entity-exact contract: 83% of real
  questions contain nothing the extractor can see.
- The gap survives on L1's home turf. Restricted to entity-bearing
  queries and using the README-documented extract-then-query path, L1
  reaches R@10 = 0.103 while BM25 reaches 0.635 on the identical
  queries. Six to one, with no extractability excuse. Max-pooling,
  IDF-only scoring, no TF, no coordination: the ranking model itself
  is the remaining deficit.
- The two L1 variants flip between scopes: direct raw-text querying
  beats the README path on the full sample (stray word-to-entity
  matches recover a little signal) and loses on the answerable subset
  (where deliberate extraction focuses the query). Neither variant is
  within an order of magnitude of a 25-line BM25.
- BM25 edges dense at R@1 on this corpus (0.358 vs 0.301): question and
  answer share vocabulary in chat logs. The production choice of pure
  embedding search is defensible but a BM25+dense hybrid would beat
  both, which is what the package's own HybridSearcher already
  implements, without SA.

Biases stated: ground truth is positional (next assistant turn); query
text is chunk-0 of the human turn (~800 char cap); the graph includes
generated-query entities (favorable to L1); relevance credit requires
hitting the exact answer turn, so all numbers understate absolute
usefulness but compare systems fairly.

## 6. Verdict

**Is spreading activation the most effective approach? No, and three
independent lines of evidence agree:**

1. Mathematically, the shipped mechanism is an inverted index with
   IDF^2-max scoring whose distinctive parameters are inert. It discards
   coordination (169-0), has no within-list ranking signal (100% ties),
   no TF, no recency.
2. The package's own architecture already demoted it: fusion weights rank
   semantic-only above SA-only, L2 reranks SA candidates, low SA
   confidence triggers semantic rescue.
3. Production on this machine routed around it entirely: the live recall
   path is pure embedding search, SA serves zero queries, and 83% of real
   queries contain nothing the entity extractor can see. Measured
   head-to-head on the real corpus, BM25 beats L1 17x on the full mix
   and 6x on L1's own entity-bearing subset.

**Is this projection? Partially, and precisely locatable.** Max-pooling
IS single-cue associative recall: the strongest cue wins and evidence
does not accumulate. That is a specific, human-feeling theory of memory,
and it matches how an RF operator recalls: one sharp discriminative cue
(a date, an error string, a callsign) keys the episode. For single-cue
entity queries the model is fine, and cue rarity is the right prior. The
projection fails at the edges: multi-cue queries (where evidence should
accumulate and does not), vague queries (where there is no entity at
all: 83% of the real mix), and tie-breaking (where a memory should
prefer recent or contextually consistent episodes and this one shrugs).
The biology in the pitch (spreading, decay, hops) is exactly the part
the code disables. What survived into production, embedding search plus
the trust layer, is the part that was never biological.

The strongest thing in this repo is not the paradigm. It is the
engineering discipline around it (the concurrency model, the audit
passes, the honest entity-exact contract) and the boundary/confidence
layer, which is novel, useful, and paradigm-independent: it would sit
just as naturally on top of BM25 or FAISS scores.

## 7. What would make the graph earn its name (falsifiable, in order)

1. **Sum-pooling (or a max/sum blend) in the kernel.** One-line change,
   restores coordination, up to ~2x MRR on multi-cue queries by the
   ablation. Cheapest possible test of whether SA-as-shipped is
   underselling itself. (~50-100k tokens including tests.)
2. **Break the ties.** Within a posting list, rank by recency (the
   timestamp already exists at ingest and is discarded) or by context
   co-activation by default. A memory system whose within-cue ordering
   is argpartition noise is leaving its purpose on the floor.
   (~100-200k tokens.)
3. **BM25-over-entities baseline in-repo.** Same extractor, same
   vocabulary, sum + TF saturation + length norm. If it beats L1 on the
   real-corpus eval, the graph is decoration; if it loses, SA has a real
   defense. Either result is worth having in research/. (~100k tokens.)
4. **The 2-hop experiment.** Add value->entity back-edges and measure
   whether doc->entity->doc association surfaces relevant docs sharing
   zero query entities on the real corpus. This is THE test of graph-ness:
   multi-hop association is the one capability an inverted index cannot
   fake. If it produces measurable recall lift, the paradigm earns its
   name; if it produces noise (the classic spreading-activation fan-out
   problem), rename L1 "entity index" and stop apologizing for it.
   (~200-400k tokens.)
5. **Wire the trust layer to what actually runs.** boundary.py's
   confidence metrics are computable from FAISS/BM25 scores. The most
   valuable IP in this repo is currently attached to the engine that
   serves zero queries. (~150-300k tokens.)
