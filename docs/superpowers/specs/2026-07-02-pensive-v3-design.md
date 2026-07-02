# Pensive v3 Design

Date: 2026-07-02. Status: draft for Gary's review.
Basis: the 2026-07-01 structural evaluation
(`research/structural-evaluation-2026-07.md`) and the brainstorming
session of 2026-07-02.

## 1. North star

Pensive is a memory state that ingests and recalls over decades, not a
document retriever. The outcome that matters: an agent partner that
carries years of accumulated context (projects, principles, people,
history) and surfaces the right piece of it at the right moment without
being asked, with honest confidence, at a token cost the context window
can afford.

Four properties define "done correctly":

- **Fast**: recall never makes an agent wait. p95 under 150 ms warm,
  end to end, rerank included.
- **Accurate**: measured on real queries against real ground truth, not
  synthetic self-evals. The eval harness is the arbiter.
- **Semantic**: natural-language queries just work. No entity-exact
  contract, no query syntax. Meaning, not surface form.
- **Invisible**: memory behaves like memory. It briefs at session
  start, surfaces mid-conversation when relevant, and captures what
  mattered without being told. Explicit tools exist and are rarely
  needed.

And one property that outranks the other four when they conflict:
**decades**. The store outlives models, machines, and runtimes. Nothing
canonical is ever unrecoverable. Time is a first-class dimension.

## 2. What v2 taught us (constraints from evidence)

- Hybrid lexical+dense is the floor: BM25 R@10 0.583 and dense 0.575 on
  1,500 real queries, with different failure modes. Fusion of the two
  plus a cross-encoder rerank is the strongest known core on this
  corpus. Spreading activation scored 0.033 and is not the core.
- 83% of real queries carry no regex-extractable entity. Entity
  extraction is an ingest-side enrichment, never the recall gate.
- The trust layer (confidence, should_trust, disambiguation gap) is
  paradigm-independent and valuable. It ports onto fusion evidence.
- Recency was captured and discarded in v2. In v3 time is structural.
- Graph roles are separated: data structure (keep), typed relationship
  data (keep, load-bearing), ranking mechanism (dead), associative
  expansion (evidence-gated experiment, section 11).

## 3. Scope

**v3.0 ships:** canonical store + schema, recall engine (fusion +
rerank + trust), MCP shadow serving with today's tool names, the
briefer, the distiller (Claude Code + Codex sources), a minimal drift
watcher, double-write from the current emit flow, the eval harness with
cutover gates, backup/restore and integrity checks.

**v3.1 and later:** auto-consolidation jobs (Dream integration), bulk
historical ingest (ChatGPT/Claude/email exports through the distiller
pipeline), Hermes as a capture source, the 2-hop association signal if
the spike earns it, IVF/HNSW migration when corpus growth demands it.

**Explicitly out:** PyPI packaging as a goal (may happen later, not a
driver), any cloud dependency, NL query parsing gymnastics (the recall
engine takes raw text), DIY auth (localhost/tailscale now; Authelia
forward-auth if ever web-exposed).

## 4. System shape

One long-lived daemon (`pensived`, working name) owning five loops, one
store, one serving surface.

```
                      +--------------------------------------+
  CC/Codex agents --->|  MCP server (pensive_recall, emits,  |
  hooks, CLI, cron -->|  v3 natives)   HTTP for hooks/CLI    |
                      +-------------------+------------------+
                                          |
                      +-------------------v------------------+
                      |             recall engine            |
                      |  FTS5-BM25 || dense ANN || facets    |
                      |  -> RRF fusion -> rerank -> trust    |
                      +-------------------+------------------+
                                          |
   distiller ---------+                   |
   (CC session-replay,|   +---------------v----------------+
    Codex logs)       +-->|        canonical store         |
   briefer <--------------|  SQLite: atoms, edges, prov,   |
   drift watcher <--------|  facets, FTS5, embeddings(model)|
   lifecycle jobs <------>|  (embeddings = derived data)    |
                          +--------------------------------+
```

Runtime: open trade, settled by a one-day spike at the top of the
implementation plan (section 13). Candidates in preference order per
Gary's language rule: (a) Go daemon with ONNX Runtime in-process for
embedding + rerank, (b) TS/Bun daemon same shape, (c) documented-
exception Python. The spike criterion: bge-small embedding and
bge-reranker scoring working end to end on this box with acceptable
latency, whichever candidate gets there without heroics wins. The
design below is runtime-neutral; SQLite and ONNX have first-class
bindings in all three.

## 5. Canonical store (the decades part)

SQLite, WAL mode, one file plus derived indexes. Chosen for archival
properties: documented stable format, single-file portability across
machines and decades, and rebuildability. Everything derived
(embeddings, ANN graphs, FTS) can be regenerated from the canonical
tables alone; a `pensive rebuild` command proves it and is part of the
test suite.

Tables (schema_version stamped, migrations forward-only):

- **atoms**: id (ulid), text, kind (atom | narrative | snapshot |
  document_chunk), project, created_at, occurred_at (nullable: when the
  remembered thing happened, vs when recorded), importance (float,
  earned; see lifecycle), status (live | superseded | tombstone),
  schema_version.
- **provenance**: atom_id, source (claude-code | codex | explicit-emit
  | bulk-import | distiller), session_id, agent, source_ref (transcript
  span / file / message id), recorded_at. Permanent; never pruned.
- **edges**: src_atom, dst_atom, type (supersedes | contradicts |
  causes | relates | same_thread), weight, created_at, provenance_id.
  Typed relationships are load-bearing: supersession drives trust and
  lifecycle; causes serves the causal-tracker integration; relates and
  same_thread serve consolidation and the association experiment.
- **facets**: atom_id, key, value (project, entity:<type>:<label>,
  tag, era). Entities arrive here via the v2 MegaExtractor at ingest
  (it survives as enrichment, with the FIRST-set guard perf fix).
- **embeddings**: atom_id, model_id, vector (blob), embedded_at.
  Keyed by model so re-embedding is additive: new model rows are
  written by an idle-GPU migration job, the recall engine cuts over
  per-model atomically, old rows are dropped only after the eval
  harness passes on the new space. Embeddings are cattle; text is pet.
- **fts** (FTS5 contentless, synced by trigger): lexical index over
  atom text. BM25 comes with FTS5; no extra dependency.

Vector search: start with a flat matrix scan per model (fine to ~500k
atoms at 384d on this hardware, and honest about it), behind a narrow
`VectorIndex` interface whose second implementation is usearch-HNSW.
The interface is in v3.0; the HNSW build triggers automatically past a
corpus-size threshold. No load-bearing path may assume flat.

Durability ops: nightly `VACUUM INTO` snapshot to a rotation, WAL
checkpoint on idle, an integrity job (Immune-style: checksums, orphan
edges, embedding/model coverage) that reports to nerve-center, and
`pensive export` producing a plain-JSONL dump of canonical tables.
Restore and rebuild are tested, not aspirational.

## 6. Recall engine

Input: raw query text plus optional structured hints (project, time
range, kinds, k, token_budget). No query language.

1. **Signals in parallel:**
   - FTS5 BM25 over text (top 200).
   - Dense cosine over the active embedding model (top 200).
   - Facet/temporal prefilter or boost: project match, explicit time
     range, entity facet hits (when the query happens to contain one,
     it helps; when it doesn't, nothing breaks).
2. **Fusion:** reciprocal-rank fusion over the signal lists. Weights
   start uniform; the eval harness owns tuning them.
3. **Priors:** fused scores modulated by importance and by a
   *two-sided* time prior: mild recency boost for same-thread
   freshness, but floor-protected so age alone never buries a
   high-importance atom. Time-scoped queries replace the prior with
   the requested window.
4. **Rerank:** bge-reranker (cross-encoder, on disk today) rescores
   the fused top-50 against the raw query. This is the single largest
   quality lever above fusion and it is local and cheap at k=50.
5. **Trust layer:** v2's boundary math ported to fusion evidence:
   signal agreement (both lexical and dense vs one), score gaps
   (disambiguation), band statistics, plus the temporal dimension:
   a superseded atom surfaces only chained to its successor, and
   confidence decays when an atom's facts are old and unconfirmed.
   Output per result: confidence (0-1), should_trust, why (one short
   plain-text reason: "both signals agree, recent, unchallenged").
6. **Payload assembly:** section 8.

Latency budget (warm, p95): signals 30 ms, fusion+priors 5 ms, rerank
80 ms, trust+assembly 10 ms. Cold start (model load) hidden behind
daemon residency.

## 7. Ambient loops

**Briefer** (session start): assembles the working set for the session:
standing principles and profile atoms (pinned), active threads by
recency+importance, project forecast context (Kairos integration as
today), open loose ends addressed to the starting agent. Feeds the
SessionStart hook. Budgeted (default 1,500 tokens), tiered per section
8. The working set is a *view*: pins, recency, importance, and usage
compute it; nothing moves or mutates in the store to produce it.

**Drift watcher** (mid-session): the harness hook posts the
conversation tail (last N exchanges, hashed for dedup) every M tool
calls; the watcher embeds it, queries the recall engine with a high
confidence floor, and injects at most one compact payload per cooldown
window when similarity crosses threshold and the hit is not already in
context. Token-budgeted and rate-limited by design; the failure mode to
engineer against is chattiness, so the v3.0 version is deliberately
conservative (long cooldown, high floor) with tuning owned by the eval
harness's injection-precision suite.

**Distiller** (capture): tails Claude Code session-replay and Codex
session logs. Two-stage: (1) cheap segmentation of new transcript
deltas into candidate spans (decisions, discoveries, failures,
corrections; heuristics + explicit emit passthrough), (2) a local-model
pass (ornith/Hermes stack, batched, idle-priority) that writes atoms in
the house format: concrete, transferable, plain register, provenance
attached. Dedup before write: embedding similarity against recent atoms
plus same-provenance suppression; near-duplicates strengthen importance
instead of duplicating. Explicit emits (engram_emit_*) bypass stage 2:
they are already curated, and remain the highest-trust source.
Distiller lag target: under 5 minutes behind live.

## 8. Payload format (the no-slop contract)

Plain text, tiered, budgeted. No markdown furniture, no headers, no
bullets-for-the-sake-of-bullets, no emoji.

- **Tier 0, handle line** (~25 tokens):
  `p3://01J... | 2026-05-12 | 0.91 | turboquant crashes on MLA models; clamp key_length 576`
- **Tier 1, atom body** (the stored text, typically 50-200 tokens),
  provenance one-liner appended.
- **Tier 2, neighborhood**: the atom plus its live edges (what
  superseded what, what it caused), for "tell me the history" asks.

Every recall answer states its confidence and, below the floor, says
"low confidence" instead of padding with weak hits. k results never
exceed the caller's token_budget; the engine drops tail hits rather
than truncating atom bodies mid-sentence.

## 9. Serving and compatibility

MCP server exposing today's names (`pensive_recall`,
`engram_emit_atom/discovery/failure/narrative/snapshot`,
`pensive_analytics`) with identical shapes during shadow, plus v3
natives (`recall` with hints/budget, `history` for edge chains,
`correct` for supersession, `pin`). HTTP endpoints for hooks, CLI, and
cron. The `pensive-recall` CLI keeps working unchanged through cutover.

## 10. Parallel run and cutover

- **Double-write:** a tee at the emit path writes to old and new
  stores from day one. Backfill of the existing ~17k atoms +
  narratives through the v3 ingest (re-embedded, provenance mapped,
  `src:` tags preserved as facets).
- **Shadow serve:** the MCP server answers from the old path while
  logging v3's answer for the same query. A/B log is the live half of
  the eval gate.
- **Cutover gates (all must pass):**
  - Real-query gate: R@10 >= 0.65 and MRR@10 >= 0.45 on the harness
    (BM25 baseline 0.583 / 0.432), on both the atom corpus and the
    chat-export benchmark.
  - Shadow win rate: v3 top-3 judged better-or-equal on >= 80% of live
    shadow queries (sampled, local-model judged, spot-checked by Gary).
  - Temporal suite: time-scoped queries resolve to the correct era;
    superseded atoms never surface unchained.
  - Latency: p95 <= 150 ms warm recall; briefing <= 2 s.
  - Re-embed drill: model swap end to end with quality within 5% and
    zero canonical loss.
  - Ops drill: kill -9 mid-write recovers clean; backup/restore and
    `pensive rebuild` round-trip byte-identical canon.
- **Cutover:** flip MCP backend, old services stay up read-only for
  two weeks, then retire. Rollback is a config flip while double-write
  continues.

## 11. The association experiment (graph as ranker, evidence-gated)

Built inside the eval harness, not the serving path: value->entity
back-edges over the facet table, specificity-damped 2-hop walk seeded
by the fused top-k (not by query entities, which don't exist for 83% of
queries), producing an expansion candidate list. Measured as a fourth
fusion signal on the real-query gate. Target query shape: "what else
was in flight around X" (association neither lexical nor dense serves).
Adopted into fusion only on measured lift; otherwise the writeup goes
in research/ next to the v2 evaluation and the back-edges remain for
Tier-2 history payloads.

## 12. Testing

Deep-tests discipline: risk model first, sabotage-checked. The high
table stakes: recall correctness under concurrent write (the v2
concurrency lessons port), supersession/trust invariants (a superseded
atom alone is a defect), payload budget enforcement, distiller dedup
(no atom storms from repetitive sessions), re-embed atomicity, restore
fidelity. Latency asserted in CI at the numbers above. The head2head
harness moves in-repo as the permanent quality regression suite;
lean-verification rules apply (touched suites per task, full suite at
phase gates only).

## 13. Implementation order (for the plan)

1. Runtime spike (one day, hard-stop): embedding + rerank through ONNX
   in Go vs TS vs Python-exception; pick and record why.
2. Store + schema + migrations + export/rebuild.
3. Recall engine (signals, fusion, priors, rerank, trust) + harness
   port + backfill; first numbers against the gate.
4. MCP shadow serving + double-write tee + CLI compat.
5. Briefer, then distiller (CC first, Codex second), then drift
   watcher.
6. Lifecycle jobs (supersession detection, importance accrual,
   integrity, re-embed pipeline) + ops drills.
7. Association experiment + tuning + cutover review.

Estimated total: 8-15M output tokens across subagent-driven execution,
dominated by phases 3, 5, and the test suites.

## 14. Open questions carried into the plan

- Repo: RESOLVED 2026-07-02. The daemon lives in this repo as a
  `daemon/` subtree; v3 is the pensive project's next major, not a new
  project (registry check ran; Gary kept the pensive name; no new
  project name created). The daemon imports the library's extractor,
  boundary math, and eval harness directly.
- Embedding model at launch: bge-small-en-v1.5 (on disk, 384d) is the
  spike default; nomic-embed-text-v1.5 evaluated during phase 3 via
  the re-embed pipeline it forces us to build anyway.
- Distiller local model: ornith-1.0-35b via the Hermes stack is the
  default; quality vs latency measured on real transcript samples in
  phase 5.
