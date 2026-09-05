# Pensive agent memory

The daemon stores authored memories and their provenance in SQLite. It retrieves
them by meaning and wording, retains correction history, and supplies session
briefs. This is the natural-language memory service used by agents. The separate
`pypensive` package under `../src/pensive/` implements entity-graph retrieval.

## Finding and following a memory

| Need | Entry point |
|---|---|
| Find relevant memory | MCP `recall` with a complete natural-language `query` |
| Consume typed results | MCP `recall_records` with IDs, content, provenance, score and trust fields |
| Read one known atom | HTTP `GET /get?id=<bare-id>&full=1`; see `serve/l1.py` for bounded fields |
| Follow a correction | MCP `history` with the atom's ID or displayed `p3://` handle |
| Correct a fact | MCP `correct` with `oldAtomId` and `newText` |
| Keep/remove a standing memory | MCP `pin` / `unpin` |
| Use the household shell integration | `pensive-recall` and `engram-emit` in the Engram repository's `tools/` directory |

`recall` defaults to L2: authored `atom`, `narrative` and `snapshot` records.
L3 also searches imported `document_chunk` records. An explicit `kinds` list
chooses the requested kinds directly. Project, agent and effective-time filters
limit the candidate universe; they do not mean "search everywhere, then discard
unrelated winners." Connection `?agent=` identifies a writer; it does not
silently scope a reader. Pass the recall `agent` argument to request that scope.

Ranking `score` and trust `confidence` have different meanings. Ranking combines
retrieval evidence; confidence is a heuristic about corroboration, separation,
age and supersession. Neither is a probability that a statement is true. A
single matching signal does not corroborate an imported chunk. The plain-text
`recall` response keeps weak matches as labelled handles. Structured records
retain their explicit trust fields for the consuming adapter.

`tokenBudget` uses the documented character-based estimate. It bounds that
estimate, not every possible model tokenizer. Consumers with a hard context
limit must use their own tokenizer. A budget too small for even a sentinel can
produce an empty payload; `lowConfidence` remains available in typed results.
For `recall_records`, the budget counts body content; a separate byte cap bounds
the serialized message including provenance. A record's `estimatedTokens` may
describe an omitted full body and exceed the request budget. The envelope's
estimate describes the body content actually returned.

## Corrections and durable state

Canonical text and provenance outlive indexes and embedding models. A correction
creates a successor and retires its live predecessor atomically. It retains
importance, pin rank and user tags, while keeping the original text and authorship
in history. Content-derived entity facets are not blindly copied to changed text.

A stale correction target is refused with lineage guidance so concurrent agents
cannot unknowingly create competing current answers. Index maintenance follows
the canonical commit; an error after that commit names the committed successor.
Read that handle before retrying. Legacy supersession forks are exposed by
history and integrity checks.

`store/export.py` and `store/rebuild.py` implement portable JSONL snapshots.
Canonical memories, provenance, relationships and retained usage history belong
in exports; embeddings and FTS are rebuilt. `store/backup.py` provides the
SQLite backup path. Verify a restore on a fresh temporary database before
replacing a working store.
Format 2 includes per-file SHA-256 digests. Editing its JSONL by hand also
requires updating those digests; mismatches are rejected. Legacy four-file
exports remain readable, with no invented usage history.

## Code map and tests

| Directory | Responsibility |
|---|---|
| `src/store/` | Canonical data, transactions, schema migrations, export and restore |
| `src/recall/` | Candidate generation, ranking, trust and bounded delivery |
| `src/serve/` | MCP/HTTP contracts and index reconciliation |
| `src/ambient/` | Session capture, briefs, deduplication and drift detection |
| `src/lifecycle/` | Importance accrual, integrity and migration jobs |
| `test/` | Daemon tests; synthetic fixtures and model-backed integration tests |

From the repository root, run `python3 -m pytest daemon/test/ -q` for the daemon
and `python3 -m pytest tests/ -q` for the standalone library. Model-backed tests
require the local embedding/reranking dependencies and cached weights. Hardware
latency tests have separate machine and load assumptions; a CPU-only correctness
run cannot certify a GPU latency target.

Use isolated stores and ports for development. `deploy.sh` includes tests and a
client-observed latency/quality gate; historical green reports do not certify a
new checkout or a loaded machine. The original v3 design lives in
`../docs/superpowers/specs/2026-07-02-pensive-v3-design.md`; later evidence documents
explain revisions, including explicit retractions.
