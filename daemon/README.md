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
| Resume one task | MCP `task_state` with its unique `taskId` |
| Record progress or completion | MCP `task_checkpoint` with `expectedRevision` and a stable `requestId` |
| Report useful memory | MCP `recall_feedback` with an admitted record's `receiptId` and an evidence note |
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

Task checkpoints are a separate append-only stream keyed by project, agent and
task ID. Reads return the latest revision, including completed or abandoned
states; historical revisions and as-of reads remain available. A stale writer
receives an error. Reusing the same request ID with the same input returns its
original result, even if the task has since advanced. Never infer completion of
the whole task from an assistant turn finishing.

For Codex hooks, a parent and its workers share `session_id`. Verify the supplied
transcript's `session_meta.payload.id` and matching session ID before using that
unique task ID. Unknown transcript formats fall back to prompt-only recall.
Legacy snapshots stay readable; they are not automatically assigned to tasks.
`/brief?taskId=<id>&project=<project>` shows explicit task state with pins and
addressed loose ends, replacing the generic active-memory section.

Household CLI examples (substitute the verified task ID and current revision):

```sh
engram-emit checkpoint --project pensive --task-id TASK_ID --caller codex \
  --expected-revision 0 --request-id REQUEST_ID --state active \
  --body 'Validator passes; next run the restore check.'
pensive-recall --state current --task-id TASK_ID --json
pensive-recall --query 'Which restore checks are required?' \
  --task-id TASK_ID --caller codex --receipt --json
engram-emit feedback --receipt-id RECEIPT_ID --atom-id p3://ATOM_ID \
  --task-id TASK_ID --caller codex --event-id EVENT_ID --type helpful \
  --note 'The saved restore check caught a missing history table.'
```

Receipts describe the final records admitted by the response budget. They keep
caller identity separate from the recall `agent` author filter. `shown` means
delivery, `used` needs evidence of use, and `helpful` needs an observed benefit.
Only helpful feedback can increase importance, once per atom/caller/task and by
at most .01 up to the ranking ceiling; older values above that ceiling remain
unchanged. Exposure counts never earn credit. Irrelevant or outdated reports do
not automatically erase, supersede or globally penalize a memory.

`store/export.py` and `store/rebuild.py` implement portable JSONL snapshots.
Canonical memories, provenance, relationships and retained usage history belong
in exports; embeddings and FTS are rebuilt. `store/backup.py` provides the
SQLite backup path. Verify a restore on a fresh temporary database before
replacing a working store.
Format 3 includes all eleven canonical/history tables with per-file SHA-256
digests. Schema 4 adds task checkpoints, receipts, exposures, feedback and
credits without rewriting old memories. Format 2 remains the historical six-file
format for schema <=3; those and legacy four-file exports remain readable.
Imported feedback must satisfy the same scope and evidence rules as live writes.
Restore publishes only after the complete snapshot validates.

## Code map and tests

| Directory | Responsibility |
|---|---|
| `src/store/` | Canonical data, transactions, schema migrations, export and restore |
| `src/recall/` | Candidate generation, ranking, trust and bounded delivery |
| `src/serve/` | MCP/HTTP contracts and index reconciliation |
| `src/ambient/` | Session capture, briefs, deduplication and drift detection |
| `src/lifecycle/` | Importance accrual, integrity and migration jobs |
| `test/` | Daemon tests; synthetic fixtures and model-backed integration tests |
| `eval/agent_outcomes/` | Frozen blind workflow replay, separate from retrieval and latency gates |

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

## Index reproducibility and startup

The complete authored-memory class uses exact Flat search through 32,000 rows.
Larger classes use HNSW with serial construction and explicit search expansion.
The daemon promotes or compacts an exact index when a write reaches its physical
row limit, so a long-running process cannot silently grow an unbounded scan.
This threshold and the four-worker BLAS budget were measured on the current
384-dimensional BGE corpus; other models and hosts need their own measurements.

The daemon caches complete HNSW builds in a private `index-cache` directory next
to its database. Fingerprints cover ordered IDs, embedding bytes, model, kinds
and index settings. It verifies the native file digest and loaded shape/key set
before use. Memory-only writes leave an unchanged document index reusable;
changed document vectors require a new deterministic build. Cache failures fall
back to rebuilding, and canonical exports do not depend on these derived files.
Two valid snapshots are retained per namespace. Corrupt or unrelated files are
preserved for inspection.

Set `PENSIVE_V3_INDEX_CACHE_DIR` to override the directory, or to an empty value
to disable snapshots. `PENSIVE_V3_BLAS_THREADS` controls this daemon's BLAS pools
(default 4, zero leaves them unchanged). The optional cross-encoder can be
preloaded with `PENSIVE_V3_PRELOAD_RERANKER=1`; ordinary served L2/L3 keep it lazy.

A new task can discover related past work with
`pensive-recall --state recent --project pensive --agent codex --json`, then
inspect the selected task ID. A prior task's checkpoint remains its own history.
