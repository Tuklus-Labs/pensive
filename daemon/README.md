# Pensive agent memory

The daemon stores authored memories and their provenance in SQLite. It retrieves
them by meaning and wording, retains correction history, and supplies session
briefs. This is the natural-language memory service used by agents. The separate
`pypensive` package under `../src/pensive/` implements entity-graph retrieval.

## Running the daemon

Developed and run on Python 3.14; older interpreters are untested. The CPU
build of torch serves the default model.

```bash
pip install -r daemon/requirements.txt
cd daemon/src
PENSIVE_V3_STORE=~/.local/share/pensive-v3/pensive.db python3 -m serve.daemon
```

Set the store path explicitly. The built-in default,
`~/.local/share/pensive-v3/shadow.db`, is a development path the code itself
labels as never a production store. Parent directories are created. On first
start the daemon opens (or creates) the SQLite store, loads the embedding
model, embeds any atoms that lack a vector for that model, builds one index per
memory class and then serves. It binds `127.0.0.1` only and logs the MCP URL
it is serving.

| Variable | Default | Meaning |
|---|---|---|
| `PENSIVE_V3_STORE` | `~/.local/share/pensive-v3/shadow.db` | SQLite store path |
| `PENSIVE_V3_PORT` | `5999` | Loopback port |
| `PENSIVE_V3_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model id (sentence-transformers) |
| `PENSIVE_V3_ONNX_MODEL` | unset | Path to an ONNX export of the same model; serves with `onnxruntime` instead of torch. The model id does not change. |
| `PENSIVE_V3_OPENAI_MODEL` | unset | OpenAI embedding model id (for example `text-embedding-3-large`). When set, its vectors fuse as a third recall signal and atom text leaves the machine. Missing key or SDK logs and disables rather than failing startup. |
| `PENSIVE_V3_AGENT` | unset | Agent name stamped into emit provenance when a caller sends none |
| `PENSIVE_V3_INDEX_CACHE_DIR` | `index-cache` beside the store | Derived HNSW snapshots; empty value disables snapshots |
| `PENSIVE_V3_BLAS_THREADS` | `4` | Process-local BLAS thread cap; `0` leaves it unchanged |
| `PENSIVE_V3_TORCH_THREADS` | `4` | Torch intra-op threads on the serve path; `0` disables the cap |
| `PENSIVE_V3_PRELOAD_RERANKER` | unset | `1` preloads the optional cross-encoder; served L2/L3 do not use it |
| `PENSIVE_V3_ALLOWED_HOSTS` | unset | Comma-separated hostnames accepted in `Host` besides loopback names, for a fronting proxy |
| `PENSIVE_V3_TEE_SECRET_FILE` | `<store dir>/tee.secret` | File holding the loopback secret the write routes require; created by `main()` with mode 0600 |
| `PENSIVE_V3_SHADOW_LOG` | `daemon/eval/shadow.jsonl` | Where `POST /shadow/recall` appends its comparison lines |

Register the MCP endpoint with any client that speaks Streamable HTTP. For
Claude Code:

```bash
claude mcp add --transport http pensive http://127.0.0.1:5999/mcp
```

A client may append `?agent=<name>` to the URL. That names the writer for
atoms it emits; it does not scope what it reads.

Check liveness with `curl -s http://127.0.0.1:5999/status`. Stop with SIGINT
or SIGTERM; uvicorn drains and exits 0.

## MCP tools

Seventeen tools are served. Argument names are exact; required arguments are
marked. Atom ids may be given bare or as the displayed `p3://` handle.

**Write**

| Tool | Required | Optional | What it does |
|---|---|---|---|
| `engram_emit_atom` | `project`, `shape`, `approach`, `outcome` (succeeded/failed/partial/abandoned), `reason`, `principle` | `tags`, `domain`, `narrative`, `trigger`, `stakes`, `dynamics`, `topic`, `agent` | Store one structured insight as a live atom with provenance |
| `engram_emit_discovery` | `project`, `principle` | `agent` | Shorthand for a positive-insight atom |
| `engram_emit_failure` | `project`, `principle` | `agent` | Shorthand for a what-did-not-work atom |
| `engram_emit_narrative` | `project`, `narrative` | `trigger`, `dynamics`, `stakes`, `topic`, `agent` | Store a prose fragment with no structured atom |
| `engram_emit_snapshot` | `project`, `hypothesis` | `dead_ends`, `next_steps`, `agent` | Store a working-state dump for compaction recovery |

Bodies are capped at 32,000 characters. `agent` overrides `PENSIVE_V3_AGENT`
for that one write.

**Read**

| Tool | Required | Optional | What it does |
|---|---|---|---|
| `recall` | `query` | `tier` (L2 default, L3), `project`, `agent`, `kinds`, `timeScope` [start, end] unix, `k` (10), `tokenBudget` (1500) | Ranked, trust-annotated plain text; weak matches stay as labelled handles |
| `recall_records` | `query` | same filters, `k` (1..32), `tokenBudget`, `includeReceipt`, `taskId`, `callerAgent` | Typed records with content, provenance, score, trust fields and, on request, a receipt for later feedback |
| `history` | `atomId` | | One atom's body, provenance, live edges and its supersession chain |
| `pensive_analytics` | | | Store counts by kind and status, plus embedding coverage for the active model |
| `pensive_recall` | `query` | `project`, `limit` | The pre-v3 recall shape, kept for callers that never upgraded |

**Maintain**

| Tool | Required | Optional | What it does |
|---|---|---|---|
| `correct` | `oldAtomId`, `newText` | `provenance` {source, agent, sessionId, sourceRef} | Write a successor and retire the predecessor atomically; both stay readable |
| `pin` / `unpin` | `atomId` | | Attach or drop a durable pin facet; idempotent |

**Tasks and feedback**

| Tool | Required | Optional | What it does |
|---|---|---|---|
| `task_checkpoint` | `project`, `taskId`, `requestId`, `expectedRevision`, `state` (active/blocked/completed/abandoned), `body` | `agent`, `sessionId`, `sourceRef` | Append task state with compare-and-swap on revision; a reused `requestId` replays its original result |
| `task_state` | | `mode` (current/history/recent), `project`, `agent`, `taskId`, `revision`, `asOf`, `afterRevision`, `limit` | Read the latest checkpoint, one revision, paged history, or recent tasks in a project |
| `recall_feedback` | `receiptId`, `atomId`, `eventId`, `feedbackType` (shown/used/helpful/irrelevant/outdated), `taskId` | `agent`, `sessionId`, `sourceRef`, `note` | Report what a recall was worth; `used` and the evaluative types need a `note` |

## HTTP routes

All routes sit behind the `Host` allow-list. The write routes additionally
require `Content-Type: application/json`, a loopback `Origin` when one is
sent, and the secret from `PENSIVE_V3_TEE_SECRET_FILE` in the
`x-pensive-tee-secret` header. `docs/SECURITY.md` explains why.

| Route | Parameters | Purpose |
|---|---|---|
| `GET /recall` | `q` (required, <= 8192 chars), `tier` (L2/L3), `k` (1..200), `budget` (1..8000) | Plain-text recall payload without MCP framing |
| `GET /get` | `id`, `full=1` for provenance | One atom by id |
| `GET /lookup` | `key`, `value`, `limit` (100) | Atoms carrying an exact facet |
| `GET /brief` | `agent`, `project`, `taskId`, `budget` | Session-start working set; with `taskId`, the explicit task state, pins and loose ends |
| `GET /status` | | Server name and request counters |
| `GET /viz`, `/viz/events`, `/viz/graph`, `/viz/history` | | Live event page and the JSON feeds behind it |
| `POST /tee/emit` | JSON emit payload | Guarded write; replays an emit into the store. Built for the v2 to v3 shadow cutover |
| `POST /shadow/recall` | JSON query and reference answer | Guarded; runs the query through v3 and appends a comparison line to the shadow log |
| `/mcp` | Streamable HTTP | The MCP endpoint |

## Recall semantics

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
`GET /brief?taskId=<id>&project=<project>` shows explicit task state with pins
and addressed loose ends, replacing the generic active-memory section.

Worked example over MCP (substitute the verified task id and current revision):

```
task_checkpoint {"project":"pensive","taskId":"TASK_ID","requestId":"REQUEST_ID",
                 "expectedRevision":0,"state":"active",
                 "body":"Validator passes; next run the restore check."}
task_state      {"mode":"current","taskId":"TASK_ID"}
recall_records  {"query":"Which restore checks are required?","taskId":"TASK_ID",
                 "callerAgent":"codex","includeReceipt":true}
recall_feedback {"receiptId":"RECEIPT_ID","atomId":"p3://ATOM_ID","eventId":"EVENT_ID",
                 "taskId":"TASK_ID","feedbackType":"helpful",
                 "note":"The saved restore check caught a missing history table."}
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
| `src/serve/` | MCP/HTTP contracts, guards and index reconciliation |
| `src/ambient/` | Session capture, briefs, deduplication and drift detection |
| `src/lifecycle/` | Importance accrual, integrity and migration jobs |
| `src/ingest/` | Backfill of a legacy store into v3 |
| `src/util/` | ULIDs and small shared helpers |
| `test/` | Daemon tests; synthetic fixtures and model-backed integration tests |
| `eval/` | Retrieval and latency gates; `eval/agent_outcomes/` is the frozen blind workflow replay |
| `gate/tiergate/` | The latency and quality gate `deploy.sh` runs |
| `tools/` | Store repair and edge-proposal utilities written against the publisher's corpus |
| `hooks/` | Session-start brief hooks for Claude Code and Grok Build, as worked examples |

From the repository root, run `python3 -m pytest daemon/test/ -q` for the daemon
and `python3 -m pytest tests/ -q` for the standalone library. Model-backed tests
need an importable `sentence_transformers` and its cached weights; without it
they error at fixture time rather than skipping. Hardware latency tests have
separate machine and load assumptions; a CPU-only correctness run cannot
certify a GPU latency target.

Use isolated stores and ports for development. `deploy.sh` is the publisher's
own deploy path: it runs the tests, rebuilds the gate binary, restarts a
`systemd --user` unit named `pensive-v3` and runs a client-observed
latency/quality gate against it. Read it as a worked example, not an installer.
Historical green reports do not certify a new checkout or a loaded machine.
The original v3 design lives in
`../docs/superpowers/specs/2026-07-02-pensive-v3-design.md`; later evidence
documents explain revisions, including explicit retractions.

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

A new task can discover related past work with `task_state`
(`{"mode":"recent","project":"pensive","agent":"codex"}`), then inspect the
selected task ID. A prior task's checkpoint remains its own history.
