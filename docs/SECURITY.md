# Security model

Two parts. The first covers the `pypensive` library under `src/pensive/`. The
second covers the agent-memory daemon under `daemon/`, which is a network
service with a different surface. Neither part describes the other.

# Part 1: the `pypensive` library

## HMAC-signed pickles

`IngestPipeline.save_graph` / `load_graph` (and the underlying
`SpreadingActivation.get_save_data` / `from_save_data`) wrap saved
graphs in an HMAC-SHA256 envelope so a hostile pickle delivered out-of-
band cannot be loaded silently.

### Key precedence

`pensive.ingestion.pipeline._load_or_create_key()` reads the HMAC key
in this order:

1. **On-disk file** at `$XDG_CONFIG_HOME/pensive/pickle.key` (default
   `~/.config/pensive/pickle.key`), mode `0600`, in a directory created
   mode `0700`. Written atomically via `O_EXCL` so the key never exists
   at umask-default permissions, even briefly.
2. **`PENSIVE_PICKLE_KEY`** environment variable, used as UTF-8 bytes.
3. **Freshly generated** 32 random bytes, persisted to the path above
   on a best-effort basis. If the persistence write fails (read-only
   filesystem, no home directory, etc.), the call still returns a key
   so save/load round-trips within a single process succeed.

Disk-first is deliberate: a hostile shell-rc could set
`PENSIVE_PICKLE_KEY` to a known weak value, then deliver a forged
"signed" pickle that passes verification under that key. Reading the
file first means an attacker has to overwrite a `0600` file in the
user's config directory before the env var is even consulted.

### Recovery: lost or compromised key

The HMAC key is per-host secret material. If it is lost or you suspect
it has been read by an attacker, **all previously-signed pickles
become unloadable** under the normal verification path. There is no
key-recovery mechanism by design -- recovering the key from a
signature would defeat the security model.

Workflow when this happens:

1. Stop trusting any existing on-disk graphs. Move them to a quarantine
   directory.
2. Delete the existing key file:
   ```
   rm ~/.config/pensive/pickle.key
   ```
   The next call to a pipeline that needs a key will generate a fresh
   one.
3. Rebuild the SpreadingActivation graph from source data. The source
   data is whatever you originally ingested -- typically ChatGPT
   exports, Facebook exports, or your own JSONL. Re-run
   `IngestPipeline.ingest_source(...)`.
4. Save with the new key:
   ```python
   pipe.save_graph(path)
   ```
   The on-disk file is now signed under the fresh key.

If the on-disk graph predates the HMAC-signed-pickle layer entirely
(legacy unsigned `.pkl`), and you trust the source, you can opt into
loading it via `trusted=True` to skip signature verification. **Only
use this if you have an out-of-band reason to trust the file**
(checksum, source provenance, etc.). The `trusted=True` path will then
re-save the graph under the current key, so subsequent loads work
without the override.

### Backup the key file

The single best mitigation against key loss is to back up
`~/.config/pensive/pickle.key` alongside any pickled graphs the key
signs. The file is 32 random bytes; storage cost is negligible.

Treat the key file like any other host-secret material:

- Keep it on the same trust boundary as the graphs it signs.
- Do not commit it to version control.
- Do not copy it to shared NFS shares without `root_squash` enabled --
  any node with root on a remote node can read `0600` files otherwise.

## Pickle verification

Loaded files must:

- Start with the `PENSIVE-SIGNED-V1\x00` magic header
- Carry a valid HMAC-SHA256 over the pickle body
- Be smaller than `DEFAULT_MAX_LOAD_SIZE` (2 GiB)

The HMAC compare uses `hmac.compare_digest` for constant-time
comparison to avoid timing oracles. (This is verified by source
inspection rather than by a functional test -- a timing test would
need timing instrumentation and would be flaky on a shared host. We
trust `hmac.compare_digest`'s documented constant-time semantics; see
[Python stdlib `hmac.compare_digest`](https://docs.python.org/3/library/hmac.html#hmac.compare_digest).)

A legacy unsigned pickle (no magic header) is rejected with a
`PickleVerificationError` unless `trusted=True` is passed at load
time. `trusted=True` is an explicit knob; there is no env-var bypass.

## Library scope

`pypensive` is a library, not a service:

- No HTTP, MCP, socket, or other network endpoint.
- No SQL, no asyncio, no subprocess, no eval/exec.
- No shell command execution.

The only attack surfaces are:

1. A maliciously-crafted pickle delivered to `load_graph` (mitigated
   by the HMAC envelope above).
2. A maliciously-crafted regex passed to `build_pattern_set` (caller
   responsibility; the library does not validate caller-supplied
   patterns against ReDoS heuristics. Typical callers pass the
   library-supplied `REAL_DATA_PATTERNS`).
3. A corrupted save file -- structural validation runs at load time
   (`_adj.check_format(full_check=True)` plus parallel-array length
   checks) and rejects corruption with a `ValueError` rather than
   surfacing as a delayed segfault inside the numba kernel.

# Part 2: the agent-memory daemon

Source of truth: `daemon/src/serve/daemon.py` and `daemon/src/serve/tee.py`.
The guards below each have tests under `daemon/test/serve/` (`test_tee.py`,
`test_host_guard.py`).

## Network position

- The daemon binds `127.0.0.1` only (`_HOST` in `serve/daemon.py`), on
  `PENSIVE_V3_PORT` (default 5999). It speaks plain HTTP. There is no TLS and
  no authentication of its own: it is a same-user, same-host service, and
  anything that can open a loopback socket as that user can read and write
  memory through `/mcp`.
- Exposing it beyond the host is the operator's job and is expected to go
  through a reverse proxy that authenticates. The daemon does not roll its own
  login.

## Browser and rebinding guards

Binding loopback keeps the daemon off the network. It does not, by itself,
protect the write routes from a browser the operator is already running: a
page can POST to loopback with no preflight, and CORS only stops the page from
reading the reply. Three independent guards close that, each covering what the
others cannot (`serve/tee.py`, `checkLocalWriteRequest`):

1. **Loopback secret in a custom header** (`x-pensive-tee-secret`) on the
   state-changing routes `POST /tee/emit` and `POST /shadow/recall`. Custom
   headers force a preflight the server never answers, and the secret lives
   in a `0600` file a page cannot read. The file is
   `$PENSIVE_V3_TEE_SECRET_FILE` or `<store dir>/tee.secret`; only the
   daemon's `main()` provisions it. A daemon that cannot load its secret
   fails those routes closed.
2. **Content-Type must be `application/json`**: the three types a
   no-preflight request may set are refused before the secret is looked at.
3. **Origin/Referer, when present, must be loopback**. Local processes send
   neither and are unaffected. This check also gates `/mcp`, which carries no
   secret because every local MCP client talks to it.
4. **Host allow-list** (`checkAllowedHost`): loopback names are always
   accepted; `PENSIVE_V3_ALLOWED_HOSTS` adds comma-separated hostnames for a
   fronting proxy; any other `Host` is a 403. This is the half that closes DNS
   rebinding, because `Host` is the one header a page cannot forge away.

Every rejection carries a machine-readable `reason` so a test can tell which
guard fired. The Host allow-list is a middleware over every route. The
Origin/Referer check also runs on the read routes `/recall`, `/get` and
`/lookup`; `/brief`, `/status` and `/viz*` carry only the Host check.

## Data at rest

- The store is one SQLite file (`PENSIVE_V3_STORE`). File permissions are the
  boundary. It holds every memory in clear text, plus `recall_log`, which
  records callers' queries verbatim. Back it up and restrict it as you would
  any private notebook.
- Portable exports (`store/export.py`) contain the same memories in clear
  JSONL with digests for integrity, not confidentiality.
- The derived index cache (`PENSIVE_V3_INDEX_CACHE_DIR`, default `index-cache`
  beside the store) holds embedding vectors and IDs. Vectors are not the text,
  but they are derived from it; treat the directory like the store.
- The daemon loads no pickles.

## Data leaving the host

By default nothing does. Embeddings run locally (sentence-transformers or the
ONNX export named by `PENSIVE_V3_ONNX_MODEL`). If `PENSIVE_V3_OPENAI_MODEL` is
set, atom text is sent to the OpenAI embeddings API as a third recall signal,
using `OPENAI_API_KEY` from the environment or `~/.keys`; leave it unset if
memory must stay on the machine.

## Reporting

Open an issue on the repository, or if the finding is sensitive, contact the
maintainers through the address on the GitHub organisation page.
