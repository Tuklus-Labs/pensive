# Security model

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

## Not in scope

Pensive is a library, not a service:

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
