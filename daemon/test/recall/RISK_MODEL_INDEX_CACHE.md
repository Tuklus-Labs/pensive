# Risk Model: bounded derived index snapshot cache

The cache is a private, disposable acceleration layer. Canonical embeddings and
atom identity remain in SQLite; every cache failure must leave a normal rebuild
possible.

## Axis: Invariants

- I1: The source fingerprint changes when ordered `(atom_id, vector_blob)` rows,
  model ID, kind scope, format version, construction policy, search policy, or
  dtype changes.
- I2: Equivalent parameter mappings and equivalent kind ordering produce the
  same namespace and source fingerprint; row ordering remains significant because
  it determines the native key mapping.
- I3: A published filename contains only validated fixed lowercase hex values,
  and its file hash equals the bytes on disk before a loader is called.
- I4: Publication is one completed binary file; no metadata sidecar can make an
  incomplete graph appear loadable.
- I5: At most two completed snapshots are retained for one namespace, while
  other namespaces and unrelated files survive.

## Axis: State transitions

- S1: A missing cache directory is created privately for save, then transitions
  through private staging to one atomically published completed file.
- S2: A corrupt, partial, stale, or loader-rejected snapshot transitions to a
  cache miss without raising into canonical startup.
- S3: A failed native save or atomic replace cleans only the helper's own staging
  path and publishes no final file.

## Axis: Boundaries

- B1: Empty and one-row sources still produce fixed-width fingerprints without
  ambiguity from concatenation.
- B2: The retention boundary keeps exactly two files after three successful
  publications for one namespace.
- B3: Invalid or path-injection-shaped namespace/fingerprint values miss safely;
  a cache directory that is a file or symlink is also a miss.
- B4: A missing cache directory is a miss on load and is created on save.

## Axis: Malformed inputs

- M1: Non-UTF8/Unicode IDs and bytes-like vector blobs are encoded consistently;
  adjacent fields cannot collide through delimiter ambiguity.
- M2: Wrong-source requests, corrupted bytes, nonregular entries, and unprivate
  preexisting directories are ignored without chmod or loader invocation.
- M3: Loader exceptions and native save exceptions are swallowed as cache misses.

## Axis: Concurrency

- C1: Staging uses a private unique path and final publication uses an atomic
  rename, so readers never observe a partially written completed filename.
- C2: Directory fsync and staging cleanup do not leave an unbounded accumulation
  of helper-owned temporary files after an interrupted publication.

## Axis: Persistence

- P1: A valid native snapshot survives process boundaries as a self-identifying
  file whose content hash is checked before loading.
- P2: A restart with a wrong source fingerprint cannot load an old graph, while a
  matching source can round-trip through the native USearch serializer.
- P3: File and directory permissions remain private (0600/0700); the helper does
  not chmod arbitrary preexisting user paths.

## Axis: Integration contracts

- X1: `indexFingerprint(rows, modelId, kinds, parameters)` has no NumPy or native
  index dependency and returns `(namespace, fingerprint)` as lowercase hex.
- X2: `loadIndexSnapshot(cacheDir, namespace, fingerprint, loader)` calls the
  caller's loader only after source filename and content hash validation, and
  returns a loaded native object or `None`.
- X3: `saveIndexSnapshot(cacheDir, namespace, fingerprint, index)` calls
  `index.save(temp_path)`, atomically publishes one final file, and returns a
  success indicator without making cache availability a startup prerequisite.
- X4: Retention recognizes only this helper's exact namespace/fingerprint/hash
  filename shape and never deletes other namespaces or arbitrary files.

## Axis: Regression traps

- [x] boundary: `boundary: off-by-one in inclusive vs exclusive range` — keeping
  two snapshots must delete only entries after the first two newest.
- [x] concurrency: `concurrency: ordering assumption without enforced
  serialization` — staged bytes must become visible under one atomic final name.
- [x] contract: `contract: API returns null where caller expects empty collection`
  — cache misses use `None`, while save failures use a false success indicator and
  never raise into the rebuild path.
- [x] encoding: `encoding: text decoded with wrong character set` — IDs use
  explicit UTF-8 bytes and vectors preserve exact byte payloads with lengths.
- [x] framework: `framework: input syntax has reserved meaning unrelated to
  intent` — path components are fixed hex, so separators and dot segments are
  rejected instead of interpreted by `Path`.
- [x] io: `io: external name does not match physical or positional order` — the
  filename hash is checked against the actual file bytes before native loading.
- [x] persistence: `persistence: transaction not rolled back on error path` — a
  failed publication leaves no completed partial snapshot and leaves canonical
  data untouched.
- [x] resource: `resource: file descriptor leak on exceptional path` — failed
  saves, corrupt loads, and directory fsync errors close handles and clean owned
  staging files.
- [x] state: `state: stage moved across its dependency` — loader invocation is
  reachable only after source/hash validation, never before it.

## Coverage Matrix

| Risk row | Test name(s) covering it |
|---|---|
| I1, I2, B1, M1, X1, encoding trap | `test_fingerprint_is_canonical_and_length_delimited` |
| I3, B3, M2, P1, P2, X2, io/state traps | `test_load_checks_source_and_content_before_loader` |
| M3, S2, X2 | `test_loader_exception_is_a_cache_miss` |
| I4, S1, C1, C2, P3, X3, resource trap | `test_save_publishes_private_atomic_snapshot` |
| I5, B2, X4, boundary trap | `test_retention_keeps_two_namespace_files_only` |
| S3, M3, C2, persistence trap | `test_failed_publication_cleans_own_staging` |
| B3, M2, P3, io/framework traps | `test_unprivate_or_non_directory_cache_is_a_miss` |
| P2, X2, framework/resource integration | `test_real_usearch_snapshot_round_trip_over_2000_vectors` |


Independent review additions: corrupt exact-shaped files must not consume a valid
retention slot; future mtimes must not evict the newly published snapshot;
wrong-typed optional paths must remain cache misses. Covered by
`test_corrupt_files_do_not_consume_valid_retention_slots`,
`test_publication_survives_clock_rollback`, and
`test_invalid_cache_path_is_optional_failure` (all observed red before fixes).
