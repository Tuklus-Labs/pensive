"""Risk-mapped tests for the private bounded derived-index snapshot cache."""

import hashlib
import os
import stat
from pathlib import Path

import pytest

from recall.index_cache import (
    indexFingerprint,
    loadIndexSnapshot,
    saveIndexSnapshot,
)


_PARAMETERS = {
    "version": 3,
    "construction": {
        "connectivity": 16,
        "expansion_add": 128,
        "threads": 1,
    },
    "search": {"expansion_search": 1024},
    "dtype": "f32",
}


class _BytesIndex:
    def __init__(self, payload, fail=False):
        self.payload = payload
        self.fail = fail

    def save(self, path):
        Path(path).write_bytes(self.payload)
        if self.fail:
            raise OSError("synthetic native save failure")


def _source(rows=None):
    return rows or (
        ("atom-001", b"\x00\x01\x02\x03"),
        ("é", memoryview(b"\x04\x05\x06\x07")),
    )


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _completed(cache):
    return sorted(cache.glob("*.usearch")) if cache.exists() else []


def test_fingerprint_is_canonical_and_length_delimited():
    """I1/I2/B1/M1/X1: config and ordered byte fields are unambiguous."""
    rows = _source()
    namespace, fingerprint = indexFingerprint(
        rows, "model-A", ("code", "memory"), _PARAMETERS
    )
    reordered_config = {
        "dtype": "f32",
        "search": {"expansion_search": 1024},
        "construction": {
            "threads": 1,
            "expansion_add": 128,
            "connectivity": 16,
        },
        "version": 3,
    }
    same_namespace, same_fingerprint = indexFingerprint(
        rows, "model-A", ("memory", "code"), reordered_config
    )
    assert namespace == same_namespace, (
        "fingerprint namespace invariant violated: equivalent kind/config order "
        f"must match, first={namespace!r} second={same_namespace!r}"
    )
    assert fingerprint == same_fingerprint, (
        "source fingerprint invariant violated: equivalent canonical inputs must "
        f"match, first={fingerprint!r} second={same_fingerprint!r}"
    )
    assert len(namespace) == 64 and len(fingerprint) == 64, (
        "fixed-hex path invariant violated: expected two SHA-256 hex values, "
        f"namespace={namespace!r} fingerprint={fingerprint!r}"
    )
    changed_row_order = indexFingerprint(
        tuple(reversed(rows)), "model-A", ("memory", "code"), _PARAMETERS
    )[1]
    assert changed_row_order != fingerprint, (
        "ordered-row identity invariant violated: changing native key order must "
        f"change the source fingerprint, original={fingerprint} reordered={changed_row_order}"
    )
    changed_vector = indexFingerprint(
        (("a", b"bc"), ("ab", b"c")), "model-A", (), _PARAMETERS
    )[1]
    collided_vector = indexFingerprint(
        (("a", b"b"), ("ab", b"c")), "model-A", (), _PARAMETERS
    )[1]
    assert changed_vector != collided_vector, (
        "length-delimited encoding invariant violated: adjacent id/vector fields "
        f"collide, changed={changed_vector} collided={collided_vector}"
    )
    assert indexFingerprint(rows, "model-B", ("memory", "code"), _PARAMETERS)[1] != fingerprint, (
        "model identity invariant violated: changing embedding model must miss "
        f"the old source, old={fingerprint}"
    )
    changed_dtype = dict(_PARAMETERS, dtype="f16")
    assert indexFingerprint(rows, "model-A", ("memory", "code"), changed_dtype)[1] != fingerprint, (
        "dtype policy invariant violated: changing native dtype must miss "
        f"the old source, old={fingerprint}"
    )
    for label, changed_parameters in (
        ("version", dict(_PARAMETERS, version=4)),
        ("construction", dict(_PARAMETERS, construction={"connectivity": 32})),
        ("search", dict(_PARAMETERS, search={"expansion_search": 512})),
    ):
        changed = indexFingerprint(
            rows, "model-A", ("memory", "code"), changed_parameters
        )[1]
        assert changed != fingerprint, (
            f"{label} policy invariant violated: changing native parameters must "
            f"miss old source, old={fingerprint} changed={changed}"
        )
    changed_kinds = indexFingerprint(rows, "model-A", ("memory",), _PARAMETERS)[1]
    assert changed_kinds != fingerprint, (
        "kind-scope identity invariant violated: changing indexed kinds must "
        f"miss old source, old={fingerprint} changed={changed_kinds}"
    )
    empty_fingerprint = indexFingerprint((), "model-A", ("memory", "code"), _PARAMETERS)[1]
    one_row_fingerprint = indexFingerprint((rows[0],), "model-A", ("memory", "code"), _PARAMETERS)[1]
    assert empty_fingerprint != one_row_fingerprint, (
        "empty/single-row boundary invariant violated: row count must distinguish "
        f"empty={empty_fingerprint} one_row={one_row_fingerprint}"
    )


def test_save_publishes_private_atomic_snapshot(tmp_path):
    """I3/I4/S1/C1/C2/P3/X3: one private completed file is published."""
    cache = tmp_path / "index-cache"
    namespace = "a" * 64
    fingerprint = "b" * 64
    index = _BytesIndex(b"native graph bytes")

    saved = saveIndexSnapshot(cache, namespace, fingerprint, index)

    assert saved is True, (
        "save success invariant violated: a valid native snapshot must publish "
        f"without raising, result={saved!r}"
    )
    assert cache.is_dir() and not cache.is_symlink(), (
        "cache directory invariant violated: save must create a real directory, "
        f"path={cache}"
    )
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700, (
        "cache directory privacy invariant violated: expected mode 0700, "
        f"mode={stat.S_IMODE(cache.stat().st_mode):04o}"
    )
    files = _completed(cache)
    assert len(files) == 1, (
        "single-file publication invariant violated: expected exactly one "
        f"completed file, files={[p.name for p in files]}"
    )
    file = files[0]
    assert stat.S_IMODE(file.stat().st_mode) == 0o600, (
        "snapshot privacy invariant violated: expected mode 0600, "
        f"file={file.name} mode={stat.S_IMODE(file.stat().st_mode):04o}"
    )
    assert file.name == f"{namespace}.{fingerprint}.{_sha256(index.payload)}.usearch", (
        "self-identifying filename invariant violated: expected source and byte "
        f"hash components, file={file.name}"
    )
    assert not list(cache.glob(".index-cache-*")), (
        "staging cleanup invariant violated: helper-owned temporary files remain, "
        f"entries={[p.name for p in cache.iterdir()]}"
    )


def test_load_checks_source_and_content_before_loader(tmp_path):
    """I3/B3/M2/P1/P2/X2: wrong source/corruption never reaches loader."""
    cache = tmp_path / "cache"
    namespace = "1" * 64
    fingerprint = "2" * 64
    assert saveIndexSnapshot(cache, namespace, fingerprint, _BytesIndex(b"valid")) is True, (
        "setup publication invariant violated: valid fixture snapshot did not save"
    )

    calls = []

    def loader(path):
        calls.append(Path(path))
        return object()

    wrong = "3" * 64
    assert loadIndexSnapshot(cache, namespace, wrong, loader) is None, (
        "wrong-source cache invariant violated: an old source fingerprint must "
        f"miss, requested={wrong}"
    )
    assert calls == [], (
        "loader ordering invariant violated: wrong-source lookup called loader "
        f"before validation, calls={calls!r}"
    )
    file = _completed(cache)[0]
    file.write_bytes(file.read_bytes() + b"corruption")
    assert loadIndexSnapshot(cache, namespace, fingerprint, loader) is None, (
        "content-hash invariant violated: corrupted bytes must be a cache miss, "
        f"file={file.name}"
    )
    assert calls == [], (
        "corruption ordering invariant violated: loader ran before byte hash "
        f"validation, calls={calls!r}"
    )


def test_loader_exception_is_a_cache_miss(tmp_path):
    """M3/S2/X2: native loader failure cannot break canonical startup."""
    cache = tmp_path / "cache"
    namespace = "4" * 64
    fingerprint = "5" * 64
    assert saveIndexSnapshot(cache, namespace, fingerprint, _BytesIndex(b"valid")) is True, (
        "setup publication invariant violated: valid fixture snapshot did not save"
    )

    def loader(_path):
        raise RuntimeError("synthetic graph corruption")

    loaded = loadIndexSnapshot(cache, namespace, fingerprint, loader)
    assert loaded is None, (
        "loader failure contract violated: native loader exceptions must become "
        f"cache misses, loaded={loaded!r}"
    )


def test_retention_keeps_two_namespace_files_only(tmp_path):
    """I5/B2/X4: three same-namespace saves retain two and preserve others."""
    cache = tmp_path / "cache"
    namespace = "6" * 64
    other_namespace = "7" * 64
    other = cache / f"{other_namespace}.{'8' * 64}.{'9' * 64}.usearch"
    foreign_shaped = cache / f"{namespace}.{'a' * 64}.{'b' * 64}.usearch"
    arbitrary = cache / "notes.txt"
    fingerprints = [f"{n:064x}" for n in (1, 2, 3)]
    for n, fingerprint in enumerate(fingerprints):
        assert saveIndexSnapshot(cache, namespace, fingerprint, _BytesIndex(bytes([n + 1]))) is True, (
            "retention setup invariant violated: each valid publication must "
            f"succeed, fingerprint={fingerprint}"
        )
        if n == 0:
            other.write_bytes(b"other namespace")
            os.chmod(other, 0o600)
            foreign_shaped.write_bytes(b"foreign content")
            os.chmod(foreign_shaped, 0o600)
            arbitrary.write_text("leave me")
            os.chmod(arbitrary, 0o600)

    retained = _completed(cache)
    own = [
        file for file in retained
        if file.name.split(".")[0] == namespace
        and file.name.split(".")[1] in fingerprints
    ]
    assert len(own) == 2, (
        "retention boundary invariant violated: exactly two same-namespace files "
        f"must remain after three saves, own={[p.name for p in own]}"
    )
    assert other.exists() and not other.is_symlink(), (
        "namespace isolation invariant violated: another namespace file was "
        f"deleted, path={other}"
    )
    assert arbitrary.exists(), (
        "arbitrary-file preservation invariant violated: unrelated cache content "
        f"was deleted, path={arbitrary}"
    )
    assert foreign_shaped.exists(), (
        "foreign-content preservation invariant violated: a private file whose "
        f"filename hash is false was deleted, path={foreign_shaped}"
    )


def test_corrupt_files_do_not_consume_valid_retention_slots(tmp_path):
    cache, namespace = tmp_path / 'cache', 'a' * 64
    for n in (1, 2):
        saveIndexSnapshot(cache, namespace, f'{n:064x}', _BytesIndex(bytes([n])))
    corrupt = next(cache.glob(f'{namespace}.{2:064x}.*'))
    corrupt.write_bytes(b'corrupted')
    saveIndexSnapshot(cache, namespace, f'{3:064x}', _BytesIndex(b'new'))
    assert loadIndexSnapshot(cache, namespace, f'{1:064x}', Path.read_bytes) == b'\x01', 'a corrupt newer entry must not evict the second loadable snapshot'
    assert loadIndexSnapshot(cache, namespace, f'{3:064x}', Path.read_bytes) == b'new', 'the newly published snapshot remains loadable'
    assert corrupt.exists(), 'retention does not erase unidentified corrupt content'


def test_publication_survives_clock_rollback(tmp_path):
    cache, namespace = tmp_path / 'cache', 'a' * 64
    for n in (1, 2):
        saveIndexSnapshot(cache, namespace, f'{n:064x}', _BytesIndex(bytes([n])))
        path = next(cache.glob(f'{namespace}.{n:064x}.*'))
        os.utime(path, (9_000_000_000 + n, 9_000_000_000 + n))
    saveIndexSnapshot(cache, namespace, f'{3:064x}', _BytesIndex(b'current'))
    assert loadIndexSnapshot(cache, namespace, f'{3:064x}', Path.read_bytes) == b'current', 'future mtimes on older snapshots cannot evict the snapshot just published'


def test_invalid_cache_path_is_optional_failure():
    namespace, fingerprint = 'a' * 64, 'b' * 64
    assert loadIndexSnapshot(object(), namespace, fingerprint, Path.read_bytes) is None, 'invalid cache configuration is a miss'
    assert saveIndexSnapshot(object(), namespace, fingerprint, _BytesIndex(b'x')) is False, 'invalid optional cache path cannot break canonical index construction'


def test_failed_publication_cleans_own_staging(tmp_path, monkeypatch):
    """S3/M3/C2: native and atomic publication failures leave no final graph."""
    cache = tmp_path / "cache"
    namespace = "a" * 64
    fingerprint = "c" * 64
    failing = _BytesIndex(b"partial", fail=True)
    assert saveIndexSnapshot(cache, namespace, fingerprint, failing) is False, (
        "native-save failure contract violated: failed save must report a cache "
        "miss indicator instead of raising"
    )
    assert _completed(cache) == [], (
        "failed-save atomicity invariant violated: partial native bytes became a "
        f"completed file, files={[p.name for p in _completed(cache)]}"
    )
    assert not list(cache.glob(".index-cache-*")), (
        "failed-save cleanup invariant violated: owned staging path remains, "
        f"entries={[p.name for p in cache.iterdir()]}"
    )

    def fail_replace(_source, _target):
        raise OSError("synthetic atomic rename failure")

    monkeypatch.setattr("recall.index_cache.os.replace", fail_replace)
    assert saveIndexSnapshot(cache, namespace, fingerprint, _BytesIndex(b"valid")) is False, (
        "rename failure contract violated: atomic publication failure must report "
        "a cache miss indicator"
    )
    assert _completed(cache) == [], (
        "rename failure atomicity invariant violated: no final graph may appear, "
        f"files={[p.name for p in _completed(cache)]}"
    )
    assert not list(cache.glob(".index-cache-*")), (
        "rename failure cleanup invariant violated: owned staging path remains, "
        f"entries={[p.name for p in cache.iterdir()]}"
    )


def test_unprivate_or_non_directory_cache_is_a_miss(tmp_path):
    """B3/M2/P3: unsafe preexisting paths remain untouched."""
    insecure = tmp_path / "insecure"
    insecure.mkdir()
    os.chmod(insecure, 0o755)
    namespace = "d" * 64
    fingerprint = "e" * 64
    assert saveIndexSnapshot(insecure, namespace, fingerprint, _BytesIndex(b"x")) is False, (
        "unprivate-directory safety invariant violated: mode 0755 must reject "
        "publication instead of being chmodded"
    )
    assert stat.S_IMODE(insecure.stat().st_mode) == 0o755, (
        "permission-preservation invariant violated: helper chmodded an existing "
        f"user directory, mode={stat.S_IMODE(insecure.stat().st_mode):04o}"
    )
    as_file = tmp_path / "cache-file"
    as_file.write_bytes(b"not a directory")
    assert loadIndexSnapshot(as_file, namespace, fingerprint, lambda _p: object()) is None, (
        "non-directory cache invariant violated: a regular file path must miss "
        f"without loader invocation, path={as_file}"
    )
    symlink_target = tmp_path / "target"
    symlink_target.mkdir()
    symlink = tmp_path / "cache-link"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    assert saveIndexSnapshot(symlink, namespace, fingerprint, _BytesIndex(b"x")) is False, (
        "symlink-cache safety invariant violated: save must reject cache-dir "
        f"symlink, path={symlink}"
    )
    assert not list(symlink_target.iterdir()), (
        "symlink safety invariant violated: helper wrote through cache-dir "
        f"symlink, target_entries={[p.name for p in symlink_target.iterdir()]}"
    )
    invalid_cache = tmp_path / "invalid-cache"
    assert loadIndexSnapshot(invalid_cache, "../" + "0" * 62, fingerprint, lambda _p: object()) is None, (
        "fixed-hex validation invariant violated: path-shaped namespace must miss "
        f"without touching disk, namespace={'../' + '0' * 62!r}"
    )
    assert not invalid_cache.exists(), (
        "invalid-input isolation invariant violated: rejected namespace caused "
        f"cache directory creation, path={invalid_cache}"
    )
    assert saveIndexSnapshot(invalid_cache, "../" + "0" * 62, fingerprint, _BytesIndex(b"x")) is False, (
        "fixed-hex save validation invariant violated: path-shaped namespace must "
        "reject publication without creating a cache directory"
    )
    candidate_cache = tmp_path / "candidate-cache"
    candidate_cache.mkdir()
    os.chmod(candidate_cache, 0o700)
    nonregular = candidate_cache / f"{namespace}.{fingerprint}.{'f' * 64}.usearch"
    nonregular.mkdir()
    calls = []
    assert loadIndexSnapshot(
        candidate_cache, namespace, fingerprint,
        lambda path: calls.append(path) or object(),
    ) is None, (
        "nonregular-snapshot safety invariant violated: a directory with a cache "
        f"filename must miss, path={nonregular}"
    )
    assert calls == [], (
        "nonregular-snapshot ordering invariant violated: loader ran for a "
        f"directory entry, calls={calls!r}"
    )


def test_real_usearch_snapshot_round_trip_over_2000_vectors(tmp_path):
    """P2/X2: a real USearch graph survives save, hash check, and native load."""
    pytest.importorskip("usearch")
    from usearch.index import Index
    import numpy as np

    count, dimension = 2_000, 16
    rng = np.random.default_rng(20260905)
    matrix = rng.standard_normal((count, dimension)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    rows = [(f"atom-{i:04d}", matrix[i].tobytes()) for i in range(count)]
    params = {
        "version": 1,
        "construction": {
            "connectivity": 16,
            "expansion_add": 128,
            "threads": 1,
        },
        "search": {"expansion_search": 1024},
        "dtype": "f32",
    }
    namespace, fingerprint = indexFingerprint(rows, "test-model", ("code",), params)
    index = Index(
        ndim=dimension,
        metric="cos",
        dtype="f32",
        connectivity=16,
        expansion_add=128,
        expansion_search=1024,
    )
    index.add(np.arange(count, dtype=np.uint64), matrix, threads=1)
    assert saveIndexSnapshot(tmp_path / "cache", namespace, fingerprint, index) is True, (
        "real-usearch publication invariant violated: 2,000-vector graph did not save"
    )

    def loader(path):
        restored = Index(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_add=128,
            expansion_search=1024,
        )
        restored.load(path)
        return restored

    restored = loadIndexSnapshot(tmp_path / "cache", namespace, fingerprint, loader)
    assert restored is not None, (
        "real-usearch load invariant violated: matching 2,000-vector snapshot "
        "must load after content-hash verification"
    )
    assert restored.size == count, (
        "real-usearch cardinality invariant violated: loaded graph size differs, "
        f"expected={count} got={restored.size}"
    )
    matches = restored.search(matrix[137], 3)
    assert int(matches.keys[0]) == 137, (
        "real-usearch key-search invariant violated: an exact row query must "
        f"return its own key first, key={int(matches.keys[0])}"
    )
