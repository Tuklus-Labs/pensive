"""Best-effort private cache for complete native vector-index snapshots."""

import hashlib
import json
import os
import re
import stat
import struct
import tempfile
from pathlib import Path


__all__ = ["indexFingerprint", "loadIndexSnapshot", "saveIndexSnapshot"]

_FORMAT_VERSION = 1
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_SNAPSHOT_NAME = re.compile(
    r"\A(?P<namespace>[0-9a-f]{64})\.(?P<fingerprint>[0-9a-f]{64})\."
    r"(?P<sha>[0-9a-f]{64})\.usearch\Z"
)
_STAGING_PREFIX = ".index-cache-"


def _field(value):
    return struct.pack(">Q", len(value)) + value


def _bytes(value, field_name):
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{field_name} must be text or bytes-like")


def _config(model_id, kinds, parameters):
    try:
        ordered_kinds = None if kinds is None else sorted(tuple(kinds))
        return json.dumps(
            {
                "format_version": _FORMAT_VERSION,
                "model_id": model_id,
                "kinds": ordered_kinds,
                "parameters": parameters,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("model, kinds, and parameters must be JSON-canonicalizable") from exc


def indexFingerprint(rows, modelId, kinds, parameters):
    """Return lowercase ``(namespace, fingerprint)`` for ordered source rows."""
    config = _config(modelId, kinds, parameters)
    namespace = hashlib.sha256(b"pensive-index-namespace\0" + _field(config)).hexdigest()
    digest = hashlib.sha256(b"pensive-index-source\0" + _field(config))
    rows = list(rows)
    digest.update(struct.pack(">Q", len(rows)))
    for row in rows:
        try:
            atom_id, vector_blob = row
        except (TypeError, ValueError) as exc:
            raise TypeError("rows must contain (atom_id, vector_blob) pairs") from exc
        digest.update(_field(_bytes(atom_id, "atom_id")))
        digest.update(_field(_bytes(vector_blob, "vector_blob")))
    return namespace, digest.hexdigest()


def _valid_hex(value):
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _private_cache_dir(cacheDir, create):
    if cacheDir is None:
        return None
    try:
        path = Path(cacheDir)
        info = path.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            path.mkdir(mode=_PRIVATE_DIR_MODE)
            info = path.lstat()
        except (OSError, ValueError):
            return None
    except (OSError, ValueError, TypeError):
        return None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return None
    return path if stat.S_IMODE(info.st_mode) == _PRIVATE_DIR_MODE else None


def _parts(path):
    match = _SNAPSHOT_NAME.fullmatch(path.name)
    return match.groupdict() if match else None


def _private_file(path):
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and stat.S_IMODE(info.st_mode) == _PRIVATE_FILE_MODE
    )


def _files(cache, namespace, fingerprint=None):
    try:
        entries = list(cache.iterdir())
    except OSError:
        return []
    found = []
    for path in entries:
        parts = _parts(path)
        if not parts or parts["namespace"] != namespace:
            continue
        if fingerprint is not None and parts["fingerprint"] != fingerprint:
            continue
        if _private_file(path):
            try:
                found.append((path, path.lstat().st_mtime_ns))
            except OSError:
                pass
    return sorted(found, key=lambda item: (item[1], item[0].name), reverse=True)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_file(path):
    with path.open("r+b") as target:
        os.fsync(target.fileno())


def _sync_dir(cache):
    descriptor = os.open(cache, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_stage(path):
    if path is None or not path.name.startswith(_STAGING_PREFIX):
        return
    try:
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            path.unlink()
    except OSError:
        pass


def _retain_two(cache, namespace, published):
    # Keep the publication even if the wall clock moved backward. Invalid
    # content remains untouched and does not consume a valid retention slot.
    files = _files(cache, namespace)
    files.sort(key=lambda item: item[0] != published)
    retained = 0
    for path, _mtime in files:
        try:
            parts = _parts(path)
            if _private_file(path) and _sha(path) == parts["sha"]:
                retained += 1
                if retained > 2:
                    path.unlink()
        except OSError:
            pass


def loadIndexSnapshot(cacheDir, namespace, fingerprint, loader):
    """Call ``loader(path)`` for the newest verified snapshot, else return None."""
    if not _valid_hex(namespace) or not _valid_hex(fingerprint):
        return None
    cache = _private_cache_dir(cacheDir, create=False)
    if cache is None:
        return None
    for path, _mtime in _files(cache, namespace, fingerprint):
        parts = _parts(path)
        try:
            if _sha(path) != parts["sha"]:
                continue
            loaded = loader(path)
            if loaded is not None:
                return loaded
        except Exception:
            continue
    return None


def saveIndexSnapshot(cacheDir, namespace, fingerprint, index):
    """Atomically publish one private snapshot; return False on cache failure."""
    if not _valid_hex(namespace) or not _valid_hex(fingerprint):
        return False
    cache = _private_cache_dir(cacheDir, create=True)
    if cache is None:
        return False
    stage = None
    try:
        fd, name = tempfile.mkstemp(prefix=_STAGING_PREFIX, suffix=".tmp", dir=cache)
        os.close(fd)
        stage = Path(name)
        os.chmod(stage, _PRIVATE_FILE_MODE)
        index.save(stage)
        if not _private_file(stage):
            return False
        os.chmod(stage, _PRIVATE_FILE_MODE)
        _sync_file(stage)
        file_sha = _sha(stage)
        final = cache / f"{namespace}.{fingerprint}.{file_sha}.usearch"
        try:
            final.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        else:
            if not _private_file(final):
                return False
        os.replace(stage, final)
        stage = None
        _sync_dir(cache)
        _retain_two(cache, namespace, final)
        _sync_dir(cache)
        return True
    except Exception:
        return False
    finally:
        _cleanup_stage(stage)
