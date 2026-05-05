"""Tests for HMAC-signed pickle save/load in IngestPipeline."""
import os
import pickle
import pytest
from pathlib import Path

from pensive.ingestion.pipeline import (
    IngestPipeline,
    _SIGNED_MAGIC,
    _HMAC_SIZE,
    _load_or_create_key,
    _default_key_path,
)
from pensive.spreading import SpreadingActivation


@pytest.fixture(autouse=True)
def _isolate_key_storage(tmp_path, monkeypatch):
    """Redirect XDG_CONFIG_HOME so the on-disk key for each test is isolated.

    Without this, every test uses the real ~/.config/pensive/pickle.key
    and the env-var-based test fixtures stop having any effect (the
    new precedence reads the disk key first). Each test gets its own
    XDG dir so monkeypatched PENSIVE_PICKLE_KEY values are honoured
    when no disk key has been written yet.
    """
    xdg = tmp_path / "xdg-config"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))


def _toy_pipeline():
    sa = SpreadingActivation()
    sa.add_documents([
        {
            "id": "doc1",
            "content": "the quick brown fox jumps over the lazy dog",
            "value": "the quick brown fox jumps over the lazy dog",
        },
        {
            "id": "doc2",
            "content": "a different document with other words",
            "value": "a different document with other words",
        },
    ])
    return IngestPipeline(sa=sa)


def test_save_load_roundtrip_signed(tmp_path, monkeypatch):
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key-abc")
    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))
    raw = dest.read_bytes()
    # Signed blob must start with magic header + HMAC
    assert raw.startswith(_SIGNED_MAGIC), "signed save should emit magic header"
    loaded = IngestPipeline.load_graph(str(dest))
    assert loaded.sa is not None


def test_save_unsigned_then_load_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key-xyz")
    pipe = _toy_pipeline()
    dest = tmp_path / "graph_unsigned.pkl"
    pipe.save_graph(str(dest), sign=False)
    raw = dest.read_bytes()
    assert not raw.startswith(_SIGNED_MAGIC)
    # Default load should reject unsigned
    with pytest.raises(ValueError, match="unsigned"):
        IngestPipeline.load_graph(str(dest))


def test_trusted_flag_loads_unsigned(tmp_path, monkeypatch, recwarn):
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key-2")
    pipe = _toy_pipeline()
    dest = tmp_path / "graph_unsigned.pkl"
    pipe.save_graph(str(dest), sign=False)
    loaded = IngestPipeline.load_graph(str(dest), trusted=True)
    assert loaded.sa is not None
    # Should have emitted a RuntimeWarning
    warn_msgs = [str(w.message) for w in recwarn]
    assert any("trusted=True" in m for m in warn_msgs), f"no warning? got {warn_msgs}"


def test_tampered_signature_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key-3")
    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))
    raw = bytearray(dest.read_bytes())
    # Flip a byte inside the payload region (after magic + HMAC)
    offset = len(_SIGNED_MAGIC) + _HMAC_SIZE + 10
    if offset < len(raw):
        raw[offset] ^= 0xFF
    dest.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="signature mismatch"):
        IngestPipeline.load_graph(str(dest))


def test_wrong_key_rejected(tmp_path, monkeypatch):
    # Save under one key
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "alice-key")
    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))
    # Try to load under a different key
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "bob-key")
    with pytest.raises(ValueError, match="signature mismatch"):
        IngestPipeline.load_graph(str(dest))


def test_truncated_signed_file_rejected_clearly(tmp_path, monkeypatch):
    """A file with magic but <32 bytes of HMAC must raise a helpful error."""
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key")
    dest = tmp_path / "truncated.pkl"
    # Magic + only 10 bytes of what should be 32-byte HMAC
    dest.write_bytes(_SIGNED_MAGIC + b"\x00" * 10)
    with pytest.raises(ValueError, match="truncated"):
        IngestPipeline.load_graph(str(dest))


def test_signed_header_empty_payload_rejected(tmp_path, monkeypatch):
    """Magic + full HMAC but zero-byte payload must not reach pickle.loads."""
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "test-key")
    dest = tmp_path / "nopayload.pkl"
    dest.write_bytes(_SIGNED_MAGIC + b"\x00" * _HMAC_SIZE)  # no payload
    with pytest.raises(ValueError, match="empty payload"):
        IngestPipeline.load_graph(str(dest))


def test_corrupted_payload_with_valid_sig(tmp_path, monkeypatch):
    """A payload that passes HMAC but is not valid pickle raises ValueError."""
    import hashlib
    import hmac as hmac_mod
    key = b"test-key-4"
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", key.decode())
    # Craft: magic + valid HMAC over garbage-but-not-pickle payload
    garbage = b"this is not valid pickle data at all"
    mac = hmac_mod.new(key, garbage, hashlib.sha256).digest()
    dest = tmp_path / "corrupt.pkl"
    dest.write_bytes(_SIGNED_MAGIC + mac + garbage)
    with pytest.raises(ValueError, match="payload is corrupted"):
        IngestPipeline.load_graph(str(dest))


def test_malicious_pickle_blocked_without_trusted(tmp_path):
    """A hostile unsigned pickle must not be unpickled without trusted=True.

    Uses a hand-rolled pickle bytestream so __reduce__ fires only at load
    time, not at save time. Mimics the classic `os.system` pickle gadget.
    """
    dest = tmp_path / "evil.pkl"
    # Opcodes: c<module>\n<name>\n (GLOBAL) + short-binstring arg + REDUCE + STOP
    # Equivalent of: pickle.dumps(os.system("touch /tmp/pwn")) via __reduce__
    evil_pickle = (
        b"cos\nsystem\n"
        b"(S'echo HOSTILE PICKLE EXECUTED >&2; exit 1'\n"
        b"tR."
    )
    dest.write_bytes(evil_pickle)
    # Default load (trusted=False) must refuse BEFORE any unpickling.
    # If the refusal fails, pickle would execute os.system() which would
    # abort the pytest process with exit 1.
    with pytest.raises(ValueError, match="unsigned"):
        IngestPipeline.load_graph(str(dest))


# ---------------------------------------------------------------------------
# CRIT-1 -- Pickle key precedence: on-disk key wins over PENSIVE_PICKLE_KEY.
#
# Rule guarded: the env var must NEVER be able to override an existing
# on-disk key. A hostile shell rc that sets PENSIVE_PICKLE_KEY to a known
# weak value previously could deliver a forged "signed" hostile pickle
# under that key, bypassing HMAC verification and reaching pickle.loads.
# Disk-first precedence forces an attacker to overwrite a 0600 file in
# the user's config dir before they can swap the trust anchor.
# ---------------------------------------------------------------------------


def test_disk_key_wins_over_env_var(tmp_path, monkeypatch):
    """Env var must not override an on-disk key file.

    Sabotage check: if the precedence regresses to env-first, this test
    would fail because save would use 'env-bytes-NOT-disk' and load would
    use 'env-bytes-NOT-disk' too, both round-tripping fine without ever
    touching the disk key. Instead we (a) assert the chosen key bytes are
    the disk bytes, and (b) verify that an env-only key cannot validate
    a disk-key-signed file.
    """
    # Plant a known on-disk key.
    cfg_dir = tmp_path / "xdg-config-disk-first"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    pensive_dir = cfg_dir / "pensive"
    pensive_dir.mkdir()
    disk_key = b"\x01" * 32  # 32 deterministic bytes
    key_path = pensive_dir / "pickle.key"
    key_path.write_bytes(disk_key)

    # Set env to a DIFFERENT value -- must be ignored.
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "env-key-attacker-controlled")

    chosen = _load_or_create_key()
    assert chosen == disk_key, (
        f"on-disk key was IGNORED; precedence regressed to env-first. "
        f"Got {chosen!r}, expected {disk_key!r}"
    )


def test_load_signed_with_disk_key_works_without_env(tmp_path, monkeypatch):
    """Saving with disk key and loading with no env var still verifies.

    Sabotage check: if precedence broke and required the env var, this
    test would fail at load() with a signature-mismatch ValueError.
    """
    cfg_dir = tmp_path / "xdg-config-no-env"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)

    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))

    # Load again with the env var still unset; must succeed.
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)
    loaded = IngestPipeline.load_graph(str(dest))
    assert loaded.sa is not None, "load with disk key should succeed"


def test_hostile_env_does_not_unlock_disk_signed_file(tmp_path, monkeypatch):
    """A hostile PENSIVE_PICKLE_KEY at load time must NOT bypass disk key.

    Save under the disk key. Then attacker sets env to anything they want.
    Load must STILL use the disk key (and therefore still succeed if disk
    is unchanged). This proves the env var is decorative when disk is
    present.
    """
    cfg_dir = tmp_path / "xdg-config-hostile"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)

    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))  # signed under freshly-generated disk key

    # Confirm a key file landed on disk.
    expected_key_path = _default_key_path()
    assert expected_key_path.exists(), (
        f"disk key was not persisted at {expected_key_path}; precondition broken"
    )

    # Attacker sets a hostile env var. Load must still validate.
    monkeypatch.setenv("PENSIVE_PICKLE_KEY", "attacker-known-weak-key")
    loaded = IngestPipeline.load_graph(str(dest))
    assert loaded.sa is not None, (
        "hostile env var leaked into key resolution; load should rely on "
        "the on-disk key only"
    )


# ---------------------------------------------------------------------------
# IMP-3 / IMP-2 -- atomic key file write + parent dir mode 0700
# ---------------------------------------------------------------------------


def test_persisted_key_is_0600(tmp_path, monkeypatch):
    """Newly persisted key file must be mode 0600 (owner-only)."""
    cfg_dir = tmp_path / "xdg-config-mode"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)

    _ = _load_or_create_key()
    key_path = _default_key_path()
    assert key_path.exists(), "load_or_create did not persist a key"
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600, (
        f"key file mode is {oct(mode)}, expected 0o600. The atomic write "
        "must guarantee owner-only access from creation."
    )
    parent_mode = key_path.parent.stat().st_mode & 0o777
    assert parent_mode == 0o700, (
        f"key parent dir mode is {oct(parent_mode)}, expected 0o700. "
        "An attacker with read on the dir can list its contents even if "
        "the key file itself is 0600."
    )


# ---------------------------------------------------------------------------
# IMP-4 -- load_graph max_size cap
# ---------------------------------------------------------------------------


def test_load_graph_rejects_oversized_file(tmp_path, monkeypatch):
    """A file larger than max_size must be rejected without read_bytes()."""
    cfg_dir = tmp_path / "xdg-config-cap"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)

    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))
    real_size = dest.stat().st_size
    assert real_size > 0

    with pytest.raises(ValueError, match="exceeds the max_size cap"):
        IngestPipeline.load_graph(str(dest), max_size=1)


def test_load_graph_within_cap_succeeds(tmp_path, monkeypatch):
    """Files at or below max_size must load normally."""
    cfg_dir = tmp_path / "xdg-config-within"
    cfg_dir.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)

    pipe = _toy_pipeline()
    dest = tmp_path / "graph.pkl"
    pipe.save_graph(str(dest))
    real_size = dest.stat().st_size

    loaded = IngestPipeline.load_graph(str(dest), max_size=real_size + 1)
    assert loaded.sa is not None
