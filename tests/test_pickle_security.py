"""Tests for HMAC-signed pickle save/load in IngestPipeline."""
import os
import pickle
import pytest
from pathlib import Path

from pensive.ingestion.pipeline import (
    IngestPipeline,
    _SIGNED_MAGIC,
    _HMAC_SIZE,
)
from pensive.spreading import SpreadingActivation


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
