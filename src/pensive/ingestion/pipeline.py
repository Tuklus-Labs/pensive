"""Ingestion pipeline orchestrator for spreading activation.

Coordinates parsing from multiple data sources, builds the SA graph
incrementally, and provides pickle-based serialization for persistence.

SECURITY NOTE on pickle:

    pickle.load() on untrusted data is arbitrary code execution. Graphs
    saved with ``save_graph()`` by default include an HMAC signature
    derived from a secret read from the ``PENSIVE_PICKLE_KEY`` environment
    variable (or a locally-generated key at ``~/.config/pensive/pickle.key``
    when the env var is not set). ``load_graph()`` verifies the signature
    before unpickling. Files with a valid signature are treated as
    originating from a trusted source.

    To load a legacy (unsigned) graph produced by an older Pensive, set
    ``trusted=True``. Do this ONLY if you are certain the file is from a
    trusted source -- a malicious .pkl can execute arbitrary code.
"""
import hashlib
import hmac
import logging
import os
import pickle
import secrets
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..spreading import SpreadingActivation, SpreadingConfig
from ..patterns import REAL_DATA_PATTERNS
from .base import BaseParser

logger = logging.getLogger(__name__)


# Magic prefix so we can distinguish signed graphs from legacy unsigned pickles.
_SIGNED_MAGIC = b"PENSIVE-SIGNED-V1\x00"
_HMAC_SIZE = 32  # sha256


def _default_key_path() -> Path:
    # os.environ.get with a default does NOT fall back if the env var is
    # set to an empty string. Treat empty as unset for a predictable path.
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if not xdg:
        xdg = str(Path.home() / ".config")
    return Path(xdg) / "pensive" / "pickle.key"


def _load_or_create_key() -> bytes:
    """Get the HMAC key from env or a locally-generated file (0600).

    The key is used only to verify that graphs were saved by this user;
    it is not a cryptographic secret in the authentication sense.

    A trailing newline / whitespace in the key file is stripped so that a
    manually-edited file matches what save_graph wrote.
    """
    env = os.environ.get("PENSIVE_PICKLE_KEY")
    if env:
        return env.encode("utf-8")
    path = _default_key_path()
    if path.exists():
        return path.read_bytes().rstrip()
    # Generate a fresh key, persist at 0600. This is best-effort; if we
    # can't write, we still return a one-shot key so save/load in the
    # same process works.
    key = secrets.token_bytes(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_suffix(".key.tmp")
        tmp.write_bytes(key)
        tmp.chmod(0o600)
        tmp.rename(path)
    except OSError as e:
        logger.warning("could not persist pensive pickle key to %s: %s", path, e)
    return key


class IngestPipeline:
    """Orchestrate ingestion from all sources into a SpreadingActivation graph."""

    def __init__(
        self,
        sa: Optional[SpreadingActivation] = None,
        batch_size: int = 1000,
        progress_callback: Optional[Callable] = None,
    ):
        self.sa = sa or SpreadingActivation(
            config=SpreadingConfig(
                max_hops=2,
                max_active=200,
            ),
            patterns=REAL_DATA_PATTERNS,
        )
        self.batch_size = batch_size
        self.progress = progress_callback or (lambda *a: None)
        self.stats: Dict[str, int] = defaultdict(int)

    def ingest_source(self, parser: BaseParser) -> int:
        """Ingest all documents from a single source."""
        name = parser.source_name()
        count = 0
        batch = []
        t0 = time.time()

        for doc in parser.parse():
            batch.append(doc.to_sa_dict())
            count += 1
            if len(batch) >= self.batch_size:
                self.sa.add_documents(batch)
                batch = []
                self.progress(name, count)

        if batch:
            self.sa.add_documents(batch)

        elapsed = time.time() - t0
        self.stats[name] = count
        logger.info("%s: ingested %d docs in %.1fs", name, count, elapsed)
        return count

    def ingest_all(self, parsers: List[BaseParser]) -> Dict[str, int]:
        """Ingest all sources sequentially."""
        for parser in parsers:
            name = parser.source_name()
            self.progress(f"Starting {name}", 0)
            n = self.ingest_source(parser)
            self.progress(f"Finished {name}", n)
        return dict(self.stats)

    def save_graph(self, path: str, sign: bool = True) -> None:
        """Serialize the SA graph to disk.

        When ``sign`` is True (default) the pickle blob is prefixed with a
        magic header + HMAC-SHA256 over the payload. ``load_graph`` will
        verify the signature before unpickling.
        """
        data = self.sa.get_save_data()
        data['pipeline_stats'] = dict(self.stats)
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path, 'wb') as f:
            if sign:
                key = _load_or_create_key()
                mac = hmac.new(key, payload, hashlib.sha256).digest()
                f.write(_SIGNED_MAGIC)
                f.write(mac)
            f.write(payload)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        logger.info("Graph saved to %s (%.1f MB, signed=%s)", path, size_mb, sign)

    @classmethod
    def load_graph(cls, path: str, trusted: bool = False) -> 'IngestPipeline':
        """Load a previously built graph.

        By default the file must carry a valid HMAC signature (produced by
        ``save_graph()``). To load a legacy unsigned graph, pass
        ``trusted=True`` -- this bypasses signature verification and will
        unpickle the file. Do this ONLY if you trust the file's origin;
        a hostile pickle executes arbitrary code when loaded.
        """
        raw = Path(path).read_bytes()
        if raw.startswith(_SIGNED_MAGIC):
            header_len = len(_SIGNED_MAGIC)
            min_size = header_len + _HMAC_SIZE
            if len(raw) < min_size:
                raise ValueError(
                    f"Graph at {path} is truncated: expected at least "
                    f"{min_size} bytes (magic + HMAC), got {len(raw)}"
                )
            mac = raw[header_len:header_len + _HMAC_SIZE]
            payload = raw[header_len + _HMAC_SIZE:]
            if not payload:
                raise ValueError(
                    f"Graph at {path} has header but empty payload"
                )
            key = _load_or_create_key()
            expected = hmac.new(key, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise ValueError(
                    f"Graph signature mismatch for {path}: this file was "
                    "not produced by this user/key. Refusing to unpickle. "
                    "If you trust the source, pass trusted=True."
                )
            try:
                data = pickle.loads(payload)
            except (EOFError, pickle.UnpicklingError) as e:
                raise ValueError(
                    f"Graph at {path} has valid signature but payload "
                    f"is corrupted: {e}"
                ) from e
        else:
            if not trusted:
                raise ValueError(
                    f"Graph at {path} is unsigned and trusted=False. "
                    "Unsigned pickles can execute arbitrary code. Either "
                    "re-save with save_graph() to sign it, or pass "
                    "trusted=True if you know the file is safe."
                )
            warnings.warn(
                "Loading unsigned graph with trusted=True. pickle.load on "
                "an untrusted file is arbitrary code execution.",
                RuntimeWarning,
                stacklevel=2,
            )
            data = pickle.loads(raw)
        sa = SpreadingActivation.from_save_data(data)
        pipe = cls(sa=sa)
        pipe.stats = data.get('pipeline_stats', data.get('stats', {}))
        return pipe
