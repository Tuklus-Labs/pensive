"""Auxiliary dense signal: a second embedding model fused as a third recall list.

The base dense signal (bge on this box) covers every kind-class; the aux signal
covers only the classes named in ``classNames`` -- the reasoning-memory class
today -- with a remote frontier embedder. Aux vectors live in the SAME
model-keyed ``embeddings`` table as the base model's (additive by schema), but
they are written externally (the batch backfill at
``~/Projects/pensive-embeddings/run_backfill.py``): the daemon only ever READS
aux vectors and embeds the QUERY at recall time. A new atom simply has no aux
vector until the next backfill and stays covered by the base signals meanwhile.

Failure posture (load-bearing): the aux signal is optional evidence, never a
dependency. Construction failing (no key, no SDK) disables the feature at
daemon startup; a query-time embed failing (network, timeout) degrades that one
call to the base signals. Neither path may ever fail a recall.
"""
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from recall.strata import KIND_CLASSES
from recall.vector_index import selectIndex

__all__ = ["OpenAIEmbedder", "AuxDense", "readOpenAiKey"]

# Known model widths; an unknown model discovers its width from the first reply.
_KNOWN_DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}

# Query-time embed budget. Fail fast: the aux signal is optional, so a slow API
# costs one bounded wait, never a hang; no SDK-level retries for the same reason.
_TIMEOUT_SECONDS = 2.5

# Single-text embed cache: recall queries repeat across hooks and sessions, and
# a hit saves a network round-trip. FIFO-evicted at this cap.
_CACHE_CAP = 512

# The box's one key store, sourced by zshrc for shells; the systemd user unit
# never sources zshrc, so the daemon parses the file itself when the env is bare.
_KEYS_FILE = Path.home() / ".keys"
_KEY_RE = re.compile(r'^\s*(?:export\s+)?OPENAI_API_KEY=(["\']?)(.+?)\1\s*$')


def readOpenAiKey():
    """OPENAI_API_KEY from the environment, else parsed from ``~/.keys``, else None.

    The key stays out of the unit file and the journal; it lives only in the
    process and the file that already held it.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        for line in _KEYS_FILE.read_text().splitlines():
            m = _KEY_RE.match(line)
            if m:
                return m.group(2)
    except OSError:
        pass
    return None


class OpenAIEmbedder:
    """Duck-type twin of ``recall.embedder.Embedder`` over the OpenAI API.

    Same three-attribute contract the engine reads: ``embed`` / ``modelId`` /
    ``dim``. Raises ``RuntimeError`` at construction when the key or SDK is
    unavailable -- the caller treats that as "feature off", never as fatal.
    ``embed`` raises on API failure; the engine catches and degrades. Returned
    vectors are unit-normalized by the API, so they satisfy the same
    dot-product-is-cosine invariant the local embedder guarantees.
    """

    def __init__(self, modelId, timeout=_TIMEOUT_SECONDS):
        key = readOpenAiKey()
        if not key:
            raise RuntimeError("no OPENAI_API_KEY in env or ~/.keys")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(f"openai SDK unavailable: {exc}")
        self._client = OpenAI(api_key=key, timeout=timeout, max_retries=0)
        self.modelId = modelId
        self.dim = _KNOWN_DIMS.get(modelId)
        self._cache = OrderedDict()

    def embed(self, texts):
        if not texts:
            return []
        if len(texts) == 1 and texts[0] in self._cache:
            self._cache.move_to_end(texts[0])
            return [self._cache[texts[0]].copy()]
        resp = self._client.embeddings.create(
            model=self.modelId, input=list(texts))
        data = sorted(resp.data, key=lambda d: d.index)
        vecs = [np.asarray(d.embedding, dtype=np.float32) for d in data]
        if self.dim is None and vecs:
            self.dim = int(vecs[0].shape[0])
        if len(texts) == 1 and vecs:
            self._cache[texts[0]] = vecs[0].copy()
            if len(self._cache) > _CACHE_CAP:
                self._cache.popitem(last=False)
        return vecs


class AuxDense:
    """A second (embedder, per-class indexes) pair the engine fuses as one more list.

    ``classNames`` scopes which kind-classes carry an aux index; every other
    class stays base-only. ``buildIndexes`` loads whatever vectors the external
    backfill has landed -- an empty backfill yields a valid empty index whose
    ``search`` returns ``[]``, which the engine reads as no aux evidence.
    ``reindex`` mirrors ``ServeContext.reindex`` scoping, so a memory-class emit
    refreshes only the memory aux index (a cheap read of existing rows, never an
    API call).
    """

    def __init__(self, embedder, classNames=("memory",)):
        self.embedder = embedder
        self.classNames = tuple(classNames)
        self.indexes = {}

    def buildIndexes(self, store):
        for name, kinds in KIND_CLASSES:
            if name in self.classNames:
                self.indexes[name] = selectIndex(
                    store, self.embedder.modelId, kinds)
        return self

    def reindex(self, store, kinds=None):
        if kinds is None:
            self.buildIndexes(store)
            return
        wanted = set(kinds)
        for name, classKinds in KIND_CLASSES:
            if name in self.classNames and wanted.intersection(classKinds):
                self.indexes[name] = selectIndex(
                    store, self.embedder.modelId, classKinds)
