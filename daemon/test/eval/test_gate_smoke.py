"""Fast smoke test that the eval gate runs against the new recall signature.

The full 1,500-query gate needs the ChatGPT export and heavy GPU; this proves
gate() is wired to the stratified recall (indexes dict) on a tiny store so a
subagent can verify without the big run. The full-gate command is in the plan.

Placed under test/eval/ (not test/recall/, per the task brief's own fallback
note) because only test/eval/conftest.py puts the daemon root on sys.path,
which `import eval.gate` needs; test/recall/ only inherits test/conftest.py
(daemon/src only), so the import would fail to resolve there.
"""
import pytest

from recall.embedder import Embedder, embedMissing
from recall.vector_index import buildClassIndexes
from store.store import openStore, putAtom
import eval.gate as gate_mod

MODEL_ID = "BAAI/bge-small-en-v1.5"

pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder(MODEL_ID)


def test_gate_runs_with_indexes_dict(tmp_path, embedder):
    store = openStore(tmp_path / "mem.db")
    try:
        ids = []
        for i in range(6):
            ids.append(putAtom(store, {
                "text": f"document chunk number {i} about routing and dispatch",
                "kind": "document_chunk", "project": "aegis",
                "importance": 0.0, "provenance": {"source": "bulk-import"},
            }))
        embedMissing(store, embedder)
        indexes = buildClassIndexes(store, MODEL_ID)
        queries = [{"query": "routing and dispatch",
                    "relevant": set(), "own": set()}]
        out = gate_mod.gate(store, indexes, embedder, queries)
        # Metrics dict is well formed and the run did not raise.
        assert "r_at_10" in out and "n" in out and out["n"] == 1
    finally:
        store.close()
