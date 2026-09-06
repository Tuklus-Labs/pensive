from types import SimpleNamespace

import numpy as np
import pytest

import serve.daemon as daemon
import serve.mcp as mcp
import recall.vector_index as indexes
from recall.embedder import vecToBlob
from store.store import openStore, putAtom


@pytest.mark.parametrize('setting,expected', [(None, 4), ('2', 2), ('0', 0), ('bad', 4)])
def test_blas_budget(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv('PENSIVE_V3_BLAS_THREADS', raising=False)
    else:
        monkeypatch.setenv('PENSIVE_V3_BLAS_THREADS', setting)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '24')
    calls = []
    monkeypatch.setattr('threadpoolctl.threadpool_limits', lambda **kw: calls.append(kw))
    assert daemon.configureBlasThreads() == expected, 'daemon BLAS budget honors default, override and opt-out'
    assert calls == ([{'limits': expected, 'user_api': 'blas'}] if expected else []), 'only BLAS pools in this process are capped'
    assert daemon.os.environ['OPENBLAS_NUM_THREADS'] == '24', 'thread control must not rewrite the inherited environment'


@pytest.mark.parametrize('override', [None, '', 'custom'])
def test_bootstrap_cache_settings(tmp_path, monkeypatch, override):
    store_path = tmp_path / 'memory.db'
    monkeypatch.setenv('PENSIVE_V3_STORE', str(store_path))
    monkeypatch.delenv('PENSIVE_V3_OPENAI_MODEL', raising=False)
    if override is None:
        monkeypatch.delenv('PENSIVE_V3_INDEX_CACHE_DIR', raising=False)
    else:
        monkeypatch.setenv('PENSIVE_V3_INDEX_CACHE_DIR', '' if not override else str(tmp_path / override))
    monkeypatch.setattr(daemon, 'openStore', lambda path: object())
    monkeypatch.setattr(daemon, 'makeEmbedder', lambda model: SimpleNamespace(device='cpu'))
    monkeypatch.setattr(daemon, '_warmReranker', lambda: True)
    monkeypatch.setattr(daemon, 'configureBlasThreads', lambda: 4)
    calls = []
    monkeypatch.setattr(daemon, 'ServeContext', lambda *a, **kw: calls.append(kw) or object())
    daemon.buildContext()
    expected = tmp_path / 'index-cache' if override is None else (tmp_path / override if override else None)
    assert calls[0]['indexCacheDir'] == expected, 'snapshot cache is per-store with an explicit opt-out'


def test_startup_does_not_preload_unused_reranker(tmp_path, monkeypatch):
    monkeypatch.setenv('PENSIVE_V3_STORE', str(tmp_path / 'memory.db'))
    monkeypatch.delenv('PENSIVE_V3_OPENAI_MODEL', raising=False)
    monkeypatch.delenv('PENSIVE_V3_PRELOAD_RERANKER', raising=False)
    monkeypatch.setattr(daemon, 'openStore', lambda path: object())
    monkeypatch.setattr(daemon, 'makeEmbedder', lambda model: SimpleNamespace(device='cpu'))
    monkeypatch.setattr(daemon, 'configureBlasThreads', lambda: 4)
    monkeypatch.setattr(daemon, 'ServeContext', lambda *a, **kw: object())
    calls = []
    monkeypatch.setattr(daemon, '_warmReranker', lambda: calls.append('loaded'))
    daemon.buildContext()
    assert calls == [], 'served L2/L3 must not allocate an unused cross-encoder at startup'
    monkeypatch.setenv('PENSIVE_V3_PRELOAD_RERANKER', '1')
    daemon.buildContext()
    assert calls == ['loaded'], 'an explicit preload remains available for workflows that use reranking'


def test_reindex_forwards_cache(tmp_path, monkeypatch):
    ctx = mcp.ServeContext.__new__(mcp.ServeContext)
    ctx.store, ctx.embedder, ctx.modelId, ctx.aux = object(), object(), 'm', None
    ctx.indexCacheDir = tmp_path / 'cache'
    calls = []
    monkeypatch.setattr(mcp, 'embedMissing', lambda *a: 0)
    monkeypatch.setattr(mcp, 'buildClassIndexes', lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr(mcp, 'selectIndex', lambda *a, **kw: calls.append(kw) or object())
    ctx.reindex()
    ctx.reindex(kinds=('atom',))
    assert calls == [{'cacheDir': ctx.indexCacheDir}] * 2, 'both full and class rebuilds use the configured snapshot cache'
    ctx.indexCacheDir = None
    calls.clear()
    ctx.reindex()
    assert calls == [{}], 'ordinary library contexts retain the uncached factory contract'


def test_hot_growth_promotes_exact_index(tmp_path, monkeypatch):
    # A small configured boundary exercises real store/index transitions without
    # constructing 32k rows just to prove an inclusive-boundary rule.
    monkeypatch.setattr(indexes, 'EXACT_MEMORY_FLAT_MAX', 3)
    store = openStore(tmp_path / 'growth.db')
    try:
        atoms = []
        for number in range(3):
            atom = putAtom(store, {'text': str(number), 'kind': 'atom',
                                  'provenance': {'source': 'test'}})
            vector = np.eye(4, dtype=np.float32)[number]
            store._conn.execute('INSERT INTO embeddings(atom_id,model_id,vector,embedded_at) VALUES (?,?,?,?)',
                                (atom, 'test', vecToBlob(vector), 1))
            store._conn.commit(); atoms.append(atom)
        embedder = SimpleNamespace(modelId='test', embed=lambda texts: np.array([[0., 0., 0., 1.]], dtype=np.float32))
        ctx = mcp.ServeContext(store, embedder, 'test')
        assert isinstance(ctx.indexes['memory'], indexes.FlatIndex), 'memory remains exact through the configured inclusive limit'
        fourth = putAtom(store, {'text': 'fourth', 'kind': 'atom', 'provenance': {'source': 'test'}})
        ctx.indexAtom(fourth, 'atom')
        from recall.hnsw_index import HnswIndex
        assert isinstance(ctx.indexes['memory'], HnswIndex), 'a live daemon must promote on the first write beyond the exact limit'
        hits = ctx.indexes['memory'].search(np.array([0., 0., 0., 1.]), 4)
        assert hits[0][0] == fourth and {i for i, _ in hits} == {*atoms, fourth}, 'promotion retains every live atom and makes the triggering write recallable'
    finally:
        store.close()


def test_optional_snapshot_failure_keeps_built_index(tmp_path, monkeypatch):
    import recall.hnsw_index as hnsw
    store = openStore(tmp_path / 'cache-failure.db')
    try:
        atom = putAtom(store, {'text': 'one', 'kind': 'atom', 'provenance': {'source': 'test'}})
        vector = np.array([1., 0.], dtype=np.float32)
        store._conn.execute('INSERT INTO embeddings(atom_id,model_id,vector,embedded_at) VALUES (?,?,?,?)',
                            (atom, 'test', vecToBlob(vector), 1))
        store._conn.commit()
        def fail(*args):
            raise RuntimeError('optional cache failure')
        monkeypatch.setattr(hnsw, 'saveIndexSnapshot', fail)
        index = hnsw.HnswIndex().build(store, 'test', cacheDir=tmp_path / 'cache')
        assert index.search(vector, 1)[0][0] == atom, 'optional snapshot failure cannot discard a correctly built canonical index'
    finally:
        store.close()
