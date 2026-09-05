"""Contracts: task scope, caller versus author, admission and explicit feedback."""
import json
from types import SimpleNamespace

import pytest

import serve.mcp as mcp
from store.store import openStore, putAtom


@pytest.fixture
def ctx(tmp_path):
    store = openStore(tmp_path / 'tools.db')
    yield SimpleNamespace(store=store, agent=None, indexes={}, embedder=None,
                          aux=None, recallLogErrors=0)
    store.close()


def call(ctx, name, args):
    result, error = mcp.dispatch(ctx, name, args)
    assert not error, f'{name} must accept this valid operation: {result}'
    return result.value if isinstance(result, mcp.StructuredResult) else json.loads(result)


def test_checkpoint_state_transport_identity_and_stale_cas(ctx, monkeypatch):
    monkeypatch.setattr(mcp, '_transportAgent', lambda: 'codex')
    args = dict(project='pensive', taskId='root-task', agent='pretend',
                requestId='one', expectedRevision=0, state='active', body='Working.')
    first = call(ctx, 'task_checkpoint', args)
    assert first['agent'] == 'codex', 'transport identity outranks caller claim on checkpoints'
    call(ctx, 'task_checkpoint', {**args, 'requestId': 'two', 'expectedRevision': 1,
                                'state': 'completed', 'body': 'Shipped.'})
    current = call(ctx, 'task_state', {'taskId': 'root-task'})
    assert [(r['revision'], r['state']) for r in current['checkpoints']] == [(2, 'completed')], 'current view returns only latest revision, including completion'
    assert call(ctx, 'task_state', {'taskId': 'worker-task'})['checkpoints'] == [], 'sibling task cannot inherit parent state'
    past = call(ctx, 'task_state', dict(taskId='root-task', project='pensive', agent='codex', revision=1))
    assert past['checkpoints'][0]['body'] == 'Working.', 'historical state remains exact'
    assert mcp.dispatch(ctx, 'task_checkpoint', {**args, 'requestId': 'stale'})[1], 'stale writers must receive an error'
    assert call(ctx, 'task_checkpoint', args) == first, 'retry of successful original write is idempotent even after progress'


def test_receipt_final_admission_caller_distinct_from_author(ctx, monkeypatch):
    ids = [putAtom(ctx.store, {'text': text, 'kind': 'atom',
        'provenance': {'source': 'explicit-emit', 'agent': 'author'}})
        for text in ['short fact', 'second fact is much longer and cannot fit this budget']]
    seen = []
    def recall(*args, **kwargs):
        seen.append(kwargs)
        return {'results': [dict(atomId=i, score=.03, confidence=.8,
                    shouldTrust=True, why='both signals') for i in ids],
                'lowConfidence': False}
    monkeypatch.setattr(mcp, 'recall', recall)
    monkeypatch.setattr(mcp, '_transportAgent', lambda: 'codex')
    page = call(ctx, 'recall_records', dict(query='fact', agent='author',
        callerAgent='claimed', taskId='task', includeReceipt=True, tokenBudget=5))
    assert [r['id'] for r in page['records']] == ids[:1], 'receipt follows final token admission'
    assert seen[0]['agent'] == 'author', 'author filter must remain separate from receipt caller'
    assert ctx.store._conn.execute('SELECT agent, task_id FROM recall_receipts').fetchall() == [('codex', 'task')], 'receipt attributes delivery to resolved caller'
    assert ctx.store._conn.execute('SELECT atom_id FROM recall_exposures').fetchall() == [(ids[0],)], 'dropped candidates cannot become exposures'
    assert ctx.store._conn.execute('SELECT atom_id FROM recall_log').fetchall() == [(ids[0],)], 'receipt request logs telemetry once'
    report = call(ctx, 'recall_feedback', dict(receiptId=page['receiptId'], atomId='p3://' + ids[0],
        taskId='task', eventId='help', feedbackType='helpful', note='Resolved the fact.', agent='claimed'))
    assert report['credited'] is True, 'explicit matching helpful feedback earns one credit'


@pytest.mark.parametrize('name,args', [
    ('task_state', {'taskId': 't', 'limit': True}),
    ('task_state', {'mode': 'history', 'taskId': 't'}),
    ('task_checkpoint', {'taskId': 't'}),
    ('recall_records', {'query': 'q', 'includeReceipt': 'yes'}),
    ('recall_records', {'query': 'q', 'includeReceipt': True, 'taskId': 't'}),
    ('recall_feedback', {'receiptId': 'r', 'source': 'forged'}),
])
def test_direct_dispatch_validates_new_arguments(ctx, name, args):
    assert mcp.dispatch(ctx, name, args)[1], f'{name} must reject invalid direct-call arguments'


def test_checkpoint_validation_error_does_not_echo_entire_body(ctx):
    result, error = mcp.dispatch(ctx, 'task_checkpoint', dict(project='p', taskId='t',
        requestId='r', expectedRevision=0, state='active', body='large' * 7000))
    assert error and 'body' in result and len(result) < 350, 'oversized checkpoint errors identify the field without echoing the whole body'


def test_history_cursor_consumes_every_revision_and_reports_truncation(ctx):
    for revision in range(3):
        call(ctx, 'task_checkpoint', dict(project='p', agent='codex', taskId='t',
            requestId=f'r{revision}', expectedRevision=revision, state='active', body=str(revision)))
    args = dict(mode='history', project='p', agent='codex', taskId='t', limit=1)
    first = call(ctx, 'task_state', args)
    second = call(ctx, 'task_state', {**args, 'afterRevision': first['nextRevision']})
    assert [first['checkpoints'][0]['revision'], second['checkpoints'][0]['revision']] == [1, 2], 'feeding the returned cursor into afterRevision must not skip a checkpoint'
    assert first['truncated'] is True, 'history pages retain the common state truncation flag for CLI consumers'
