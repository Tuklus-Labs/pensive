import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from store.store import openStore, putAtom, getAtom, logRecall
from store.feedback import recordRecallReceipt, recordRecallFeedback
from lifecycle.importance import accrueImportance


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'feedback.db'
    store = openStore(path)
    atom = putAtom(store, {'text': 'Inspect the source before changing a route.',
                          'kind': 'atom', 'importance': .95,
                          'provenance': {'source': 'explicit-emit', 'agent': 'author'}})
    yield store, atom, path
    store.close()


def receipt(store, atom, receiptId='receipt', taskId='task'):
    return recordRecallReceipt(store, receiptId=receiptId, query='route?',
        project=None, agent='codex', taskId=taskId, sourceRef='mcp.recall_records',
        records=[{'id': atom, 'score': .031, 'delivery': 'body'}])


def feedback(store, atom, **changes):
    args = dict(eventId='event', receiptId='receipt', atomId=atom,
                feedbackType='helpful', agent='codex', taskId='task',
                source='mcp-feedback', note='Source inspection prevented a wrong route change.')
    args.update(changes)
    return recordRecallFeedback(store, **args)


def test_receipt_scope_and_exposure(data):
    store, atom, _ = data
    receipt(store, atom)
    assert store._conn.execute('SELECT atom_id, rank FROM recall_exposures').fetchall() == [(atom, 1)], 'receipt contains only final admitted records in rank order'
    assert store._conn.execute('SELECT atom_id FROM recall_log').fetchall() == [(atom,)], 'receipt preserves legacy traffic accounting exactly once'
    for changes in ({'agent': 'worker'}, {'taskId': 'sibling'}, {'atomId': 'not-exposed'}):
        with pytest.raises(ValueError, match='(?i)not exposed' if 'atomId' in changes else '(?i)receipt'):
            feedback(store, atom, **changes)
    assert store._conn.execute('SELECT count(*) FROM recall_feedback').fetchone()[0] == 0, 'wrong scope or unexposed atom must leave no event'


def test_only_helpful_credits_once_per_task(data):
    store, atom, _ = data
    receipt(store, atom)
    for kind in ('shown', 'used', 'irrelevant', 'outdated'):
        feedback(store, atom, eventId=kind, feedbackType=kind)
    logRecall(store, [atom] * 8)
    accrueImportance(store)
    assert getAtom(store, atom)['importance'] == .95, 'neither exposure nor non-helpful feedback earns importance'
    feedback(store, atom)
    feedback(store, atom, eventId='second-helpful')
    receipt(store, atom, receiptId='again')
    feedback(store, atom, eventId='third-helpful', receiptId='again')
    assert getAtom(store, atom)['importance'] == pytest.approx(.96), 'all helpful reports in one task share a single .01 credit'
    receipt(store, atom, receiptId='different', taskId='different')
    feedback(store, atom, eventId='fourth', receiptId='different', taskId='different')
    assert getAtom(store, atom)['importance'] == pytest.approx(.97), 'a distinct task can independently validate usefulness'
    assert getAtom(store, atom)['text'] == 'Inspect the source before changing a route.', 'feedback never rewrites canonical memory'


def test_receipt_validation_is_atomic(data):
    store, atom, _ = data
    for records in ([{'id': atom, 'score': float('nan'), 'delivery': 'body'}],
                    [{'id': atom, 'score': 0., 'delivery': 'body'}] * 2,
                    [{'id': 'absent', 'score': 0., 'delivery': 'body'}]):
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            recordRecallReceipt(store, receiptId='bad', query='q', project=None,
                agent='codex', taskId='task', sourceRef='test', records=records)
    assert store._conn.execute('SELECT count(*) FROM recall_receipts').fetchone()[0] == 0, 'invalid receipt writes roll back completely'
    recordRecallReceipt(store, receiptId='empty', query='q', project=None,
        agent='codex', taskId='task', sourceRef='test', records=[])
    assert store._conn.execute('SELECT id FROM recall_receipts').fetchall() == [('empty',)], 'abstention can have an empty receipt'


@pytest.mark.parametrize('changes', [
    {'feedbackType': 'successful'}, {'note': ''}, {'note': '   '},
    {'agent': ' '}, {'eventId': 'x' * 257}, {'taskId': True},
    {'feedbackType': 'used', 'note': None},
])
def test_feedback_validation(data, changes):
    store, atom, _ = data
    receipt(store, atom)
    field = 'note' if 'note' in changes else next(iter(changes))
    with pytest.raises((ValueError, TypeError), match=field):
        feedback(store, atom, **changes)
    assert getAtom(store, atom)['importance'] == .95, 'malformed feedback cannot change ranking'


@pytest.mark.parametrize('field,maximum', [
    ('eventId', 256), ('receiptId', 256), ('atomId', 256), ('taskId', 256),
    ('source', 256), ('agent', 64), ('sessionId', 256), ('sourceRef', 2048), ('note', 2048),
])
def test_feedback_metadata_bounds_name_the_field(data, field, maximum):
    store, atom, _ = data
    receipt(store, atom)
    with pytest.raises(ValueError, match=field):
        feedback(store, atom, **{field: 'x' * (maximum + 1)})
    assert store._conn.execute('SELECT count(*) FROM recall_feedback').fetchone()[0] == 0, 'overlong metadata is rejected before writing an event'


def test_feedback_preserves_evidence_and_allows_shown_without_note(data):
    store, atom, _ = data
    receipt(store, atom)
    delivery = feedback(store, atom, feedbackType='shown', note=None)
    assert delivery['note'] is None and not delivery['credited'], 'shown is valid without a fabricated usefulness explanation'
    details = {'eventId': 'details', 'source': 's' * 256, 'sessionId': 'i' * 256,
               'sourceRef': 'r' * 2048, 'note': 'é' * 2048}
    report = feedback(store, atom, **details)
    assert {key: report[key] for key in details} == details, 'maximum-length evidence metadata remains verbatim including Unicode'


def test_replay_and_failure_atomicity(data):
    store, atom, path = data
    receipt(store, atom)
    first = feedback(store, atom)
    other = openStore(path)
    try:
        assert feedback(other, atom) == first, 'exact event replay survives reopen and returns original result'
        with pytest.raises(ValueError, match='replay'):
            feedback(other, atom, note='changed')
    finally:
        other.close()
    store._conn.executescript("CREATE TRIGGER reject_credit BEFORE INSERT ON memory_credits BEGIN SELECT RAISE(ABORT, 'fault'); END;")
    receipt(store, atom, receiptId='fault', taskId='fault')
    with pytest.raises(sqlite3.IntegrityError, match='fault'):
        feedback(store, atom, eventId='fault', receiptId='fault', taskId='fault')
    assert store._conn.execute("SELECT count(*) FROM recall_feedback WHERE event_id='fault'").fetchone()[0] == 0, 'credit failure must roll back feedback event and processed marker'
    assert getAtom(store, atom)['importance'] == pytest.approx(.96), 'failed credit must preserve committed importance'


def test_concurrent_credit(data):
    store, atom, path = data
    receipt(store, atom)
    def report(event):
        connection = openStore(path)
        try:
            return feedback(connection, atom, eventId=event)
        finally:
            connection.close()
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(report, ['worker-1', 'worker-2']))
    assert getAtom(store, atom)['importance'] == pytest.approx(.96), 'competing connections cannot double-credit the same task'
    assert store._conn.execute('SELECT count(*) FROM recall_feedback').fetchone()[0] == 2, 'both independent reports remain in the audit history'


def test_helpful_credit_never_reduces_legacy_importance(data):
    store, atom, _ = data
    store._conn.execute('UPDATE atoms SET importance=5.0 WHERE id=?', (atom,))
    store._conn.commit()
    receipt(store, atom)
    feedback(store, atom)
    assert getAtom(store, atom)['importance'] == 5.0, 'the ranking ceiling is not permission to lower canonical legacy importance'
