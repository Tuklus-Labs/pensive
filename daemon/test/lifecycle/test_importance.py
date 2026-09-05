"""RISK_MODEL_FEEDBACK: exposure is inert; pending helpful credits recover once."""
import pytest

from lifecycle.importance import accrueImportance
from store.feedback import recordRecallReceipt, processPendingFeedback
from store.store import openStore, putAtom, getAtom, logRecall


@pytest.fixture
def data(tmp_path):
    store = openStore(tmp_path / 'importance.db')
    atom = putAtom(store, {'text': 'validated lesson', 'kind': 'atom', 'importance': .995,
                          'provenance': {'source': 'codex'}})
    yield store, atom
    store.close()


def pending(store, atom, event, task='task', kind='helpful'):
    recordRecallReceipt(store, receiptId=event, query='q', project=None,
        agent='codex', taskId=task, sourceRef='test',
        records=[{'id': atom, 'score': .03, 'delivery': 'body'}])
    store._conn.execute('INSERT INTO recall_feedback '
        '(event_id, receipt_id, atom_id, feedback_type, source, agent, task_id, note, recorded_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (event, event, atom, kind, 'mcp-feedback', 'codex', task, 'Observed correct action.', 100))
    store._conn.commit()


def test_legacy_exposure_log_is_never_rewarded(data):
    store, atom = data
    logRecall(store, [atom] * 100, weight=100)
    assert accrueImportance(store) == {'processed': 0, 'updated': 0, 'missingAtoms': 0}, 'old exposure counts are not helpful feedback'
    assert getAtom(store, atom)['importance'] == .995, 'mere retrieval never earns importance'


def test_pending_feedback_caps_credit_and_replays_once(data):
    store, atom = data
    pending(store, atom, 'one')
    pending(store, atom, 'two')
    report = accrueImportance(store)
    assert report == {'processed': 2, 'updated': 1, 'missingAtoms': 0}, 'one task can credit once and both reports are processed'
    assert getAtom(store, atom)['importance'] == 1., 'helpful credit respects importance ceiling'
    assert accrueImportance(store)['processed'] == 0, 'recovery cannot process an event twice'
    assert store._conn.execute('SELECT count(*) FROM memory_credits').fetchone()[0] == 1, 'one durable credit survives all repeated reports'


def test_pending_batch_is_bounded_and_shown_is_inert(data):
    store, atom = data
    pending(store, atom, 'one', kind='shown')
    pending(store, atom, 'two', kind='shown')
    assert processPendingFeedback(store, limit=1)['processed'] == 1, 'one batch only processes its selected event'
    assert store._conn.execute('SELECT count(*) FROM recall_feedback WHERE processed_at IS NULL').fetchone()[0] == 1, 'unselected events remain recoverable'
    assert getAtom(store, atom)['importance'] == .995, 'shown events remain inert even through recovery'


def test_pending_import_cannot_bypass_receipt_scope(data):
    store, atom = data
    pending(store, atom, 'corrupt')
    store._conn.execute("UPDATE recall_feedback SET task_id='wrong' WHERE event_id='corrupt'")
    store._conn.commit()
    with pytest.raises(ValueError, match='receipt scope'):
        accrueImportance(store)
    assert getAtom(store, atom)['importance'] == .995, 'imported feedback cannot bypass receipt caller/task invariants'
