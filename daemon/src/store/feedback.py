"""Recall delivery receipts and explicit usefulness, separate from exposure counts."""
import math
import time

from util.ulid import ulid

FEEDBACK_TYPES = ('shown', 'used', 'helpful', 'irrelevant', 'outdated')
_FIELDS = ('eventId', 'receiptId', 'atomId', 'feedbackType', 'source', 'agent',
           'taskId', 'sessionId', 'sourceRef', 'note', 'recordedAt', 'processedAt')
_COLUMNS = ('event_id, receipt_id, atom_id, feedback_type, source, agent, '
            'task_id, session_id, source_ref, note, recorded_at, processed_at')


def _text(value, name, maximum=256, optional=False):
    if optional and value is None:
        return value
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be nonblank text of at most {maximum} characters')
    return value


def _begin(conn):
    if conn.in_transaction:
        raise RuntimeError('feedback requires its own transaction')
    conn.execute('BEGIN IMMEDIATE')


def recordRecallReceipt(store, *, receiptId, query, project, agent, taskId,
                        sourceRef, records):
    """Commit final served records and legacy traffic telemetry together.

    The server allocates receiptId before measuring the wire envelope. Failures
    propagate rather than returning a receipt the caller cannot use.
    """
    _text(receiptId, 'receiptId')
    _text(query, 'query', 8192)
    _text(project, 'project', optional=True)
    _text(agent, 'agent', 64)
    _text(taskId, 'taskId')
    _text(sourceRef, 'sourceRef')
    if not isinstance(records, list) or len(records) > 32:
        raise ValueError('records must be an array of at most 32 admitted records')
    seen = set()
    for record in records:
        atom = _text(record['id'], 'atomId')
        if atom in seen:
            raise ValueError('receipt records must not contain duplicate atoms')
        seen.add(atom)
        score = record.get('score')
        if score is not None and (isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('receipt score must be finite or null')
        if record.get('delivery') not in ('body', 'handle'):
            raise ValueError('delivery must be body or handle')
    conn, now = store._conn, int(time.time())
    _begin(conn)
    try:
        conn.execute('INSERT INTO recall_receipts '
            '(id, query, project, agent, task_id, source_ref, recorded_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (receiptId, query, project, agent, taskId, sourceRef, now))
        conn.executemany('INSERT INTO recall_exposures '
            '(receipt_id, atom_id, rank, score, delivery) VALUES (?, ?, ?, ?, ?)',
            [(receiptId, r['id'], i, r.get('score'), r['delivery'])
             for i, r in enumerate(records, 1)])
        conn.executemany('INSERT INTO recall_log '
            '(id, atom_id, query, source_ref, weight, recorded_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            [(ulid(), r['id'], query, sourceRef, 1., now) for r in records])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return receiptId


def _applyCredit(conn, row, now):
    """Credit once per atom/caller/task, within the owning write transaction."""
    eventId, _receipt, atomId, kind, _source, agent, taskId = row[:7]
    if kind not in FEEDBACK_TYPES or (kind != 'shown' and not (row[9] or '').strip()):
        raise ValueError('pending feedback lacks valid type or evidence note')
    scope = conn.execute('SELECT r.agent, r.task_id FROM recall_receipts r '
        'JOIN recall_exposures e ON e.receipt_id=r.id '
        'WHERE r.id=? AND e.atom_id=?', (_receipt, atomId)).fetchone()
    if scope != (agent, taskId):
        raise ValueError('pending feedback does not match its receipt scope')
    updated = missing = 0
    if kind == 'helpful':
        atom = conn.execute('SELECT importance FROM atoms WHERE id=?', (atomId,)).fetchone()
        if atom is None:
            missing = 1
        else:
            inserted = conn.execute('INSERT OR IGNORE INTO memory_credits '
                '(atom_id, agent, task_id, feedback_id, awarded_at) VALUES (?, ?, ?, ?, ?)',
                (atomId, agent, taskId, eventId, now)).rowcount
            if inserted:
                # Older imports may exceed the ranking ceiling. Helpful
                # feedback must never lower their canonical importance.
                after = atom[0] if atom[0] >= 1. else min(1., atom[0] + .01)
                if after != atom[0]:
                    conn.execute('UPDATE atoms SET importance=? WHERE id=?', (after, atomId))
                    updated = 1
    conn.execute('UPDATE recall_feedback SET processed_at=? WHERE event_id=?', (now, eventId))
    return updated, missing


def _result(conn, row):
    result = dict(zip(_FIELDS, row))
    result['credited'] = conn.execute(
        'SELECT 1 FROM memory_credits WHERE atom_id=? AND agent=? '
        'AND task_id=? AND feedback_id=?', (row[2], row[5], row[6], row[0]),
    ).fetchone() is not None
    return result


def recordRecallFeedback(store, *, eventId, receiptId, atomId, feedbackType,
                         agent, taskId, source, sessionId=None, sourceRef=None,
                         note=None):
    """Append an evidenced report; exact retries are idempotent across restarts."""
    for name, value in (('eventId', eventId), ('receiptId', receiptId),
                        ('atomId', atomId), ('taskId', taskId), ('source', source)):
        _text(value, name)
    _text(agent, 'agent', 64)
    _text(sessionId, 'sessionId', optional=True)
    _text(sourceRef, 'sourceRef', 2048, optional=True)
    _text(note, 'note', 2048, optional=feedbackType == 'shown')
    if feedbackType not in FEEDBACK_TYPES:
        raise ValueError(f'feedbackType must be one of {FEEDBACK_TYPES}')
    values = (eventId, receiptId, atomId, feedbackType, source, agent, taskId,
              sessionId, sourceRef, note)
    conn, now = store._conn, int(time.time())
    _begin(conn)
    try:
        original = conn.execute(f'SELECT {_COLUMNS} FROM recall_feedback WHERE event_id=?', (eventId,)).fetchone()
        if original is not None:
            if tuple(original[:10]) != values:
                raise ValueError('feedback event replay differs from original input')
            result = _result(conn, original)
            conn.commit()
            return result
        receipt = conn.execute('SELECT agent, task_id FROM recall_receipts WHERE id=?', (receiptId,)).fetchone()
        if receipt != (agent, taskId):
            raise ValueError('feedback caller and task must match its receipt')
        exposed = conn.execute('SELECT 1 FROM recall_exposures WHERE receipt_id=? AND atom_id=?', (receiptId, atomId)).fetchone()
        if exposed is None:
            raise ValueError('feedback atom was not exposed by this receipt')
        row = (*values, now, None)
        conn.execute(f'INSERT INTO recall_feedback ({_COLUMNS}) VALUES ({",".join("?" for _ in row)})', row)
        _applyCredit(conn, row, now)
        result = _result(conn, (*values, now, now))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return result


def processPendingFeedback(store, *, limit=1000):
    """Recover pending feedback in bounded batches; old exposure logs are inert."""
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('limit must be an integer in 1..10000')
    conn, now = store._conn, int(time.time())
    _begin(conn)
    try:
        rows = conn.execute(f'SELECT {_COLUMNS} FROM recall_feedback '
            'WHERE processed_at IS NULL ORDER BY recorded_at, event_id LIMIT ?', (limit,)).fetchall()
        updated = missing = 0
        for row in rows:
            count, absent = _applyCredit(conn, row, now)
            updated += count
            missing += absent
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {'processed': len(rows), 'updated': updated, 'missingAtoms': missing}
