"""Strict task-state and feedback contracts shared by native MCP handlers."""
from jsonschema import Draft202012Validator

from store.checkpoints import (putTaskCheckpoint, getTaskStates,
                               listTaskHistory, listRecentTaskStates)
from store.feedback import FEEDBACK_TYPES, recordRecallFeedback


def _text(maximum=256):
    return {'type': 'string', 'minLength': 1, 'maxLength': maximum, 'pattern': r'.*\S.*'}


def _integer(minimum=0, maximum=9_007_199_254_740_991):
    return {'type': 'integer', 'minimum': minimum, 'maximum': maximum}


def _schema(properties, required):
    return {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': required}


SCHEMAS = {
    'task_checkpoint': _schema({
        'project': _text(), 'agent': _text(64), 'taskId': _text(),
        'requestId': _text(), 'expectedRevision': _integer(),
        'state': {'enum': ['active', 'blocked', 'completed', 'abandoned']},
        'body': {'type': 'string', 'maxLength': 32000},
        'sessionId': _text(), 'sourceRef': _text(),
    }, ['project', 'taskId', 'requestId', 'expectedRevision', 'state', 'body']),
    'task_state': _schema({
        'mode': {'enum': ['current', 'history', 'recent'], 'default': 'current'},
        'project': _text(), 'agent': _text(64), 'taskId': _text(),
        'revision': _integer(1), 'asOf': _integer(),
        'afterRevision': _integer(), 'limit': _integer(1, 100),
    }, []),
    'recall_feedback': _schema({
        'receiptId': _text(), 'atomId': _text(), 'eventId': _text(),
        'feedbackType': {'enum': list(FEEDBACK_TYPES)}, 'agent': _text(64),
        'taskId': _text(), 'sessionId': _text(), 'sourceRef': _text(2048),
        'note': _text(2048),
    }, ['receiptId', 'atomId', 'eventId', 'feedbackType', 'taskId']),
}

DESCRIPTIONS = {
    'task_checkpoint': 'Append explicit current task state with expectedRevision compare-and-swap; requestId makes retries idempotent. Task IDs must identify one task, not a shared parent session.',
    'task_state': 'Read latest task checkpoints, exact revision/asOf, paged history or recent project tasks. Completed tasks remain visible. Agent is an explicit state-owner filter.',
    'recall_feedback': 'Report shown, explicitly used, helpful, irrelevant or outdated memory from a recall receipt. Used and evaluative feedback require an evidence note. Only helpful earns credit, once per caller/task/atom.',
}


def runMemoryTool(store, name, args, caller=None):
    error = next(Draft202012Validator(SCHEMAS[name]).iter_errors(args), None)
    if error is not None:
        field = '.'.join(str(part) for part in error.absolute_path) or 'arguments'
        raise ValueError(f'{name}: {field}: {error.message[:240]}')
    if name == 'task_state':
        mode = args.get('mode', 'current')
        scope = {k: args[k] for k in ('project', 'agent', 'taskId', 'limit') if k in args}
        if mode == 'current':
            if 'afterRevision' in args:
                raise ValueError('afterRevision is only valid for history')
            scope.update({k: args[k] for k in ('revision', 'asOf') if k in args})
            return getTaskStates(store, **scope)
        if any(k in args for k in ('revision', 'asOf')):
            raise ValueError('revision/asOf is only valid for current mode')
        if mode == 'history':
            return listTaskHistory(store, **scope, afterRevision=args.get('afterRevision', 0))
        if 'taskId' in args or 'afterRevision' in args:
            raise ValueError('recent mode accepts project, agent and limit only')
        return listRecentTaskStates(store, **scope)
    if not caller:
        raise ValueError('a resolved caller agent is required')
    payload = {k: v for k, v in args.items() if k != 'agent'}
    if name == 'task_checkpoint':
        return putTaskCheckpoint(store, **payload, agent=caller, source='mcp-checkpoint')
    atom = payload['atomId']
    if atom.startswith('p3://'):
        payload['atomId'] = atom[5:]
    return recordRecallFeedback(store, **payload, agent=caller, source='mcp-feedback')
