#!/usr/bin/env python3
"""Prepare blind reader packets using real Pensive retrieval on an isolated store.

Run once with the baseline checkout/plugin and once with the candidate. Each
reader receives identical questions and recent dialogue; only memory varies.
Expected answers never enter a reader packet. No model endpoint is contacted.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


def load_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('outcome_hook', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--plugin', required=True, type=Path)
    parser.add_argument('--condition', choices=['baseline', 'candidate'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--onnx', required=True, type=Path)
    args = parser.parse_args()
    fixture_path = Path(__file__).with_name('fixtures.v1.json')
    fixture_bytes = fixture_path.read_bytes()
    expected_hash = fixture_path.with_suffix('.sha256').read_text().strip()
    if hashlib.sha256(fixture_bytes).hexdigest() != expected_hash:
        raise ValueError('frozen fixture hash changed')
    cases = json.loads(fixture_bytes)['cases']
    sys.path.insert(0, str(args.repo / 'daemon/src'))
    os.environ['PENSIVE_V3_ONNX_MODEL'] = str(args.onnx)
    from store.store import openStore, putAtom
    import store.store as canonical
    from util.ulid import _encode
    import itertools
    ids = itertools.count(1)
    canonical.ulid = lambda: _encode((1788566400000 << 80) | next(ids), 26)
    canonical._now = lambda: 1788566400
    from recall.embedder import makeEmbedder, embedMissing
    from recall.vector_index import buildClassIndexes
    from serve.mcp import dispatch
    hook = load_module(args.plugin / 'hooks/pensive_hook.py')
    packets, metrics, evidence = [], [], {}
    with tempfile.TemporaryDirectory(prefix='pensive-outcome-') as temporary:
        root = Path(temporary)
        store = openStore(root / 'memory.db')
        for case in cases:
            evidence[case['id']] = []
            for order, memory in enumerate(case['memories']):
                atom = putAtom(store, {'text': memory['body'], 'kind': memory['kind'],
                    'project': case['project'], 'importance': 0., 'occurredAt': 1788566400 + order,
                    'provenance': {'source': 'explicit-emit', 'agent': 'codex'}})
                evidence[case['id']].append(atom)
        # Shared ordinary distractors, identical in both conditions.
        for index in range(64):
            putAtom(store, {'text': f'Workshop project {index}: report exports, imports, release notes, validation and build tasks were discussed. No deployment decision was recorded.',
                'kind': 'atom', 'project': f'workshop-{index}',
                'provenance': {'source': 'explicit-emit', 'agent': 'other'}})
        if args.condition == 'candidate':
            from store.checkpoints import putTaskCheckpoint, getTaskStates
            for case in cases:
                for revision, (state, body) in enumerate(case['checkpoints']):
                    checkpoint = putTaskCheckpoint(store, project=case['project'], agent='codex',
                        taskId=case['taskId'], expectedRevision=revision,
                        requestId=f'{case["id"]}-{revision}', state=state, body=body, source='eval-fixture')
                    evidence[case['id']].append(checkpoint['id'])
                for who, state, body in case['siblings']:
                    putTaskCheckpoint(store, project=case['project'], agent='codex',
                        taskId=case['project'] + '-' + who, expectedRevision=0,
                        requestId=case['id'] + who, state=state, body=body, source='eval-fixture')
        embedder = makeEmbedder('BAAI/bge-small-en-v1.5')
        embedMissing(store, embedder)
        ctx = SimpleNamespace(store=store, indexes=buildClassIndexes(store, embedder.modelId),
            embedder=embedder, aux=None, agent=None, recallLogErrors=0)
        for case in cases:
            times, pages = [], []
            def query(text):
                started = time.perf_counter_ns()
                result, error = dispatch(ctx, 'recall_records',
                    {'query': text, 'kinds': ['atom', 'narrative', 'snapshot'], 'k': 4, 'tokenBudget': 8000})
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if error:
                    raise RuntimeError(result)
                native = result.value
                normalized = [{**r, 'handle': 'p3://' + r['id'], 'type': r['kind'],
                    'summary': r['content'], 'scoreKind': 'fused'} for r in native['records']]
                snapshot = next((r for r in normalized if r['kind'] == 'snapshot'), None)
                page = {'project': None, 'queries': [text],
                    'snapshot': snapshot['content'] if snapshot else None, 'snapshotRecord': snapshot,
                    'results': [r for r in normalized if r['kind'] != 'snapshot'],
                    'pages': [{k: native[k] for k in ['query', 'lowConfidence', 'truncated']} | {'servedIds': [r['id'] for r in normalized]}]}
                times.append(elapsed)
                pages.append({'query': text, 'recordIds': [r['id'] for r in normalized],
                              'lowConfidence': native['lowConfidence'], 'latencyMs': elapsed})
                if args.condition == 'candidate':
                    page['taskStates'] = getTaskStates(store, taskId=case['taskId'], agent='codex')
                    return page
                return hook._render_recall_json(page, '', 2500)
            transcript = root / (case['id'] + '.jsonl')
            records = [{'type': 'session_meta', 'payload': {'id': case['taskId'], 'session_id': 'eval-family'}}]
            for index, exchange in enumerate(case['dialogue']):
                records.append({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                    'content': [{'type': 'input_text', 'text': exchange['user']}],
                    'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.text'], 'turn_id': str(index)}}})
                records.append({'type': 'event_msg', 'payload': {'type': 'task_complete', 'turn_id': str(index), 'last_agent_message': exchange['assistant']}})
            transcript.write_text(''.join(json.dumps(record) + '\n' for record in records))
            payload = {'prompt': case['question'], 'session_id': 'eval-family', 'turn_id': 'current',
                       'cwd': '/tmp/eval-workspace', 'transcript_path': str(transcript)}
            kwargs = {'run_recall_query': query}
            if args.condition == 'candidate':
                kwargs.update(run_task_state=lambda task: {'taskStates': getTaskStates(store, taskId=task, agent='codex')},
                              state_dir=root / 'cache')
            output = hook.user_prompt_submit_output(payload, **kwargs)
            context = json.loads(output)['hookSpecificOutput']['additionalContext']
            # Protocol boilerplate is identical reader instruction, not task evidence.
            marker = 'Auto-recalled memory for this prompt'
            if marker in context:
                memory = context[context.index(marker):]
                before = context[:context.index(marker)]
                if 'Current task' in before:
                    memory = before[before.index('Current task'):] + '\n' + memory
            else:
                # State-only delivery remains valid evidence.
                memory = context[context.index('Current task'):] if 'Current task' in context else ''
            packets.append({'id': case['id'], 'question': case['question'],
                            'recentDialogue': case['dialogue'], 'memory': memory})
            metrics.append({'id': case['id'], 'queries': pages, 'memoryChars': len(memory),
                            'estimatedMemoryTokens': (len(memory) + 3) // 4})
        store.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'reader.json').write_text(json.dumps(packets, indent=2) + '\n')
    if args.condition == 'baseline':
        no_memory = [{**p, 'memory': ''} for p in packets]
        (args.output / 'no-memory-reader.json').write_text(json.dumps(no_memory, indent=2) + '\n')
    (args.output / 'metrics.json').write_text(json.dumps({'fixtureSha256': expected_hash,
        'condition': args.condition, 'cases': metrics, 'supportIds': evidence}, indent=2) + '\n')


if __name__ == '__main__':
    main()
