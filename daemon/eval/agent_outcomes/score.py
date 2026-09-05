#!/usr/bin/env python3
"""Score exact workflow answers and observable support from blind reader output."""
import argparse
import json
from pathlib import Path
import re


def normalize(answer):
    return ' '.join(answer.strip().strip('`').split()).casefold() if isinstance(answer, str) else answer


def score(cases, packets, responses, support):
    expected_ids = {case['id'] for case in cases}
    for label, rows in [('packets', packets), ('responses', responses)]:
        ids = [row['id'] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f'{label} must contain exactly one row per fixture case')
    by_id = {row['id']: row for row in responses}
    packet_by_id = {row['id']: row for row in packets}
    results = []
    for case in cases:
        row, packet = by_id[case['id']], packet_by_id[case['id']]
        if type(row.get('abstained')) is not bool or not isinstance(row.get('evidence'), list):
            raise ValueError('reader must provide boolean abstained and an evidence array')
        answer = row.get('answer')
        if answer is not None and not isinstance(answer, str):
            raise ValueError('answer must be exact text or null')
        if row['abstained'] != (answer is None):
            raise ValueError('abstained must agree with a null answer')
        correct = (row['abstained'] == case['expected']['abstained']
                   and normalize(answer) == normalize(case['expected']['answer']))
        allowed = set(support.get(case['id'], []))
        displayed = set(re.findall(r'p3://([0-7][0-9A-HJKMNP-TV-Z]{25})', packet['memory']))
        state_ref = f'task:{case["taskId"]}:r{len(case["checkpoints"])}'
        if 'Current task state' in packet['memory']:
            allowed.add(state_ref); displayed.add(state_ref)
        if case['category'] == 'fresh-user':
            allowed.add('recentDialogue'); displayed.add('recentDialogue')
        fact_support = allowed & displayed
        if case['dialogue']:
            # A follow-up uses dialogue to resolve its subject, but still needs
            # a displayed memory fact unless the new user message supplied it.
            allowed.add('recentDialogue'); displayed.add('recentDialogue')
        citations = {str(item).removeprefix('p3://') for item in row['evidence']}
        supported = row['abstained'] or bool(citations & fact_support and citations <= allowed & displayed)
        irrelevant = len(displayed - allowed - {'recentDialogue'})
        results.append({'id': case['id'], 'category': case['category'], 'correct': correct,
                        'supported': supported, 'abstained': row['abstained'],
                        'incorrectNonAbstention': not correct and not row['abstained'],
                        'irrelevantHandles': irrelevant, 'memoryChars': len(packet['memory'])})
    return {'cases': results, 'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'correctSupported': sum(r['correct'] and r['supported'] for r in results),
            'incorrectNonAbstentions': sum(r['incorrectNonAbstention'] for r in results),
            'irrelevantHandles': sum(r['irrelevantHandles'] for r in results),
            'memoryChars': sum(r['memoryChars'] for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', required=True, type=Path)
    parser.add_argument('--responses', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name('fixtures.v1.json').read_text())['cases']
    result = score(cases, json.loads(args.packets.read_text()),
                   json.loads(args.responses.read_text()),
                   json.loads(args.metrics.read_text())['supportIds'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
