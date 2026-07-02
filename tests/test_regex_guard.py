"""Guard derivation must never change what a mega-regex matches.

Risk model:
- A too-narrow guard silently drops matches (worst case: entities vanish
  from graphs with no error). Covered by the guarded-vs-unguarded
  property tests over every shipped pattern set and a torture corpus.
- The guard must not shift capturing-group numbers, or entities get the
  wrong type. Covered by the lastindex checks.
- Optional leading elements (v?\\d+) must contribute both the optional
  char and what follows. Covered directly.
- Anything unprovable must bail to None (unguarded compile), never to a
  wrong guard. Covered for dot, negated class, backref, empty-matchable.
"""
import re

import pytest

from pensive.mega_extract import MegaExtractor, _split_literal_patterns
from pensive.patterns import (ALL_PATTERNS, REAL_DATA_PATTERNS,
                              SYNTHETIC_PATTERNS)
from pensive.regex_guard import first_charset_guard

TORTURE = [
    '',
    ' ',
    'a',
    'no entities here at all just plain words and more words',
    'Jan 5, 2025 jan 5 2025 JAN 5,2025 Dec 31,1999',
    'v1 v1.2 v1.2.3-rc1 1.2 40MB 99% 12ms 5tok/s $5M $1,234.56',
    'llama.cpp qwen3-32B mistral 7B gemini pro 1.5 phi-4 gpt-4o',
    'flash attention kv-cache kv cache spreading activation',
    'Dr. Smith met Employee Jones at Project Alpha (Core Engineering)',
    'ACME-1234 prod-web-01 eu-west-1 badge 12345 the Phoenix initiative',
    'Boeing Industries Seattle Office North Star Alice Johnson',
    'https://x.io/a?b=c /home/aegis/x.txt ~/notes.md /usr/lib/y',
    'İstanbul ß café naïve → 90% 45% -> 72% 120 --> 212tok/s',
    'edge$5 x$6M mid$word 7900xtx 7900 XTX 7900\tXTX',
    'word\nJan 1, 2020\nword',
    'CamelCase ALLCAPS lowercase Wordy AB12345 A-12',
]


def _branch_sources(patterns, ci):
    subset = [(r, et) for r, et, c in patterns if c is ci]
    regex_pats, _ = _split_literal_patterns(subset, case_insensitive=ci)
    return [r for r, _ in regex_pats]


@pytest.mark.parametrize('patterns', [REAL_DATA_PATTERNS, SYNTHETIC_PATTERNS,
                                      ALL_PATTERNS],
                         ids=['real', 'synthetic', 'all'])
@pytest.mark.parametrize('ci', [True, False], ids=['ci', 'cs'])
def test_guard_preserves_matches_and_groups(patterns, ci):
    srcs = _branch_sources(patterns, ci)
    if not srcs:
        pytest.skip('no regex branches after literal split')
    flags = re.IGNORECASE if ci else 0
    guard = first_charset_guard(srcs, flags)
    assert guard is not None, 'shipped pattern sets must stay analyzable'

    mega = '|'.join(f'(?:{r})' for r in srcs)
    plain = re.compile(mega, flags)
    guarded = re.compile(guard + '(?:' + mega + ')', flags)
    for text in TORTURE:
        expect = [(m.start(), m.lastindex, m.group(m.lastindex))
                  for m in plain.finditer(text)]
        got = [(m.start(), m.lastindex, m.group(m.lastindex))
               for m in guarded.finditer(text)]
        assert got == expect, f'guard changed matches on {text!r}'


def test_optional_lead_includes_both_first_chars():
    guard = first_charset_guard([r'\b(v?\d+\.\d+)\b'], 0)
    assert guard is not None
    cls = guard[3:-2]
    assert 'v' in cls
    assert r'\d' in cls or '0' in cls


@pytest.mark.parametrize('src', [
    r'(.)x',            # ANY consumes first
    r'([^x]+)',         # negated class: complement unbounded
    r'(a)\1',           # fine lead, but exercises the walker past groups
    r'(a*)',            # can match empty
], ids=['dot', 'negated', 'backref-tail', 'empty'])
def test_unprovable_or_empty_leads(src):
    guard = first_charset_guard([src], 0)
    if src == r'(a)\1':
        # lead is a plain literal; backref sits after it and is never reached
        assert guard == '(?=[a])'
    elif src == r'(a*)':
        assert guard is None
    else:
        assert guard is None


def test_mixed_branches_one_unprovable_bails_all():
    assert first_charset_guard([r'(\d+)', r'(.)'], 0) is None


def test_case_insensitive_guard_admits_both_cases():
    guard = first_charset_guard([r'\b(Jan\w*)\b'], re.IGNORECASE)
    assert guard is not None
    compiled = re.compile(guard + r'(?:\b(Jan\w*)\b)', re.IGNORECASE)
    assert compiled.search('january')
    assert compiled.search('JANUARY')


def test_extractor_output_unchanged_by_guard(monkeypatch):
    """MegaExtractor with guarding disabled must extract identically."""
    baseline = MegaExtractor(ALL_PATTERNS)
    import pensive.regex_guard as rg
    import pensive.mega_extract as me
    monkeypatch.setattr(rg, '_parser', None)
    if hasattr(me, '_parser'):
        monkeypatch.setattr(me, '_parser', None)
    unguarded = MegaExtractor(ALL_PATTERNS)
    for text in TORTURE:
        assert baseline.extract(text) == unguarded.extract(text)
        assert (baseline.extract_with_spans(text)
                == unguarded.extract_with_spans(text))
