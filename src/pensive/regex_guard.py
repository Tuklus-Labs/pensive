"""FIRST-set guard derivation for mega-regex alternations.

CPython's sre engine attempts every alternation branch at every text
position; the leading \\b in typical entity patterns defeats its
literal-prefix skip, so a 20-branch mega-regex costs ~20 branch attempts
per character. Prepending a single-charclass lookahead guard
``(?=[...])`` built from the union of every branch's possible first
characters lets the engine reject most positions with one class test
(measured 1.7-3.7x on the shipped pattern sets).

Safety contract: the derived set must be a SUPERSET of the true first
characters. Every handler below either overapproximates or raises
``_Bail``, and ``first_charset_guard`` returns None on any bail, in
which case the caller compiles unguarded. A too-broad guard only costs
speed; a too-narrow one would silently drop matches, so nothing here is
allowed to guess.
"""
import re
from typing import List, Optional, Set, Tuple

try:
    import re._parser as _parser
except ImportError:  # future stdlib reshuffle: run unguarded, stay correct
    _parser = None

# A RANGE wider than this stays in the class as a range instead of being
# expanded into per-character case pairs.
_MAX_RANGE_EXPAND = 512


class _Bail(Exception):
    pass


def _case_pair(ch: str) -> Set[str]:
    return {ch, ch.lower(), ch.upper()}


def _collect_in(items, chars: Set[str], classes: Set[str],
                ignorecase: bool) -> None:
    """Union a parsed character-class body into (chars, classes)."""
    for iop, iav in items:
        iname = str(iop)
        if iname == 'NEGATE':
            raise _Bail
        if iname == 'LITERAL':
            ch = chr(iav)
            chars.update(_case_pair(ch) if ignorecase else {ch})
        elif iname == 'RANGE':
            lo, hi = iav
            if hi - lo + 1 > _MAX_RANGE_EXPAND:
                raise _Bail
            for cp in range(lo, hi + 1):
                ch = chr(cp)
                chars.update(_case_pair(ch) if ignorecase else {ch})
        elif iname == 'CATEGORY':
            cat = str(iav)
            if cat.endswith('CATEGORY_DIGIT'):
                classes.add(r'\d')
            elif cat.endswith('CATEGORY_WORD'):
                classes.add(r'\w')
            elif cat.endswith('CATEGORY_SPACE'):
                classes.add(r'\s')
            else:
                raise _Bail  # negated categories: complement is unbounded
        else:
            raise _Bail


def _first_of_seq(seq, ignorecase: bool) -> Tuple[Set[str], Set[str]]:
    """FIRST set of a parsed subpattern sequence.

    Returns (chars, classes) where classes holds verbatim class atoms
    like '\\d'. Raises _Bail when the first consuming element cannot be
    proven, including a sequence that can match empty (a guard would
    then wrongly require a next character).
    """
    chars: Set[str] = set()
    classes: Set[str] = set()
    for op, av in seq:
        name = str(op)
        if name in ('AT', 'ASSERT', 'ASSERT_NOT'):
            continue  # zero-width: constrains but consumes nothing
        if name == 'LITERAL':
            ch = chr(av)
            chars.update(_case_pair(ch) if ignorecase else {ch})
            return chars, classes
        if name == 'IN':
            _collect_in(av, chars, classes, ignorecase)
            return chars, classes
        if name == 'SUBPATTERN':
            # av = (group, add_flags, del_flags, body)
            sub_ic = ignorecase or bool(av[1] & re.IGNORECASE)
            if av[2] & re.IGNORECASE:
                sub_ic = False
            c, k = _first_of_seq(av[3], sub_ic)
            chars |= c
            classes |= k
            return chars, classes
        if name == 'BRANCH':
            for alt in av[1]:
                c, k = _first_of_seq(alt, ignorecase)
                chars |= c
                classes |= k
            return chars, classes
        if name in ('MAX_REPEAT', 'MIN_REPEAT', 'POSSESSIVE_REPEAT'):
            lo, _hi, body = av
            c, k = _first_of_seq(body, ignorecase)
            chars |= c
            classes |= k
            if lo >= 1:
                return chars, classes
            continue  # optional: what follows can also start the match
        if name == 'ATOMIC_GROUP':
            c, k = _first_of_seq(av, ignorecase)
            chars |= c
            classes |= k
            return chars, classes
        raise _Bail  # ANY, NOT_LITERAL, GROUPREF, anything unrecognized
    raise _Bail  # ran off the end: sequence can match empty here


def first_charset_guard(pattern_srcs: List[str], flags: int) -> Optional[str]:
    """Derive a ``(?=[...])`` guard for an alternation of pattern sources.

    Returns None when any branch defeats the analysis; callers must then
    compile the alternation unguarded.
    """
    if _parser is None or not pattern_srcs:
        return None
    ignorecase = bool(flags & re.IGNORECASE)
    chars: Set[str] = set()
    classes: Set[str] = set()
    try:
        for src in pattern_srcs:
            parsed = _parser.parse(src, flags)
            c, k = _first_of_seq(list(parsed), ignorecase)
            chars |= c
            classes |= k
    except _Bail:
        return None
    except Exception:
        # A parser change or exotic pattern must never break compilation;
        # the guard is an optimization, unguarded is always correct.
        return None
    if not chars and not classes:
        return None
    if r'\w' in classes:
        classes.discard(r'\d')
        chars = {c for c in chars if not (c.isalnum() or c == '_')}
    if r'\d' in classes:
        chars = {c for c in chars if not c.isdigit()}
    body = ''.join(sorted(classes)) + ''.join(re.escape(c) for c in sorted(chars))
    return '(?=[' + body + '])'
