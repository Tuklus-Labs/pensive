"""REFACTOR-4: broad-catch fallbacks must keep tracebacks recoverable.

The hybrid layer's degradation boundaries (SA query, L2 query, candidate
L2 query, cross-encoder rerank) catch Exception and return a degraded
result. That is the right call for a daily-ops engine, but a one-line
warning with no traceback makes an unexpected KeyError from a refactor
indistinguishable from an expected backend hiccup. The contract: at
WARNING level the log stays one line; when the logger is enabled for
DEBUG, the warning carries exc_info so the traceback is recoverable.
"""
import logging

import pytest

from pensive.parallel_hybrid import ParallelHybrid


class _ExplodingL2:
    def query(self, query, top_k=None):
        raise KeyError("internal-bug-marker")


@pytest.fixture
def hybrid():
    return ParallelHybrid(
        spreading_activation=None,
        l2_handler=_ExplodingL2(),
        enable_pattern_learning=False,
    )


def test_fallback_logs_traceback_at_debug(hybrid, caplog):
    with caplog.at_level(logging.DEBUG, logger="pensive.parallel_hybrid"):
        assert hybrid._query_l2("anything", top_k=5) == []

    records = [r for r in caplog.records if "L2 query failed" in r.message]
    assert records, "expected a 'L2 query failed' warning record"
    assert records[0].exc_info is not None, (
        "at DEBUG level the degradation warning must carry exc_info so "
        "the traceback is recoverable"
    )
    assert "internal-bug-marker" in str(records[0].exc_info[1])


def test_fallback_stays_one_line_at_warning(hybrid, caplog):
    with caplog.at_level(logging.WARNING, logger="pensive.parallel_hybrid"):
        assert hybrid._query_l2("anything", top_k=5) == []

    records = [r for r in caplog.records if "L2 query failed" in r.message]
    assert records, "expected a 'L2 query failed' warning record"
    assert not records[0].exc_info, (
        "at WARNING level the degradation log must stay a one-liner "
        "(no traceback spam in production logs)"
    )
