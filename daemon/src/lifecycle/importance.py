"""Recover explicit usefulness credits. Retrieval frequency is not usefulness."""
from store.feedback import processPendingFeedback

__all__ = ['accrueImportance']


def accrueImportance(store):
    """Process a bounded batch of helpful feedback, never legacy recall logs.

    Normal feedback credits immediately in its write transaction. This entry
    point also handles pending imported events without awarding twice.
    """
    return processPendingFeedback(store)
