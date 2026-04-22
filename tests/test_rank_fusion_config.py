"""Tests for ParallelHybrid rank-fusion weight configurability."""
import pytest

from pensive.parallel_hybrid import ParallelHybrid


def test_default_weights_match_documented_constants():
    h = ParallelHybrid()
    assert h.rank_fusion_agreement == 100.0
    assert h.rank_fusion_l2_only == 50.0
    assert h.rank_fusion_sa_only == 30.0


def test_override_weights_applied():
    h = ParallelHybrid(
        rank_fusion_agreement=200.0,
        rank_fusion_l2_only=80.0,
        rank_fusion_sa_only=40.0,
    )
    assert h.rank_fusion_agreement == 200.0
    assert h.rank_fusion_l2_only == 80.0
    assert h.rank_fusion_sa_only == 40.0


def test_partial_override_keeps_defaults_for_others():
    h = ParallelHybrid(rank_fusion_agreement=150.0)
    assert h.rank_fusion_agreement == 150.0
    assert h.rank_fusion_l2_only == 50.0
    assert h.rank_fusion_sa_only == 30.0


def test_invariant_violation_rejects_equal_weights():
    # agreement must be strictly greater than l2_only
    with pytest.raises(ValueError, match="agreement > l2_only"):
        ParallelHybrid(
            rank_fusion_agreement=50.0,
            rank_fusion_l2_only=50.0,
            rank_fusion_sa_only=30.0,
        )


def test_invariant_violation_rejects_inverted_order():
    with pytest.raises(ValueError, match="agreement > l2_only > sa_only"):
        ParallelHybrid(
            rank_fusion_agreement=20.0,
            rank_fusion_l2_only=50.0,
            rank_fusion_sa_only=80.0,
        )


def test_class_level_defaults_can_be_monkeypatched():
    """Users can subclass and change defaults without passing per-instance."""
    class Tuned(ParallelHybrid):
        RANK_FUSION_AGREEMENT = 500.0
        RANK_FUSION_L2_ONLY = 100.0
        RANK_FUSION_SA_ONLY = 50.0

    h = Tuned()
    assert h.rank_fusion_agreement == 500.0
    assert h.rank_fusion_l2_only == 100.0
    assert h.rank_fusion_sa_only == 50.0
