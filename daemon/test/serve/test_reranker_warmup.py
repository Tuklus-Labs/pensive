"""Startup must not die because the unused cross-encoder cannot import."""
import serve.daemon as daemon


def test_warm_reranker_fail_open_on_import_error(monkeypatch):
    def boom():
        raise RuntimeError("torchcodec ffmpeg mismatch")

    monkeypatch.setattr("recall.rerank._getReranker", boom)
    assert daemon._warmReranker() is False, (
        "reranker-warmup fail-open rule violated: an unusable cross-encoder "
        "must not prevent the daemon from serving"
    )


def test_warm_reranker_true_when_loader_returns(monkeypatch):
    monkeypatch.setattr("recall.rerank._getReranker", lambda: object())
    assert daemon._warmReranker() is True, (
        "reranker-warmup success rule violated: a working loader must report True"
    )
