# Loudness Audit: bounded derived index snapshot cache

AST-style review of `test_index_cache.py` found 49 assertion statements. Every
assertion has an explicit message that names the fingerprint, source/hash order,
private permission, atomic publication, retention, namespace isolation, loader
failure, or native round-trip rule and includes the relevant path, hash, count,
mode, or returned value. The messages are present-tense and distinct enough to
grep from a failed test run.

The `pytest.importorskip("usearch")` guard is an intentional environment gate:
the real native integration test runs when USearch is installed and skips when
the optional daemon dependency is unavailable; unit coverage remains active.

Exemptions: none.
