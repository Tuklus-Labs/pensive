"""Pure helpers for the Phase A corpus repair (parsing, ref mapping)."""
import sys
from pathlib import Path

# daemon/tools is not a package on sys.path by default; add it the same way
# the tools themselves add daemon/src (parents: tools -> test -> daemon).
_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from repair_lib import parseFilesSummary, abspathToRef, refToProject


def test_parseFilesSummary_extracts_path():
    s = ("[files] file /home/aegis/Projects/mission-control/dashboard_metrics.go "
         "Created dashboard_metrics.go")
    assert parseFilesSummary(s) == \
        "/home/aegis/Projects/mission-control/dashboard_metrics.go"


def test_parseFilesSummary_rejects_non_files_summaries():
    assert parseFilesSummary("[claude] on GPU-context-creation something") is None
    assert parseFilesSummary("[claude] [Narrative: pensive] blah") is None
    assert parseFilesSummary("") is None
    assert parseFilesSummary(None) is None


def test_parseFilesSummary_rejects_relative_path():
    assert parseFilesSummary("[files] file not/an/abs/path Created x") is None


def test_abspathToRef_projects_root():
    ref, project = abspathToRef(
        "/home/aegis/Projects/mission-control/dashboard_metrics.go")
    assert ref == "projects/mission-control/dashboard_metrics.go"
    assert project == "mission-control"


def test_abspathToRef_claude_and_codex_home():
    assert abspathToRef("/home/aegis/.claude/hooks/emit.py") == \
        ("claude-home/hooks/emit.py", None)
    assert abspathToRef("/home/aegis/.codex/config.toml") == \
        ("codex-home/config.toml", None)


def test_abspathToRef_unknown_root_returns_none():
    assert abspathToRef("/etc/passwd") is None
    assert abspathToRef("/home/aegis/Downloads/x.bin") is None


def test_abspathToRef_bare_file_under_projects_returns_none_project():
    result = abspathToRef("/home/aegis/Projects/barefile.go")
    assert result is not None
    ref, project = result
    assert ref == "projects/barefile.go"
    assert project is None


def test_refToProject_variants():
    assert refToProject("projects/obol/internal/api/x.go#c11") == "obol"
    assert refToProject("reference-library/53-hardware.md#c2") == "Aegis"
    assert refToProject("claude-home/hooks/x.py") is None
    assert refToProject("kv_cache/vector_meta.db#rowid=7") is None
