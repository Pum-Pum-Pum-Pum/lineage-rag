import subprocess
import sys

from scripts.run_qdrant_query_search import _requires_qdrant_collection


def test_qdrant_query_search_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_qdrant_query_search.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--min-top-score" in result.stdout
    assert "configured retrieval mode" in result.stdout


def test_qdrant_query_search_collection_requirement_depends_on_retrieval_mode() -> None:
    assert _requires_qdrant_collection("dense") is True
    assert _requires_qdrant_collection("hybrid") is True
    assert _requires_qdrant_collection("lexical") is False
