import subprocess
import sys


def test_qdrant_query_search_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_qdrant_query_search.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--min-top-score" in result.stdout
