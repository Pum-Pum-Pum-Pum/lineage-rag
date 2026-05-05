import subprocess
import sys


def test_retrieval_comparison_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_retrieval_comparison.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--lexical-artifact-dir" in result.stdout
    assert "--output-file" in result.stdout