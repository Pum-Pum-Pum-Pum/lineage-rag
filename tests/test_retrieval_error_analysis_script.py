import subprocess
import sys


def test_retrieval_error_analysis_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_retrieval_error_analysis.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--comparison-report" in result.stdout
    assert "--include-both-pass" in result.stdout