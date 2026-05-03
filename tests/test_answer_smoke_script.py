import subprocess
import sys


def test_answer_smoke_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_answer_smoke_test.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--query" in result.stdout
    assert "--min-top-score" in result.stdout
