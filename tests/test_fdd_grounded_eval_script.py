import subprocess
import sys


def test_fdd_grounded_eval_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_fdd_grounded_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--allow-unreviewed" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--case-id" in result.stdout
