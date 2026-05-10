import subprocess
import sys


def test_hybrid_weight_experiments_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_hybrid_weight_experiments.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--weights" in result.stdout
    assert "--candidate-limit" in result.stdout