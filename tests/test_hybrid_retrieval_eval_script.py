import subprocess
import sys


def test_hybrid_retrieval_eval_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_hybrid_retrieval_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dense-weight" in result.stdout
    assert "--lexical-weight" in result.stdout