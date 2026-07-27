from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    return CI_WORKFLOW_PATH.read_text(encoding="utf-8")


def test_ci_runs_for_pushes_and_pull_requests_with_minimal_permissions() -> None:
    workflow = _workflow_text()

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "timeout-minutes: 15" in workflow


def test_ci_uses_pinned_uv_and_project_python() -> None:
    workflow = _workflow_text()

    assert "actions/checkout@v7" in workflow
    assert (
        "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
        in workflow
    )
    assert 'version: "0.11.32"' in workflow
    assert 'python-version: "3.12"' in workflow
    assert 'cache-dependency-glob: "uv.lock"' in workflow


def test_ci_rejects_lock_drift_and_runs_only_locked_project_commands() -> None:
    workflow = _workflow_text()

    assert "run: uv lock --check" in workflow
    assert "run: uv sync --locked --dev" in workflow
    assert "run: uv run --locked pytest -q" in workflow
    assert "${{ secrets." not in workflow
