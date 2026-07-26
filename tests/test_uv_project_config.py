from pathlib import Path


def test_uv_project_declares_runtime_and_dev_dependencies() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.12,<3.13"' in pyproject
    assert '"fastapi>=0.140,<1"' in pyproject
    assert '"httpx>=0.28,<1"' in pyproject
    assert '"openai>=2.48,<3"' in pyproject
    assert '"pydantic>=2.13,<3"' in pyproject
    assert '"pydantic-settings>=2.14,<3"' in pyproject
    assert '"python-docx>=1.2,<2"' in pyproject
    assert '"qdrant-client>=1.18,<2"' in pyproject
    assert '"streamlit>=1.60,<2"' in pyproject
    assert '"uvicorn>=0.51,<1"' in pyproject
    assert "[dependency-groups]" in pyproject
    assert '"pytest>=9.1,<10"' in pyproject
    assert '"pandas"' not in pyproject


def test_uv_project_commits_lock_and_python_version() -> None:
    assert Path("uv.lock").is_file()
    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_requirements_file_is_not_a_second_dependency_source() -> None:
    assert not Path("requirements.txt").exists()


def test_readme_documents_locked_uv_workflow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "uv sync --locked" in readme
    assert "uv run --locked pytest -q" in readme
    assert "uv lock --check" in readme
    assert "uv sync --locked --no-dev" in readme
    assert "uv export --locked --no-dev" in readme
