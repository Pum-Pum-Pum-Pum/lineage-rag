import json
import zipfile
from pathlib import Path

from app.deployment.native_package import build_native_package


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "app").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "deployment").mkdir()
    (root / "config").mkdir()
    for path, content in {
        ".env.example": "OPENAI_API_KEY=\n",
        "README.md": "runtime",
        "config/ingestion_sources.toml": "schema_version = 'ingestion_source_policy_v1'\n",
        "config/code_analysis.toml": "schema_version = 'code_analysis_policy_v1'\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
        "app/main.py": "print('app')\n",
        "scripts/start.py": "print('script')\n",
        "deployment/native_runtime.json": "{}\n",
    }.items():
        target = root / path
        target.write_text(content, encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "private.docx").write_bytes(b"private")
    return root


def test_native_package_is_deterministic_and_excludes_state_and_secrets(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    first = build_native_package(
        project_root=root,
        output_path=tmp_path / "first.zip",
    )
    second = build_native_package(
        project_root=root,
        output_path=tmp_path / "second.zip",
    )

    assert first.archive_sha256 == second.archive_sha256
    with zipfile.ZipFile(first.archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(
            archive.read("deployment/package_manifest.json")
        )

    assert ".env.example" in names
    assert "config/ingestion_sources.toml" in names
    assert "config/code_analysis.toml" in names
    assert ".env" not in names
    assert "data/private.docx" not in names
    assert "app/main.py" in names
    assert manifest["python_version"] == "3.12"
    assert manifest["dependency_lock"] == "uv.lock"
    assert manifest["mutable_state_bundled"] is False
    assert {item["path"] for item in manifest["files"]} == (
        names - {"deployment/package_manifest.json"}
    )
