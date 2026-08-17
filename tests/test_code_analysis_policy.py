from __future__ import annotations

from pathlib import Path

import pytest

from app.code_ingestion.analysis_policy import load_code_analysis_policy


def test_versioned_policy_normalizes_boundaries_and_has_stable_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    content = """schema_version = "code_analysis_policy_v3"
[boundaries]
custom_program_unit_suffixes = ["_custom", "_main"]
infer_noncustom_qualified_packages_as_kernel = true
kernel_package_names = ["kernel_claim"]
kernel_package_prefixes = ["kernel_"]
external_package_prefixes = ["dbms_"]
ignored_builtin_calls = ["nvl"]
"""
    first.write_text(content, encoding="utf-8")
    second.write_text(content.replace('["kernel_"]', '["KERNEL_"]'), encoding="utf-8")

    first_policy = load_code_analysis_policy(first)
    second_policy = load_code_analysis_policy(second)

    assert first_policy.boundaries.kernel_package_prefixes == ("KERNEL_",)
    assert first_policy.boundaries.custom_program_unit_suffixes == ("_CUSTOM", "_MAIN")
    assert first_policy.sha256 == second_policy.sha256


def test_duplicate_or_unknown_policy_entries_fail_closed(tmp_path: Path) -> None:
    policy = tmp_path / "invalid.toml"
    policy.write_text(
        """schema_version = "code_analysis_policy_v3"
unexpected = true
[boundaries]
custom_program_unit_suffixes = ["_CUSTOM", "_custom"]
infer_noncustom_qualified_packages_as_kernel = true
kernel_package_names = []
kernel_package_prefixes = ["KERNEL_", "kernel_"]
external_package_prefixes = []
ignored_builtin_calls = []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_code_analysis_policy(policy)


def test_missing_policy_fails_before_analysis(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_code_analysis_policy(tmp_path / "missing.toml")
