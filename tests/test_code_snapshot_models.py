from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.code_ingestion.snapshot_models import SnapshotRequest


def test_snapshot_request_normalizes_expected_paths_and_is_frozen() -> None:
    request = SnapshotRequest(
        module_set="fci-custom",
        svn_revision="12345",
        application_build="14.7.1",
        reviewer="SME",
        expected_changed_packages=("packages\\pkg_customer.prc",),
    )

    assert request.expected_changed_packages == ("packages/pkg_customer.prc",)
    with pytest.raises(ValidationError, match="frozen"):
        request.svn_revision = "12346"  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_path",
    ["../outside.sql", "/absolute.sql", "C:\\outside.sql", ""],
)
def test_snapshot_request_rejects_unsafe_expected_paths(invalid_path: str) -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        SnapshotRequest(
            module_set="fci-custom",
            svn_revision="12345",
            application_build="14.7.1",
            reviewer="SME",
            expected_changed_packages=(invalid_path,),
        )


def test_snapshot_request_rejects_duplicate_case_insensitive_paths() -> None:
    with pytest.raises(ValidationError, match="duplicate paths"):
        SnapshotRequest(
            module_set="fci-custom",
            svn_revision="12345",
            application_build="14.7.1",
            reviewer="SME",
            expected_changed_packages=("PKG_A.PRC", "pkg_a.prc"),
        )


def test_snapshot_request_rejects_unknown_contract_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SnapshotRequest.model_validate(
            {
                "module_set": "fci-custom",
                "svn_revision": "12345",
                "application_build": "14.7.1",
                "reviewer": "SME",
                "unreviewed_override": True,
            }
        )

