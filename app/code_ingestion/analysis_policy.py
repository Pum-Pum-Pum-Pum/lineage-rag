from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


DEFAULT_CODE_ANALYSIS_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "code_analysis.toml"


class AnalysisBoundaries(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    custom_program_unit_suffixes: tuple[str, ...] = ("_CUSTOM", "_MAIN")
    infer_noncustom_qualified_packages_as_kernel: bool = False
    kernel_program_unit_suffixes: tuple[str, ...] = ("_KERNEL",)
    kernel_package_names: tuple[str, ...] = ()
    kernel_package_prefixes: tuple[str, ...] = ()
    external_package_prefixes: tuple[str, ...] = ("DBMS_", "UTL_")
    external_object_type_names: tuple[str, ...] = (
        "JSON_ARRAY_T",
        "JSON_ELEMENT_T",
        "JSON_OBJECT_T",
    )
    infrastructure_utility_calls: tuple[str, ...] = ()
    ignored_builtin_calls: tuple[str, ...] = ()

    @field_validator(
        "custom_program_unit_suffixes",
        "kernel_program_unit_suffixes",
        "kernel_package_names",
        "kernel_package_prefixes",
        "external_package_prefixes",
        "external_object_type_names",
        "infrastructure_utility_calls",
        "ignored_builtin_calls",
    )
    @classmethod
    def normalize_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip().upper() for value in values)
        if any(not value for value in normalized):
            raise ValueError("Analysis boundary entries must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Analysis boundary entries must be unique")
        return normalized


class CodeAnalysisPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["code_analysis_policy_v5"] = "code_analysis_policy_v5"
    boundaries: AnalysisBoundaries

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_code_analysis_policy(path: Path = DEFAULT_CODE_ANALYSIS_POLICY_PATH) -> CodeAnalysisPolicy:
    if not path.is_file():
        raise FileNotFoundError(f"Code analysis policy not found: {path}")
    with path.open("rb") as handle:
        return CodeAnalysisPolicy.model_validate(tomllib.load(handle))
