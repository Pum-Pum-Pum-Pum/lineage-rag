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

    kernel_package_prefixes: tuple[str, ...] = ()
    external_package_prefixes: tuple[str, ...] = ("DBMS_", "UTL_")
    ignored_builtin_calls: tuple[str, ...] = ()

    @field_validator(
        "kernel_package_prefixes",
        "external_package_prefixes",
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

    schema_version: Literal["code_analysis_policy_v1"] = "code_analysis_policy_v1"
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
