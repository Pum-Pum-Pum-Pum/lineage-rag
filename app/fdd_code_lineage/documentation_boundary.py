"""Reviewed diagnostics for visible code without an approved FDD/code mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentationBoundaryCase(FrozenModel):
    schema_version: Literal["code_documentation_boundary_case_v1"] = (
        "code_documentation_boundary_case_v1"
    )
    case_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9-]+$")
    question: str = Field(min_length=10)
    expected_code_paths: tuple[str, ...] = Field(min_length=1)
    expected_code_symbols: tuple[str, ...] = Field(min_length=1)
    expected_code_symbol_policy: Literal["all", "any"] = "all"
    expected_documentation_state: Literal["no_reviewed_fdd_lineage"] = (
        "no_reviewed_fdd_lineage"
    )
    sme_reviewed: bool = False
    review_status: Literal["draft", "reviewed"] = "draft"
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_review_status(self) -> "DocumentationBoundaryCase":
        if self.review_status == "reviewed" and not self.sme_reviewed:
            raise ValueError("Reviewed boundary cases require sme_reviewed=true")
        if self.sme_reviewed and self.review_status != "reviewed":
            raise ValueError("SME-reviewed boundary cases must use review_status=reviewed")
        return self


class DocumentationBoundaryCaseReport(FrozenModel):
    case_id: str
    question: str
    expected_code_paths: tuple[str, ...]
    retrieved_code_paths: tuple[str, ...]
    expected_code_symbols: tuple[str, ...]
    retrieved_code_symbols: tuple[str, ...]
    fdd_candidate_document_ids: tuple[str, ...]
    fdd_evidence_status: Literal["none", "unreviewed_candidate"]
    reviewed_mapping_ids: tuple[str, ...]
    documentation_state: Literal["no_reviewed_fdd_lineage"]
    passed: bool
    failures: tuple[str, ...]


def load_documentation_boundary_cases(path: Path) -> list[DocumentationBoundaryCase]:
    cases: list[DocumentationBoundaryCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = DocumentationBoundaryCase.model_validate_json(line)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid documentation-boundary case at line {line_number}: {error}"
            ) from error
        if case.case_id in seen:
            raise ValueError(f"Duplicate documentation-boundary case ID: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Documentation-boundary manifest did not contain any cases")
    return cases


def require_reviewed_documentation_boundary_cases(
    cases: Sequence[DocumentationBoundaryCase], *, allow_unreviewed: bool
) -> None:
    if allow_unreviewed:
        return
    unreviewed = [case.case_id for case in cases if not case.sme_reviewed]
    if unreviewed:
        raise ValueError(
            "Documentation-boundary evaluation contains unreviewed cases: "
            + ", ".join(unreviewed)
        )


def build_documentation_boundary_case_report(
    *, case: DocumentationBoundaryCase, retrieval: CombinedRetrievalResult
) -> DocumentationBoundaryCaseReport:
    if retrieval.query != case.question:
        raise ValueError("Retrieval query does not match documentation-boundary case")
    paths = _unique(item.source_path for item in retrieval.code_evidence)
    symbols = _unique(item.display_name for item in retrieval.code_evidence)
    missing_paths = tuple(sorted(set(case.expected_code_paths).difference(paths)))
    matched_symbols = tuple(
        symbol
        for symbol in case.expected_code_symbols
        if symbol.casefold() in {retrieved.casefold() for retrieved in symbols}
    )
    missing_symbols = tuple(
        symbol for symbol in case.expected_code_symbols if symbol not in matched_symbols
    )
    failures: list[str] = []
    if missing_paths:
        failures.append(f"Missing code paths: {list(missing_paths)}")
    if case.expected_code_symbol_policy == "all" and missing_symbols:
        failures.append(f"Missing code symbols: {list(missing_symbols)}")
    if case.expected_code_symbol_policy == "any" and not (
        matched_symbols
    ):
        failures.append(
            "None of the expected code symbols were retrieved: "
            f"{list(case.expected_code_symbols)}"
        )
    # A combined response can legitimately contain an FDD document which has a
    # reviewed relationship to *other* code in the corpus.  The boundary is
    # about the requested code path, so only a mapping whose selected units
    # overlap that path can contradict a no-reviewed-lineage assertion.
    expected_unit_ids = {
        item.unit_id
        for item in retrieval.code_evidence
        if item.source_path in set(case.expected_code_paths)
    }
    mapping_ids = tuple(
        item.mapping_id
        for item in retrieval.reviewed_lineage
        if expected_unit_ids.intersection(item.code_unit_ids)
    )
    if mapping_ids:
        failures.append(
            "A reviewed FDD-to-code mapping was returned for a no-reviewed-lineage case"
        )
    fdd_ids = _unique(item.document_id for item in retrieval.fdd_evidence)
    return DocumentationBoundaryCaseReport(
        case_id=case.case_id,
        question=case.question,
        expected_code_paths=case.expected_code_paths,
        retrieved_code_paths=paths,
        expected_code_symbols=case.expected_code_symbols,
        retrieved_code_symbols=symbols,
        fdd_candidate_document_ids=fdd_ids,
        fdd_evidence_status="unreviewed_candidate" if fdd_ids else "none",
        reviewed_mapping_ids=mapping_ids,
        documentation_state="no_reviewed_fdd_lineage",
        passed=not failures,
        failures=tuple(failures),
    )


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))
