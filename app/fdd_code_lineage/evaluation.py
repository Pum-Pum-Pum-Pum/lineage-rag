from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_retrieval.models import CodeEvidence, CodeRetrievalResult
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeCombinedEvalCase(FrozenModel):
    schema_version: Literal[
        "code_combined_eval_case_v1", "code_combined_eval_case_v2"
    ] = (
        "code_combined_eval_case_v1"
    )
    case_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9-]+$")
    mode: Literal["code", "combined"]
    question: str = Field(min_length=10)
    analysis_kind: Literal["explanation", "impact_analysis"] = "explanation"
    expected_claims: tuple[str, ...] = ()
    expected_code_paths: tuple[str, ...] = ()
    expected_code_symbols: tuple[str, ...] = ()
    expected_code_symbol_policy: Literal["all", "any", "advisory"] = "all"
    expected_fdd_document_ids: tuple[str, ...] = ()
    require_reviewed_lineage: bool = False
    should_abstain: bool = False
    expected_unknown_kinds: tuple[
        Literal[
            "parser_degradation",
            "conditional_unknown",
            "kernel_unavailable",
            "dynamic_sql_unknown",
            "external_schema",
            "missing_snapshot",
        ],
        ...,
    ] = ()
    sme_reviewed: bool = False
    review_status: Literal["draft", "reviewed"] = "draft"
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_contract(self) -> "CodeCombinedEvalCase":
        if self.mode == "code" and (
            self.expected_fdd_document_ids or self.require_reviewed_lineage
        ):
            raise ValueError("Code-only cases cannot require FDD evidence or lineage")
        if self.mode == "combined" and not self.should_abstain:
            if not self.expected_fdd_document_ids or not self.expected_code_paths:
                raise ValueError(
                    "Answered combined cases require FDD documents and code paths"
                )
        if self.should_abstain and (
            self.expected_claims
            or self.expected_code_paths
            or self.expected_code_symbols
            or self.expected_fdd_document_ids
            or self.require_reviewed_lineage
        ):
            raise ValueError(
                "Abstention cases must not declare positive evidence expectations"
            )
        if self.review_status == "reviewed" and not self.sme_reviewed:
            raise ValueError("Reviewed cases require sme_reviewed=true")
        if self.sme_reviewed and self.review_status != "reviewed":
            raise ValueError("SME-reviewed cases must use review_status=reviewed")
        return self


class EvidenceIdentity(FrozenModel):
    rank: int = Field(gt=0)
    unit_id: str
    source_path: str
    display_name: str
    parent_unit_id: str | None = None
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    score: float


class CodeCombinedRetrievalCaseReport(FrozenModel):
    case_id: str
    mode: Literal["code", "combined"]
    question: str
    should_abstain: bool
    positive_gate_applicable: bool
    positive_gate_passed: bool | None
    expected_code_paths: tuple[str, ...]
    retrieved_code_paths: tuple[str, ...]
    missing_code_paths: tuple[str, ...]
    expected_code_symbols: tuple[str, ...]
    retrieved_code_symbols: tuple[str, ...]
    missing_code_symbols: tuple[str, ...]
    expected_fdd_document_ids: tuple[str, ...]
    retrieved_fdd_document_ids: tuple[str, ...]
    missing_fdd_document_ids: tuple[str, ...]
    require_reviewed_lineage: bool
    reviewed_mapping_ids: tuple[str, ...]
    code_recall_at_k: float | None
    fdd_recall_at_k: float | None
    code_evidence: tuple[EvidenceIdentity, ...]
    direct_dense_candidates: tuple[EvidenceIdentity, ...] = ()
    direct_lexical_candidates: tuple[EvidenceIdentity, ...] = ()
    mapped_dense_candidates: tuple[EvidenceIdentity, ...] = ()
    mapped_lexical_candidates: tuple[EvidenceIdentity, ...] = ()
    failures: tuple[str, ...]


def load_code_combined_eval_cases(path: str | Path) -> list[CodeCombinedEvalCase]:
    cases: list[CodeCombinedEvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = CodeCombinedEvalCase.model_validate_json(line)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid code/combined evaluation case at line {line_number}: {error}"
            ) from error
        if case.case_id in seen:
            raise ValueError(
                f"Duplicate code/combined evaluation case_id at line {line_number}: "
                f"{case.case_id}"
            )
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation manifest did not contain any cases")
    return cases


def require_reviewed_code_combined_cases(
    cases: Sequence[CodeCombinedEvalCase], *, allow_unreviewed: bool
) -> None:
    unreviewed = [case.case_id for case in cases if not case.sme_reviewed]
    if unreviewed and not allow_unreviewed:
        raise ValueError(
            "Evaluation contains unreviewed cases and is not release-gate eligible: "
            + ", ".join(unreviewed)
        )


def build_code_combined_retrieval_case_report(
    *,
    case: CodeCombinedEvalCase,
    retrieval: CodeRetrievalResult | CombinedRetrievalResult,
) -> CodeCombinedRetrievalCaseReport:
    if retrieval.query != case.question:
        raise ValueError("Retrieval query does not match evaluation case question")
    if isinstance(retrieval, CombinedRetrievalResult):
        if case.mode != "combined":
            raise ValueError("Combined retrieval cannot evaluate a code-only case")
        code_evidence = retrieval.code_evidence
        fdd_document_ids = _unique(item.document_id for item in retrieval.fdd_evidence)
        mapping_ids = tuple(item.mapping_id for item in retrieval.reviewed_lineage)
    else:
        if case.mode != "code":
            raise ValueError("Code retrieval cannot evaluate a combined case")
        code_evidence = retrieval.evidence
        fdd_document_ids = ()
        mapping_ids = ()

    code_paths = _unique(item.source_path for item in code_evidence)
    code_symbols = _unique(item.display_name for item in code_evidence)
    missing_paths = tuple(sorted(set(case.expected_code_paths).difference(code_paths)))
    missing_symbols = tuple(
        sorted(set(case.expected_code_symbols).difference(code_symbols))
    )
    missing_documents = tuple(
        sorted(set(case.expected_fdd_document_ids).difference(fdd_document_ids))
    )
    failures: list[str] = []
    if missing_paths:
        failures.append(f"Missing code paths: {list(missing_paths)}")
    if missing_symbols and case.expected_code_symbol_policy == "all":
        failures.append(f"Missing code symbols: {list(missing_symbols)}")
    if (
        case.expected_code_symbols
        and case.expected_code_symbol_policy == "any"
        and not set(case.expected_code_symbols).intersection(code_symbols)
    ):
        failures.append(
            "None of the alternative expected code symbols were retrieved: "
            f"{list(case.expected_code_symbols)}"
        )
    if missing_documents:
        failures.append(f"Missing FDD document IDs: {list(missing_documents)}")
    if case.require_reviewed_lineage and not mapping_ids:
        failures.append("No reviewed FDD-to-code mapping was followed")

    applicable = not case.should_abstain
    return CodeCombinedRetrievalCaseReport(
        case_id=case.case_id,
        mode=case.mode,
        question=case.question,
        should_abstain=case.should_abstain,
        positive_gate_applicable=applicable,
        positive_gate_passed=not failures if applicable else None,
        expected_code_paths=case.expected_code_paths,
        retrieved_code_paths=code_paths,
        missing_code_paths=missing_paths,
        expected_code_symbols=case.expected_code_symbols,
        retrieved_code_symbols=code_symbols,
        missing_code_symbols=missing_symbols,
        expected_fdd_document_ids=case.expected_fdd_document_ids,
        retrieved_fdd_document_ids=fdd_document_ids,
        missing_fdd_document_ids=missing_documents,
        require_reviewed_lineage=case.require_reviewed_lineage,
        reviewed_mapping_ids=mapping_ids,
        code_recall_at_k=_recall(case.expected_code_paths, code_paths),
        fdd_recall_at_k=_recall(case.expected_fdd_document_ids, fdd_document_ids),
        code_evidence=_summarize_code_items(code_evidence),
        direct_dense_candidates=(
            _summarize_code_items(retrieval.direct_dense_candidates)
            if isinstance(retrieval, CombinedRetrievalResult)
            else _summarize_code_items(retrieval.dense_candidates)
        ),
        direct_lexical_candidates=(
            _summarize_code_items(retrieval.direct_lexical_candidates)
            if isinstance(retrieval, CombinedRetrievalResult)
            else _summarize_code_items(retrieval.lexical_candidates)
        ),
        mapped_dense_candidates=(
            _summarize_code_items(retrieval.mapped_dense_candidates)
            if isinstance(retrieval, CombinedRetrievalResult)
            else ()
        ),
        mapped_lexical_candidates=(
            _summarize_code_items(retrieval.mapped_lexical_candidates)
            if isinstance(retrieval, CombinedRetrievalResult)
            else ()
        ),
        failures=tuple(failures),
    )


def build_code_combined_retrieval_report(
    *,
    metadata: dict[str, Any],
    cases: Sequence[CodeCombinedRetrievalCaseReport],
    minimum_positive_pass_rate: float,
) -> dict[str, Any]:
    if not 0.0 <= minimum_positive_pass_rate <= 1.0:
        raise ValueError("minimum_positive_pass_rate must be between 0 and 1")
    positive = [item for item in cases if item.positive_gate_applicable]
    passed = sum(item.positive_gate_passed is True for item in positive)
    pass_rate = passed / len(positive) if positive else 0.0
    reviewed = bool(metadata.get("reviewed_manifest"))
    return {
        "schema_version": "code_combined_retrieval_gate_v1",
        "metadata": metadata,
        "summary": {
            "total_cases": len(cases),
            "positive_cases": len(positive),
            "abstention_diagnostic_cases": len(cases) - len(positive),
            "positive_cases_passed": passed,
            "positive_case_pass_rate": pass_rate,
            "minimum_positive_pass_rate": minimum_positive_pass_rate,
            "retrieval_threshold_passed": pass_rate >= minimum_positive_pass_rate,
            "reviewed_manifest": reviewed,
            "release_gate_eligible": reviewed
            and pass_rate >= minimum_positive_pass_rate,
        },
        "cases": [item.model_dump(mode="json") for item in cases],
    }


def write_json_report_no_overwrite(report: dict[str, Any], path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Evaluation report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _unique(values: Sequence[str] | Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _recall(expected: Sequence[str], retrieved: Sequence[str]) -> float | None:
    if not expected:
        return None
    return len(set(expected).intersection(retrieved)) / len(set(expected))


def _summarize_code_items(items: Sequence[Any]) -> tuple[EvidenceIdentity, ...]:
    return tuple(
        EvidenceIdentity(
            rank=rank,
            unit_id=item.unit_id,
            source_path=item.source_path,
            display_name=item.display_name,
            parent_unit_id=item.parent_unit_id,
            start_line=item.start_line,
            end_line=item.end_line,
            score=item.score,
        )
        for rank, item in enumerate(items, start=1)
    )
