from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.code_retrieval.models import CodeEvidence, CodeRetrievalResult
from app.fdd_code_lineage.combined_retrieval import (
    CombinedRetrievalResult,
    FddEvidence,
    ReviewedLineageUse,
)
from app.fdd_code_lineage.evaluation import (
    CodeCombinedEvalCase,
    build_code_combined_retrieval_case_report,
    build_code_combined_retrieval_report,
    load_code_combined_eval_cases,
    require_reviewed_code_combined_cases,
    write_json_report_no_overwrite,
)


def test_eval_contract_separates_code_and_combined_expectations() -> None:
    with pytest.raises(ValidationError, match="Code-only cases cannot require FDD"):
        _case(mode="code", expected_fdd_document_ids=("doc-1",))
    with pytest.raises(ValidationError, match="require FDD documents and code paths"):
        _case(mode="combined", expected_code_paths=())
    with pytest.raises(ValidationError, match="must not declare positive evidence"):
        _case(mode="code", should_abstain=True)


def test_manifest_loader_rejects_duplicate_ids_and_unreviewed_gate(tmp_path: Path) -> None:
    case = _case()
    path = tmp_path / "cases.jsonl"
    line = case.model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_code_combined_eval_cases(path)
    with pytest.raises(ValueError, match="unreviewed"):
        require_reviewed_code_combined_cases([case], allow_unreviewed=False)
    require_reviewed_code_combined_cases([case], allow_unreviewed=True)


def test_combined_gate_requires_both_lanes_and_reviewed_mapping() -> None:
    case = _case(mode="combined")
    passing = build_code_combined_retrieval_case_report(
        case=case,
        retrieval=_combined_retrieval(case.question),
    )
    assert passing.positive_gate_passed is True
    assert passing.code_recall_at_k == 1.0
    assert passing.fdd_recall_at_k == 1.0

    failed_retrieval = _combined_retrieval(case.question).model_copy(
        update={"fdd_evidence": (), "reviewed_lineage": ()}
    )
    failed = build_code_combined_retrieval_case_report(
        case=case, retrieval=failed_retrieval
    )
    assert failed.positive_gate_passed is False
    assert "No reviewed FDD-to-code mapping" in failed.failures[-1]


def test_code_gate_fails_on_nearby_wrong_symbol() -> None:
    case = _case(mode="code")
    wrong = _code_retrieval(case.question).model_copy(
        update={
            "evidence": (
                _evidence().model_copy(update={"display_name": "nearby_routine"}),
            )
        }
    )
    report = build_code_combined_retrieval_case_report(case=case, retrieval=wrong)
    assert report.positive_gate_passed is False
    assert report.missing_code_symbols == ("process_aml",)


def test_abstention_is_diagnostic_and_draft_cannot_be_release_gate() -> None:
    case = _case(
        mode="code",
        should_abstain=True,
        expected_code_paths=(),
        expected_code_symbols=(),
        expected_claims=(),
        expected_unknown_kinds=("kernel_unavailable",),
    )
    case_report = build_code_combined_retrieval_case_report(
        case=case, retrieval=_code_retrieval(case.question)
    )
    report = build_code_combined_retrieval_report(
        metadata={"reviewed_manifest": False},
        cases=[case_report],
        minimum_positive_pass_rate=0.9,
    )
    assert case_report.positive_gate_applicable is False
    assert report["summary"]["release_gate_eligible"] is False


def test_report_write_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json_report_no_overwrite({"ok": True}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(FileExistsError):
        write_json_report_no_overwrite({"ok": False}, path)


def _case(
    *,
    mode: str = "code",
    expected_claims: tuple[str, ...] = ("Explain the flow.",),
    expected_code_paths: tuple[str, ...] = ("pkgaml_custom.sql",),
    expected_code_symbols: tuple[str, ...] = ("process_aml",),
    expected_fdd_document_ids: tuple[str, ...] = (),
    should_abstain: bool = False,
    expected_unknown_kinds: tuple[str, ...] = (),
) -> CodeCombinedEvalCase:
    if mode == "combined" and not expected_fdd_document_ids:
        expected_fdd_document_ids = ("doc-aml",)
    return CodeCombinedEvalCase(
        case_id=f"{mode}-case-001",
        mode=mode,
        question="How does the approved AML integration work?",
        expected_claims=expected_claims,
        expected_code_paths=expected_code_paths,
        expected_code_symbols=expected_code_symbols,
        expected_fdd_document_ids=expected_fdd_document_ids,
        require_reviewed_lineage=mode == "combined" and not should_abstain,
        should_abstain=should_abstain,
        expected_unknown_kinds=expected_unknown_kinds,
        rationale="This case checks the approved visible AML integration path.",
    )


def _evidence() -> CodeEvidence:
    return CodeEvidence(
        unit_id="unit-aml",
        point_id="point-aml",
        score=0.9,
        retrieval_method="lexical",
        snapshot_id="snapshot-1",
        module_id="fci-custom",
        source_path="pkgaml_custom.sql",
        source_kind="procedure",
        display_name="process_aml",
        package_name="PKG_AML_CUSTOM",
        start_line=10,
        end_line=20,
        parser_state="full_parse",
        conditional_state="unconditional",
        text="PROCEDURE process_aml IS BEGIN NULL; END;",
    )


def _code_retrieval(query: str) -> CodeRetrievalResult:
    return CodeRetrievalResult(
        query=query,
        mode="lexical",
        snapshot_id="snapshot-1",
        artifact_identity_sha256="a" * 64,
        evidence=(_evidence(),),
    )


def _combined_retrieval(query: str) -> CombinedRetrievalResult:
    return CombinedRetrievalResult(
        query=query,
        fdd_generation="functional_specs_v5",
        code_snapshot_id="snapshot-1",
        fdd_evidence=(
            FddEvidence(
                unit_id="fdd-unit",
                document_id="doc-aml",
                document_family="neo-aml",
                release_label="R22",
                source_kind="paragraph",
                score=0.8,
                text="The system sends AML data.",
            ),
        ),
        code_evidence=(_evidence(),),
        direct_code_evidence=(_evidence(),),
        mapped_code_evidence=(_evidence(),),
        reviewed_lineage=(
            ReviewedLineageUse(
                mapping_id="b" * 64,
                fdd_document_id="doc-aml",
                code_unit_ids=("unit-aml",),
            ),
        ),
    )
