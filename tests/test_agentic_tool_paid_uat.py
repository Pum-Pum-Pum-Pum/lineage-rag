from __future__ import annotations

from app.agentic_tools.evaluation import execute_local_lexical_tools
from app.agentic_tools.paid_uat import build_paid_case, retrieval_from_local_uat
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.uat import ManualToolUatCase, build_local_uat_report
from app.code_retrieval.models import CodeRetrievalResult
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult
from tests.test_agentic_tool_evaluation import _code_artifact, _fdd_documents, _lineage


def _case(mode: str) -> ManualToolUatCase:
    return ManualToolUatCase(
        case_id=f"paid-{mode}-case-001",
        source_reviewed_case_id="source-reviewed-001",
        knowledge_mode=mode,
        question="Explain the reviewed AML transaction integration evidence.",
        limit=5,
        expected_outcome="evidence",
        expected_fdd_document_ids=("FDD-AML-R24",) if mode != "code" else (),
        expected_code_paths=("pkg_aml_custom.sql",) if mode != "fdd" else (),
        expected_code_symbols=("SP_PROCESS_AML",) if mode != "fdd" else (),
        require_reviewed_lineage=mode == "combined",
        review_status="reviewed",
        sme_reviewed=True,
        rationale="Tests conversion from preserved local evidence into paid prompt input.",
    )


def _report(case: ManualToolUatCase):
    artifact = _code_artifact()
    lineage = _lineage(artifact)
    policy = load_agentic_tools_policy()
    execution = execute_local_lexical_tools(
        knowledge_mode=case.knowledge_mode,
        question=case.question,
        limit=case.limit,
        policy=policy,
        fdd_documents=_fdd_documents(),
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        lineage_artifact=lineage,
    )
    return artifact, build_local_uat_report(
        knowledge_mode=case.knowledge_mode,
        question=case.question,
        fdd_generation="functional_specs_v5",
        code_snapshot_id=artifact.snapshot_id,
        lineage_artifact_identity_sha256=lineage.artifact_identity_sha256,
        policy_sha256=policy.sha256,
        execution=execution,
    )


def test_paid_uat_reuses_code_evidence_without_query_embedding() -> None:
    case = _case("code")
    artifact, report = _report(case)
    retrieval = retrieval_from_local_uat(case=case, report=report, artifact=artifact)
    assert isinstance(retrieval, CodeRetrievalResult)
    assert retrieval.mode == "lexical"
    assert retrieval.evidence[0].display_name == "SP_PROCESS_AML"
    assert build_paid_case(case).analysis_kind == "explanation"


def test_paid_uat_preserves_combined_lineage_and_fdd_only_boundary() -> None:
    combined_case = _case("combined")
    artifact, combined_report = _report(combined_case)
    combined = retrieval_from_local_uat(
        case=combined_case, report=combined_report, artifact=artifact
    )
    assert isinstance(combined, CombinedRetrievalResult)
    assert combined.reviewed_lineage
    assert combined.code_evidence

    fdd_case = _case("fdd")
    artifact, fdd_report = _report(fdd_case)
    fdd = retrieval_from_local_uat(case=fdd_case, report=fdd_report, artifact=artifact)
    assert isinstance(fdd, CombinedRetrievalResult)
    assert fdd.fdd_evidence
    assert not fdd.code_evidence
    assert "did not request" in fdd.unknowns[0]
